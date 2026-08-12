"""
参数网格实验运行器。

每轮实验只允许改动一个维度：按 2-3 条/天的发布节奏，一周约 20 个样本，
两分支已经只有 10 个样本，再多分支就完全淹没在 Shorts 的分发方差里。
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import subprocess
import sys
import tempfile
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Sequence

import yaml

MAX_VARIANTS = 3
REQUIRED_KEYS = ("name", "base", "subjects", "grid")

DEFAULT_RESULTS_PATH = "experiments/results.jsonl"
VALID_OUTCOMES = ("hit", "mid", "flop")

# 随仓库发布的示例 spec 带着占位符（例如 voice_name 里的 elevenlabs 声音 id）。
# load_spec 不校验它：声音 id 是运行期问题，装载期报错会让示例文件根本读不出来。
# 但 run 必须在第一个子进程之前拦下，否则整轮视频都用坏掉的旁白生成。
PLACEHOLDER_MARKER = "REPLACE_ME"

# ai_image_* 参数通过 config.toml 生效，不是 cli.py 的入参，因此不映射成命令行标志，
# 改为经环境变量传给子进程（见 build_subprocess_env）。
_NON_CLI_PREFIXES = ("ai_image_",)

# upload_post 发布到 YouTube 时会把标题截断到 100 字符（upload_post.py:64）。
# results.jsonl 里存的是人工回查用的键，必须与用户在 YouTube 上看到的完全一致，
# 否则超长标题永远匹配不上 mark。
PUBLISHED_TITLE_MAX = 100

AIIMAGE_SOURCE = "aiimage"
# AI 图片的分镜是按叙事顺序规划的（plan_prompts 一次性看完整脚本才排得出这个
# 顺序），generate_clips 也按序返回。而 video_concat_mode 默认是 random，
# video.py 会把片段打乱——神话就被讲乱了。aiimage 只能配顺序拼接。
REQUIRED_CONCAT_MODE = "sequential"


class SpecError(ValueError):
    """实验配置不合法。"""


def _grid_values(grid: Any, axis: str) -> list:
    """取某个网格轴的取值列表；轴不存在或类型不对时返回空列表。"""
    if not isinstance(grid, dict):
        return []
    values = grid.get(axis)
    return list(values) if isinstance(values, list) else []


def _declared_values(base: Any, grid: Any, key: str) -> list:
    """一个参数在 spec 里的全部取值来源：base 里写死的，加上网格轴上的。"""
    values = []
    if isinstance(base, dict) and key in base:
        values.append(base[key])
    values.extend(_grid_values(grid, key))
    return values


def uses_aiimage(base: Any, grid: Any) -> bool:
    """
    spec 是否会以 aiimage 生成——base 写死的和网格轴上的都算。

    只看 base 是不够的：``grid: {video_source: [aiimage, pexels]}``（"AI 图片
    对比图库素材"）是最顺理成章的第二轮实验，而这样一条 spec 会绕过全部
    aiimage 校验——pexels key 检查、兜底源的无限递归防护、以及生成前的
    ai_image_enabled 前置检查，也就是重新打开前面三个提交刚关上的
    无限递归与计费风险。
    """
    return AIIMAGE_SOURCE in _declared_values(base, grid, "video_source")


def load_spec(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as spec_file:
        spec = yaml.safe_load(spec_file)

    if not isinstance(spec, dict):
        raise SpecError("experiment spec must be a YAML mapping")

    for key in REQUIRED_KEYS:
        if key not in spec:
            raise SpecError(f"experiment spec is missing required key: {key}")

    # 配置文件由人手写，格式错误属于预期输入而不是异常。逐项校验类型，保证
    # 调用方只需要捕获 SpecError，而不会收到 AttributeError / TypeError 之类
    # 带着无用堆栈的内建异常。
    base = spec.get("base") or {}
    if not isinstance(base, dict):
        raise SpecError(
            f"base must be a mapping of task parameters, got {type(base).__name__}"
        )

    video_count = base.get("video_count", 1)
    if isinstance(video_count, bool) or not isinstance(video_count, int):
        raise SpecError(
            f"base.video_count must be an integer, got {video_count!r}"
        )
    if video_count != 1:
        raise SpecError(
            "base.video_count must be 1 — task.py passes audio_duration * video_count "
            "into download_videos, so a higher count multiplies the image bill"
        )

    subjects = spec.get("subjects") or []
    # 字符串是可迭代的真值，直接放行会按字符展开成一堆单字母主题。
    if not isinstance(subjects, list):
        raise SpecError(
            f"subjects must be a list, got {type(subjects).__name__} — "
            "a single subject still needs to be written as a one-item list"
        )
    if not subjects:
        raise SpecError("experiment spec needs at least one subject")

    grid = spec.get("grid") or {}
    if not isinstance(grid, dict):
        raise SpecError(
            f"grid must be a mapping of one axis name to its values, "
            f"got {type(grid).__name__}"
        )
    if len(grid) != 1:
        raise SpecError(
            f"exactly one grid axis is allowed per round, found {len(grid)} — "
            "more axes than that cannot be resolved at this sample size"
        )

    ((axis, values),) = grid.items()
    # 同样的字符串陷阱：len("abc") == 3 能通过下面的数量上限，然后按字符展开。
    if not isinstance(values, list):
        raise SpecError(
            f"grid axis {axis} must be a list of values, got {type(values).__name__}"
        )
    if not values:
        raise SpecError(
            f"grid axis {axis} has no values, so this round would expand to zero "
            "runs and generate nothing — give the axis at least one value"
        )
    if len(values) > MAX_VARIANTS:
        raise SpecError(
            f"grid expands to {len(values)} variants, above the ceiling of {MAX_VARIANTS}"
        )

    if axis.startswith(_NON_CLI_PREFIXES):
        # ai_image_* 不是 cli.py 的入参，只能靠环境变量送进子进程，而只有经
        # ai_image._setting 读取的配置项才会看那个环境变量。轴若不在其中，
        # 变量导出了却没人读：两个分支实际用同一个取值生成，results.jsonl 却
        # 标成不同分支——数据看着干净，描述的却是一个从未变化过的变量。
        # 集合定义在 ai_image 里并在此导入，两边不会各自漂移。
        from app.services.ai_image import ENV_ROUTED_SETTINGS

        if axis not in ENV_ROUTED_SETTINGS:
            allowed = ", ".join(sorted(ENV_ROUTED_SETTINGS))
            raise SpecError(
                f"grid axis {axis} cannot be varied per run: it is not read through "
                f"ai_image._setting, so the value would be exported to the subprocess "
                f"and then ignored, and every arm would silently generate with the "
                f"config.toml value while results.jsonl labelled them differently. "
                f"Varyable ai_image settings: {allowed}"
            )

    if uses_aiimage(base, grid):
        from app.config import config

        # 分镜是按叙事顺序生成的，拼接却默认随机：schema.py 的默认值是 random，
        # cli.py 不传就是 None，video.py 于是把片段打乱，神话被讲乱了。
        # 显式要求写出来而不是悄悄注入默认值，spec 才能自证它测的是什么。
        concat_modes = _declared_values(base, grid, "video_concat_mode")
        if not concat_modes:
            raise SpecError(
                f"video_source={AIIMAGE_SOURCE} requires an explicit "
                f'video_concat_mode: "{REQUIRED_CONCAT_MODE}" in base — AI image '
                "beats are planned in narrative order, and the default (random) "
                "shuffles them, so the myth is told out of order"
            )
        wrong = [mode for mode in concat_modes if mode != REQUIRED_CONCAT_MODE]
        if wrong:
            raise SpecError(
                f"video_source={AIIMAGE_SOURCE} requires "
                f'video_concat_mode = "{REQUIRED_CONCAT_MODE}", got {wrong!r} — '
                "AI image beats are narrative-ordered (plan_prompts makes one LLM "
                "call over the whole script precisely to order them), and any other "
                "mode shuffles the clips into the wrong story order"
            )

        # AI 图片的兜底路径走图库下载，没有 key 时被安全策略拦截的分镜会直接失败。
        # 与其跑到一半才崩，不如在装载配置时就拒绝。
        if not config.app.get("pexels_api_keys"):
            raise SpecError(
                "video_source=aiimage requires pexels_api_keys in config.toml — "
                "blocked or failed beats fall back to stock footage, and without a "
                "key that fallback cannot run"
            )

        # 同一个理由的另一面：兜底源再指回 aiimage 就是无限递归。
        # ai_image._fallback_clip 把该配置原样传给 material.download_videos，
        # 而 download_videos 又会把 "aiimage" 派发回 ai_image。
        #
        # 该配置现在也经 _setting 读取，spec 的 base 和网格轴都能通过环境变量
        # 覆盖它，因此三个来源都要查，不能只看 config.toml。
        fallback_sources = [config.app.get("ai_image_fallback_source", "pexels")]
        fallback_sources.extend(
            _declared_values(base, grid, "ai_image_fallback_source")
        )
        if AIIMAGE_SOURCE in fallback_sources:
            raise SpecError(
                'ai_image_fallback_source must not be "aiimage" — the fallback value '
                "is passed straight back into material.download_videos, which "
                "dispatches aiimage to ai_image again; that recursion never "
                'terminates and keeps billing Imagen. Use "pexels" or "pixabay"'
            )

    return spec


def expand_variants(spec: dict) -> list[dict[str, Any]]:
    """展开成待生成的运行列表；每个变体都跑一遍全部主题，避免主题混淆效应。"""
    base = dict(spec.get("base") or {})
    grid = spec.get("grid") or {}
    axis, values = next(iter(grid.items()))

    runs = []
    for value, subject in itertools.product(values, spec["subjects"]):
        variant = {axis: value}
        runs.append({
            "variant": variant,
            "subject": subject,
            "params": {**base, **variant},
        })
    return runs


def build_cli_argv(run: dict, task_id: str) -> list[str]:
    argv = [sys.executable, "cli.py", "--task-id", task_id,
            "--video-subject", run["subject"]]

    for key, value in (run.get("params") or {}).items():
        if key.startswith(_NON_CLI_PREFIXES):
            continue
        flag = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            # cli.py 的布尔开关都用 BooleanOptionalAction，且部分默认为真
            # （subtitle_enabled）。False 若不发出任何参数，spec 写的关闭就会被
            # 默认值悄悄翻回打开：视频与记录里的变体不一致，整轮归因作废。
            argv.append(flag if value else f"--no-{key.replace('_', '-')}")
            continue
        argv.extend([flag, str(value)])
    return argv


def build_subprocess_env(run: dict) -> dict[str, str]:
    """
    把不走命令行的参数转成子进程环境变量。

    build_cli_argv 会跳过 ``ai_image_*``：它们只能从 config.toml 读。但整台机器
    共用同一个 config.toml，若不另行传递，网格里所有分支都会用同一个风格生成，
    而 results.jsonl 却把它们标成不同分支——正是这个文件存在的意义所在被推翻。
    ai_image._setting 因此优先读取 ``MPT_<KEY大写>``。

    键从 run["params"] 通用推导，所以将来新增 ai_image_motion 之类的轴无需改动
    此处；用 params 而不是 variant，是为了让 base 里写死的 ai_image_* 同样生效。
    这样每个参数要么变成命令行标志，要么变成环境变量，不存在被静默丢弃的项。
    """
    from app.services.ai_image import ENV_SETTING_PREFIX

    return {
        ENV_SETTING_PREFIX + key.upper(): str(value)
        for key, value in (run.get("params") or {}).items()
        if key.startswith(_NON_CLI_PREFIXES)
    }


def _placeholder_errors(spec: dict) -> list[str]:
    """找出仍带占位符的参数值。"""
    candidates: list[tuple[str, Any]] = list((spec.get("base") or {}).items())
    candidates.extend(("subjects", subject) for subject in spec.get("subjects") or [])
    for axis, values in (spec.get("grid") or {}).items():
        candidates.extend((axis, value) for value in values)

    return [
        f"{key} still contains the {PLACEHOLDER_MARKER} placeholder: {value!r}"
        for key, value in candidates
        if isinstance(value, str) and PLACEHOLDER_MARKER in value
    ]


def preflight_errors(spec: dict) -> list[str]:
    """
    返回生成前必须修复的问题；空列表表示可以开跑。

    load_spec 只保证 spec 自身合法，以及 aiimage 的**兜底**路径可用
    （pexels_api_keys）。主路径的前置条件没人检查：ai_image.is_enabled() 为假时
    generate_clips 会直接返回空列表，每个变体都会在素材阶段失败——整批跑完
    一无所获。和占位符一样，必须在起第一个子进程前报错。
    """
    errors = _placeholder_errors(spec)

    if uses_aiimage(spec.get("base") or {}, spec.get("grid") or {}):
        from app.config import config
        from app.services import ai_image

        if not ai_image.is_enabled():
            if not config.app.get("ai_image_enabled", False):
                errors.append(
                    "video_source=aiimage requires ai_image_enabled = true in "
                    "config.toml — ai_image.generate_clips returns no clips while it "
                    "is false, so every variant fails at the materials stage"
                )
            if not config.app.get("gemini_api_key", ""):
                errors.append(
                    "video_source=aiimage requires gemini_api_key in config.toml — "
                    "Imagen cannot be reached without it, so ai_image is disabled and "
                    "every variant fails at the materials stage"
                )
            if not errors:
                errors.append(
                    "video_source=aiimage but ai_image.is_enabled() is false; "
                    "check the [app] ai_image settings in config.toml"
                )

    return errors


def record_result(results_path: str, record: dict) -> None:
    os.makedirs(os.path.dirname(results_path) or ".", exist_ok=True)
    with open(results_path, "a", encoding="utf-8") as results_file:
        results_file.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_results(results_path: str) -> list[dict]:
    if not os.path.exists(results_path):
        return []
    records = []
    with open(results_path, "r", encoding="utf-8") as results_file:
        for line in results_file:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _rewrite_results(results_path: str, records: list[dict]) -> None:
    """
    原子重写结果文件：先写同目录临时文件，再 os.replace 覆盖。

    ``open(path, "w")`` 会先截断再写。中途崩溃、磁盘写满或被中断，就会留下一个
    半截或空的 results.jsonl —— 而人工评级按定义无法重跑，用户手输的判断只存在
    于这个文件里。临时文件必须与目标同目录，os.replace 才能保证原子替换语义
    （与 task_artifacts._write_json_atomic 同一模式）。
    """
    directory = os.path.dirname(results_path) or "."
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=directory,
            prefix=f".{os.path.basename(results_path)}.",
            suffix=".tmp",
            delete=False,
        ) as temp_file:
            temp_path = temp_file.name
            for record in records:
                temp_file.write(json.dumps(record, ensure_ascii=False) + "\n")
        os.replace(temp_path, results_path)
        temp_path = None
    finally:
        # 替换成功前原文件一字未动；失败时只清理本次的临时文件。
        if temp_path is not None:
            try:
                os.unlink(temp_path)
            except OSError:
                pass


def mark_outcome(results_path: str, key: str, outcome: str) -> bool:
    records = load_results(results_path)
    matched = False
    for record in records:
        # 标题恒存在但可能是空字符串（llm.generate_social_metadata 的兜底分支）。
        # 空标题不能当匹配键，否则一次 mark 会把所有标题为空的记录一起改掉。
        by_title = bool(key) and record.get("title") == key
        if record.get("task_id") == key or by_title:
            record["outcome"] = outcome
            matched = True
    if matched:
        _rewrite_results(results_path, records)
    return matched


def build_report(records: list[dict]) -> str:
    arms: dict[str, list[dict]] = defaultdict(list)
    excluded = 0
    for record in records:
        if record.get("fallback_beats", 0) > 0:
            excluded += 1
            continue
        arm = ", ".join(f"{k}={v}" for k, v in sorted((record.get("variant") or {}).items()))
        arms[arm or "(no variant)"].append(record)

    lines = []
    for arm, arm_records in sorted(arms.items()):
        tally: dict[str, int] = defaultdict(int)
        for record in arm_records:
            tally[record.get("outcome") or "unrated"] += 1
        summary = "  ".join(f"{name}={count}" for name, count in sorted(tally.items()))
        lines.append(f"{arm}\n    n={len(arm_records)}  {summary}")

    lines.append(f"\n{excluded} sample(s) excluded for fallback_beats > 0 (contaminated).")
    lines.append(
        "At this sample size, only a stark difference is meaningful. "
        "A narrow gap is noise — re-run rather than act on it."
    )
    return "\n".join(lines)


def run_experiment(spec_path: str, results_path: str = DEFAULT_RESULTS_PATH) -> int:
    from app.services import task_artifacts

    try:
        spec = load_spec(spec_path)
    except SpecError as exc:
        # 配置错误是预期输入，堆栈对用户没有信息量。
        print(f"invalid experiment spec {spec_path}: {exc}")
        return 2
    except (OSError, yaml.YAMLError) as exc:
        print(f"cannot read experiment spec {spec_path}: {exc}")
        return 2

    errors = preflight_errors(spec)
    if errors:
        print(f"refusing to run {spec.get('name', spec_path)}:")
        for error in errors:
            print(f"  - {error}")
        return 2

    runs = expand_variants(spec)
    print(f"{spec['name']}: {len(runs)} video(s) to generate")

    failures = 0
    for index, run in enumerate(runs, start=1):
        task_id = str(uuid.uuid4())
        argv = build_cli_argv(run, task_id)
        print(f"\n[{index}/{len(runs)}] {run['subject']} | {run['variant']}")

        completed = subprocess.run(argv, env={**os.environ, **build_subprocess_env(run)})
        script_data = task_artifacts.read_script_data(task_id) or {}
        metadata = script_data.get("social_metadata") or {}
        # 标题这个键总是存在，失败时是空字符串而不是缺键，所以判空值而不是判缺键。
        # 截断到发布端的长度，否则超长标题在 YouTube 上显示的是截断版，
        # 而 mark 用完整版做精确匹配，永远对不上。
        title = (metadata.get("title") or "")[:PUBLISHED_TITLE_MAX]
        # 分镜总数和降级数必须一起记录：只有降级数时，"7 个降级 1 个"和"模型 id
        # 写错、7 个全降级"在 results.jsonl 里是同一种东西，而后者代表这条视频
        # 里一张 AI 图片都没有，标称的变量根本没被测到。
        beats = script_data.get("ai_image_beats", 0)
        fallback_beats = script_data.get("ai_image_fallback_beats", 0)

        record_result(results_path, {
            "task_id": task_id,
            "experiment": spec["name"],
            "niche": spec.get("niche", ""),
            "variant": run["variant"],
            "subject": run["subject"],
            "title": title,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "beats": beats,
            "fallback_beats": fallback_beats,
            "succeeded": completed.returncode == 0,
            "outcome": None,
        })

        if beats and fallback_beats >= beats:
            print(
                f"  WARNING: all {beats} beat(s) fell back to stock footage — this "
                "video contains no AI imagery, so it measures nothing about "
                "this arm. Check ai_image_model, the API key, and the safety "
                "filter warnings in the log above."
            )

        if not title:
            # 人工反馈靠发布标题查回记录，标题为空时只能用 task_id。
            print(f"  no published title recorded — mark this one by task_id {task_id}")

        if completed.returncode != 0:
            failures += 1
            print(f"  variant failed (exit {completed.returncode}) — continuing batch")

    print(f"\nDone. Results appended to {results_path}")
    if failures:
        # 批次里有失败就不能报成功退出：这条命令每天跑两三次，静默的 0
        # 会让用户以为素材都拿到了。
        print(f"{failures}/{len(runs)} variant(s) failed")
        return 1
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="MoneyPrinterTurbo experiment runner")
    parser.add_argument("--results", default=DEFAULT_RESULTS_PATH)
    sub = parser.add_subparsers(dest="command", required=True)

    run_parser = sub.add_parser("run", help="generate and publish every variant in a spec")
    run_parser.add_argument("spec")

    mark_parser = sub.add_parser("mark", help="record a manual outcome for a published video")
    mark_parser.add_argument("key", help="task_id or published title")
    mark_parser.add_argument("--outcome", required=True, choices=VALID_OUTCOMES)

    report_parser = sub.add_parser("report", help="summarise outcomes per variant arm")
    report_parser.add_argument("experiment")

    args = parser.parse_args(argv)

    if args.command == "run":
        return run_experiment(args.spec, args.results)
    if args.command == "mark":
        if mark_outcome(args.results, args.key, args.outcome):
            print(f"marked {args.key} as {args.outcome}")
            return 0
        print(f"no record found for: {args.key}")
        return 1

    records = [r for r in load_results(args.results) if r.get("experiment") == args.experiment]
    if not records:
        print(f"no records for experiment: {args.experiment}")
        return 1
    print(build_report(records))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
