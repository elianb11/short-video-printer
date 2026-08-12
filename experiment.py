"""
参数网格实验运行器。

每轮实验只允许改动一个维度：按 2-3 条/天的发布节奏，一周约 20 个样本，
两分支已经只有 10 个样本，再多分支就完全淹没在 Shorts 的分发方差里。
"""

from __future__ import annotations

import itertools
from typing import Any

import yaml

MAX_VARIANTS = 3
REQUIRED_KEYS = ("name", "base", "subjects", "grid")


class SpecError(ValueError):
    """实验配置不合法。"""


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

    if base.get("video_source") == "aiimage":
        from app.config import config

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
        if config.app.get("ai_image_fallback_source", "pexels") == "aiimage":
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
