"""
AI 图片素材生成 —— 用 Imagen 生成分镜图，再用 ffmpeg 做 Ken Burns 运镜，
输出普通 mp4 片段，因此对下游合成流程完全透明。

该模块伪装成一个素材提供方：``generate_clips`` 的返回值与
``material.download_videos`` 一致（本地视频文件路径列表），所以拼接、转场、
字幕和音频阶段都不需要任何改动。

配置（config.toml 的 [app] 段，沿用 twelvelabs_* 的前缀约定）:
    ai_image_enabled = true
    ai_image_model = "imagen-3.0-generate-002"
    ai_image_style = "..."            # 风格锁，实验变量
    ai_image_motion = "random"        # zoom_in|zoom_out|pan_left|pan_right|random
    ai_image_max_images_per_task = 12 # 成本熔断
    ai_image_fallback_source = "pexels"
"""

import hashlib
import json
import math
import os
import re
import subprocess

from loguru import logger

from app.config import config
from app.models.schema import VideoAspect
from app.services import task_artifacts
from app.services.llm import _generate_response
from app.utils import utils

DEFAULT_MODEL = "imagen-3.0-generate-002"
DEFAULT_STYLE = (
    "peinture à l'huile cinématographique, clair-obscur dramatique, "
    "palette or et bleu profond"
)
MOTIONS = ("zoom_in", "zoom_out", "pan_left", "pan_right")
DEFAULT_MAX_IMAGES = 12

_ASPECT_RATIOS = {
    VideoAspect.portrait.value: "9:16",
    VideoAspect.landscape.value: "16:9",
    VideoAspect.square.value: "1:1",
}


def _aspect_value(video_aspect) -> str:
    return video_aspect.value if isinstance(video_aspect, VideoAspect) else str(video_aspect)


def is_enabled() -> bool:
    """仅当显式开启且配置了 Gemini key 时才启用。"""
    if not config.app.get("ai_image_enabled", False):
        return False
    return bool(config.app.get("gemini_api_key", ""))


def beat_count(audio_duration: float, clip_duration: int) -> int:
    """按音频时长和单片时长推算分镜数量，并受成本上限约束。"""
    clip_duration = max(int(clip_duration or 1), 1)
    count = math.ceil(max(float(audio_duration), 0.0) / clip_duration)
    count = max(count, 1)
    ceiling = int(config.app.get("ai_image_max_images_per_task", DEFAULT_MAX_IMAGES))
    if count > ceiling:
        logger.warning(
            f"beat count {count} exceeds ai_image_max_images_per_task={ceiling}, capping"
        )
        count = ceiling
    return count


def aspect_ratio(video_aspect) -> str:
    value = _aspect_value(video_aspect)
    if value not in _ASPECT_RATIOS:
        raise ValueError(f"unsupported video aspect for ai_image: {value}")
    return _ASPECT_RATIOS[value]


def resolution(video_aspect) -> tuple[int, int]:
    return VideoAspect(_aspect_value(video_aspect)).to_resolution()


def _style() -> str:
    return str(config.app.get("ai_image_style", DEFAULT_STYLE)).strip() or DEFAULT_STYLE


def _build_planning_prompt(script: str, count: int) -> str:
    return f"""Tu es directeur artistique pour une vidéo courte de mythologie.

Voici la narration:
\"\"\"{script}\"\"\"

Génère EXACTEMENT {count} descriptions d'images, une par plan, dans l'ordre de la narration.

Règles impératives:
1. Chaque description couvre le moment narratif de son plan.
2. Varie les cadrages entre les plans (plan large, gros plan, contre-plongée) pour éviter la répétition.
3. Reste ATMOSPHÉRIQUE et SYMBOLIQUE. Ne décris jamais de violence explicite, de sang,
   de mutilation ni de nudité — suggère par l'ombre, la ruine, la lumière et le symbole.
4. Pas de texte ni de lettres dans l'image.
5. Ne mentionne aucun style artistique: il est ajouté automatiquement.

Réponds UNIQUEMENT avec un tableau JSON de {count} chaînes, sans texte autour.
Exemple: ["une silhouette au sommet d'une falaise", "un temple en ruine sous l'orage"]"""


def _frame_symbolically(base: str) -> str:
    """降级路径的关键词同样要象征化包装，否则 "sacrifice"、"bataille"
    这类原词会直接触发图像模型的安全过滤——而缓解措施正是为这条路径存在的。"""
    return (
        f"évocation atmosphérique et symbolique de {base}, suggérée par l'ombre, "
        "la ruine et la lumière, sans violence explicite ni nudité, sans texte"
    )


def _parse_prompt_list(response: str) -> list[str]:
    text = (response or "").strip()
    # llm._generate_response provider 报错时返回 "Error: ..." 字符串而不抛异常，
    # 必须显式拦截：否则错误文本里若恰好含 JSON 数组，会被当成提示词用。
    if text.startswith("Error:"):
        raise ValueError(f"planning response reported a provider error: {text[:200]}")
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if not match:
        raise ValueError("no JSON array found in planning response")
    items = json.loads(match.group(0))
    # 只接受字符串元素：dict/数字被 str() 化后会变成 "{'prompt': 'x'}" 这类垃圾提示词，
    # 宁可返回空列表让调用方走降级分支。
    return [item.strip() for item in items if isinstance(item, str) and item.strip()]


def plan_prompts(task_id: str, count: int) -> list[str]:
    """生成 count 条已锁定风格的画面提示词。LLM 失败时降级为关键词提示。"""
    script_data = task_artifacts.read_script_data(task_id) or {}
    script = str(script_data.get("script", "")).strip()
    search_terms = [str(term) for term in script_data.get("search_terms", []) if term]

    bases: list[str] = []
    response = ""
    try:
        response = _generate_response(prompt=_build_planning_prompt(script, count))
        bases = _parse_prompt_list(response)
    except Exception as exc:
        # 带上截断的响应体，否则真正的 provider 错误会被日志吞掉、无从排查。
        logger.warning(
            "ai image prompt planning failed: "
            f"task_id={task_id}, error={type(exc).__name__}, detail={exc}, "
            f"response={str(response)[:300]!r}"
        )

    if not bases:
        logger.warning(
            f"ai image prompt planning produced no usable prompts, "
            f"falling back to symbolic keyword prompts: task_id={task_id}"
        )
        fallback = search_terms or [script[:200] or "scène mythologique"]
        bases = [_frame_symbolically(term) for term in fallback]

    # 数量对齐：多则截断，少则循环补齐，保证时间线每个分镜都有画面。
    bases = bases[:count]
    # cycle 必须是补齐开始前的快照：直接对 bases 取模会因 len % len == 0 恒取首条。
    cycle = list(bases)
    while len(bases) < count:
        bases.append(cycle[len(bases) % len(cycle)])

    style = _style()
    return [f"{base}, {style}" for base in bases]


class RaiBlockedError(RuntimeError):
    """模型因安全策略拒绝生成。神话题材容易触发，需要单独区分于网络故障。"""


def _imagen_client():
    # 延迟导入，保持与 llm.py 中 gemini 分支一致的依赖策略。
    from google import genai

    return genai.Client(api_key=config.app.get("gemini_api_key", ""))


def image_cache_path(task_id: str, prompt: str) -> str:
    digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]
    image_dir = os.path.join(utils.task_dir(task_id), "images")
    os.makedirs(image_dir, exist_ok=True)
    return os.path.join(image_dir, f"{digest}.jpg")


def generate_image(prompt: str, ratio: str, out_path: str) -> str:
    """生成单张图片并写盘；命中缓存时直接返回。"""
    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        logger.info(f"ai image cache hit: {out_path}")
        return out_path

    from google.genai import types

    client = _imagen_client()
    response = client.models.generate_images(
        model=config.app.get("ai_image_model", DEFAULT_MODEL),
        prompt=prompt,
        config=types.GenerateImagesConfig(
            number_of_images=1,
            aspect_ratio=ratio,
            output_mime_type="image/jpeg",
        ),
    )

    images = getattr(response, "generated_images", None) or []
    if not images or not getattr(images[0].image, "image_bytes", None):
        raise RaiBlockedError(f"image generation returned no image for prompt: {prompt[:80]}")

    with open(out_path, "wb") as image_file:
        image_file.write(images[0].image.image_bytes)
    logger.success(f"ai image generated: {out_path}")
    return out_path


# zoompan 逐帧推进 zoom/x/y。'on' 是输出帧序号，总帧数 = duration * FPS。
# 先放大到 4 倍再 zoompan，可以避免 zoompan 在整数像素上取整造成的抖动。
_FPS = 30
_SUPERSAMPLE = 4


def pick_motion(index: int) -> str:
    configured = str(config.app.get("ai_image_motion", "random")).strip().lower()
    if configured == "random":
        return MOTIONS[index % len(MOTIONS)]
    if configured not in MOTIONS:
        logger.warning(f"unknown ai_image_motion={configured}, using zoom_in")
        return "zoom_in"
    return configured


def _zoompan_expr(motion: str, frames: int) -> tuple[str, str, str]:
    """返回 (zoom, x, y) 三个 zoompan 表达式。"""
    if motion == "zoom_in":
        return (f"min(1+0.25*on/{frames},1.25)", "iw/2-(iw/zoom/2)", "ih/2-(ih/zoom/2)")
    if motion == "zoom_out":
        return (f"max(1.25-0.25*on/{frames},1.0)", "iw/2-(iw/zoom/2)", "ih/2-(ih/zoom/2)")
    if motion == "pan_left":
        return ("1.15", f"(iw-iw/zoom)*(1-on/{frames})", "ih/2-(ih/zoom/2)")
    return ("1.15", f"(iw-iw/zoom)*(on/{frames})", "ih/2-(ih/zoom/2)")


def render_ken_burns(
    image_path: str, out_path: str, duration: int, motion: str, width: int, height: int
) -> str:
    frames = max(int(duration) * _FPS, 1)
    zoom, pan_x, pan_y = _zoompan_expr(motion, frames)
    video_filter = (
        f"scale={width * _SUPERSAMPLE}:{height * _SUPERSAMPLE},"
        f"zoompan=z='{zoom}':x='{pan_x}':y='{pan_y}':"
        f"d={frames}:s={width}x{height}:fps={_FPS},"
        f"format=yuv420p"
    )
    # -hide_banner: 编译信息有 ~370 字节，会把真正的报错挤出下面 stderr[-500:] 的窗口。
    argv = [
        "ffmpeg", "-hide_banner", "-y", "-loop", "1", "-i", image_path,
        "-t", str(duration),
        "-vf", video_filter,
        "-c:v", "libx264", "-preset", "medium", "-crf", "20",
        "-pix_fmt", "yuv420p", "-r", str(_FPS),
        out_path,
    ]
    completed = subprocess.run(argv, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(
            f"ffmpeg ken burns failed (exit {completed.returncode}): {completed.stderr[-500:]}"
        )
    return out_path


def _fallback_clip(task_id: str, index: int, video_aspect, clip_duration: int) -> str | None:
    """
    单个分镜失败时退回图库素材，保证成片仍能交付。

    这里延迟导入 material，避免 material -> ai_image -> material 的循环导入。
    """
    from app.services import material

    script_data = task_artifacts.read_script_data(task_id) or {}
    terms = [str(term) for term in script_data.get("search_terms", []) if term]
    if not terms:
        logger.warning(f"no search terms available for fallback: task_id={task_id}")
        return None

    term = terms[index % len(terms)]
    try:
        paths = material.download_videos(
            task_id=task_id,
            search_terms=[term],
            source=config.app.get("ai_image_fallback_source", "pexels"),
            video_aspect=video_aspect,
            audio_duration=float(clip_duration),
            max_clip_duration=clip_duration,
        )
    except Exception as exc:
        logger.error(
            f"fallback material download failed: task_id={task_id}, "
            f"term={term}, error={type(exc).__name__}, detail={exc}"
        )
        return None
    return paths[0] if paths else None


def generate_clips(
    task_id: str, video_aspect, audio_duration: float, clip_duration: int
) -> list[str]:
    """生成整条时间线的 AI 图片片段，返回值与 download_videos 一致。

    单个分镜失败不得中断整条流水线：先退回图库素材，连图库也拿不到时丢弃该
    分镜，其余片段照常交付。渲染耗 CPU（4 倍超采样），因此按顺序执行。
    """
    count = beat_count(audio_duration, clip_duration)
    ratio = aspect_ratio(video_aspect)
    width, height = resolution(video_aspect)
    # render_ken_burns 用 int(duration) 算帧数却把原值传给 -t，浮点会让运镜中途重启。
    duration = int(clip_duration)
    prompts = plan_prompts(task_id, count)

    clip_dir = os.path.join(utils.task_dir(task_id), "images")
    os.makedirs(clip_dir, exist_ok=True)

    clips: list[str] = []
    fallback_beats = 0

    for index, prompt in enumerate(prompts):
        try:
            image_path = generate_image(prompt, ratio, image_cache_path(task_id, prompt))
            # 输出路径按分镜序号命名：提示词可能重复（缓存命中），
            # 若按图片路径派生，两个分镜会互相覆盖。
            clip_path = os.path.join(clip_dir, f"beat-{index:02d}.mp4")
            rendered = render_ken_burns(
                image_path, clip_path, duration, pick_motion(index), width, height
            )
            # ffmpeg 遇到 duration=0 之类的输入会以退出码 0 产出空文件，
            # 没有抛异常并不代表片段可用，必须校验落盘结果。
            if not rendered or not os.path.exists(rendered) or os.path.getsize(rendered) <= 0:
                raise RuntimeError(f"ken burns produced an empty clip: {rendered}")
            clips.append(rendered)
            continue
        except RaiBlockedError as exc:
            logger.warning(f"beat {index} blocked by safety filter: {exc}")
        except Exception as exc:
            logger.error(
                f"beat {index} generation failed: {type(exc).__name__}, detail={exc}"
            )

        fallback_beats += 1
        fallback = _fallback_clip(task_id, index, video_aspect, duration)
        if fallback:
            clips.append(fallback)
        else:
            logger.error(f"beat {index} dropped: fallback unavailable")

    task_artifacts.patch_script_data(task_id, ai_image_fallback_beats=fallback_beats)
    logger.info(
        f"ai image clips ready: task_id={task_id}, clips={len(clips)}, "
        f"fallback_beats={fallback_beats}"
    )
    return clips
