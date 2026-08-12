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

import json
import math
import re

from loguru import logger

from app.config import config
from app.models.schema import VideoAspect
from app.services import task_artifacts
from app.services.llm import _generate_response

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
