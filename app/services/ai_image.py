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

import math

from loguru import logger

from app.config import config
from app.models.schema import VideoAspect

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
