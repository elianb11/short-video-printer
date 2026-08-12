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

    base = spec.get("base") or {}
    if int(base.get("video_count", 1)) != 1:
        raise SpecError(
            "base.video_count must be 1 — task.py passes audio_duration * video_count "
            "into download_videos, so a higher count multiplies the image bill"
        )

    subjects = spec.get("subjects") or []
    if not subjects:
        raise SpecError("experiment spec needs at least one subject")

    grid = spec.get("grid") or {}
    if len(grid) != 1:
        raise SpecError(
            f"exactly one grid axis is allowed per round, found {len(grid)} — "
            "more axes than that cannot be resolved at this sample size"
        )

    (values,) = grid.values()
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
