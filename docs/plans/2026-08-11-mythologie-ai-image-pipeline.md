# Mythologie AI-Image Video Pipeline — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate French mythology Shorts using AI-generated images with Ken Burns motion instead of stock footage, driven by a parameter-grid experiment runner that records which parameters produced which published video.

**Architecture:** A new `ai_image` service impersonates a stock-footage provider — it returns a `List[str]` of local mp4 paths from `material.download_videos()`, so every downstream stage (concat, transitions, subtitles, audio, social metadata, cross-post) runs unmodified. A top-level `experiment.py` expands a YAML parameter grid into variants, invokes `cli.py` per variant as a subprocess, and appends attribution records to `experiments/results.jsonl`.

**Tech Stack:** Python 3.11+, `google-genai==2.11.0` (already a dependency, used for Imagen), ffmpeg `zoompan` filter via subprocess, PyYAML (already a dependency), `unittest` + `pytest==9.1.1`.

**Spec:** `docs/specs/2026-08-11-mythologie-video-experiment-design.md`

## Global Constraints

- **Never edit `video.py` or `webui/Main.py`.** They are upstream's most-churned files (`webui/Main.py`: +5044 lines in the v1.2.9→v1.3.3 sync). Every edit becomes recurring rebase cost.
- **Config keys live in `config.app` with an `ai_image_` prefix**, NOT a new `[ai_image]` TOML section. `config.py:515-521` hardcodes section names, so a new section requires editing `config.py`. Follow the `twelvelabs_*` precedent (`twelvelabs.py:26-30`).
- **Gemini API key is reused** from the existing `gemini_api_key` config key (`config.example.toml:98`). Do not add a second key.
- **All external APIs are mocked in tests.** No test may make a network call. Follow `test/services/test_twelvelabs.py`.
- **Tests use `unittest.TestCase`**, with `sys.path.insert(0, str(Path(__file__).parent.parent.parent))` at the top, matching every file in `test/services/`.
- **`video_count` must be 1** for any AI-image task. `task.py:594` passes `audio_duration * params.video_count` into `download_videos`, so a higher count silently multiplies the beat count and the image bill.
- **Pexels is a hard prerequisite**, not an alternative: the per-beat fallback path calls Pexels, so `pexels_api_keys` must be configured even though `video_source` is `aiimage`.
- Run tests with: `uv run pytest test/services/<file> -v`

## Prerequisites (do before Task 1)

These are environment setup, not code:

- [ ] `sudo apt install ffmpeg` — **not currently installed**; Task 6 cannot be verified without it
- [ ] `config.toml`: set `gemini_api_key`
- [ ] `config.toml`: set `pexels_api_keys = ["..."]` (required for the fallback path)
- [ ] `config.toml`: set `[elevenlabs] api_key`, then run the WebUI once to list voices via `voice.get_elevenlabs_voices()` and pick a French narrator voice id
- [ ] `config.toml`: set `upload_post_api_key`, `upload_post_username`, `upload_post_enabled = true`

---

### Task 1: Read persisted script data

`ai_image.py` needs the full narration script for beat-level prompt context. It is already written to `tasks/<id>/script.json` at pipeline step 3, but `task_artifacts.py` only has writers.

**Files:**
- Modify: `app/services/task_artifacts.py` (add function after `write_script_data`, line 59-62)
- Test: `test/services/test_task_artifacts.py`

**Interfaces:**
- Consumes: nothing
- Produces: `task_artifacts.read_script_data(task_id: str) -> dict | None` — returns the parsed `script.json` payload, or `None` if the file is missing or unreadable. Never raises.

- [ ] **Step 1: Write the failing test**

Append to `test/services/test_task_artifacts.py`:

```python
class TestReadScriptData(unittest.TestCase):
    def setUp(self):
        self.task_id = f"test-read-{uuid4().hex}"

    def tearDown(self):
        shutil.rmtree(utils.task_dir(self.task_id), ignore_errors=True)

    def test_returns_payload_written_by_write_script_data(self):
        payload = {"script": "Prométhée vola le feu.", "search_terms": ["feu", "titan"]}
        task_artifacts.write_script_data(self.task_id, payload)

        result = task_artifacts.read_script_data(self.task_id)

        self.assertEqual(result["script"], "Prométhée vola le feu.")
        self.assertEqual(result["search_terms"], ["feu", "titan"])

    def test_returns_none_when_file_missing(self):
        self.assertIsNone(task_artifacts.read_script_data("nonexistent-task-id"))

    def test_returns_none_on_corrupt_json(self):
        target = Path(utils.task_dir(self.task_id)) / "script.json"
        target.write_text("{not valid json", encoding="utf-8")

        self.assertIsNone(task_artifacts.read_script_data(self.task_id))
```

Ensure the file's imports include `shutil`, `unittest`, `from pathlib import Path`, `from uuid import uuid4`, `from app.services import task_artifacts`, `from app.utils import utils`.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest test/services/test_task_artifacts.py -v -k ReadScriptData`
Expected: FAIL with `AttributeError: module 'app.services.task_artifacts' has no attribute 'read_script_data'`

- [ ] **Step 3: Write minimal implementation**

Add to `app/services/task_artifacts.py` after `write_script_data`:

```python
def read_script_data(task_id: str) -> dict | None:
    """
    读取任务的 ``script.json``，文件缺失或损坏时返回 ``None``。

    该入口供素材生成等辅助流程复用已持久化的脚本内容，不应因为读取失败
    中断主流程，因此所有异常都降级为 ``None`` 并记录日志。
    """
    try:
        with _script_file(task_id).open("r", encoding="utf-8") as script_file:
            payload = json.load(script_file)
        if not isinstance(payload, dict):
            raise ValueError("task script data must be a JSON object")
        return payload
    except FileNotFoundError:
        logger.debug(f"task script data not found: task_id={task_id}")
        return None
    except Exception as exc:
        logger.warning(
            "failed to read task script data: "
            f"task_id={task_id}, error={type(exc).__name__}, detail={exc}"
        )
        return None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest test/services/test_task_artifacts.py -v -k ReadScriptData`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add app/services/task_artifacts.py test/services/test_task_artifacts.py
git commit -m "feat(task-artifacts): add read_script_data for reusing persisted scripts"
```

---

### Task 2: Persist social metadata so published titles are recoverable

The YouTube title generated at `task.py:865` is passed straight to the Upload-Post API and never stored. `state.get_task` is in-memory only, so after a subprocess `cli.py` run the title is unrecoverable — but it is the manual lookup key for matching a published video back to its parameters.

**Files:**
- Modify: `app/services/task.py:865-877` (inside `_run_cross_post`)
- Test: `test/services/test_task.py`

**Interfaces:**
- Consumes: `task_artifacts.patch_script_data` (already exists, `task_artifacts.py:64`)
- Produces: `script.json` gains a `social_metadata` key: `{"title": str, "caption": str, "hashtags": list[str]}`. Task 10 reads this.

- [ ] **Step 1: Write the failing test**

Append to `test/services/test_task.py`:

```python
class TestCrossPostPersistsSocialMetadata(unittest.TestCase):
    def setUp(self):
        self.task_id = f"test-meta-{uuid4().hex}"
        task_artifacts.write_script_data(self.task_id, {"script": "s", "search_terms": []})

    def tearDown(self):
        shutil.rmtree(utils.task_dir(self.task_id), ignore_errors=True)

    @patch("app.services.task.upload_post.cross_post_video")
    @patch("app.services.task.llm.generate_social_metadata")
    def test_youtube_metadata_is_written_to_script_json(self, mock_meta, mock_post):
        mock_meta.return_value = {
            "title": "Le mythe de Prométhée",
            "caption": "Le titan qui défia les dieux.",
            "hashtags": ["#mythologie", "#grece"],
        }
        mock_post.return_value = {"success": True, "request_id": "req-1"}

        task._run_cross_post(
            task_id=self.task_id,
            video_paths=("/tmp/fake.mp4",),
            video_subject="Prométhée",
            video_script="s",
            video_language="fr-FR",
            platforms=("youtube",),
            youtube_privacy_status="public",
        )

        saved = task_artifacts.read_script_data(self.task_id)
        self.assertEqual(saved["social_metadata"]["title"], "Le mythe de Prométhée")
        self.assertEqual(saved["social_metadata"]["hashtags"], ["#mythologie", "#grece"])
```

The signature is `_run_cross_post(task_id, video_paths: tuple[str, ...], video_subject, video_script, video_language, platforms: tuple[str, ...], youtube_privacy_status)` — `task.py:831`. Both sequence parameters are tuples.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest test/services/test_task.py -v -k CrossPostPersistsSocialMetadata`
Expected: FAIL — `saved` has no `social_metadata` key (`KeyError`)

- [ ] **Step 3: Write minimal implementation**

In `app/services/task.py`, immediately after the `youtube_extra = {...}` block (ends ~line 877), add:

```python
            # 发布标题由 LLM 生成且只发送给 Upload-Post，进程结束后无法回溯。
            # 实验流程需要用它把已发布视频对应回生成参数，因此落盘保存。
            task_artifacts.patch_script_data(task_id, social_metadata=metadata)
```

`task_artifacts` is already imported in `task.py`. `patch_script_data` never raises, so this cannot break the cross-post flow.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest test/services/test_task.py -v -k CrossPostPersistsSocialMetadata`
Expected: PASS

Then run the full existing task suite to confirm no regression:
Run: `uv run pytest test/services/test_task.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add app/services/task.py test/services/test_task.py
git commit -m "feat(task): persist generated social metadata to script.json"
```

---

### Task 3: ai_image config accessors, beat count, aspect mapping

Pure functions with no I/O — the foundation the rest of the module builds on.

**Files:**
- Create: `app/services/ai_image.py`
- Modify: `config.example.toml` (document new keys in the `[app]` section)
- Test: `test/services/test_ai_image.py`

**Interfaces:**
- Consumes: `task_artifacts.read_script_data` (Task 1)
- Produces:
  - `ai_image.is_enabled() -> bool`
  - `ai_image.beat_count(audio_duration: float, clip_duration: int) -> int`
  - `ai_image.aspect_ratio(video_aspect) -> str` — accepts `VideoAspect` or its string value, returns `"9:16"` / `"16:9"` / `"1:1"`
  - `ai_image.resolution(video_aspect) -> tuple[int, int]`
  - Constants: `DEFAULT_MODEL`, `DEFAULT_STYLE`, `MOTIONS`

- [ ] **Step 1: Write the failing test**

Create `test/services/test_ai_image.py`:

```python
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.config import config
from app.models.schema import VideoAspect
from app.services import ai_image


class TestAiImageConfig(unittest.TestCase):
    def setUp(self):
        self.original_app_config = dict(config.app)

    def tearDown(self):
        config.app.clear()
        config.app.update(self.original_app_config)

    def test_disabled_by_default(self):
        config.app.pop("ai_image_enabled", None)
        self.assertFalse(ai_image.is_enabled())

    def test_disabled_without_gemini_key(self):
        config.app["ai_image_enabled"] = True
        config.app["gemini_api_key"] = ""
        self.assertFalse(ai_image.is_enabled())

    def test_enabled_with_flag_and_key(self):
        config.app["ai_image_enabled"] = True
        config.app["gemini_api_key"] = "fake-key"
        self.assertTrue(ai_image.is_enabled())


class TestBeatCount(unittest.TestCase):
    def test_exact_division(self):
        self.assertEqual(ai_image.beat_count(35.0, 5), 7)

    def test_rounds_up_partial_beat(self):
        self.assertEqual(ai_image.beat_count(33.0, 5), 7)

    def test_minimum_one_beat(self):
        self.assertEqual(ai_image.beat_count(1.0, 5), 1)

    def test_zero_duration_yields_one_beat(self):
        self.assertEqual(ai_image.beat_count(0.0, 5), 1)

    def test_capped_by_max_images_per_task(self):
        with patch.dict(config.app, {"ai_image_max_images_per_task": 4}):
            self.assertEqual(ai_image.beat_count(60.0, 5), 4)


class TestAspectMapping(unittest.TestCase):
    def test_portrait(self):
        self.assertEqual(ai_image.aspect_ratio(VideoAspect.portrait), "9:16")
        self.assertEqual(ai_image.resolution(VideoAspect.portrait), (1080, 1920))

    def test_landscape(self):
        self.assertEqual(ai_image.aspect_ratio(VideoAspect.landscape), "16:9")

    def test_square(self):
        self.assertEqual(ai_image.aspect_ratio(VideoAspect.square), "1:1")

    def test_accepts_raw_string_value(self):
        self.assertEqual(ai_image.aspect_ratio("9:16"), "9:16")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest test/services/test_ai_image.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.services.ai_image'`

- [ ] **Step 3: Write minimal implementation**

Create `app/services/ai_image.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest test/services/test_ai_image.py -v`
Expected: PASS (12 tests)

- [ ] **Step 5: Document the config keys**

In `config.example.toml`, inside the `[app]` section near the `twelvelabs_*` keys, add:

```toml
# -----------------------------------------------------------------------------
# AI image materials / AI 图片素材
# -----------------------------------------------------------------------------
# Generate images per script beat with Imagen and animate them with a Ken Burns
# effect, instead of downloading stock footage. Set video_source = "aiimage".
# Requires gemini_api_key above. pexels_api_keys is ALSO required: blocked or
# failed beats fall back to a stock clip so the video still ships.
ai_image_enabled = false
# Imagen model id.
ai_image_model = "imagen-3.0-generate-002"
# Style suffix appended to every prompt. Without it, each generated image lands
# in a different art style and the video looks incoherent.
ai_image_style = "peinture à l'huile cinématographique, clair-obscur dramatique, palette or et bleu profond"
# Ken Burns motion: zoom_in, zoom_out, pan_left, pan_right, or random.
ai_image_motion = "random"
# Hard ceiling on images per task — cost guard.
ai_image_max_images_per_task = 12
# Provider used when an image is blocked or fails.
ai_image_fallback_source = "pexels"
```

- [ ] **Step 6: Commit**

```bash
git add app/services/ai_image.py test/services/test_ai_image.py config.example.toml
git commit -m "feat(ai-image): add config accessors, beat count and aspect mapping"
```

---

### Task 4: Visual prompt planning with style lock

One LLM call produces all N prompts, so the model sees the whole narrative arc and varies shot composition instead of repeating itself.

**Files:**
- Modify: `app/services/ai_image.py`
- Test: `test/services/test_ai_image.py`

**Interfaces:**
- Consumes: `task_artifacts.read_script_data` (Task 1), `ai_image.beat_count` (Task 3)
- Produces: `ai_image.plan_prompts(task_id: str, count: int) -> list[str]` — returns exactly `count` prompts, each already carrying the style suffix. Falls back to script-derived prompts if the LLM fails.

- [ ] **Step 1: Write the failing test**

Append to `test/services/test_ai_image.py`:

```python
class TestPlanPrompts(unittest.TestCase):
    def setUp(self):
        self.original_app_config = dict(config.app)
        config.app["ai_image_style"] = "STYLE-LOCK"

    def tearDown(self):
        config.app.clear()
        config.app.update(self.original_app_config)

    @patch("app.services.ai_image.task_artifacts.read_script_data")
    @patch("app.services.ai_image._generate_response")
    def test_returns_exact_count_with_style_appended(self, mock_llm, mock_script):
        mock_script.return_value = {"script": "Prométhée vola le feu.", "search_terms": ["feu"]}
        mock_llm.return_value = '["un titan enchaîné", "une flamme volée", "un aigle"]'

        prompts = ai_image.plan_prompts("task-1", 3)

        self.assertEqual(len(prompts), 3)
        for prompt in prompts:
            self.assertTrue(prompt.endswith("STYLE-LOCK"))
        self.assertIn("un titan enchaîné", prompts[0])

    @patch("app.services.ai_image.task_artifacts.read_script_data")
    @patch("app.services.ai_image._generate_response")
    def test_truncates_when_llm_returns_too_many(self, mock_llm, mock_script):
        mock_script.return_value = {"script": "s", "search_terms": []}
        mock_llm.return_value = '["a", "b", "c", "d", "e"]'

        self.assertEqual(len(ai_image.plan_prompts("task-1", 2)), 2)

    @patch("app.services.ai_image.task_artifacts.read_script_data")
    @patch("app.services.ai_image._generate_response")
    def test_pads_when_llm_returns_too_few(self, mock_llm, mock_script):
        mock_script.return_value = {"script": "s", "search_terms": ["feu"]}
        mock_llm.return_value = '["a"]'

        prompts = ai_image.plan_prompts("task-1", 3)
        self.assertEqual(len(prompts), 3)

    @patch("app.services.ai_image.task_artifacts.read_script_data")
    @patch("app.services.ai_image._generate_response")
    def test_falls_back_to_search_terms_when_llm_fails(self, mock_llm, mock_script):
        mock_script.return_value = {"script": "s", "search_terms": ["feu", "titan"]}
        mock_llm.side_effect = RuntimeError("llm down")

        prompts = ai_image.plan_prompts("task-1", 2)
        self.assertEqual(len(prompts), 2)
        for prompt in prompts:
            self.assertTrue(prompt.endswith("STYLE-LOCK"))

    @patch("app.services.ai_image.task_artifacts.read_script_data")
    @patch("app.services.ai_image._generate_response")
    def test_prompt_instructs_symbolic_framing(self, mock_llm, mock_script):
        mock_script.return_value = {"script": "s", "search_terms": []}
        mock_llm.return_value = '["a", "b"]'

        ai_image.plan_prompts("task-1", 2)

        sent = mock_llm.call_args.kwargs.get("prompt") or mock_llm.call_args.args[0]
        self.assertIn("symbolique", sent.lower())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest test/services/test_ai_image.py -v -k PlanPrompts`
Expected: FAIL with `AttributeError: module 'app.services.ai_image' has no attribute 'plan_prompts'`

- [ ] **Step 3: Write minimal implementation**

Add to `app/services/ai_image.py` — imports first:

```python
import json
import re

from app.services import task_artifacts
from app.services.llm import _generate_response
```

Then the functions:

```python
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


def _parse_prompt_list(response: str) -> list[str]:
    match = re.search(r"\[.*\]", response or "", re.DOTALL)
    if not match:
        raise ValueError("no JSON array found in planning response")
    items = json.loads(match.group(0))
    return [str(item).strip() for item in items if str(item).strip()]


def plan_prompts(task_id: str, count: int) -> list[str]:
    """生成 count 条已锁定风格的画面提示词。LLM 失败时降级为关键词提示。"""
    script_data = task_artifacts.read_script_data(task_id) or {}
    script = str(script_data.get("script", "")).strip()
    search_terms = [str(term) for term in script_data.get("search_terms", []) if term]

    bases: list[str] = []
    try:
        response = _generate_response(prompt=_build_planning_prompt(script, count))
        bases = _parse_prompt_list(response)
    except Exception as exc:
        logger.warning(
            "ai image prompt planning failed, falling back to search terms: "
            f"task_id={task_id}, error={type(exc).__name__}, detail={exc}"
        )

    if not bases:
        bases = search_terms or [script[:200] or "scène mythologique"]

    # 数量对齐：多则截断，少则循环补齐，保证时间线每个分镜都有画面。
    bases = bases[:count]
    while len(bases) < count:
        bases.append(bases[len(bases) % max(len(bases), 1)])

    style = _style()
    return [f"{base}, {style}" for base in bases]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest test/services/test_ai_image.py -v -k PlanPrompts`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add app/services/ai_image.py test/services/test_ai_image.py
git commit -m "feat(ai-image): plan per-beat visual prompts with style lock"
```

---

### Task 5: Imagen generation with caching and RAI classification

**Files:**
- Modify: `app/services/ai_image.py`
- Test: `test/services/test_ai_image.py`

**Interfaces:**
- Consumes: `ai_image.aspect_ratio` (Task 3)
- Produces:
  - `class ai_image.RaiBlockedError(RuntimeError)` — raised when the model returns no image (safety filter)
  - `ai_image.generate_image(prompt: str, ratio: str, out_path: str) -> str` — writes the image, returns `out_path`; raises `RaiBlockedError` on a safety block, other exceptions on transport failure
  - `ai_image.image_cache_path(task_id: str, prompt: str) -> str` — `tasks/<id>/images/<sha256>.jpg`

- [ ] **Step 1: Write the failing test**

Append to `test/services/test_ai_image.py` (add `import hashlib`, `import shutil`, `from unittest.mock import MagicMock`, `from uuid import uuid4`, `from app.utils import utils` to the imports):

```python
class TestGenerateImage(unittest.TestCase):
    def setUp(self):
        self.original_app_config = dict(config.app)
        config.app["gemini_api_key"] = "fake-key"
        config.app["ai_image_model"] = "imagen-test"
        self.task_id = f"test-img-{uuid4().hex}"

    def tearDown(self):
        config.app.clear()
        config.app.update(self.original_app_config)
        shutil.rmtree(utils.task_dir(self.task_id), ignore_errors=True)

    def _client_returning(self, image_bytes):
        generated = MagicMock()
        generated.image.image_bytes = image_bytes
        response = MagicMock()
        response.generated_images = [generated] if image_bytes else []
        client = MagicMock()
        client.models.generate_images.return_value = response
        return client

    def test_cache_path_is_stable_for_same_prompt(self):
        first = ai_image.image_cache_path(self.task_id, "un titan")
        second = ai_image.image_cache_path(self.task_id, "un titan")
        self.assertEqual(first, second)
        self.assertIn(hashlib.sha256("un titan".encode("utf-8")).hexdigest()[:16], first)

    def test_cache_path_differs_for_different_prompts(self):
        self.assertNotEqual(
            ai_image.image_cache_path(self.task_id, "a"),
            ai_image.image_cache_path(self.task_id, "b"),
        )

    @patch("app.services.ai_image._imagen_client")
    def test_writes_image_bytes_to_disk(self, mock_client):
        mock_client.return_value = self._client_returning(b"JPEGDATA")
        out_path = ai_image.image_cache_path(self.task_id, "un titan")

        result = ai_image.generate_image("un titan", "9:16", out_path)

        self.assertEqual(result, out_path)
        self.assertEqual(Path(out_path).read_bytes(), b"JPEGDATA")

    @patch("app.services.ai_image._imagen_client")
    def test_passes_aspect_ratio_to_api(self, mock_client):
        client = self._client_returning(b"X")
        mock_client.return_value = client
        ai_image.generate_image("p", "9:16", ai_image.image_cache_path(self.task_id, "p"))

        kwargs = client.models.generate_images.call_args.kwargs
        self.assertEqual(kwargs["model"], "imagen-test")
        self.assertEqual(kwargs["config"].aspect_ratio, "9:16")

    @patch("app.services.ai_image._imagen_client")
    def test_empty_result_raises_rai_blocked(self, mock_client):
        mock_client.return_value = self._client_returning(None)

        with self.assertRaises(ai_image.RaiBlockedError):
            ai_image.generate_image("gore", "9:16", ai_image.image_cache_path(self.task_id, "gore"))

    @patch("app.services.ai_image._imagen_client")
    def test_transport_error_is_not_rai_blocked(self, mock_client):
        client = MagicMock()
        client.models.generate_images.side_effect = ConnectionError("boom")
        mock_client.return_value = client

        with self.assertRaises(ConnectionError):
            ai_image.generate_image("p", "9:16", ai_image.image_cache_path(self.task_id, "p"))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest test/services/test_ai_image.py -v -k GenerateImage`
Expected: FAIL with `AttributeError: module 'app.services.ai_image' has no attribute 'image_cache_path'`

- [ ] **Step 3: Write minimal implementation**

Add to `app/services/ai_image.py` (add `import hashlib`, `import os`, `from app.utils import utils` to imports):

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest test/services/test_ai_image.py -v -k GenerateImage`
Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
git add app/services/ai_image.py test/services/test_ai_image.py
git commit -m "feat(ai-image): generate images via Imagen with caching and RAI detection"
```

---

### Task 6: Ken Burns rendering via ffmpeg

Requires ffmpeg installed (see Prerequisites). Tests mock `subprocess.run`, but Step 6 verifies against real ffmpeg.

**Files:**
- Modify: `app/services/ai_image.py`
- Test: `test/services/test_ai_image.py`

**Interfaces:**
- Consumes: `ai_image.MOTIONS` (Task 3)
- Produces: `ai_image.render_ken_burns(image_path: str, out_path: str, duration: int, motion: str, width: int, height: int) -> str` — returns `out_path`; raises `RuntimeError` if ffmpeg exits non-zero. `ai_image.pick_motion(index: int) -> str` resolves the configured motion, cycling deterministically through `MOTIONS` when set to `random`.

- [ ] **Step 1: Write the failing test**

Append to `test/services/test_ai_image.py`:

```python
class TestKenBurns(unittest.TestCase):
    def setUp(self):
        self.original_app_config = dict(config.app)

    def tearDown(self):
        config.app.clear()
        config.app.update(self.original_app_config)

    def test_pick_motion_uses_configured_value(self):
        config.app["ai_image_motion"] = "zoom_in"
        self.assertEqual(ai_image.pick_motion(0), "zoom_in")
        self.assertEqual(ai_image.pick_motion(3), "zoom_in")

    def test_pick_motion_cycles_when_random(self):
        config.app["ai_image_motion"] = "random"
        picked = [ai_image.pick_motion(i) for i in range(len(ai_image.MOTIONS) + 1)]
        self.assertEqual(len(set(picked)), len(ai_image.MOTIONS))
        self.assertEqual(picked[0], picked[len(ai_image.MOTIONS)])

    def test_invalid_motion_falls_back_to_zoom_in(self):
        config.app["ai_image_motion"] = "nonsense"
        self.assertEqual(ai_image.pick_motion(0), "zoom_in")

    @patch("app.services.ai_image.subprocess.run")
    def test_builds_zoompan_command_with_output_size(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stderr="")

        result = ai_image.render_ken_burns("/in.jpg", "/out.mp4", 5, "zoom_in", 1080, 1920)

        self.assertEqual(result, "/out.mp4")
        argv = mock_run.call_args.args[0]
        self.assertEqual(argv[0], "ffmpeg")
        self.assertIn("/in.jpg", argv)
        self.assertIn("/out.mp4", argv)
        filter_arg = argv[argv.index("-vf") + 1]
        self.assertIn("zoompan", filter_arg)
        self.assertIn("1080x1920", filter_arg)

    @patch("app.services.ai_image.subprocess.run")
    def test_zoom_out_differs_from_zoom_in(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stderr="")
        ai_image.render_ken_burns("/in.jpg", "/out.mp4", 5, "zoom_in", 1080, 1920)
        zoom_in_filter = mock_run.call_args.args[0][mock_run.call_args.args[0].index("-vf") + 1]

        ai_image.render_ken_burns("/in.jpg", "/out.mp4", 5, "zoom_out", 1080, 1920)
        zoom_out_filter = mock_run.call_args.args[0][mock_run.call_args.args[0].index("-vf") + 1]

        self.assertNotEqual(zoom_in_filter, zoom_out_filter)

    @patch("app.services.ai_image.subprocess.run")
    def test_nonzero_exit_raises(self, mock_run):
        mock_run.return_value = MagicMock(returncode=1, stderr="bad input")

        with self.assertRaises(RuntimeError):
            ai_image.render_ken_burns("/in.jpg", "/out.mp4", 5, "zoom_in", 1080, 1920)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest test/services/test_ai_image.py -v -k KenBurns`
Expected: FAIL with `AttributeError: module 'app.services.ai_image' has no attribute 'pick_motion'`

- [ ] **Step 3: Write minimal implementation**

Add to `app/services/ai_image.py` (add `import subprocess` to imports):

```python
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
    argv = [
        "ffmpeg", "-y", "-loop", "1", "-i", image_path,
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest test/services/test_ai_image.py -v -k KenBurns`
Expected: PASS (6 tests)

- [ ] **Step 5: Verify against real ffmpeg**

The mocked tests prove the command is *built* correctly, not that ffmpeg *accepts* it. Verify manually:

```bash
uv run python -c "
from PIL import Image
Image.new('RGB', (1080, 1920), (40, 30, 90)).save('/tmp/kb-test.jpg')
from app.services import ai_image
print(ai_image.render_ken_burns('/tmp/kb-test.jpg', '/tmp/kb-test.mp4', 5, 'zoom_in', 1080, 1920))
"
ffprobe -v error -show_entries stream=width,height,duration -of csv /tmp/kb-test.mp4
```

Expected: `1080,1920,5.0…`. Repeat for `zoom_out`, `pan_left`, `pan_right`, and watch one to confirm the motion is smooth and in the right direction. **If ffmpeg rejects a filter expression, fix it here — this is the step that catches it.**

- [ ] **Step 6: Commit**

```bash
git add app/services/ai_image.py test/services/test_ai_image.py
git commit -m "feat(ai-image): render Ken Burns clips from stills via ffmpeg zoompan"
```

---

### Task 7: generate_clips orchestration with per-beat fallback

**Files:**
- Modify: `app/services/ai_image.py`
- Test: `test/services/test_ai_image.py`

**Interfaces:**
- Consumes: everything from Tasks 3-6
- Produces: `ai_image.generate_clips(task_id: str, video_aspect, audio_duration: float, clip_duration: int) -> list[str]` — a list of local mp4 paths, same contract as `material.download_videos`. Records `ai_image_fallback_beats` (int) into `script.json` via `patch_script_data`. Never raises for a single failed beat.

- [ ] **Step 1: Write the failing test**

Append to `test/services/test_ai_image.py`:

```python
class TestGenerateClips(unittest.TestCase):
    def setUp(self):
        self.original_app_config = dict(config.app)
        config.app["gemini_api_key"] = "fake-key"
        config.app["ai_image_enabled"] = True
        self.task_id = f"test-clips-{uuid4().hex}"

    def tearDown(self):
        config.app.clear()
        config.app.update(self.original_app_config)
        shutil.rmtree(utils.task_dir(self.task_id), ignore_errors=True)

    @patch("app.services.ai_image.task_artifacts.patch_script_data")
    @patch("app.services.ai_image.render_ken_burns")
    @patch("app.services.ai_image.generate_image")
    @patch("app.services.ai_image.plan_prompts")
    def test_returns_one_clip_per_beat(self, mock_plan, mock_gen, mock_render, mock_patch):
        mock_plan.return_value = ["p1", "p2", "p3"]
        mock_gen.side_effect = lambda prompt, ratio, out: out
        mock_render.side_effect = lambda img, out, *a, **k: out

        clips = ai_image.generate_clips(self.task_id, VideoAspect.portrait, 15.0, 5)

        self.assertEqual(len(clips), 3)
        self.assertEqual(mock_gen.call_count, 3)
        mock_patch.assert_called_with(self.task_id, ai_image_fallback_beats=0)

    @patch("app.services.ai_image.task_artifacts.patch_script_data")
    @patch("app.services.ai_image._fallback_clip")
    @patch("app.services.ai_image.render_ken_burns")
    @patch("app.services.ai_image.generate_image")
    @patch("app.services.ai_image.plan_prompts")
    def test_blocked_beat_falls_back_to_stock(
        self, mock_plan, mock_gen, mock_render, mock_fallback, mock_patch
    ):
        mock_plan.return_value = ["p1", "p2"]
        mock_gen.side_effect = [ai_image.RaiBlockedError("blocked"), "/img2.jpg"]
        mock_render.side_effect = lambda img, out, *a, **k: out
        mock_fallback.return_value = "/stock.mp4"

        clips = ai_image.generate_clips(self.task_id, VideoAspect.portrait, 10.0, 5)

        self.assertEqual(len(clips), 2)
        self.assertIn("/stock.mp4", clips)
        mock_patch.assert_called_with(self.task_id, ai_image_fallback_beats=1)

    @patch("app.services.ai_image.task_artifacts.patch_script_data")
    @patch("app.services.ai_image._fallback_clip")
    @patch("app.services.ai_image.render_ken_burns")
    @patch("app.services.ai_image.generate_image")
    @patch("app.services.ai_image.plan_prompts")
    def test_beat_is_dropped_when_fallback_also_fails(
        self, mock_plan, mock_gen, mock_render, mock_fallback, mock_patch
    ):
        mock_plan.return_value = ["p1", "p2"]
        mock_gen.side_effect = ai_image.RaiBlockedError("blocked")
        mock_fallback.return_value = None

        clips = ai_image.generate_clips(self.task_id, VideoAspect.portrait, 10.0, 5)

        self.assertEqual(clips, [])

    @patch("app.services.ai_image.task_artifacts.patch_script_data")
    @patch("app.services.ai_image._fallback_clip")
    @patch("app.services.ai_image.render_ken_burns")
    @patch("app.services.ai_image.generate_image")
    @patch("app.services.ai_image.plan_prompts")
    def test_network_error_also_falls_back(
        self, mock_plan, mock_gen, mock_render, mock_fallback, mock_patch
    ):
        mock_plan.return_value = ["p1"]
        mock_gen.side_effect = ConnectionError("down")
        mock_fallback.return_value = "/stock.mp4"

        self.assertEqual(ai_image.generate_clips(self.task_id, VideoAspect.portrait, 5.0, 5), ["/stock.mp4"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest test/services/test_ai_image.py -v -k GenerateClips`
Expected: FAIL with `AttributeError: module 'app.services.ai_image' has no attribute 'generate_clips'`

- [ ] **Step 3: Write minimal implementation**

Add to `app/services/ai_image.py`:

```python
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


def generate_clips(task_id, video_aspect, audio_duration: float, clip_duration: int) -> list[str]:
    """生成整条时间线的 AI 图片片段，返回值与 download_videos 一致。"""
    count = beat_count(audio_duration, clip_duration)
    ratio = aspect_ratio(video_aspect)
    width, height = resolution(video_aspect)
    prompts = plan_prompts(task_id, count)

    clip_dir = os.path.join(utils.task_dir(task_id), "images")
    os.makedirs(clip_dir, exist_ok=True)

    clips: list[str] = []
    fallback_beats = 0

    for index, prompt in enumerate(prompts):
        try:
            image_path = generate_image(prompt, ratio, image_cache_path(task_id, prompt))
            clip_path = os.path.join(clip_dir, f"beat-{index:02d}.mp4")
            clips.append(
                render_ken_burns(
                    image_path, clip_path, clip_duration, pick_motion(index), width, height
                )
            )
            continue
        except RaiBlockedError as exc:
            logger.warning(f"beat {index} blocked by safety filter: {exc}")
        except Exception as exc:
            logger.error(
                f"beat {index} generation failed: {type(exc).__name__}, detail={exc}"
            )

        fallback_beats += 1
        fallback = _fallback_clip(task_id, index, video_aspect, clip_duration)
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest test/services/test_ai_image.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add app/services/ai_image.py test/services/test_ai_image.py
git commit -m "feat(ai-image): orchestrate clip generation with per-beat stock fallback"
```

---

### Task 8: Wire `aiimage` into the material dispatch

**Files:**
- Modify: `app/services/material.py:769-777` (top of `download_videos`)
- Test: `test/services/test_material_aiimage.py` (new file, so the large existing material tests stay untouched)

**Interfaces:**
- Consumes: `ai_image.generate_clips` (Task 7)
- Produces: `material.download_videos(source="aiimage", ...)` returns AI-generated clips. All other sources unchanged.

- [ ] **Step 1: Write the failing test**

Create `test/services/test_material_aiimage.py`:

```python
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.models.schema import VideoAspect
from app.services import material


class TestAiImageSourceDispatch(unittest.TestCase):
    @patch("app.services.ai_image.generate_clips")
    def test_aiimage_source_delegates_to_ai_image(self, mock_generate):
        mock_generate.return_value = ["/a.mp4", "/b.mp4"]

        result = material.download_videos(
            task_id="t1",
            search_terms=["feu"],
            source="aiimage",
            video_aspect=VideoAspect.portrait,
            audio_duration=10.0,
            max_clip_duration=5,
        )

        self.assertEqual(result, ["/a.mp4", "/b.mp4"])
        kwargs = mock_generate.call_args.kwargs
        self.assertEqual(kwargs["task_id"], "t1")
        self.assertEqual(kwargs["audio_duration"], 10.0)
        self.assertEqual(kwargs["clip_duration"], 5)

    @patch("app.services.ai_image.generate_clips")
    @patch("app.services.material.search_videos_pexels")
    def test_pexels_source_does_not_call_ai_image(self, mock_pexels, mock_generate):
        mock_pexels.return_value = []

        material.download_videos(
            task_id="t2",
            search_terms=["feu"],
            source="pexels",
            video_aspect=VideoAspect.portrait,
            audio_duration=10.0,
            max_clip_duration=5,
        )

        mock_generate.assert_not_called()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest test/services/test_material_aiimage.py -v`
Expected: FAIL — the first test returns `[]` (falls through to the Pexels path) instead of the AI clips

- [ ] **Step 3: Write minimal implementation**

In `app/services/material.py`, as the **first statements** inside `download_videos` (immediately after the signature, before `provider = "pexels"`):

```python
    if source == "aiimage":
        # 延迟导入避免 ai_image -> material 的循环依赖。
        from app.services import ai_image

        return ai_image.generate_clips(
            task_id=task_id,
            video_aspect=video_aspect,
            audio_duration=audio_duration,
            clip_duration=max_clip_duration,
        )

```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest test/services/test_material_aiimage.py -v`
Expected: PASS (2 tests)

Then confirm no regression in the existing material suite:
Run: `uv run pytest test/services/test_material.py test/services/test_material_cache.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add app/services/material.py test/services/test_material_aiimage.py
git commit -m "feat(material): dispatch aiimage source to the AI image generator"
```

---

### Task 9: Experiment spec loading, validation and grid expansion

**Files:**
- Create: `experiment.py` (repo root, alongside `cli.py`)
- Create: `experiments/round-01.yaml`
- Test: `test/test_experiment.py`

**Interfaces:**
- Consumes: nothing from earlier tasks
- Produces:
  - `experiment.load_spec(path: str) -> dict` — parses and validates; raises `SpecError` on violation
  - `experiment.expand_variants(spec: dict) -> list[dict]` — each item is `{"variant": {...}, "subject": str, "params": {...}}` where `params` is `base` merged with the variant overrides
  - `class experiment.SpecError(ValueError)`
  - Constant `experiment.MAX_VARIANTS = 3`

- [ ] **Step 1: Write the failing test**

Create `test/test_experiment.py`:

```python
import sys
import unittest
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

import experiment


def _write_spec(tmp_path: Path, spec: dict) -> str:
    target = tmp_path / "spec.yaml"
    target.write_text(yaml.safe_dump(spec, allow_unicode=True), encoding="utf-8")
    return str(target)


BASE_SPEC = {
    "name": "myth-fr-round-01",
    "niche": "mythologie",
    "base": {
        "video_language": "fr-FR",
        "video_source": "aiimage",
        "video_count": 1,
        "video_clip_duration": 5,
    },
    "subjects": ["Prométhée", "Cerbère", "Sisyphe"],
    "grid": {"ai_image_style": ["style-a", "style-b"]},
}


class TestLoadSpec(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(__file__).parent / "_tmp_experiment"
        self.tmp.mkdir(exist_ok=True)

    def tearDown(self):
        for child in self.tmp.iterdir():
            child.unlink()
        self.tmp.rmdir()

    def test_loads_valid_spec(self):
        spec = experiment.load_spec(_write_spec(self.tmp, BASE_SPEC))
        self.assertEqual(spec["name"], "myth-fr-round-01")

    def test_rejects_video_count_above_one(self):
        bad = {**BASE_SPEC, "base": {**BASE_SPEC["base"], "video_count": 2}}
        with self.assertRaises(experiment.SpecError) as ctx:
            experiment.load_spec(_write_spec(self.tmp, bad))
        self.assertIn("video_count", str(ctx.exception))

    def test_rejects_more_than_one_grid_axis(self):
        bad = {**BASE_SPEC, "grid": {"ai_image_style": ["a"], "ai_image_motion": ["b"]}}
        with self.assertRaises(experiment.SpecError) as ctx:
            experiment.load_spec(_write_spec(self.tmp, bad))
        self.assertIn("one grid axis", str(ctx.exception))

    def test_rejects_too_many_variants(self):
        bad = {**BASE_SPEC, "grid": {"ai_image_style": ["a", "b", "c", "d"]}}
        with self.assertRaises(experiment.SpecError) as ctx:
            experiment.load_spec(_write_spec(self.tmp, bad))
        self.assertIn("variant", str(ctx.exception))

    def test_rejects_missing_subjects(self):
        bad = {**BASE_SPEC, "subjects": []}
        with self.assertRaises(experiment.SpecError):
            experiment.load_spec(_write_spec(self.tmp, bad))

    def test_aiimage_spec_requires_pexels_keys_for_fallback(self):
        from app.config import config

        original = dict(config.app)
        try:
            config.app["pexels_api_keys"] = []
            with self.assertRaises(experiment.SpecError) as ctx:
                experiment.load_spec(_write_spec(self.tmp, BASE_SPEC))
            self.assertIn("pexels", str(ctx.exception).lower())
        finally:
            config.app.clear()
            config.app.update(original)

    def test_non_aiimage_spec_does_not_require_pexels_keys(self):
        from app.config import config

        original = dict(config.app)
        try:
            config.app["pexels_api_keys"] = []
            local = {**BASE_SPEC, "base": {**BASE_SPEC["base"], "video_source": "local"}}
            experiment.load_spec(_write_spec(self.tmp, local))
        finally:
            config.app.clear()
            config.app.update(original)


class TestExpandVariants(unittest.TestCase):
    def test_one_run_per_subject_per_variant(self):
        runs = experiment.expand_variants(BASE_SPEC)
        self.assertEqual(len(runs), 6)  # 3 subjects x 2 variants

    def test_each_variant_sees_every_subject(self):
        runs = experiment.expand_variants(BASE_SPEC)
        for style in ("style-a", "style-b"):
            subjects = {r["subject"] for r in runs if r["variant"]["ai_image_style"] == style}
            self.assertEqual(subjects, {"Prométhée", "Cerbère", "Sisyphe"})

    def test_params_merge_base_with_variant(self):
        run = experiment.expand_variants(BASE_SPEC)[0]
        self.assertEqual(run["params"]["video_source"], "aiimage")
        self.assertIn("ai_image_style", run["params"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest test/test_experiment.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'experiment'`

- [ ] **Step 3: Write minimal implementation**

Create `experiment.py`:

```python
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

    # AI 图片的兜底路径走图库下载，没有 key 时被安全策略拦截的分镜会直接失败。
    # 与其跑到一半才崩，不如在装载配置时就拒绝。
    if base.get("video_source") == "aiimage":
        from app.config import config

        if not config.app.get("pexels_api_keys"):
            raise SpecError(
                "video_source=aiimage requires pexels_api_keys in config.toml — "
                "blocked or failed beats fall back to stock footage, and without a "
                "key that fallback cannot run"
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest test/test_experiment.py -v`
Expected: PASS (10 tests)

- [ ] **Step 5: Create the round-1 spec file**

Create `experiments/round-01.yaml`:

```yaml
name: myth-fr-round-01
niche: mythologie
base:
  video_language: fr-FR
  video_source: aiimage
  video_count: 1
  video_aspect: "9:16"
  video_clip_duration: 5
  paragraph_number: 3
  subtitle_enabled: true
  # Replace with a real voice id from voice.get_elevenlabs_voices()
  voice_name: "elevenlabs:REPLACE_ME:Narrateur"
subjects:
  - "Le mythe de Prométhée"
  - "Pourquoi Cerbère garde les Enfers"
  - "La malédiction de Sisyphe"
  - "Le fil d'Ariane et le Minotaure"
  - "La boîte de Pandore"
grid:
  ai_image_style:
    - "peinture à l'huile cinématographique, clair-obscur dramatique, palette or et bleu profond"
    - "fresque antique érodée, ocre et terre cuite, lumière rasante"
```

- [ ] **Step 6: Commit**

```bash
git add experiment.py test/test_experiment.py experiments/round-01.yaml
git commit -m "feat(experiment): add spec loading, validation and grid expansion"
```

---

### Task 10: Experiment run/mark/report commands

**Files:**
- Modify: `experiment.py`
- Test: `test/test_experiment.py`

**Interfaces:**
- Consumes: `experiment.load_spec`, `experiment.expand_variants` (Task 9); `task_artifacts.read_script_data` (Task 1); `social_metadata` in `script.json` (Task 2); `ai_image_fallback_beats` in `script.json` (Task 7)
- Produces:
  - `experiment.build_cli_argv(run: dict, task_id: str) -> list[str]`
  - `experiment.record_result(results_path: str, record: dict) -> None`
  - `experiment.load_results(results_path: str) -> list[dict]`
  - `experiment.mark_outcome(results_path: str, key: str, outcome: str) -> bool`
  - `experiment.build_report(records: list[dict]) -> str`
  - CLI: `run <spec>`, `mark <key> --outcome hit|mid|flop`, `report <experiment>`

- [ ] **Step 1: Write the failing test**

Append to `test/test_experiment.py`:

```python
class TestBuildCliArgv(unittest.TestCase):
    def test_maps_params_to_cli_flags(self):
        run = {
            "subject": "Prométhée",
            "variant": {"ai_image_style": "style-a"},
            "params": {"video_source": "aiimage", "video_clip_duration": 5,
                       "subtitle_enabled": True},
        }
        argv = experiment.build_cli_argv(run, "task-123")

        self.assertIn("--video-subject", argv)
        self.assertEqual(argv[argv.index("--video-subject") + 1], "Prométhée")
        self.assertEqual(argv[argv.index("--task-id") + 1], "task-123")
        self.assertEqual(argv[argv.index("--video-source") + 1], "aiimage")
        self.assertEqual(argv[argv.index("--video-clip-duration") + 1], "5")

    def test_true_booleans_become_bare_flags(self):
        run = {"subject": "s", "variant": {}, "params": {"subtitle_enabled": True}}
        argv = experiment.build_cli_argv(run, "t")
        self.assertIn("--subtitle-enabled", argv)
        self.assertNotIn("True", argv)

    def test_ai_image_params_are_not_cli_flags(self):
        run = {"subject": "s", "variant": {"ai_image_style": "x"},
               "params": {"ai_image_style": "x", "video_source": "aiimage"}}
        argv = experiment.build_cli_argv(run, "t")
        self.assertNotIn("--ai-image-style", argv)


class TestResultsStore(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(__file__).parent / "_tmp_results"
        self.tmp.mkdir(exist_ok=True)
        self.results = str(self.tmp / "results.jsonl")

    def tearDown(self):
        for child in self.tmp.iterdir():
            child.unlink()
        self.tmp.rmdir()

    def test_append_and_load_roundtrip(self):
        experiment.record_result(self.results, {"task_id": "a", "title": "T1"})
        experiment.record_result(self.results, {"task_id": "b", "title": "T2"})

        records = experiment.load_results(self.results)
        self.assertEqual([r["task_id"] for r in records], ["a", "b"])

    def test_load_missing_file_returns_empty(self):
        self.assertEqual(experiment.load_results(str(self.tmp / "none.jsonl")), [])

    def test_mark_outcome_by_task_id(self):
        experiment.record_result(self.results, {"task_id": "a", "title": "T1", "outcome": None})

        self.assertTrue(experiment.mark_outcome(self.results, "a", "hit"))
        self.assertEqual(experiment.load_results(self.results)[0]["outcome"], "hit")

    def test_mark_outcome_by_title(self):
        experiment.record_result(self.results, {"task_id": "a", "title": "Le mythe", "outcome": None})

        self.assertTrue(experiment.mark_outcome(self.results, "Le mythe", "flop"))
        self.assertEqual(experiment.load_results(self.results)[0]["outcome"], "flop")

    def test_mark_unknown_key_returns_false(self):
        self.assertFalse(experiment.mark_outcome(self.results, "nope", "hit"))


class TestBuildReport(unittest.TestCase):
    def test_groups_outcomes_per_variant_arm(self):
        records = [
            {"experiment": "e1", "variant": {"ai_image_style": "a"}, "outcome": "hit",
             "fallback_beats": 0, "title": "t1"},
            {"experiment": "e1", "variant": {"ai_image_style": "a"}, "outcome": "flop",
             "fallback_beats": 0, "title": "t2"},
            {"experiment": "e1", "variant": {"ai_image_style": "b"}, "outcome": "flop",
             "fallback_beats": 0, "title": "t3"},
        ]
        report = experiment.build_report(records)

        self.assertIn("ai_image_style=a", report)
        self.assertIn("ai_image_style=b", report)

    def test_flags_contaminated_samples(self):
        records = [
            {"experiment": "e1", "variant": {"ai_image_style": "a"}, "outcome": "hit",
             "fallback_beats": 2, "title": "t1"},
        ]
        report = experiment.build_report(records)
        self.assertIn("excluded", report.lower())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest test/test_experiment.py -v -k "BuildCliArgv or ResultsStore or BuildReport"`
Expected: FAIL with `AttributeError: module 'experiment' has no attribute 'build_cli_argv'`

- [ ] **Step 3: Write minimal implementation**

Add to `experiment.py` (extend imports with `argparse`, `json`, `os`, `subprocess`, `sys`, `uuid`, `from collections import defaultdict`, `from datetime import datetime, timezone`):

```python
DEFAULT_RESULTS_PATH = "experiments/results.jsonl"
VALID_OUTCOMES = ("hit", "mid", "flop")

# ai_image_* 参数通过 config.toml 生效，不是 cli.py 的入参，因此不映射成命令行标志。
_NON_CLI_PREFIXES = ("ai_image_",)


def build_cli_argv(run: dict, task_id: str) -> list[str]:
    argv = [sys.executable, "cli.py", "--task-id", task_id,
            "--video-subject", run["subject"]]

    for key, value in (run.get("params") or {}).items():
        if key.startswith(_NON_CLI_PREFIXES):
            continue
        flag = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            if value:
                argv.append(flag)
            continue
        argv.extend([flag, str(value)])
    return argv


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
    with open(results_path, "w", encoding="utf-8") as results_file:
        for record in records:
            results_file.write(json.dumps(record, ensure_ascii=False) + "\n")


def mark_outcome(results_path: str, key: str, outcome: str) -> bool:
    records = load_results(results_path)
    matched = False
    for record in records:
        if record.get("task_id") == key or record.get("title") == key:
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
        tally = defaultdict(int)
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
```

Then the run command and CLI entry point:

```python
def run_experiment(spec_path: str, results_path: str = DEFAULT_RESULTS_PATH) -> int:
    from app.services import task_artifacts

    spec = load_spec(spec_path)
    runs = expand_variants(spec)
    print(f"{spec['name']}: {len(runs)} video(s) to generate")

    for index, run in enumerate(runs, start=1):
        task_id = str(uuid.uuid4())
        argv = build_cli_argv(run, task_id)
        print(f"\n[{index}/{len(runs)}] {run['subject']} | {run['variant']}")

        completed = subprocess.run(argv)
        script_data = task_artifacts.read_script_data(task_id) or {}
        metadata = script_data.get("social_metadata") or {}

        record_result(results_path, {
            "task_id": task_id,
            "experiment": spec["name"],
            "niche": spec.get("niche", ""),
            "variant": run["variant"],
            "subject": run["subject"],
            "title": metadata.get("title", ""),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "fallback_beats": script_data.get("ai_image_fallback_beats", 0),
            "succeeded": completed.returncode == 0,
            "outcome": None,
        })

        if completed.returncode != 0:
            print(f"  variant failed (exit {completed.returncode}) — continuing batch")

    print(f"\nDone. Results appended to {results_path}")
    return 0


def main() -> int:
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

    args = parser.parse_args()

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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest test/test_experiment.py -v`
Expected: all PASS

- [ ] **Step 5: Verify the CLI wiring end to end**

```bash
uv run python experiment.py report myth-fr-round-01   # expect "no records"
uv run python experiment.py run experiments/round-01.yaml --help
```

Then a single real video, with cross-posting disabled in `config.toml` (`upload_post_enabled = false`) so nothing is published while verifying:

```bash
uv run python experiment.py run experiments/round-01.yaml
```

Watch for: images appearing under `storage/tasks/<id>/images/`, a finished mp4, and one line appended to `experiments/results.jsonl` per video. Confirm `fallback_beats` is 0 or low — a high count means the safety-filter mitigation in Task 4's prompt needs strengthening.

- [ ] **Step 6: Commit**

```bash
git add experiment.py test/test_experiment.py
git commit -m "feat(experiment): add run, mark and report commands with results store"
```

---

## Final verification

- [ ] Full test suite: `uv run pytest test/ -v` — all PASS
- [ ] Rebase surface check: `git diff --stat upstream/main -- app/ webui/ cli.py` should show only `app/services/material.py` (+~10), `app/services/task.py` (+~3), `app/services/task_artifacts.py` (+~18), plus the new `app/services/ai_image.py`. **`webui/Main.py` and `video.py` must not appear.**
- [ ] Enable `upload_post_enabled = true` and run one full variant end to end, confirming it reaches TikTok/Instagram/YouTube
- [ ] Confirm the published YouTube title matches `title` in `results.jsonl` — this is the manual-feedback lookup key and the thing most likely to be subtly wrong

## Deferred (not in this plan)

Per the spec: automated YouTube Analytics ingestion, AI video generation as a source, native still rendering in `video.py`, a second niche, and WebUI surfacing.
