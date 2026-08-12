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
