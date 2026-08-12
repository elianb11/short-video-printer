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
    def test_pads_by_cycling_through_bases(self, mock_llm, mock_script):
        """补齐必须循环取值，而不是反复重复第一条。"""
        mock_script.return_value = {"script": "s", "search_terms": []}
        mock_llm.return_value = '["alpha", "beta", "gamma"]'

        prompts = ai_image.plan_prompts("task-1", 5)

        self.assertEqual(len(prompts), 5)
        self.assertIn("alpha", prompts[3])
        self.assertIn("beta", prompts[4])
        self.assertNotIn("alpha", prompts[4])

    @patch("app.services.ai_image.task_artifacts.read_script_data")
    @patch("app.services.ai_image._generate_response")
    def test_fallback_prompts_carry_symbolic_framing(self, mock_llm, mock_script):
        """降级路径同样要带象征化框架，否则安全缓解在最需要时消失。"""
        mock_script.return_value = {"script": "s", "search_terms": ["sacrifice", "bataille"]}
        mock_llm.side_effect = RuntimeError("llm down")

        prompts = ai_image.plan_prompts("task-1", 2)

        self.assertEqual(len(prompts), 2)
        for prompt in prompts:
            self.assertIn("symbolique", prompt.lower())
            self.assertTrue(prompt.endswith("STYLE-LOCK"))
        self.assertIn("sacrifice", prompts[0])

    @patch("app.services.ai_image.task_artifacts.read_script_data")
    @patch("app.services.ai_image._generate_response")
    def test_rejects_non_string_items_and_falls_back(self, mock_llm, mock_script):
        """字典元素不得被 str() 化成垃圾提示词，应直接降级。"""
        mock_script.return_value = {"script": "s", "search_terms": ["feu"]}
        mock_llm.return_value = '[{"prompt": "x"}]'

        prompts = ai_image.plan_prompts("task-1", 1)

        self.assertEqual(len(prompts), 1)
        self.assertNotIn("'prompt'", prompts[0])
        self.assertIn("feu", prompts[0])

    @patch("app.services.ai_image.task_artifacts.read_script_data")
    @patch("app.services.ai_image._generate_response")
    def test_error_string_response_falls_back(self, mock_llm, mock_script):
        """_generate_response 失败时返回 "Error: ..." 而非抛异常，必须显式识别。"""
        mock_script.return_value = {"script": "s", "search_terms": ["feu"]}
        mock_llm.return_value = "Error: rate limited"

        prompts = ai_image.plan_prompts("task-1", 2)

        self.assertEqual(len(prompts), 2)
        for prompt in prompts:
            self.assertNotIn("rate limited", prompt)
            self.assertIn("symbolique", prompt.lower())

    @patch("app.services.ai_image.task_artifacts.read_script_data")
    @patch("app.services.ai_image._generate_response")
    def test_error_string_embedding_json_array_falls_back(self, mock_llm, mock_script):
        """错误信息里恰好含 JSON 数组时，也不能被当成提示词解析。"""
        mock_script.return_value = {"script": "s", "search_terms": ["feu"]}
        mock_llm.return_value = 'Error: invalid payload ["boom"]'

        prompts = ai_image.plan_prompts("task-1", 1)

        self.assertEqual(len(prompts), 1)
        self.assertNotIn("boom", prompts[0])
        self.assertIn("feu", prompts[0])

    @patch("app.services.ai_image.task_artifacts.read_script_data")
    @patch("app.services.ai_image._generate_response")
    def test_prompt_instructs_symbolic_framing(self, mock_llm, mock_script):
        mock_script.return_value = {"script": "s", "search_terms": []}
        mock_llm.return_value = '["a", "b"]'

        ai_image.plan_prompts("task-1", 2)

        sent = mock_llm.call_args.kwargs.get("prompt") or mock_llm.call_args.args[0]
        self.assertIn("symbolique", sent.lower())
