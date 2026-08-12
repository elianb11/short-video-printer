import hashlib
import shutil
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.config import config
from app.models.schema import VideoAspect
from app.services import ai_image
from app.utils import utils


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
