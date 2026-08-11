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
