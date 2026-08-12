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
