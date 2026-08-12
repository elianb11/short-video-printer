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
        from app.config import config

        # 校验读取的是进程级全局配置，仓库里的 config.toml 未必配好 key。
        # 在 setUp 里做快照、在 tearDown 里整体还原，保证无论断言是否失败、
        # 用例内部是否再次改动配置，都不会把脏状态泄漏给其它测试文件。
        self.original_app_config = dict(config.app)
        config.app["pexels_api_keys"] = ["test-pexels-key"]
        config.app["ai_image_fallback_source"] = "pexels"

        self.tmp = Path(__file__).parent / "_tmp_experiment"
        self.tmp.mkdir(exist_ok=True)

    def tearDown(self):
        from app.config import config

        config.app.clear()
        config.app.update(self.original_app_config)

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

    def test_rejects_empty_grid_axis(self):
        # 空轴能通过 len(grid) == 1 和数量上限检查，但 expand_variants 会展开出
        # 零个运行：配置装载成功、却什么都不生成，是最糟糕的失败方式。
        bad = {**BASE_SPEC, "grid": {"ai_image_style": []}}
        with self.assertRaises(experiment.SpecError) as ctx:
            experiment.load_spec(_write_spec(self.tmp, bad))
        self.assertIn("ai_image_style", str(ctx.exception))

    def test_rejects_non_mapping_base(self):
        bad = {**BASE_SPEC, "base": "hello"}
        with self.assertRaises(experiment.SpecError) as ctx:
            experiment.load_spec(_write_spec(self.tmp, bad))
        self.assertIn("base", str(ctx.exception))

    def test_rejects_non_mapping_grid(self):
        bad = {**BASE_SPEC, "grid": "a"}
        with self.assertRaises(experiment.SpecError) as ctx:
            experiment.load_spec(_write_spec(self.tmp, bad))
        self.assertIn("grid", str(ctx.exception))

    def test_rejects_scalar_grid_axis_values(self):
        # 字符串是真值，len() 数的是字符数：三个字符的字符串甚至能通过数量上限，
        # 随后按字符展开成变体。必须显式拒绝。
        bad = {**BASE_SPEC, "grid": {"ai_image_style": "abc"}}
        with self.assertRaises(experiment.SpecError) as ctx:
            experiment.load_spec(_write_spec(self.tmp, bad))
        self.assertIn("ai_image_style", str(ctx.exception))

    def test_rejects_non_numeric_video_count(self):
        bad = {**BASE_SPEC, "base": {**BASE_SPEC["base"], "video_count": "one"}}
        with self.assertRaises(experiment.SpecError) as ctx:
            experiment.load_spec(_write_spec(self.tmp, bad))
        self.assertIn("video_count", str(ctx.exception))

    def test_rejects_scalar_subjects(self):
        bad = {**BASE_SPEC, "subjects": "Prométhée"}
        with self.assertRaises(experiment.SpecError) as ctx:
            experiment.load_spec(_write_spec(self.tmp, bad))
        self.assertIn("subjects", str(ctx.exception))

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

    def test_rejects_aiimage_fallback_pointing_back_at_aiimage(self):
        # ai_image._fallback_clip 把 ai_image_fallback_source 直接交给
        # material.download_videos，而 download_videos 现在又会把 "aiimage"
        # 派发回 ai_image，形成无限递归并持续消耗 Imagen 额度。
        from app.config import config

        original = dict(config.app)
        try:
            config.app["ai_image_fallback_source"] = "aiimage"
            with self.assertRaises(experiment.SpecError) as ctx:
                experiment.load_spec(_write_spec(self.tmp, BASE_SPEC))
            message = str(ctx.exception)
            self.assertIn("ai_image_fallback_source", message)
            self.assertIn("recursion", message.lower())
        finally:
            config.app.clear()
            config.app.update(original)

    def test_non_aiimage_spec_ignores_fallback_source(self):
        from app.config import config

        original = dict(config.app)
        try:
            config.app["ai_image_fallback_source"] = "aiimage"
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
