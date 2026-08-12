import json
import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

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
        # aiimage 的分镜按叙事顺序生成，随机拼接会把神话讲乱，load_spec 强制要求。
        "video_concat_mode": "sequential",
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

    def test_rejects_grid_axis_on_an_unrouted_ai_image_setting(self):
        # ai_image_enabled 不经 _setting 读取：这样的轴会导出一个没人读的环境
        # 变量，两个分支实际用同一个取值生成，却被标成不同分支。
        bad = {**BASE_SPEC, "grid": {"ai_image_enabled": [True, False]}}
        with self.assertRaises(experiment.SpecError) as ctx:
            experiment.load_spec(_write_spec(self.tmp, bad))
        self.assertIn("ai_image_enabled", str(ctx.exception))

    def test_accepts_grid_axis_on_every_routed_ai_image_setting(self):
        from app.services.ai_image import ENV_ROUTED_SETTINGS

        for axis in sorted(ENV_ROUTED_SETTINGS):
            with self.subTest(axis=axis):
                values = ["pexels", "pixabay"] if axis == "ai_image_fallback_source" \
                    else ["a", "b"]
                spec = {**BASE_SPEC, "grid": {axis: values}}
                self.assertEqual(
                    experiment.load_spec(_write_spec(self.tmp, spec))["name"],
                    "myth-fr-round-01",
                )

    def test_rejects_fallback_source_recursion_declared_in_the_spec(self):
        # 该配置现在经 _setting 读取，spec 能用环境变量覆盖 config.toml，
        # 所以只查 config.toml 的旧检查已经不够。
        for spec in (
            {**BASE_SPEC, "base": {**BASE_SPEC["base"], "ai_image_fallback_source": "aiimage"}},
            {**BASE_SPEC, "grid": {"ai_image_fallback_source": ["pexels", "aiimage"]}},
        ):
            with self.subTest(spec=spec["grid"]):
                with self.assertRaises(experiment.SpecError) as ctx:
                    experiment.load_spec(_write_spec(self.tmp, spec))
                self.assertIn("recursion", str(ctx.exception).lower())

    def test_requires_sequential_concat_mode_for_aiimage(self):
        # plan_prompts 之所以对整篇脚本只做一次 LLM 调用，就是为了让分镜按叙事
        # 顺序排列；video_concat_mode 默认 random 会把它们打乱，神话被讲乱。
        for mode in ("random", "shuffle"):
            with self.subTest(mode=mode):
                bad = {**BASE_SPEC,
                       "base": {**BASE_SPEC["base"], "video_concat_mode": mode}}
                with self.assertRaises(experiment.SpecError) as ctx:
                    experiment.load_spec(_write_spec(self.tmp, bad))
                self.assertIn("video_concat_mode", str(ctx.exception))

    def test_requires_concat_mode_to_be_declared_explicitly(self):
        base = {k: v for k, v in BASE_SPEC["base"].items() if k != "video_concat_mode"}
        bad = {**BASE_SPEC, "base": base}
        with self.assertRaises(experiment.SpecError) as ctx:
            experiment.load_spec(_write_spec(self.tmp, bad))
        self.assertIn("video_concat_mode", str(ctx.exception))

    def test_rejects_random_concat_mode_declared_on_a_grid_axis(self):
        bad = {**BASE_SPEC, "grid": {"video_concat_mode": ["sequential", "random"]}}
        with self.assertRaises(experiment.SpecError) as ctx:
            experiment.load_spec(_write_spec(self.tmp, bad))
        self.assertIn("video_concat_mode", str(ctx.exception))

    def test_non_aiimage_spec_may_use_any_concat_mode(self):
        local = {**BASE_SPEC,
                 "base": {**BASE_SPEC["base"], "video_source": "local",
                          "video_concat_mode": "random"}}
        self.assertEqual(
            experiment.load_spec(_write_spec(self.tmp, local))["name"],
            "myth-fr-round-01",
        )

    def test_video_source_grid_axis_does_not_bypass_the_aiimage_checks(self):
        # "AI 图片 vs 图库素材" 是最顺理成章的第二轮实验，而这条 spec 的
        # base 里没有 aiimage：只看 base 的旧检查会整块跳过，重新打开
        # 无限递归和计费风险。两扇门都必须走同一套校验。
        from app.config import config

        grid_spec = {
            **BASE_SPEC,
            "base": {k: v for k, v in BASE_SPEC["base"].items() if k != "video_source"},
            "grid": {"video_source": ["aiimage", "pexels"]},
        }
        # 门 1：兜底源指回 aiimage（无限递归）。
        original = dict(config.app)
        try:
            config.app["ai_image_fallback_source"] = "aiimage"
            with self.assertRaises(experiment.SpecError) as ctx:
                experiment.load_spec(_write_spec(self.tmp, grid_spec))
            self.assertIn("recursion", str(ctx.exception).lower())
        finally:
            config.app.clear()
            config.app.update(original)

        # 门 2：兜底路径缺 pexels key。
        original = dict(config.app)
        try:
            config.app["pexels_api_keys"] = []
            with self.assertRaises(experiment.SpecError) as ctx:
                experiment.load_spec(_write_spec(self.tmp, grid_spec))
            self.assertIn("pexels", str(ctx.exception).lower())
        finally:
            config.app.clear()
            config.app.update(original)

        # 门 3：顺序拼接同样必须显式声明。
        no_concat = {
            **grid_spec,
            "base": {k: v for k, v in grid_spec["base"].items()
                     if k != "video_concat_mode"},
        }
        with self.assertRaises(experiment.SpecError) as ctx:
            experiment.load_spec(_write_spec(self.tmp, no_concat))
        self.assertIn("video_concat_mode", str(ctx.exception))

        # 三个前置条件都满足时，这条 spec 本身是合法的。
        self.assertEqual(
            experiment.load_spec(_write_spec(self.tmp, grid_spec))["name"],
            "myth-fr-round-01",
        )

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

    def test_false_booleans_become_negated_flags(self):
        # cli.py 的布尔开关用 BooleanOptionalAction，且 --subtitle-enabled 默认为真。
        # 若 False 不发出任何参数，spec 里写 subtitle_enabled: false 的分支会照样
        # 带字幕生成：实验记录的变体与实际视频不符，整轮归因作废。
        run = {"subject": "s", "variant": {}, "params": {"subtitle_enabled": False}}
        argv = experiment.build_cli_argv(run, "t")
        self.assertIn("--no-subtitle-enabled", argv)
        self.assertNotIn("--subtitle-enabled", argv)
        self.assertNotIn("False", argv)

    def test_ai_image_params_are_not_cli_flags(self):
        run = {"subject": "s", "variant": {"ai_image_style": "x"},
               "params": {"ai_image_style": "x", "video_source": "aiimage"}}
        argv = experiment.build_cli_argv(run, "t")
        self.assertNotIn("--ai-image-style", argv)

    def test_cli_accepts_the_aiimage_video_source(self):
        # build_cli_argv 发出 --video-source aiimage；若 cli.py 的 choices 不含它，
        # argparse 会在做任何事之前退出 2，整个实验一个视频也生成不出来。
        import cli

        args = cli.parse_args(["--video-subject", "s", "--video-source", "aiimage"])
        self.assertEqual(args.video_source, "aiimage")


class TestBuildSubprocessEnv(unittest.TestCase):
    """ai_image_* 不是命令行参数，只能靠环境变量按变体区分子进程。"""

    def test_ai_image_params_become_env_overrides(self):
        run = {"subject": "s", "variant": {"ai_image_style": "style-a"},
               "params": {"ai_image_style": "style-a", "video_source": "aiimage"}}
        env = experiment.build_subprocess_env(run)
        self.assertEqual(env, {"MPT_AI_IMAGE_STYLE": "style-a"})

    def test_exported_settings_are_exported_and_read(self):
        # 导出一个没人读的环境变量，等于给一个从未变化的变量贴上不同的分支标签。
        # 因此每个导出的键都必须落在 ai_image 真正经 _setting 读取的集合里。
        from app.services import ai_image

        run = {"subject": "s", "variant": {"ai_image_motion": "pan_left"},
               "params": {"ai_image_motion": "pan_left", "ai_image_model": "m"}}
        env = experiment.build_subprocess_env(run)
        self.assertEqual(env,
                         {"MPT_AI_IMAGE_MOTION": "pan_left", "MPT_AI_IMAGE_MODEL": "m"})

        for key in env:
            setting = key[len(ai_image.ENV_SETTING_PREFIX):].lower()
            self.assertIn(setting, ai_image.ENV_ROUTED_SETTINGS)

        with patch.dict("os.environ", env):
            self.assertEqual(ai_image.pick_motion(0), "pan_left")
            self.assertEqual(ai_image._setting("ai_image_model", "default"), "m")

    def test_cli_params_are_not_duplicated_into_the_environment(self):
        run = {"subject": "s", "variant": {}, "params": {"video_clip_duration": 5}}
        self.assertEqual(experiment.build_subprocess_env(run), {})

    def test_env_keys_match_the_prefix_ai_image_reads(self):
        from app.services import ai_image

        run = {"subject": "s", "variant": {}, "params": {"ai_image_style": "x"}}
        key, = experiment.build_subprocess_env(run)
        self.assertTrue(key.startswith(ai_image.ENV_SETTING_PREFIX))


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

    def test_failed_rewrite_leaves_the_store_intact(self):
        # 人工评级无法重跑：results.jsonl 一旦被截断就永久丢失。
        experiment.record_result(self.results, {"task_id": "a", "title": "T1", "outcome": None})
        experiment.record_result(self.results, {"task_id": "b", "title": "T2", "outcome": "hit"})
        before = Path(self.results).read_text(encoding="utf-8")

        real_dumps = json.dumps
        calls = []

        def failing_dumps(*args, **kwargs):
            calls.append(1)
            if len(calls) == 2:
                raise OSError("no space left on device")
            return real_dumps(*args, **kwargs)

        with patch("experiment.json.dumps", side_effect=failing_dumps):
            with self.assertRaises(OSError):
                experiment.mark_outcome(self.results, "a", "flop")

        self.assertEqual(Path(self.results).read_text(encoding="utf-8"), before)
        self.assertEqual(experiment.load_results(self.results)[1]["outcome"], "hit")
        # 临时文件不能留在目录里。
        self.assertEqual([child.name for child in self.tmp.iterdir()], ["results.jsonl"])

    def test_empty_title_is_not_a_lookup_key(self):
        # social_metadata["title"] 恒存在但可能是空字符串，此时 title 不能当作
        # 匹配键，否则一次 mark 会误伤所有标题为空的记录。
        experiment.record_result(self.results, {"task_id": "a", "title": "", "outcome": None})
        experiment.record_result(self.results, {"task_id": "b", "title": "", "outcome": None})

        self.assertFalse(experiment.mark_outcome(self.results, "", "hit"))
        self.assertTrue(experiment.mark_outcome(self.results, "b", "hit"))
        outcomes = [r["outcome"] for r in experiment.load_results(self.results)]
        self.assertEqual(outcomes, [None, "hit"])


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


class _RunExperimentTestCase(unittest.TestCase):
    """run 相关用例共用的配置固定与临时目录处理。"""

    def setUp(self):
        from app.config import config

        self.original_app_config = dict(config.app)
        config.app["pexels_api_keys"] = ["test-pexels-key"]
        config.app["ai_image_fallback_source"] = "pexels"
        config.app["ai_image_enabled"] = True
        config.app["gemini_api_key"] = "test-gemini-key"

        self.tmp = Path(__file__).parent / "_tmp_run"
        self.tmp.mkdir(exist_ok=True)
        self.results = str(self.tmp / "results.jsonl")

    def tearDown(self):
        from app.config import config

        config.app.clear()
        config.app.update(self.original_app_config)

        for child in self.tmp.iterdir():
            child.unlink()
        self.tmp.rmdir()

    def _spec_path(self, spec: dict) -> str:
        return _write_spec(self.tmp, spec)

    @staticmethod
    def _completed(returncode: int = 0):
        return SimpleNamespace(returncode=returncode)


class TestRunPreflight(_RunExperimentTestCase):
    """生成前的拦截：任何一次误跑都要真金白银地烧 Imagen 额度。"""

    def _run(self, spec: dict):
        with patch("experiment.subprocess.run") as mock_run:
            code = experiment.run_experiment(self._spec_path(spec), self.results)
        return code, mock_run

    def test_rejects_placeholder_voice_name_before_generating(self):
        spec = {**BASE_SPEC, "base": {**BASE_SPEC["base"],
                                      "voice_name": "elevenlabs:REPLACE_ME:Narrateur"}}
        code, mock_run = self._run(spec)

        self.assertEqual(code, 2)
        mock_run.assert_not_called()
        self.assertEqual(experiment.load_results(self.results), [])

    def test_rejects_placeholder_in_grid_values(self):
        spec = {**BASE_SPEC, "grid": {"ai_image_style": ["REPLACE_ME"]}}
        code, mock_run = self._run(spec)

        self.assertEqual(code, 2)
        mock_run.assert_not_called()

    def test_rejects_aiimage_spec_when_ai_image_disabled(self):
        from app.config import config

        config.app["ai_image_enabled"] = False
        code, mock_run = self._run(BASE_SPEC)

        self.assertEqual(code, 2)
        mock_run.assert_not_called()

    def test_rejects_aiimage_spec_without_gemini_key(self):
        from app.config import config

        config.app["gemini_api_key"] = ""
        code, mock_run = self._run(BASE_SPEC)

        self.assertEqual(code, 2)
        mock_run.assert_not_called()

    def test_non_aiimage_spec_does_not_need_ai_image_config(self):
        from app.config import config

        config.app["ai_image_enabled"] = False
        config.app["gemini_api_key"] = ""
        spec = {**BASE_SPEC, "base": {**BASE_SPEC["base"], "video_source": "local"}}

        with patch("experiment.subprocess.run", return_value=self._completed()) as mock_run, \
                patch("app.services.task_artifacts.read_script_data", return_value=None):
            code = experiment.run_experiment(self._spec_path(spec), self.results)

        self.assertEqual(code, 0)
        self.assertTrue(mock_run.called)

    def test_video_source_grid_axis_still_requires_ai_image_enabled(self):
        # 前置检查曾只看 base["video_source"]，一条 video_source 网格轴就能整块
        # 绕过它，然后每个 aiimage 分支都在素材阶段失败。
        from app.config import config

        config.app["ai_image_enabled"] = False
        spec = {
            **BASE_SPEC,
            "base": {k: v for k, v in BASE_SPEC["base"].items() if k != "video_source"},
            "grid": {"video_source": ["aiimage", "pexels"]},
        }
        code, mock_run = self._run(spec)

        self.assertEqual(code, 2)
        mock_run.assert_not_called()

    def test_reports_spec_error_without_traceback(self):
        bad = {**BASE_SPEC, "grid": {"ai_image_style": []}}
        code, mock_run = self._run(bad)

        self.assertEqual(code, 2)
        mock_run.assert_not_called()


class TestRunExperiment(_RunExperimentTestCase):
    SPEC = {**BASE_SPEC, "subjects": ["Prométhée"],
            "grid": {"ai_image_style": ["style-a", "style-b"]}}

    def test_records_one_result_per_run(self):
        script_data = {
            "social_metadata": {"title": "Le mythe", "description": "d", "tags": ["t"]},
            "ai_image_fallback_beats": 1,
        }
        with patch("experiment.subprocess.run", return_value=self._completed()) as mock_run, \
                patch("app.services.task_artifacts.read_script_data", return_value=script_data):
            code = experiment.run_experiment(self._spec_path(self.SPEC), self.results)

        self.assertEqual(code, 0)
        self.assertEqual(mock_run.call_count, 2)

        records = experiment.load_results(self.results)
        self.assertEqual(len(records), 2)
        self.assertEqual([r["variant"]["ai_image_style"] for r in records],
                         ["style-a", "style-b"])
        self.assertEqual(records[0]["experiment"], "myth-fr-round-01")
        self.assertEqual(records[0]["niche"], "mythologie")
        self.assertEqual(records[0]["subject"], "Prométhée")
        self.assertEqual(records[0]["title"], "Le mythe")
        self.assertEqual(records[0]["fallback_beats"], 1)
        self.assertTrue(records[0]["succeeded"])
        self.assertIsNone(records[0]["outcome"])
        self.assertTrue(records[0]["task_id"])

    def test_records_the_beat_total_next_to_the_fallback_count(self):
        # 只记降级数时，"7 个降级 1 个"和"7 个全降级"在记录里无法区分，
        # 而后者代表这条视频里一张 AI 图片都没有。
        script_data = {
            "social_metadata": {"title": "Le mythe", "description": "d", "tags": []},
            "ai_image_beats": 7,
            "ai_image_fallback_beats": 1,
        }
        with patch("experiment.subprocess.run", return_value=self._completed()), \
                patch("app.services.task_artifacts.read_script_data", return_value=script_data):
            experiment.run_experiment(self._spec_path(self.SPEC), self.results)

        record = experiment.load_results(self.results)[0]
        self.assertEqual(record["beats"], 7)
        self.assertEqual(record["fallback_beats"], 1)

    def test_warns_when_every_beat_fell_back(self):
        script_data = {
            "social_metadata": {"title": "Le mythe", "description": "d", "tags": []},
            "ai_image_beats": 7,
            "ai_image_fallback_beats": 7,
        }
        with patch("experiment.subprocess.run", return_value=self._completed()), \
                patch("app.services.task_artifacts.read_script_data", return_value=script_data), \
                patch("builtins.print") as mock_print:
            experiment.run_experiment(self._spec_path(self.SPEC), self.results)

        printed = " ".join(str(call.args[0]) for call in mock_print.call_args_list if call.args)
        self.assertIn("no AI imagery", printed)
        self.assertIn("7", printed)

    def test_does_not_warn_when_some_beats_survived(self):
        script_data = {
            "social_metadata": {"title": "Le mythe", "description": "d", "tags": []},
            "ai_image_beats": 7,
            "ai_image_fallback_beats": 6,
        }
        with patch("experiment.subprocess.run", return_value=self._completed()), \
                patch("app.services.task_artifacts.read_script_data", return_value=script_data), \
                patch("builtins.print") as mock_print:
            experiment.run_experiment(self._spec_path(self.SPEC), self.results)

        printed = " ".join(str(call.args[0]) for call in mock_print.call_args_list if call.args)
        self.assertNotIn("no AI imagery", printed)

    def test_missing_beat_total_does_not_warn(self):
        # 非 aiimage 的运行不会写入 ai_image_beats，缺席不能被读成"全部降级"。
        with patch("experiment.subprocess.run", return_value=self._completed()), \
                patch("app.services.task_artifacts.read_script_data", return_value=None), \
                patch("builtins.print") as mock_print:
            experiment.run_experiment(self._spec_path(self.SPEC), self.results)

        printed = " ".join(str(call.args[0]) for call in mock_print.call_args_list if call.args)
        self.assertNotIn("no AI imagery", printed)
        self.assertEqual(experiment.load_results(self.results)[0]["beats"], 0)

    def test_passes_generated_task_id_to_the_subprocess(self):
        with patch("experiment.subprocess.run", return_value=self._completed()) as mock_run, \
                patch("app.services.task_artifacts.read_script_data", return_value=None):
            experiment.run_experiment(self._spec_path(self.SPEC), self.results)

        argv = mock_run.call_args_list[0].args[0]
        record = experiment.load_results(self.results)[0]
        self.assertEqual(argv[argv.index("--task-id") + 1], record["task_id"])

    def test_records_failed_run_and_continues_the_batch(self):
        with patch("experiment.subprocess.run", return_value=self._completed(1)), \
                patch("app.services.task_artifacts.read_script_data", return_value=None):
            code = experiment.run_experiment(self._spec_path(self.SPEC), self.results)

        records = experiment.load_results(self.results)
        self.assertEqual(code, 1)
        self.assertEqual(len(records), 2)
        self.assertFalse(records[0]["succeeded"])
        self.assertEqual(records[0]["title"], "")
        self.assertEqual(records[0]["fallback_beats"], 0)

    def test_each_variant_gets_its_own_style_in_the_subprocess_environment(self):
        # 这条用例钉死核心缺陷：ai_image_style 只能从 config.toml 读，两个子进程
        # 共用同一份配置，若不按变体注入环境变量，两个分支会用完全相同的风格
        # 生成，而 results.jsonl 却把它们标成 style-a / style-b。
        with patch("experiment.subprocess.run", return_value=self._completed()) as mock_run, \
                patch("app.services.task_artifacts.read_script_data", return_value=None):
            experiment.run_experiment(self._spec_path(self.SPEC), self.results)

        envs = [call.kwargs["env"] for call in mock_run.call_args_list]
        styles = [env["MPT_AI_IMAGE_STYLE"] for env in envs]
        self.assertEqual(styles, ["style-a", "style-b"])
        self.assertEqual(len(set(styles)), 2)

        recorded = [r["variant"]["ai_image_style"] for r in experiment.load_results(self.results)]
        self.assertEqual(recorded, styles)

    def test_model_axis_genuinely_reaches_ai_image(self):
        # 与 style 同形：从被 mock 的 subprocess.run 上取下真实环境，再把它套到
        # ai_image 的读取点上，证明这个轴不只是被导出，而是真的被读到了。
        from app.services import ai_image

        spec = {**BASE_SPEC, "subjects": ["Prométhée"],
                "grid": {"ai_image_model": ["imagen-3.0-generate-002", "imagen-4.0-ultra"]}}

        with patch("experiment.subprocess.run", return_value=self._completed()) as mock_run, \
                patch("app.services.task_artifacts.read_script_data", return_value=None):
            experiment.run_experiment(self._spec_path(spec), self.results)

        seen = []
        for call in mock_run.call_args_list:
            with patch.dict("os.environ", call.kwargs["env"], clear=True):
                seen.append(ai_image._setting("ai_image_model", ai_image.DEFAULT_MODEL))

        self.assertEqual(seen, ["imagen-3.0-generate-002", "imagen-4.0-ultra"])
        self.assertEqual(len(set(seen)), 2)

        recorded = [r["variant"]["ai_image_model"] for r in experiment.load_results(self.results)]
        self.assertEqual(recorded, seen)

    def test_subprocess_environment_extends_rather_than_replaces_os_environ(self):
        with patch("experiment.subprocess.run", return_value=self._completed()) as mock_run, \
                patch("app.services.task_artifacts.read_script_data", return_value=None):
            experiment.run_experiment(self._spec_path(self.SPEC), self.results)

        env = mock_run.call_args_list[0].kwargs["env"]
        for key in os.environ:
            self.assertIn(key, env)

    def test_stores_the_published_truncation_of_a_long_title(self):
        # upload_post 发到 YouTube 时截断到 100 字符，存全长会让人工 mark 永远匹配不上。
        long_title = "Le mythe de Prométhée et le feu volé aux dieux de l'Olympe " * 3
        self.assertGreater(len(long_title), experiment.PUBLISHED_TITLE_MAX)
        script_data = {"social_metadata": {"title": long_title, "description": "d", "tags": []}}

        with patch("experiment.subprocess.run", return_value=self._completed()), \
                patch("app.services.task_artifacts.read_script_data", return_value=script_data):
            experiment.run_experiment(self._spec_path(self.SPEC), self.results)

        stored = experiment.load_results(self.results)[0]["title"]
        self.assertEqual(stored, long_title[:experiment.PUBLISHED_TITLE_MAX])
        self.assertEqual(len(stored), experiment.PUBLISHED_TITLE_MAX)
        self.assertTrue(experiment.mark_outcome(self.results, stored, "hit"))

    def test_short_titles_are_stored_unchanged(self):
        script_data = {"social_metadata": {"title": "Le mythe", "description": "", "tags": []}}
        with patch("experiment.subprocess.run", return_value=self._completed()), \
                patch("app.services.task_artifacts.read_script_data", return_value=script_data):
            experiment.run_experiment(self._spec_path(self.SPEC), self.results)

        self.assertEqual(experiment.load_results(self.results)[0]["title"], "Le mythe")

    def test_empty_title_is_reported_as_a_lookup_gap(self):
        # llm.generate_social_metadata 始终返回三个键，标题失败时是空字符串而不是
        # 缺键；空标题意味着人工反馈只能靠 task_id 查回，必须当场提示。
        script_data = {"social_metadata": {"title": "", "description": "", "tags": []}}
        with patch("experiment.subprocess.run", return_value=self._completed()), \
                patch("app.services.task_artifacts.read_script_data", return_value=script_data), \
                patch("builtins.print") as mock_print:
            experiment.run_experiment(self._spec_path(self.SPEC), self.results)

        printed = " ".join(str(call.args[0]) for call in mock_print.call_args_list if call.args)
        self.assertIn("task_id", printed)
        self.assertEqual(experiment.load_results(self.results)[0]["title"], "")


class TestMainCli(_RunExperimentTestCase):
    def test_report_on_missing_store_returns_one(self):
        code = experiment.main(["--results", self.results, "report", "myth-fr-round-01"])
        self.assertEqual(code, 1)

    def test_report_prints_arms_for_a_known_experiment(self):
        experiment.record_result(self.results, {
            "experiment": "e1", "variant": {"ai_image_style": "a"},
            "outcome": "hit", "fallback_beats": 0, "title": "t1"})

        with patch("builtins.print") as mock_print:
            code = experiment.main(["--results", self.results, "report", "e1"])

        printed = " ".join(str(call.args[0]) for call in mock_print.call_args_list if call.args)
        self.assertEqual(code, 0)
        self.assertIn("ai_image_style=a", printed)

    def test_mark_unknown_key_returns_one(self):
        experiment.record_result(self.results, {"task_id": "a", "title": "T", "outcome": None})
        code = experiment.main(["--results", self.results, "mark", "zzz", "--outcome", "hit"])
        self.assertEqual(code, 1)

    def test_mark_known_key_returns_zero(self):
        experiment.record_result(self.results, {"task_id": "a", "title": "T", "outcome": None})
        code = experiment.main(["--results", self.results, "mark", "a", "--outcome", "mid"])
        self.assertEqual(code, 0)
        self.assertEqual(experiment.load_results(self.results)[0]["outcome"], "mid")

    def test_rejects_unknown_outcome(self):
        with self.assertRaises(SystemExit):
            experiment.main(["--results", self.results, "mark", "a", "--outcome", "great"])
