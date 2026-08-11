# Handoff — mythologie AI-image pipeline (paused after Task 3)

Tracked copy of the SDD ledger, so the resume state survives a machine change.
The live ledger lives at `.superpowers/sdd/2026-08-11-mythologie-ai-image-pipeline/progress.md`,
which is git-ignored and therefore does NOT transfer. Recreate it there from this file
when resuming, or just work from this file.

**Plan:** `docs/plans/2026-08-11-mythologie-ai-image-pipeline.md`
**Spec:** `docs/specs/2026-08-11-mythologie-video-experiment-design.md`
**Branch:** `feat/ai-image-pipeline`, based on `04756ec` on `main`

## State at pause

Tasks 1-3 of 10 complete, each reviewed clean (SPEC ✅ / QUALITY APPROVED).
Tasks 4-10 not started. No agents in flight; working tree clean.

| Task | Commit | Deliverable |
|---|---|---|
| 1 | `2d64b19` | `task_artifacts.read_script_data()` |
| 2 | `df7b5b6` | Social metadata persisted to `script.json` |
| 3 | `cc9da57` | `ai_image.py` foundations + `ai_image_*` config keys |

Test suite: **546 passed, 11 skipped, 0 failures** (baseline before this branch was 530).

## Resume instructions

Dispatch Task 4 (visual prompt planning with style lock).

- BASE for its review package = `cc9da5720709acaf41c7cf190eb3cc4c5eb23fb4`
- Task briefs are generated with
  `<superpowers>/skills/subagent-driven-development/scripts/task-brief docs/plans/2026-08-11-mythologie-ai-image-pipeline.md <N>`
  (briefs 4-10 were pre-generated on the old machine but are git-ignored — regenerate them)
- Carry into the Task 4 dispatch: `ai_image.py` must import `_generate_response` from
  `app.services.llm`, so the test's patch of `app.services.ai_image._generate_response`
  binds correctly.

## Findings that MUST carry forward

These were established by review and contradict what the plan or an implementer claimed.
Ignoring them produces real bugs.

**For Task 7 — `beat_count` is NOT guaranteed >= 1.** It applies `max(count, 1)` *before*
the ceiling clamp, so `ai_image_max_images_per_task <= 0` makes it return 0 or negative.
The Task 3 report claimed callers can rely on `>= 1`; they cannot. Proper fix is
`max(min(count, ceiling), 1)`.

**For Task 10 — `social_metadata["title"]` is ALWAYS present.** `generate_social_metadata`
always returns a dict with all three keys (`llm.py:892`, fallback at `llm.py:907`). It may
be an EMPTY STRING, never absent. Guard on falsy, not on a missing key. The Task 2
implementer's report claimed the opposite; the reviewer corrected it.

**Plan defect, applies to all remaining tasks — the `sys.path.insert` constraint is false.**
The plan's Global Constraints say tests use it "matching every file in `test/services/`".
19 of 34 files there omit it. `test/__init__.py` and `test/services/__init__.py` both exist,
so pytest's rootdir handling already covers it. Tell implementers to match neighbouring
files rather than applying this verbatim.

**Plan defect hit in Task 2 (already worked around, noted in case it recurs).** The brief's
test would have failed against a correct implementation: `_run_cross_post` calls
`_patch_cross_post_state` first (`task.py:843`), and `MemoryState.patch_task` returns `False`
for an unregistered id (`state.py:81`), so it returns early at `task.py:849` before the LLM
call. Fix was registering the task via `patch.object(tm.sm, "state", MemoryState())`, matching
`test_task.py:321,345,362`.

## Deferred minors (for the final whole-branch review to triage)

- Task 1: new tests use real `storage/tasks/` instead of the `TemporaryDirectory` patch used
  by `TestTaskArtifacts` in the same file (plan-mandated verbatim)
- Task 1: `test_returns_none_when_file_missing` uses a fixed non-uuid task id — the only
  shared-global-state test in the file; would go flaky if anything ever writes a `script.json` there
- Task 1: `read_script_data` creates a directory as a side effect of a read, via
  `_script_file` -> `utils.task_dir` -> `makedirs` (pre-existing behaviour, inherited)
- Task 1: no test covers the non-dict guard (e.g. `script.json` holding `[1,2]`)
- Task 1: round-trip test subscripts `result` without `assertIsNotNone` first, so a regression
  yields `TypeError` rather than a clean assertion failure
- Task 2: test asserts title and hashtags but not caption, and lacks
  `mock_post.assert_called_once()` — so the "persistence must never gate cross-posting"
  constraint is itself untested
- Task 3: `TestBeatCount`'s four non-cap tests read live `config.app` without patching —
  spurious failure on a machine whose real `config.toml` sets `ai_image_max_images_per_task` below 7
- Task 3: `int(config.app.get("ai_image_max_images_per_task", ...))` raises on a `None` or
  non-numeric configured value instead of falling back to `DEFAULT_MAX_IMAGES`

## Pre-existing bug found (NOT ours — do not fix on this branch)

`task.py:873` — `metadata.get("title", video_subject)` can never fire its default, because the
key is always present. An empty LLM title therefore publishes as `""` rather than falling back
to `video_subject`. Belongs in a separate change against upstream.

## Environment setup needed on the new machine

The old machine's setup does not transfer. On the new one:

```sh
curl -LsSf https://astral.sh/uv/install.sh | sh   # if uv is missing
uv python install 3.11
uv sync --frozen
```

Plus **ffmpeg with the `zoompan` filter** (Task 6 needs it; verify with
`ffmpeg -filters | grep zoompan` and that `ffprobe` exists). On the old machine — AlmaLinux 9 —
neither `apt` nor `dnf` could provide it (RHEL excludes ffmpeg over patent licensing);
`brew install ffmpeg` was the working no-sudo route. On Debian/Ubuntu, `apt install ffmpeg`
is fine.

Still unset, and blocking ONLY the end-to-end verification steps in Tasks 6 and 10 (every unit
test is mocked, so Tasks 4-10 can be implemented and reviewed without them):
`gemini_api_key`, `pexels_api_keys`, an ElevenLabs voice id, and Upload-Post credentials
in `config.toml`.
