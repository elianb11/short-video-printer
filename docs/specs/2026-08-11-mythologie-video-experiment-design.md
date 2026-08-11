# Mythologie video experiment pipeline — design

**Date:** 2026-08-11
**Status:** approved, ready for implementation planning
**Fork:** `elianb11/short-video-printer` (fork of `harry0703/MoneyPrinterTurbo`, synced to v1.3.3)

## Goal

Produce French-language mythology Shorts at 2-3/day on one channel, published to
TikTok, Instagram and YouTube Shorts, and find which generation parameters produce
videos that hold attention. Round 1 uses manual outcome reporting; automated
analytics ingestion is explicitly deferred.

## What already exists

The upstream pipeline is end-to-end already. `task.py::_run_pipeline` runs:

1. `generate_script` — LLM writes the narration
2. `generate_terms` — LLM extracts search keywords
3. `save_script_data` — persists script + terms + params to `tasks/<id>/script.json`
4. `generate_audio` — TTS, returns `audio_duration`
5. `generate_subtitle` — Whisper `large-v3`, `word_timestamps=True`
6. `get_video_materials` — fetches clips
7. `generate_final_videos` — MoviePy composite
8. social metadata — `llm.py:825` generates per-platform title/caption/hashtags
9. cross-post — Upload-Post to TikTok/Instagram/YouTube Shorts, persists `request_id`
   into the task record via `_patch_cross_post_state`

`cli.py` drives this with ~40 flags and a `--stop-at` stage selector.
ElevenLabs TTS is already implemented (`voice.py:1278`, `[elevenlabs]` config section,
`eleven_multilingual_v2` default) and needs **no code change** — only configuration.

Nothing in this design re-implements any of the above.

## Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Niche | Mythologie (grecque, nordique, égyptienne) | AI-generated imagery is a moat here — stock footage cannot depict Cerbère or Ragnarök. Evergreen, so the back catalogue compounds at low publish volume. No YMYL or monetization risk. |
| Language | French | Far thinner AI-content-farm competition than English, so cleaner signal per video. |
| Visuals | AI images + Ken Burns motion | Unlocks topic-specific visuals that stock cannot serve. Far cheaper than AI video generation. |
| TTS | ElevenLabs (`eleven_multilingual_v2`) | Materially better French than edge-tts; voice is a major retention factor in a narration-driven format. |
| Publishing | All platforms via Upload-Post | Already implemented; no work required. |
| Feedback | Manual outcome reporting | Defers the riskiest integration. Automated analytics is a later round. |
| Volume | 2-3/day, one channel | Safe for a new channel; ~20 samples/week. |

## Architecture

The load-bearing insight: `material.download_videos()` returns `List[str]` — plain
local video file paths (`material.py:768`). Nothing downstream cares about their
origin. AI images therefore enter the system by impersonating a stock provider, and
the rest of the pipeline is untouched.

```
experiment.py  ──►  cli.py  ──►  task.py::_run_pipeline        [UNCHANGED]
                                        │
                                   step 3 writes tasks/<id>/script.json
                                        │
                                   step 6 get_video_materials
                                        ▼
                          material.download_videos(source="aiimage")
                                        │  early-return branch (new)
                                        ▼
                          ai_image.generate_clips(task_id, aspect,
                                                  audio_duration, clip_duration)
                                        │   reads script.json for beat context
                                        │   a) LLM: script → N visual prompts
                                        │   b) Imagen: prompt → 1080×1920 image
                                        │   c) ffmpeg zoompan: image → N-sec mp4
                                        ▼
                              List[str] of ordinary mp4 paths
                                        │
                                        ▼
                    combine_videos / transitions / subtitles / audio  [UNCHANGED]
                    social metadata / Upload-Post cross-post          [UNCHANGED]
                                        │
                                        ▼
                              experiments/results.jsonl
```

### Files

| File | Change | Rebase conflict risk |
|---|---|---|
| `app/services/ai_image.py` | new, ~200 lines | none |
| `experiment.py` | new, top-level runner | none |
| `app/services/material.py` | +3 lines, early-return branch at the `source` dispatch (`material.py:771`) | negligible |
| `app/services/task_artifacts.py` | +4 lines, `read_script_data()` | negligible |
| `config.example.toml` | new `[ai_image]` section | trivial |
| `video.py`, `webui/Main.py`, `task.py` | **zero edits** | — |

Keeping `video.py` and `webui/Main.py` untouched is a deliberate constraint, not an
accident: they are upstream's most-churned files (`webui/Main.py` changed by +5044
lines in the v1.2.9 → v1.3.3 sync). Every edit there becomes recurring merge cost.

### Why `ai_image.py` reads `script.json` rather than taking the script as an argument

`download_videos()`'s signature carries only `search_terms` — terse keywords that make
weak image prompts. The full script is already persisted at pipeline step 3 under the
same `task_id`. Reading it from disk gives beat-level narrative context **without
changing any function signature in `task.py`**, which is what holds the rebase surface
at zero.

## Component: `ai_image.py`

**Entry point:** `generate_clips(task_id, video_aspect, audio_duration, clip_duration) -> List[str]`

**1. Beat planning.** `N = ceil(audio_duration / clip_duration)`. Read `script.json`,
then make **one** LLM call returning exactly N visual prompts, each covering the
narrative beat that plays in its slot. One call rather than N so the model sees the
whole arc and varies shot composition (wide → close-up → wide) instead of producing N
near-identical images.

> **Implementation trap:** `get_video_materials` passes `audio_duration * params.video_count`
> into `download_videos` (`task.py:594`), not the raw duration. `ai_image.py` receives that
> product. Experiment specs must therefore pin `video_count: 1`, and the runner asserts it —
> otherwise the beat count silently multiplies and so does the image bill.

**2. Style lock.** Every prompt receives the same style suffix, e.g.
`"peinture à l'huile cinématographique, clair-obscur dramatique, palette or et bleu profond"`.
Without this, N independent generations produce N unrelated art styles and the video
reads as broken. The style string is a config key and a first-class experiment variable.

**3. Generation.** `client.models.generate_images(model=<config>, prompt=...,
config=types.GenerateImagesConfig(number_of_images=1, aspect_ratio="9:16",
output_mime_type="image/jpeg"))` via `google-genai`, already a dependency
(`google-genai==2.11.0`). `VideoAspect` maps to `9:16` / `16:9` / `1:1`, all natively
supported, so no cropping. Images cached at
`tasks/<id>/images/<sha256(prompt)>.jpg` — re-running a variant costs nothing.
The exact Imagen model id is a config key, verified current at implementation time.

**4. Ken Burns.** Still → `clip_duration`-second mp4 via **ffmpeg's `zoompan` filter**,
not MoviePy: ffmpeg is already a hard dependency, it is much faster for this, and it
avoids MoviePy 2.x's renamed API (`resized` / `with_position`) which upstream may churn.
Four motion presets — `zoom_in`, `zoom_out`, `pan_left`, `pan_right` — assigned per
clip, and an experiment variable.

### Safety-filter risk and graceful degradation

Mythology will trip Imagen's safety filters. Greek and Norse myth is saturated with
violence and nudity (Kronos devouring his children, Prométhée's liver, classical nudity).
A nontrivial share of prompts will return RAI-blocked. Two mitigations:

- The prompt-planning LLM call is instructed to compose **atmospheric and symbolic**
  framing rather than literal depiction — *"l'ombre d'un titan au-dessus d'un temple en
  ruine"* rather than the gore. This is also better filmmaking.
- **Per-beat fallback to stock:** any beat that fails generation or is blocked falls
  back to a Pexels clip for that slot only. The video still ships. The count is recorded
  as `fallback_beats` on the result record so degraded videos can be excluded from
  comparisons rather than misread as a style effect.

RAI blocks are logged distinctly from network errors, so "prompt too spicy" is
distinguishable from "API down".

**This makes Pexels a hard prerequisite, not an alternative.** `pexels_api_keys` must be
configured even though `video_source` is `aiimage`, because the fallback path calls
`search_videos_pexels`. If the fallback itself cannot run, a blocked beat fails the whole
task. The runner validates this at spec-load time rather than failing mid-batch.

### Configuration

New `[ai_image]` section in `config.example.toml`:

| Key | Purpose |
|---|---|
| `enabled` | master switch |
| `model` | Imagen model id; pinned in config, confirmed current at implementation time |
| `style` | the style-lock suffix — the round-1 experiment variable |
| `motion` | `zoom_in` \| `zoom_out` \| `pan_left` \| `pan_right` \| `random` |
| `max_images_per_task` | runaway-cost guard |
| `fallback_source` | provider for blocked beats, default `pexels` |

The Gemini API key is reused from the existing `[gemini]`/LLM configuration rather than
duplicated.

### Cost

~7 images/video at current Imagen pricing ≈ **$0.25/video** → ~$23/month at 3/day.
ElevenLabs: a 35s French script ≈ 600 characters → ~54k chars/month, within the
Creator tier (~$22/month, 100k chars). Script LLM is pennies; edge-tts unused; Whisper
runs locally. **Total ≈ $45/month.** Cost does not constrain the experiment design.
Config carries a `max_images_per_task` ceiling as a runaway guard.

## Component: `experiment.py`

At 2-3 videos/day you get ~20 samples/week. Shorts distribution variance is large, so
an undisciplined "change everything each run" approach can never attribute an outcome
to a parameter. The runner exists to enforce that discipline.

### Spec format (`experiments/round-01.yaml`)

```yaml
name: myth-fr-round-01
niche: mythologie
base:
  video_language: fr-FR
  video_source: aiimage
  voice_name: "elevenlabs:<voice_id>:Narrateur"
  video_aspect: "9:16"
  video_clip_duration: 5
  video_count: 1          # required — see the beat-count trap above
  paragraph_number: 3
  subtitle_enabled: true
subjects:
  - "Le mythe de Prométhée"
  - "Pourquoi Cerbère garde les Enfers"
  - "La malédiction de Sisyphe"
grid:
  ai_image_style:
    - "peinture à l'huile cinématographique, clair-obscur"
    - "fresque antique érodée, ocre et terre cuite"
```

### Behaviour

Expands `grid` into variants, pairs each with a rotating subject, invokes the existing
pipeline per variant sequentially, and appends one record per video to
`experiments/results.jsonl`:

```json
{"task_id": "...", "experiment": "myth-fr-round-01",
 "variant": {"ai_image_style": "..."},
 "subject": "Le mythe de Prométhée",
 "title": "<generated social title>",
 "generated_at": "2026-08-11T10:00:00Z",
 "fallback_beats": 0,
 "video_path": "...",
 "cross_post": {"youtube": {"request_id": "..."}},
 "outcome": null}
```

`title` is recorded because it is the **manual lookup key**: you see a video perform on
YouTube, and `experiment.py report` maps that title back to the exact parameters.

### Enforced rules

- **One grid axis per round.** ~20 samples/week over a 2-variant axis is ~10 per arm,
  already marginal; a 2×2 grid gives ~5 per arm, which is noise. The runner rejects a
  spec whose grid expands past a configurable variant ceiling.
- **Subject rotation across variants**, so an arm cannot win by drawing better topics.

### Commands

- `experiment.py run <spec.yaml>` — expand, generate, publish, record
- `experiment.py mark <title|task_id> --outcome hit|mid|flop` — record manual judgment
- `experiment.py report <experiment>` — per-arm outcome tally, n per arm, and count
  excluded for `fallback_beats > 0`

A failed variant is recorded and the batch continues; it does not abort the run.

## Statistical honesty

**At ~10 samples per arm this will not produce statistically significant results.**
Retention variance between identical Shorts is large, and manual hit/mid/flop is coarser
still. Round 1 can reliably say *"this arm is grossly worse, kill it"*. It cannot say
*"this arm is 8% better"*. Experiments eliminate losers; they do not fine-tune winners.
Any arm difference that is not stark should be re-run, not acted on.

This is recorded because the tempting failure mode is over-reading a three-video
difference and locking in a wrong recipe for months.

## Testing

Follows existing `test/services/` conventions (`test_twelvelabs.py`,
`test_upload_post.py` as models); all external APIs mocked.

- `test_ai_image.py` — beat-count math, aspect mapping, prompt planning and style-lock
  application, cache-hit behaviour, per-beat fallback path, RAI-vs-network error
  classification
- `test_experiment.py` — grid expansion, variant-ceiling enforcement, subject rotation,
  result record shape, continue-past-failure

## Explicitly out of scope for round 1

- Automated analytics ingestion (YouTube Analytics API, OAuth, `request_id` →
  `videoId` join resolution, snapshot cadence). `results.jsonl` is shaped so this drops
  in later without migration.
- AI video generation as a material source.
- Native still-image rendering inside `video.py`.
- A second niche (histoire) — shares this visual pipeline once the recipe is validated.
- WebUI surfacing of any of the above.
