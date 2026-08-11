# short-video-printer — Setup & Catch-up

A fork of [harry0703/MoneyPrinterTurbo](https://github.com/harry0703/MoneyPrinterTurbo).
Generates short videos from a topic: LLM script → stock footage → TTS voiceover →
subtitles → composited video.

## Where things stand

- ✅ Forked to `github.com/elianb11/short-video-printer`
- ✅ Cloned here. Remotes: `origin` = my fork, `upstream` = original repo
- ✅ Synced with upstream **v1.3.3** (2026-08-11) — fork = upstream + this doc
- ✅ `config.toml` regenerated from the new `config.example.toml` (git-ignored — safe for keys)
- ⬜ API keys not filled in yet
- ⬜ Not run yet

## Next step: fill in `config.toml`

| Line | Field | What to put |
|------|-------|-------------|
| 37 | `video_source`      | `"pexels"` (default), `"pixabay"`, `"coverr"`, or `"local"` |
| 44 | `pexels_api_keys`   | Free key from https://www.pexels.com/api/ → `["yourkey"]` |
| 82 | `llm_provider`      | Ships as `"moonshot"` — set to `"openai"` or another provider |
| 92 | `openai_api_key`    | Your LLM API key |
| 94 | `openai_model_name` | e.g. `gpt-4o-mini` |

Other sources: `pixabay_api_keys` (line 47), `coverr_api_keys` (line 50).
Each provider has its own `*_api_key` / `*_model_name` block further down the file —
fill in the one matching your `llm_provider`.

## How to run

### Option A — Docker (easiest; bundles ffmpeg + Python)
```sh
docker compose up
```
- Web UI:  http://127.0.0.1:8501
- API docs: http://127.0.0.1:8080/docs

### Option B — Local dev with uv
Requires ffmpeg installed locally (`sudo apt install ffmpeg`); uv handles Python 3.11.
```sh
uv python install 3.11
uv sync --frozen
./webui.sh          # Streamlit UI on :8501
# or: uv run python main.py   # FastAPI only, on :8080
```

### Option C — One-shot from the command line
```sh
uv run python cli.py --video-subject "How AI is changing everyday life"
uv run python cli.py --help
```

## Architecture (MVC)

| Part | Entry | Port |
|------|-------|------|
| REST API (FastAPI) | `main.py` → `app/asgi.py` → `app/controllers/v1/` | 8080 |
| Web UI (Streamlit) | `webui/Main.py` (via `webui.sh`) | 8501 |
| CLI | `cli.py` | — |

Pipeline services live in `app/services/`:
`llm.py` (script) · `material.py` (stock footage) · `voice.py` (TTS) ·
`subtitle.py` (whisper) · `video.py` (moviepy compositing) · `task.py` (orchestration) ·
`bgm.py` (background music) · `upload_post.py` (TikTok/Instagram/YouTube Shorts publishing) ·
`twelvelabs.py` (optional semantic material ranking) · `material_cache.py` · `version_checker.py`.

## Local environment notes

- System Python is 3.9; project needs **≥ 3.11** → use `uv` (installed) or Docker.
- `ffmpeg` is **not** installed locally → needed only for Options B/C.
- Docker 28.5.0 is installed → Option A works out of the box.

## Staying in sync with upstream

This fork carries exactly one local commit (this file), rebased on top of upstream:
```sh
git fetch upstream
git rebase upstream/main
git push --force-with-lease origin main
```
