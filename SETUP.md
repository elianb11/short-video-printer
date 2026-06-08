# short-video-printer — Setup & Catch-up

A fork of [harry0703/MoneyPrinterTurbo](https://github.com/harry0703/MoneyPrinterTurbo).
Generates short videos from a topic: LLM script → stock footage → TTS voiceover →
subtitles → composited video.

## Where things stand

- ✅ Forked to `github.com/elianb11/short-video-printer`
- ✅ Cloned here. Remotes: `origin` = my fork, `upstream` = original repo
- ✅ `config.toml` created from `config.example.toml` (git-ignored — safe for keys)
- ⬜ API keys not filled in yet
- ⬜ Not run yet

## Next step: fill in `config.toml`

| Line | Field | What to put |
|------|-------|-------------|
| 2  | `video_source`     | `"pexels"` (default) or `"pixabay"` |
| 23 | `pexels_api_keys`  | Free key from https://www.pexels.com/api/ → `["yourkey"]` |
| 49 | `llm_provider`     | `"openai"` (default) or another provider |
| 70 | `openai_api_key`   | Your LLM API key |
| 78 | `openai_model_name`| Defaults to `gpt-4o-mini` |

If using Pixabay instead, set `video_source = "pixabay"` and fill `pixabay_api_keys` (line 30).

## How to run

### Option A — Docker (easiest; bundles ffmpeg + Python 3.11)
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

## Architecture (MVC)

| Part | Entry | Port |
|------|-------|------|
| REST API (FastAPI) | `main.py` → `app/asgi.py` → `app/controllers/v1/` | 8080 |
| Web UI (Streamlit) | `webui/Main.py` (via `webui.sh`) | 8501 |

Pipeline services live in `app/services/`:
`llm.py` (script) · `material.py` (stock footage) · `voice.py` (TTS) ·
`subtitle.py` (whisper) · `video.py` (moviepy compositing) · `task.py` (orchestration).

## Local environment notes

- System Python is 3.9; project needs 3.11–3.12 → use `uv` (installed) or Docker.
- `ffmpeg` is **not** installed locally → needed only for Option B.
- Docker 28.5.0 is installed → Option A works out of the box.

## Staying in sync with upstream
```sh
git fetch upstream
git merge upstream/main
```
