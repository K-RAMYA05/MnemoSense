# MnemoSense (Ollama LLM + Voice UI + Video Summaries)

- 24/7 audio logging → **summary-only** storage
- **Voice UI**: click 🎙️ to ask; answer is **spoken** + displayed (no timestamps)
- **Video**: sample URL or live camera every 10s → captions → RAG
- **Free local LLM** via **Ollama** (default model: `llama3.1`)

## Install
```bash
cd MnemoSense
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
brew install ffmpeg  # for Whisper
# Install Ollama from https://ollama.com
ollama pull llama3.1
```

## Run always-on + UI
```bash
python src/runner.py
# open http://127.0.0.1:5005
```

## Test video ingestion (URL)
```bash
python src/video_ingest_sample.py --url "http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/TearsOfSteel.mp4" --every-seconds 10
```

## Live camera (every 10s)
```bash
python src/video_ingest_rt.py --device 0 --every-seconds 10
```
