# Part 3 — Multi-model AI server

One FastAPI process hosts VQA, OCR, and image classification behind a small `ModelRegistry`.

## What's new

- `registry.py` — register / load / get / unload / list (~50 lines)
- Lazy load (default) vs eager: `LOAD_MODE=lazy|eager`
- `GET /v1/models` — loaded status + approx memory MB (+ CUDA allocated if GPU)
- `POST /v1/models/{name}/unload` — free memory (`del` + `gc` + `empty_cache`)
- Three endpoints: `/v1/vqa/predict`, `/v1/ocr/predict`, `/v1/classification/predict`

## Run

```bash
cd 03-multi-model
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# lazy (default): first request per model pays the load cost
uvicorn app:app --host 0.0.0.0 --port 8000

# or eager: load all three at startup
LOAD_MODE=eager uvicorn app:app --host 0.0.0.0 --port 8000
```

## Client (times first vs second call)

```bash
python client.py
```

## Example curls

```bash
curl -X POST http://localhost:8000/v1/vqa/predict \
  -F "image=@../samples/coco_000000039769.jpg" \
  -F "question=How many cats are there?"

curl -X POST http://localhost:8000/v1/ocr/predict \
  -F "image=@../samples/sign_text.png"

curl -X POST http://localhost:8000/v1/classification/predict \
  -F "image=@../samples/coco_000000039769.jpg"

curl http://localhost:8000/v1/models
curl -X POST http://localhost:8000/v1/models/ocr/unload
```

## Expected response shapes

```json
{"answer": "...", "score": 0.0, "latency_ms": 0.0}
{"text": "...", "latency_ms": 0.0}
{"predictions": [{"label": "...", "score": 0.0}], "latency_ms": 0.0}
{"models": [{"name": "vqa", "loaded": true, "device": "cpu", "approx_memory_mb": 0.0}], "device": "cpu"}
```

## Docker

```bash
docker build -t vqa-part3 .
docker run --rm -p 8000:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -e LOAD_MODE=lazy \
  vqa-part3
# optional GPU: add --gpus all
```
