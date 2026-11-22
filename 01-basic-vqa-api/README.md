# Part 1 — Basic VQA inference API

Sync FastAPI endpoint that loads ViLT once at startup and answers one visual question per request.

## What's new

- Model loaded in FastAPI `lifespan`, stored on `app.state`
- `POST /v1/vqa/predict` with image upload + question form field
- 400 on a blank question or a non-image file (a missing or empty `question` field gets the FastAPI 422)
- `GET /health`

## Run

```bash
cd 01-basic-vqa-api
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8000
```

In another terminal:

```bash
python client.py
# or
curl -X POST http://localhost:8000/v1/vqa/predict \
  -F "image=@../samples/coco_000000039769.jpg" \
  -F "question=How many cats are there?"
```

## Expected response shape

```json
{"answer": "...", "score": 0.0, "latency_ms": 0.0}
```

## Docker

```bash
docker build -t vqa-part1 .
docker run --rm -p 8000:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  vqa-part1
# optional GPU: add --gpus all
```
