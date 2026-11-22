# Part 2 — Async job API

Same idea as Part 1, but with a slower BLIP model and a job queue so the client does not block on long inference.

## What's new

- `MODEL_NAME` switch at top of `app.py` (default BLIP)
- Sync `POST /v1/vqa/predict` still available for comparison
- Job API: `POST /v1/vqa/jobs` → 202, `GET /v1/vqa/jobs/{id}` to poll
- One background worker thread + in-memory `jobs` dict
- Optional `callback_url` webhook when a job finishes
- `compare_latency.py` — ViLT vs BLIP mean ms on 5 fixed pairs

## Run the API

```bash
cd 02-async-jobs
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8000
```

## Polling client

```bash
python client_poll.py
```

## Webhook demo

```bash
# terminal A
uvicorn webhook_receiver:app --host 0.0.0.0 --port 9000

# terminal B (API already running on 8000)
curl -X POST http://localhost:8000/v1/vqa/jobs \
  -F "image=@../samples/coco_000000039769.jpg" \
  -F "question=How many cats are there?" \
  -F "callback_url=http://localhost:9000/webhook"
```

## Compare latency

```bash
python compare_latency.py
```

## Expected shapes

Job created:

```json
{"job_id": "...", "status": "queued"}
```

Job completed:

```json
{"job_id": "...", "status": "completed", "answer": "...", "latency_ms": 0.0}
```

## Docker

```bash
docker build -t vqa-part2 .
docker run --rm -p 8000:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  vqa-part2
# optional GPU: add --gpus all
```
