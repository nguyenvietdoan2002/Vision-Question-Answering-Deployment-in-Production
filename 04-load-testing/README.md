# Part 4 — Load testing and batching

Locust against the Part 2 server, a ViLT batch micro-benchmark, and an optional batch endpoint.

## What's new

- `locustfile_sync.py` — POST `/v1/vqa/predict`
- `locustfile_jobs.py` — submit job, poll, fire custom `job_total` request event
- `run_locust.sh` — headless loop over 1 / 5 / 10 / 20 users
- `gpu_monitor.sh` — `nvidia-smi` CSV every 1s → `results/nvidia_smi.csv`
- `batch_benchmark.py` — ViLT batch sizes 1–16, latency + samples/s
- `app_batch.py` — Part 1 copy + `POST /v1/vqa/predict_batch`

`results/` is git-ignored; the scripts create it when you run them.

## Prerequisites

Start the Part 2 API (BLIP + jobs) on port 8000:

```bash
cd ../02-async-jobs
uvicorn app:app --host 0.0.0.0 --port 8000
```

Install Part 4 deps:

```bash
cd ../04-load-testing
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Locust

```bash
# optional GPU sampling in another terminal
chmod +x gpu_monitor.sh run_locust.sh
./gpu_monitor.sh

# sync endpoint
./run_locust.sh locustfile_sync.py

# job API
./run_locust.sh locustfile_jobs.py
```

CSV files land in `results/<name>_u<N>_stats.csv`. Useful columns:

| Column | Meaning |
|--------|---------|
| `Name` | request name |
| `Request Count` | total requests |
| `Failure Count` | failures |
| `Average Response Time` | avg latency (ms) |
| `95%` | P95 latency (ms) |
| `Requests/s` | throughput |

Compare sync vs jobs on avg, P95, req/s, and failures at the same user count.

## Batch benchmark

```bash
python batch_benchmark.py
```

## Batch API (optional)

```bash
uvicorn app_batch:app --host 0.0.0.0 --port 8000

curl -X POST http://localhost:8000/v1/vqa/predict_batch \
  -F "images=@../samples/coco_000000039769.jpg" \
  -F "images=@../samples/coco_000000000139.jpg" \
  -F "questions=How many cats?" \
  -F "questions=Is there a TV?"
```

Expected shape:

```json
{"results": [{"answer": "...", "score": 0.0}], "latency_ms": 0.0, "batch_size": 2}
```
