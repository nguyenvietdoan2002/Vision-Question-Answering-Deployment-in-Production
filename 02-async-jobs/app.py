"""Part 2: sync predict + async job API. Default model is slower BLIP."""
import io
import queue
import time
import uuid
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from PIL import Image
from transformers import (
    BlipForQuestionAnswering,
    BlipProcessor,
    ViltForQuestionAnswering,
    ViltProcessor,
)

from schemas import JobCreated, JobStatus, VQAResponse
from worker import start_worker

# Switch models here. BLIP is slower → good demo for async jobs.
MODEL_NAME = "Salesforce/blip-vqa-capfilt-large"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

jobs: dict = {}
job_queue: queue.Queue = queue.Queue()


def load_model(name: str):
    if "blip" in name.lower():
        processor = BlipProcessor.from_pretrained(name)
        model = BlipForQuestionAnswering.from_pretrained(name).to(DEVICE)
    else:
        processor = ViltProcessor.from_pretrained(name)
        model = ViltForQuestionAnswering.from_pretrained(name).to(DEVICE)
    model.eval()
    return processor, model


def run_vqa(image_bytes: bytes, question: str) -> dict:
    pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    t0 = time.perf_counter()

    if "blip" in MODEL_NAME.lower():
        inputs = app.state.processor(pil, question, return_tensors="pt")
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        with torch.no_grad():
            out = app.state.model.generate(**inputs)
        answer = app.state.processor.decode(out[0], skip_special_tokens=True)
        latency_ms = (time.perf_counter() - t0) * 1000
        return {"answer": answer, "latency_ms": latency_ms}

    inputs = app.state.processor(pil, question, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = app.state.model(**inputs)
    idx = outputs.logits.argmax(-1).item()
    answer = app.state.model.config.id2label[idx]
    score = torch.softmax(outputs.logits, dim=-1)[0, idx].item()
    latency_ms = (time.perf_counter() - t0) * 1000
    return {"answer": answer, "score": score, "latency_ms": latency_ms}


@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"Loading {MODEL_NAME} on {DEVICE}...")
    app.state.processor, app.state.model = load_model(MODEL_NAME)
    start_worker(job_queue, jobs, run_vqa)
    print("Ready.")
    yield


app = FastAPI(title="Async VQA Jobs", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_NAME, "device": DEVICE}


@app.post("/v1/vqa/predict", response_model=VQAResponse)
async def predict(image: UploadFile = File(...), question: str = Form(...)):
    if not question.strip():
        raise HTTPException(400, "question must not be empty")
    if not (image.content_type or "").startswith("image/"):
        raise HTTPException(400, "file must be an image")
    raw = await image.read()
    try:
        Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        raise HTTPException(400, "could not decode image")
    return VQAResponse(**run_vqa(raw, question))


@app.post("/v1/vqa/jobs", status_code=202, response_model=JobCreated)
async def create_job(
    image: UploadFile = File(...),
    question: str = Form(...),
    callback_url: str = Form(None),
):
    if not question.strip():
        raise HTTPException(400, "question must not be empty")
    if not (image.content_type or "").startswith("image/"):
        raise HTTPException(400, "file must be an image")
    raw = await image.read()
    job_id = str(uuid.uuid4())
    jobs[job_id] = {"job_id": job_id, "status": "queued"}
    job_queue.put((job_id, raw, question, callback_url))
    return JobCreated(job_id=job_id, status="queued")


@app.get("/v1/vqa/jobs/{job_id}", response_model=JobStatus)
def get_job(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, "unknown job_id")
    return JobStatus(**jobs[job_id])
