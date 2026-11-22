"""Part 1: sync VQA API. Load ViLT once, answer one question per request."""
import io
import time
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from PIL import Image
from transformers import ViltForQuestionAnswering, ViltProcessor

from schemas import VQAResponse

MODEL_NAME = "dandelin/vilt-b32-finetuned-vqa"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load once at startup — model stays warm for every request.
    print(f"Loading {MODEL_NAME} on {DEVICE}...")
    app.state.processor = ViltProcessor.from_pretrained(MODEL_NAME)
    app.state.model = ViltForQuestionAnswering.from_pretrained(MODEL_NAME).to(DEVICE)
    app.state.model.eval()
    print("Ready.")
    yield


app = FastAPI(title="Basic VQA API", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok", "device": DEVICE}


@app.post("/v1/vqa/predict", response_model=VQAResponse)
async def predict(image: UploadFile = File(...), question: str = Form(...)):
    if not question.strip():
        raise HTTPException(400, "question must not be empty")
    if not (image.content_type or "").startswith("image/"):
        raise HTTPException(400, "file must be an image")

    raw = await image.read()
    try:
        pil = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        raise HTTPException(400, "could not decode image")

    t0 = time.perf_counter()
    inputs = app.state.processor(pil, question, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = app.state.model(**inputs)
    idx = outputs.logits.argmax(-1).item()
    answer = app.state.model.config.id2label[idx]
    score = torch.softmax(outputs.logits, dim=-1)[0, idx].item()
    latency_ms = (time.perf_counter() - t0) * 1000

    return VQAResponse(answer=answer, score=score, latency_ms=latency_ms)
