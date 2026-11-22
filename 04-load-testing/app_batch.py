"""Part 1-style app plus POST /v1/vqa/predict_batch. No queues."""
import io
import time
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from PIL import Image
from transformers import ViltForQuestionAnswering, ViltProcessor

MODEL_NAME = "dandelin/vilt-b32-finetuned-vqa"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"Loading {MODEL_NAME} on {DEVICE}...")
    app.state.processor = ViltProcessor.from_pretrained(MODEL_NAME)
    app.state.model = ViltForQuestionAnswering.from_pretrained(MODEL_NAME).to(DEVICE)
    app.state.model.eval()
    print("Ready.")
    yield


app = FastAPI(title="Batch VQA API", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok", "device": DEVICE}


@app.post("/v1/vqa/predict")
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
    return {"answer": answer, "score": score, "latency_ms": (time.perf_counter() - t0) * 1000}


@app.post("/v1/vqa/predict_batch")
async def predict_batch(
    images: list[UploadFile] = File(...),
    questions: list[str] = Form(...),
):
    if len(images) != len(questions):
        raise HTTPException(400, "images and questions must have the same length")
    if any(not q.strip() for q in questions):
        raise HTTPException(400, "questions must not be empty")

    pils = []
    for img in images:
        if not (img.content_type or "").startswith("image/"):
            raise HTTPException(400, "every file must be an image")
        raw = await img.read()
        try:
            pils.append(Image.open(io.BytesIO(raw)).convert("RGB"))
        except Exception:
            raise HTTPException(400, "could not decode image")

    t0 = time.perf_counter()
    inputs = app.state.processor(pils, questions, return_tensors="pt", padding=True)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = app.state.model(**inputs)
    probs = torch.softmax(outputs.logits, dim=-1)
    idxs = outputs.logits.argmax(-1)
    results = []
    for i, idx in enumerate(idxs):
        results.append({
            "answer": app.state.model.config.id2label[idx.item()],
            "score": probs[i, idx].item(),
        })
    return {"results": results, "latency_ms": (time.perf_counter() - t0) * 1000, "batch_size": len(pils)}
