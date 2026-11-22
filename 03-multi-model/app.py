"""Part 3: one server, three models, lazy or eager load via LOAD_MODE."""
import io
import os
import time
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from PIL import Image

from models import classify, ocr, vqa
from registry import ModelRegistry

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# LOAD_MODE=eager loads all at startup; lazy (default) waits for first request.
LOAD_ALL_AT_STARTUP = os.getenv("LOAD_MODE", "lazy") == "eager"

registry = ModelRegistry(DEVICE)
registry.register("vqa", vqa.load)
registry.register("ocr", ocr.load)
registry.register("classify", classify.load)


def read_image(upload: UploadFile, raw: bytes) -> Image.Image:
    if not (upload.content_type or "").startswith("image/"):
        raise HTTPException(400, "file must be an image")
    try:
        return Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        raise HTTPException(400, "could not decode image")


@asynccontextmanager
async def lifespan(app: FastAPI):
    if LOAD_ALL_AT_STARTUP:
        for name in ("vqa", "ocr", "classify"):
            registry.load(name)
    yield


app = FastAPI(title="Multi-model AI server", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok", "device": DEVICE, "load_mode": "eager" if LOAD_ALL_AT_STARTUP else "lazy"}


@app.get("/v1/models")
def list_models():
    info = {"models": registry.list(), "device": DEVICE}
    if torch.cuda.is_available():
        info["cuda_memory_allocated_mb"] = round(torch.cuda.memory_allocated() / 1e6, 1)
    return info


@app.post("/v1/models/{name}/unload")
def unload_model(name: str):
    if name not in ("vqa", "ocr", "classify"):
        raise HTTPException(404, "unknown model")
    registry.unload(name)
    return {"unloaded": name, "models": registry.list()}


@app.post("/v1/vqa/predict")
async def predict_vqa(image: UploadFile = File(...), question: str = Form(...)):
    if not question.strip():
        raise HTTPException(400, "question must not be empty")
    raw = await image.read()
    pil = read_image(image, raw)
    t0 = time.perf_counter()
    result = vqa.predict(registry.get("vqa"), pil, question, DEVICE)
    result["latency_ms"] = (time.perf_counter() - t0) * 1000
    return result


@app.post("/v1/ocr/predict")
async def predict_ocr(image: UploadFile = File(...)):
    raw = await image.read()
    pil = read_image(image, raw)
    t0 = time.perf_counter()
    result = ocr.predict(registry.get("ocr"), pil, DEVICE)
    result["latency_ms"] = (time.perf_counter() - t0) * 1000
    return result


@app.post("/v1/classification/predict")
async def predict_classify(image: UploadFile = File(...)):
    raw = await image.read()
    pil = read_image(image, raw)
    t0 = time.perf_counter()
    result = classify.predict(registry.get("classify"), pil, DEVICE)
    result["latency_ms"] = (time.perf_counter() - t0) * 1000
    return result
