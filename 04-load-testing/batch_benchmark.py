"""ViLT batch sizes 1..16: print batch latency ms and samples/s."""
import time

import torch
from PIL import Image
from transformers import ViltForQuestionAnswering, ViltProcessor

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "dandelin/vilt-b32-finetuned-vqa"
IMAGE = "../samples/coco_000000039769.jpg"
QUESTION = "How many cats are there?"
BATCH_SIZES = [1, 2, 4, 8, 16]

print(f"Device: {DEVICE}")
processor = ViltProcessor.from_pretrained(MODEL_NAME)
model = ViltForQuestionAnswering.from_pretrained(MODEL_NAME).to(DEVICE)
model.eval()

pil = Image.open(IMAGE).convert("RGB")


def run_batch(n: int) -> float:
    images = [pil] * n
    questions = [QUESTION] * n
    inputs = processor(images, questions, return_tensors="pt", padding=True)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        model(**inputs)
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000


# Warmup
run_batch(1)

print(f"{'batch':>6} {'latency_ms':>12} {'samples/s':>12}")
for n in BATCH_SIZES:
    ms = run_batch(n)
    sps = n / (ms / 1000)
    print(f"{n:>6} {ms:>12.1f} {sps:>12.1f}")
