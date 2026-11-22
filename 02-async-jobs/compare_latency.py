"""Load ViLT and BLIP, run the same 5 (image, question) pairs, print mean ms."""
import time

import torch
from PIL import Image
from transformers import (
    BlipForQuestionAnswering,
    BlipProcessor,
    ViltForQuestionAnswering,
    ViltProcessor,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PAIRS = [
    ("../samples/coco_000000039769.jpg", "How many cats are there?"),
    ("../samples/coco_000000039769.jpg", "What color is the couch?"),
    ("../samples/coco_000000000139.jpg", "Is there a TV?"),
    ("../samples/coco_000000000632.jpg", "What room is this?"),
    ("../samples/coco_000000000632.jpg", "Is there a bed?"),
]


def timed_vilt(processor, model, path, question):
    pil = Image.open(path).convert("RGB")
    inputs = processor(pil, question, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        model(**inputs)
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000


def timed_blip(processor, model, path, question):
    pil = Image.open(path).convert("RGB")
    inputs = processor(pil, question, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        model.generate(**inputs)
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000


print(f"Device: {DEVICE}")
print("Loading ViLT...")
vilt_p = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
vilt_m = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa").to(DEVICE)
vilt_m.eval()

print("Loading BLIP...")
blip_p = BlipProcessor.from_pretrained("Salesforce/blip-vqa-capfilt-large")
blip_m = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-capfilt-large").to(DEVICE)
blip_m.eval()

# Warmup
timed_vilt(vilt_p, vilt_m, *PAIRS[0])
timed_blip(blip_p, blip_m, *PAIRS[0])

vilt_ms, blip_ms = [], []
for path, q in PAIRS:
    vilt_ms.append(timed_vilt(vilt_p, vilt_m, path, q))
    blip_ms.append(timed_blip(blip_p, blip_m, path, q))

print(f"{'model':<10} {'mean_ms':>10} {'device':>8}")
print(f"{'ViLT':<10} {sum(vilt_ms)/len(vilt_ms):>10.1f} {DEVICE:>8}")
print(f"{'BLIP':<10} {sum(blip_ms)/len(blip_ms):>10.1f} {DEVICE:>8}")
