"""Call VQA, OCR, classify, and /v1/models — time first vs second call."""
import time

import requests

BASE = "http://localhost:8000"


def timed_post(url, files, data=None):
    t0 = time.perf_counter()
    r = requests.post(url, files=files, data=data)
    ms = (time.perf_counter() - t0) * 1000
    return r, ms


print("=== /v1/models (before) ===")
print(requests.get(f"{BASE}/v1/models").json())

# VQA — first call may load the model (lazy); second is warm.
with open("../samples/coco_000000039769.jpg", "rb") as f:
    r1, ms1 = timed_post(f"{BASE}/v1/vqa/predict", {"image": ("cats.jpg", f, "image/jpeg")}, {"question": "How many cats?"})
with open("../samples/coco_000000039769.jpg", "rb") as f:
    r2, ms2 = timed_post(f"{BASE}/v1/vqa/predict", {"image": ("cats.jpg", f, "image/jpeg")}, {"question": "How many cats?"})
print(f"VQA first={ms1:.0f}ms {r1.json()}  second={ms2:.0f}ms {r2.json()}")

# OCR
with open("../samples/sign_text.png", "rb") as f:
    r1, ms1 = timed_post(f"{BASE}/v1/ocr/predict", {"image": ("sign.png", f, "image/png")})
with open("../samples/sign_text.png", "rb") as f:
    r2, ms2 = timed_post(f"{BASE}/v1/ocr/predict", {"image": ("sign.png", f, "image/png")})
print(f"OCR first={ms1:.0f}ms {r1.json()}  second={ms2:.0f}ms {r2.json()}")

# Classify
with open("../samples/coco_000000039769.jpg", "rb") as f:
    r1, ms1 = timed_post(f"{BASE}/v1/classification/predict", {"image": ("cats.jpg", f, "image/jpeg")})
with open("../samples/coco_000000039769.jpg", "rb") as f:
    r2, ms2 = timed_post(f"{BASE}/v1/classification/predict", {"image": ("cats.jpg", f, "image/jpeg")})
print(f"Classify first={ms1:.0f}ms {r1.json()}  second={ms2:.0f}ms {r2.json()}")

print("=== /v1/models (after) ===")
print(requests.get(f"{BASE}/v1/models").json())
