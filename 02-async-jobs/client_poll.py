"""Submit a job, poll every 0.5s, print status changes and final answer."""
import time

import requests

BASE = "http://localhost:8000"
IMAGE = "../samples/coco_000000039769.jpg"
QUESTION = "How many cats are there?"

with open(IMAGE, "rb") as f:
    r = requests.post(
        f"{BASE}/v1/vqa/jobs",
        files={"image": ("cats.jpg", f, "image/jpeg")},
        data={"question": QUESTION},
    )
print("submit:", r.status_code, r.json())
job_id = r.json()["job_id"]

last = None
while True:
    s = requests.get(f"{BASE}/v1/vqa/jobs/{job_id}").json()
    if s["status"] != last:
        print("status:", s["status"], s)
        last = s["status"]
    if s["status"] in ("completed", "failed"):
        break
    time.sleep(0.5)
