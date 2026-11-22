"""Hit the Part 1 API with one sample image + question."""
import requests

URL = "http://localhost:8000/v1/vqa/predict"
IMAGE = "../samples/coco_000000039769.jpg"
QUESTION = "How many cats are there?"

with open(IMAGE, "rb") as f:
    r = requests.post(URL, files={"image": ("cats.jpg", f, "image/jpeg")}, data={"question": QUESTION})

print(r.status_code, r.json())
