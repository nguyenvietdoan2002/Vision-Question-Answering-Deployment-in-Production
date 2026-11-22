"""Locust: hammer Part 2 sync POST /v1/vqa/predict."""
from locust import HttpUser, between, task


class SyncVQAUser(HttpUser):
    wait_time = between(0.1, 0.5)

    @task
    def predict(self):
        with open("../samples/coco_000000039769.jpg", "rb") as f:
            self.client.post(
                "/v1/vqa/predict",
                files={"image": ("cats.jpg", f, "image/jpeg")},
                data={"question": "How many cats are there?"},
            )
