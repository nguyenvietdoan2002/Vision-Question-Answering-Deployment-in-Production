"""Locust: submit a job, poll until done, report total job time as a custom request."""
import time

from locust import HttpUser, between, task


class JobVQAUser(HttpUser):
    wait_time = between(0.1, 0.5)

    @task
    def submit_and_poll(self):
        t0 = time.perf_counter()
        with open("../samples/coco_000000039769.jpg", "rb") as f:
            r = self.client.post(
                "/v1/vqa/jobs",
                files={"image": ("cats.jpg", f, "image/jpeg")},
                data={"question": "How many cats are there?"},
                name="/v1/vqa/jobs [submit]",
            )
        if r.status_code != 202:
            return
        job_id = r.json()["job_id"]

        # Poll until completed or failed.
        while True:
            s = self.client.get(
                f"/v1/vqa/jobs/{job_id}",
                name="/v1/vqa/jobs [poll]",
            )
            status = s.json().get("status")
            if status in ("completed", "failed"):
                break
            time.sleep(0.5)

        # Report end-to-end job time as its own metric.
        total_ms = (time.perf_counter() - t0) * 1000
        self.environment.events.request.fire(
            request_type="JOB",
            name="job_total",
            response_time=total_ms,
            response_length=0,
            exception=None if status == "completed" else Exception(status),
            context={},
        )
