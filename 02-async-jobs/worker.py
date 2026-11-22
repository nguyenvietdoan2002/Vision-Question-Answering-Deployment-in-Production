"""One background thread: pull jobs from a queue, run predict, optional webhook."""
import queue
import threading

import requests


def start_worker(job_queue: queue.Queue, jobs: dict, predict_fn):
    """Spawn a daemon thread that processes jobs forever."""

    def loop():
        while True:
            job_id, image_bytes, question, callback_url = job_queue.get()
            jobs[job_id]["status"] = "processing"
            try:
                result = predict_fn(image_bytes, question)
                jobs[job_id].update(status="completed", **result)
            except Exception as e:
                jobs[job_id].update(status="failed", error=str(e))
            # Optional webhook: POST the final job dict to the caller's URL.
            if callback_url:
                try:
                    requests.post(callback_url, json=jobs[job_id], timeout=5)
                except Exception:
                    pass
            job_queue.task_done()

    t = threading.Thread(target=loop, daemon=True)
    t.start()
    return t
