"""Tiny receiver on :9000 — prints webhook payloads from the job worker."""
from fastapi import FastAPI, Request

app = FastAPI(title="Webhook receiver")


@app.post("/webhook")
async def webhook(request: Request):
    body = await request.json()
    print("webhook received:", body)
    return {"ok": True}
