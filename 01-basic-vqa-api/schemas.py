from pydantic import BaseModel


class VQAResponse(BaseModel):
    answer: str
    score: float
    latency_ms: float
