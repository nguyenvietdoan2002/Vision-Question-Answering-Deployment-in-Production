from typing import Optional

from pydantic import BaseModel


class VQAResponse(BaseModel):
    answer: str
    latency_ms: float
    score: Optional[float] = None  # ViLT has a score; BLIP generate does not


class JobCreated(BaseModel):
    job_id: str
    status: str


class JobStatus(BaseModel):
    job_id: str
    status: str
    answer: Optional[str] = None
    latency_ms: Optional[float] = None
    score: Optional[float] = None
    error: Optional[str] = None
