"""TrOCR printed-text OCR: load + predict."""
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

MODEL_NAME = "microsoft/trocr-small-printed"


def load(device: str):
    processor = TrOCRProcessor.from_pretrained(MODEL_NAME)
    model = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME).to(device)
    model.eval()
    return {"processor": processor, "model": model}


def predict(bundle, pil_image, device: str) -> dict:
    # TrOCR expects RGB; pixel_values only (no text input).
    pixel_values = bundle["processor"](pil_image, return_tensors="pt").pixel_values.to(device)
    with torch.no_grad():
        ids = bundle["model"].generate(pixel_values)
    text = bundle["processor"].batch_decode(ids, skip_special_tokens=True)[0]
    return {"text": text}
