"""ViLT VQA: load + predict."""
import torch
from transformers import ViltForQuestionAnswering, ViltProcessor

MODEL_NAME = "dandelin/vilt-b32-finetuned-vqa"


def load(device: str):
    processor = ViltProcessor.from_pretrained(MODEL_NAME)
    model = ViltForQuestionAnswering.from_pretrained(MODEL_NAME).to(device)
    model.eval()
    return {"processor": processor, "model": model}


def predict(bundle, pil_image, question: str, device: str) -> dict:
    inputs = bundle["processor"](pil_image, question, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = bundle["model"](**inputs)
    idx = outputs.logits.argmax(-1).item()
    answer = bundle["model"].config.id2label[idx]
    score = torch.softmax(outputs.logits, dim=-1)[0, idx].item()
    return {"answer": answer, "score": score}
