"""ViT ImageNet classifier: load + predict top-3."""
import torch
from transformers import ViTForImageClassification, ViTImageProcessor

MODEL_NAME = "google/vit-base-patch16-224"


def load(device: str):
    processor = ViTImageProcessor.from_pretrained(MODEL_NAME)
    model = ViTForImageClassification.from_pretrained(MODEL_NAME).to(device)
    model.eval()
    return {"processor": processor, "model": model}


def predict(bundle, pil_image, device: str) -> dict:
    inputs = bundle["processor"](pil_image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = bundle["model"](**inputs).logits[0]
    probs = torch.softmax(logits, dim=-1)
    top = torch.topk(probs, 3)
    labels = [
        {"label": bundle["model"].config.id2label[i.item()], "score": s.item()}
        for i, s in zip(top.indices, top.values)
    ]
    return {"predictions": labels}
