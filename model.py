# model.py
import torch
from transformers import AutoModelForImageClassification, AutoImageProcessor, AutoConfig

# You can swap this for another HF model later
MODEL_NAME = "prithivMLmods/deepfake-detector-model-v1"
# Notes: this model’s card states id2label {"0":"fake","1":"real"} (reversed from what many people assume).

def load_model(device="cpu"):
    cfg = AutoConfig.from_pretrained(MODEL_NAME)
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    model = AutoModelForImageClassification.from_pretrained(MODEL_NAME)
    model.to(device)
    model.eval()

    # Determine which index is "fake" and which is "real" from the config
    id2label = {int(k): v.lower() for k, v in getattr(cfg, "id2label", {}).items()}
    fake_idx = None
    real_idx = None
    for i, name in id2label.items():
        if any(k in name for k in ["fake", "deepfake", "synthetic"]):
            fake_idx = i
        if "real" in name or "authentic" in name:
            real_idx = i

    # Sensible defaults if model lacks labels
    if fake_idx is None:
        fake_idx = 1
    if real_idx is None:
        real_idx = 0

    mapping = {"fake_idx": fake_idx, "real_idx": real_idx, "id2label": id2label}
    print(f"[Model] Using mapping: {mapping}")
    return model, processor, mapping
