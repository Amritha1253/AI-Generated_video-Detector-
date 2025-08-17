# utils.py
import cv2
import numpy as np
from PIL import Image
import torch
from facenet_pytorch import MTCNN

# ---------- Frame sampling & quality ----------
def _video_total_frames(cap):
    try:
        return int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    except Exception:
        return None

def _variance_of_laplacian(gray):
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def sample_frame_indices(n_total, n_samples):
    if not n_total or n_total <= 0:
        return list(range(min(32, n_samples)))  # fallback
    n_samples = min(n_samples, n_total)
    return [int(i) for i in np.linspace(0, n_total - 1, n_samples)]

# ---------- Face detection & cropping ----------
class FaceCropper:
    def __init__(self, device="cpu"):
        self.mtcnn = MTCNN(keep_all=True, device=device)

    def crop_largest_face(self, img_pil, margin=0.2):
        # img_pil: RGB PIL
        boxes, _ = self.mtcnn.detect(img_pil)
        if boxes is None or len(boxes) == 0:
            # fallback: center crop square
            w, h = img_pil.size
            s = min(w, h)
            left = (w - s) // 2
            top = (h - s) // 2
            return img_pil.crop((left, top, left + s, top + s))

        # pick largest face
        areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in boxes]
        b = boxes[int(np.argmax(areas))]

        # expand box by margin
        x1, y1, x2, y2 = b
        w, h = img_pil.size
        bw, bh = x2 - x1, y2 - y1
        x1 = max(0, int(x1 - margin * bw))
        y1 = max(0, int(y1 - margin * bh))
        x2 = min(w, int(x2 + margin * bw))
        y2 = min(h, int(y2 + margin * bh))
        return img_pil.crop((x1, y1, x2, y2))

# ---------- Main prediction ----------
def predict_video(model, processor, mapping, video_path, device="cpu",
                  max_frames=64, batch_size=16, blur_thresh=30.0,
                  decision_threshold=0.55, topk_ratio=0.3):
    """
    Returns (score_fake, label_str, details_dict)
    score_fake: 0..1 probability-like score for 'fake'
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0.0, "Error: cannot open video", {}

    n_total = _video_total_frames(cap)
    frame_idxs = sample_frame_indices(n_total, max_frames)

    face_cropper = FaceCropper(device=device)

    frame_scores = []
    batch_imgs = []

    # Seek and read chosen frames
    for idx in frame_idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame_bgr = cap.read()
        if not ok:
            continue

        # quality filter (blur)
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        if _variance_of_laplacian(gray) < blur_thresh:
            continue

        # to RGB PIL
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(frame_rgb)

        # face crop (fallback to center crop if no face)
        face_img = face_cropper.crop_largest_face(img_pil)

        batch_imgs.append(face_img)

        # run in batches
        if len(batch_imgs) == batch_size:
            frame_scores.extend(_infer_batch(model, processor, mapping, batch_imgs, device))
            batch_imgs = []

    # last partial batch
    if batch_imgs:
        frame_scores.extend(_infer_batch(model, processor, mapping, batch_imgs, device))

    cap.release()

    if not frame_scores:
        return 0.0, "Low-quality/No-face frames", {"frames_used": 0}

    # Robust aggregation: median + top-k mean
    scores = np.array(frame_scores, dtype=np.float32)
    median = float(np.median(scores))
    k = max(1, int(len(scores) * topk_ratio))
    topk_mean = float(np.mean(np.sort(scores)[-k:]))
    video_score = 0.6 * median + 0.4 * topk_mean

    label = "AI-Generated (Fake)" if video_score >= decision_threshold else "Real"
    details = {
        "frames_used": len(scores),
        "median": round(median, 4),
        "topk_mean": round(topk_mean, 4),
        "decision_threshold": decision_threshold
    }
    return float(video_score), label, details

def _infer_batch(model, processor, mapping, pil_list, device):
    inputs = processor(images=pil_list, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**inputs)
        probs = torch.softmax(out.logits, dim=1).cpu().numpy()
    fake_idx = mapping.get("fake_idx", 1)
    return probs[:, fake_idx].tolist()
