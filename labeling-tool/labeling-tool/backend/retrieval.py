import os
import json
import glob
import torch
import numpy as np
from PIL import Image
from typing import List, Dict
from tqdm import tqdm
from pathlib import Path
import open_clip
from sklearn.metrics.pairwise import cosine_similarity
from safetensors.torch import load_file as safe_load_file

# ============================================================
# ENVIRONMENT SETUP (OFFLINE)
# ============================================================
os.environ["HF_HOME"] = "/root/.cache/huggingface"
os.environ["OPENCLIP_CACHE_DIR"] = "/root/.cache/huggingface"
os.environ["TORCH_HOME"] = "/root/.cache/torch"
os.environ["HF_HUB_OFFLINE"] = "1"

# ============================================================
# MODEL CONFIGURATION
# ============================================================
MODEL_NAME = "ViT-B-16"
BASE_MODEL_DIR = "/root/.cache/huggingface/hub/models--laion--CLIP-ViT-B-16-laion2B-s34B-b88K"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Find safetensors weights automatically ---
safetensors_files = glob.glob(
    os.path.join(BASE_MODEL_DIR, "snapshots", "*", "open_clip_model.safetensors")
)
if not safetensors_files:
    raise RuntimeError(
        f"[retrieval] ❌ No safetensors weights found under {BASE_MODEL_DIR}. "
        f"Make sure the model is mounted correctly."
    )

MODEL_PATH = safetensors_files[0]
print(f"[retrieval] Loading OpenCLIP model from local weights: {MODEL_PATH}")
print(f"[retrieval] Running on device: {DEVICE}")

# ============================================================
# LOAD MODEL MANUALLY (SAFETENSORS)
# ============================================================
try:
    model = open_clip.create_model(MODEL_NAME, pretrained=None)
    state_dict = safe_load_file(MODEL_PATH)
    model.load_state_dict(state_dict, strict=False)
    model = model.to(DEVICE).eval()
    _, _, preprocess = open_clip.create_model_and_transforms(MODEL_NAME)
    EMBED_DIM = int(getattr(getattr(model, "visual", model), "output_dim", 512))
    print(f"[retrieval] ✅ Model loaded successfully (offline safetensors). EMBED_DIM={EMBED_DIM}")
except Exception as e:
    raise RuntimeError(f"[retrieval] ❌ Failed to load safetensors model from {MODEL_PATH}: {e}")

# ============================================================
# CACHE CONFIGURATION
# ============================================================
CACHE_DIR = Path("/data/embeddings")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
EMB_FILE = CACHE_DIR / "embeddings.npy"
PATH_FILE = CACHE_DIR / "paths.json"

_embs: np.ndarray = np.zeros((0, 1), dtype=np.float32)
_paths: List[str] = []

# ============================================================
# UTILITY FUNCTIONS
# ============================================================
def _norm_path(p: str) -> str:
    return os.path.abspath(os.path.normpath(p))

def _load_cache():
    global _embs, _paths
    if EMB_FILE.exists() and PATH_FILE.exists():
        try:
            arr = np.load(EMB_FILE)
            with open(PATH_FILE, "r", encoding="utf-8") as f:
                paths = json.load(f)
            if arr.ndim == 2 and len(paths) == arr.shape[0]:
                _embs = arr.astype("float32", copy=False)
                _paths = [_norm_path(p) for p in paths]
                print(f"[retrieval] Loaded {_embs.shape[0]} embeddings from cache.")
                return
        except Exception as e:
            print(f"[retrieval] Failed to load cache: {e}. Resetting.")
    _embs = np.zeros((0, EMBED_DIM), dtype=np.float32)
    _paths = []

def _save_cache():
    np.save(EMB_FILE, _embs)
    PATH_FILE.write_text(json.dumps(_paths, indent=2))
    print(f"[retrieval] Saved {_embs.shape[0]} embeddings to cache.")

def _embed_image(img_path: str) -> np.ndarray:
    """Compute embedding for a single image."""
    try:
        # Resolve relative paths (handle both /images/... and /data/images/...)
        if not os.path.isabs(img_path):
            img_path = os.path.join("/data/images", img_path)
        elif img_path.startswith("/images/"):
            img_path = img_path.replace("/images/", "/data/images/")

        if not os.path.exists(img_path):
            raise FileNotFoundError(img_path)

        img = Image.open(img_path).convert("RGB")
        img_tensor = preprocess(img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            emb = model.encode_image(img_tensor)
            emb /= emb.norm(dim=-1, keepdim=True)
        return emb.cpu().numpy().astype("float32")[0]

    except Exception as e:
        print(f"[retrieval] ⚠️ Failed to embed {img_path}: {e}")
        return np.zeros((EMBED_DIM,), dtype=np.float32)


def _ensure_embeddings(paths: List[str]):
    global _embs, _paths
    _load_cache()
    paths = [_norm_path(p) for p in paths]
    known = set(_paths)
    new_paths = [p for p in paths if p not in known]
    if not new_paths:
        return
    print(f"[retrieval] Computing embeddings for {len(new_paths)} new images...")
    new_embs = []
    for p in tqdm(new_paths, desc="Embedding new images"):
        new_embs.append(_embed_image(p))
    new_embs = np.stack(new_embs, axis=0)
    if _embs.size == 0:
        _embs = new_embs
        _paths = list(new_paths)
    else:
        _embs = np.concatenate([_embs, new_embs], axis=0)
        _paths.extend(new_paths)
    _save_cache()

# ============================================================
# PUBLIC API
# ============================================================
def top_k_similar(query_path: str, all_paths: List[str], k: int = 5, exclude_self: bool = True) -> List[Dict[str, float]]:
    all_paths = [_norm_path(p) for p in all_paths]
    _ensure_embeddings(all_paths)

    path_to_idx = {p: i for i, p in enumerate(_paths)}
    q_abs = _norm_path(query_path)
    if q_abs in path_to_idx:
        q_emb = _embs[path_to_idx[q_abs]]
    else:
        q_emb = _embed_image(q_abs)

    idxs = [path_to_idx[p] for p in all_paths if p in path_to_idx]
    if not idxs:
        print("[retrieval] No valid embeddings for provided paths.")
        return []

    candidates = _embs[idxs]
    sims = cosine_similarity(q_emb.reshape(1, -1), candidates)[0]

    if exclude_self and q_abs in path_to_idx and path_to_idx[q_abs] in idxs:
        sims[idxs.index(path_to_idx[q_abs])] = -1.0

    order = np.argsort(-sims)[:k]
    top_scores = sims[order]
    top_paths = [all_paths[i] for i in order]

    probs = np.clip(top_scores, 0, None)
    if probs.sum() > 0:
        probs = probs / probs.sum()

    return [{"path": top_paths[i], "score": float(probs[i])} for i in range(len(top_paths))]

# ============================================================
# MANUAL TEST
# ============================================================
if __name__ == "__main__":
    import glob
    imgs = sorted(glob.glob("/data/images/**/*.[jp][pn]g", recursive=True))
    if not imgs:
        print("No images found in /data/images/")
        raise SystemExit(0)
    q = imgs[0]
    print(f"[test] Query: {q}")
    results = top_k_similar(q, imgs, k=5)
    for r in results:
        print(f"  {r['path']} → {r['score']:.4f}")
