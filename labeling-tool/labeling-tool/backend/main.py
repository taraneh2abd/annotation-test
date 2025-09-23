# main.py
import os
import time
from pathlib import Path
from typing import List
# ---------------- Upload ZIP APIs ----------------
from fastapi import UploadFile, File
import shutil

from fastapi import FastAPI, HTTPException, Query, Path as PathParam, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from pymongo import MongoClient, ASCENDING
from pymongo.errors import PyMongoError, ServerSelectionTimeoutError

from auth import login_user, LoginRequest, LoginResponse, verify_token
from batches import BATCHES

# ---------------- Env ----------------
MONGO_HOST = os.getenv("MONGO_HOST", "mongo")
MONGO_PORT = int(os.getenv("MONGO_PORT", "27017"))
MONGO_USER = os.getenv("MONGO_INITDB_ROOT_USERNAME", "root")
MONGO_PASS = os.getenv("MONGO_INITDB_ROOT_PASSWORD", "example")
MONGO_DB = os.getenv("MONGO_DB", "labeldb")
MONGO_COLLECTION = os.getenv("MONGO_COLLECTION", "labels")
IMAGE_ROOT = os.getenv("IMAGE_ROOT", "/data/images")
PAGE_SIZE_DEFAULT = int(os.getenv("PAGE_SIZE", "20"))

UPLOAD_ROOT = os.getenv("UPLOAD_ROOT", "/data/uploads")
BATCH_DIR = Path(UPLOAD_ROOT) / "batches"
NON_LABELED_DIR = Path(UPLOAD_ROOT) / "non_labeled"
BATCH_DIR.mkdir(parents=True, exist_ok=True)
NON_LABELED_DIR.mkdir(parents=True, exist_ok=True)

MONGO_URI = f"mongodb://{MONGO_USER}:{MONGO_PASS}@{MONGO_HOST}:{MONGO_PORT}/"

# ---------------- App ----------------
app = FastAPI(title="Image Labeling API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- Static Images ----------------
Path(IMAGE_ROOT).mkdir(parents=True, exist_ok=True)
app.mount("/images", StaticFiles(directory=IMAGE_ROOT), name="images")

# ---------------- Database ----------------
try:
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=2000)
    db = client[MONGO_DB]
    labels_col = db[MONGO_COLLECTION]
    # NEW: collection to track per-image positive/negative tallies
    image_stats_col = db["image_stats"]
    image_stats_col.create_index("image", unique=True)
except Exception as e:
    print(f"[error] MongoDB connection failed: {e}")
    raise e

def ensure_indexes():
    try:
        labels_col.create_index([("query_image", ASCENDING)], unique=True)
        labels_col.create_index([("ts", ASCENDING)])
        # (image_stats unique index created above)
    except PyMongoError as e:
        print(f"[warn] failed to create index: {e}")

def ping_mongo() -> bool:
    try:
        client.admin.command("ping")
        return True
    except ServerSelectionTimeoutError:
        return False
    except Exception:
        return False

ensure_indexes()

# ---------------- Image Scan ----------------
def scan_images(root: str) -> List[str]:
    exts = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"}
    files: List[str] = []
    for base, _, names in os.walk(root):
        for n in names:
            if Path(n).suffix.lower() in exts:
                rel = os.path.relpath(os.path.join(base, n), root)
                files.append(rel.replace("\\", "/"))
    return sorted(files)

IMAGE_LIST: List[str] = scan_images(IMAGE_ROOT)
print(f"[startup] images found: {len(IMAGE_LIST)} in {IMAGE_ROOT}")
print(f"[startup] mongo reachable: {ping_mongo()} (host={MONGO_HOST}:{MONGO_PORT})")

# ---------------- Schemas ----------------
class SessionResponse(BaseModel):
    queryImage: str
    images: List[str]

class SaveLabelsBody(BaseModel):
    queryImage: str
    positives: List[str] = []
    negatives: List[str] = []

# NEW: bulk stats request/response
class StatsBulkRequest(BaseModel):
    images: List[str]

class ImageStat(BaseModel):
    positive_count: int = 0
    negative_count: int = 0

class StatsBulkResponse(BaseModel):
    # map image -> counts
    stats: dict

# ---------------- Auth Endpoint ----------------
@app.post("/api/login", response_model=LoginResponse)
def login(req: LoginRequest):
    return login_user(req)

# ---------------- Health ----------------
@app.get("/api/health")
def health():
    return {"status": "ok", "images": len(IMAGE_LIST), "mongo": ping_mongo(), "image_root": IMAGE_ROOT}

@app.post("/api/refresh")
def refresh():
    global IMAGE_LIST
    IMAGE_LIST = scan_images(IMAGE_ROOT)
    return {"ok": True, "count": len(IMAGE_LIST)}

# ---------------- Session & Labels ----------------
@app.get("/api/session", response_model=SessionResponse)
def get_session(offset: int = Query(0, ge=0), limit: int = Query(PAGE_SIZE_DEFAULT, ge=1, le=200), user=Depends(verify_token)):
    if not IMAGE_LIST:
        raise HTTPException(status_code=404, detail=f"No images found in {IMAGE_ROOT}")
    query_rel = IMAGE_LIST[offset % len(IMAGE_LIST)]
    start = (offset + 1) % len(IMAGE_LIST)
    page = IMAGE_LIST[start:start + limit]
    if len(page) < limit and len(IMAGE_LIST) > 0:
        page += IMAGE_LIST[0:limit - len(page)]
    page = [p for p in page if p != query_rel]
    return {"queryImage": f"/images/{query_rel}", "images": [f"/images/{p}" for p in page]}

@app.post("/api/labels/save")
def save_labels(body: SaveLabelsBody, user=Depends(verify_token)):
    """
    Save labels for a query and update global per-image stats.

    Rule: every (queryImage, candidate) pair counts for BOTH images.
    - positive match: +1 positive for queryImage and candidate
    - negative match: +1 negative for queryImage and candidate
    Re-saves adjust via diffs so totals remain correct.
    """
    if not ping_mongo():
        raise HTTPException(status_code=503, detail="MongoDB not reachable")

    query_image = body.queryImage
    # de-duplicate & sanitize overlaps
    new_pos = set(body.positives or [])
    new_neg = set(body.negatives or [])
    # Remove any accidental overlaps; prefer last state semantics (neg wins if overlapping sets provided)
    # but since front won't overlap, this is a safety net:
    overlap = new_pos & new_neg
    if overlap:
        # drop overlapped from positives
        new_pos -= overlap

    # fetch previous state BEFORE overwrite
    prev_doc = labels_col.find_one({"query_image": query_image}) or {"positives": [], "negatives": []}
    prev_pos = set(prev_doc.get("positives", []))
    prev_neg = set(prev_doc.get("negatives", []))

    # compute diffs
    added_pos = new_pos - prev_pos
    removed_pos = prev_pos - new_pos
    added_neg = new_neg - prev_neg
    removed_neg = prev_neg - new_neg

    # Build increments for both sides (candidate and query image)
    from collections import defaultdict
    inc = defaultdict(lambda: {"positive_count": 0, "negative_count": 0})

    def bump(img, dp=0, dn=0):
        inc[img]["positive_count"] += dp
        inc[img]["negative_count"] += dn

    # Helper to apply change to both candidate and query
    def pair_delta(candidate_img, dp=0, dn=0):
        if candidate_img == query_image:
            # Shouldn't happen (page excludes query), but guard anyway: count once.
            bump(candidate_img, dp, dn)
        else:
            bump(candidate_img, dp, dn)
            bump(query_image, dp, dn)

    for img in added_pos:
        pair_delta(img, dp=1, dn=0)
    for img in removed_pos:
        pair_delta(img, dp=-1, dn=0)
    for img in added_neg:
        pair_delta(img, dp=0, dn=1)
    for img in removed_neg:
        pair_delta(img, dp=0, dn=-1)

    # write the updated labels first (so the doc reflects current state)
    doc = {"query_image": query_image, "positives": sorted(list(new_pos)), "negatives": sorted(list(new_neg)), "ts": int(time.time())}
    try:
        labels_col.update_one({"query_image": query_image}, {"$set": doc}, upsert=True)
    except PyMongoError as e:
        raise HTTPException(status_code=500, detail=str(e))

    # then apply stat increments (upserts)
    try:
        for img, delta in inc.items():
            to_inc = {}
            if delta["positive_count"] != 0:
                to_inc["positive_count"] = delta["positive_count"]
            if delta["negative_count"] != 0:
                to_inc["negative_count"] = delta["negative_count"]
            if not to_inc:
                continue
            image_stats_col.update_one(
                {"image": img},
                {"$inc": to_inc},
                upsert=True
            )
    except PyMongoError as e:
        print(f"[warn] failed to update stats: {e}")

    return {
        "ok": True,
        "changed": {
            "added_pos": len(added_pos), "removed_pos": len(removed_pos),
            "added_neg": len(added_neg), "removed_neg": len(removed_neg)
        }
    }

@app.get("/api/batch/{index}", response_model=SessionResponse)
def get_batch(index: int = PathParam(..., ge=0), user=Depends(verify_token)):
    if index >= len(BATCHES):
        raise HTTPException(status_code=404, detail="Batch index out of range")
    return BATCHES[index]

@app.get("/api/batches/count")
def get_batches_count(user=Depends(verify_token)):
    return {"total": len(BATCHES)}

# -------- NEW: Bulk stats endpoint (front uses this to render badges) --------
@app.post("/api/image_stats/bulk", response_model=StatsBulkResponse)
def image_stats_bulk(req: StatsBulkRequest, user=Depends(verify_token)):
    images = list(set(req.images or []))
    if not images:
        return {"stats": {}}
    try:
        # fetch stats for requested images
        stats_map = {img: {"positive_count": 0, "negative_count": 0} for img in images}
        cursor = image_stats_col.find({"image": {"$in": images}})
        for d in cursor:
            stats_map[d["image"]] = {
                "positive_count": int(d.get("positive_count", 0)),
                "negative_count": int(d.get("negative_count", 0)),
            }
        return {"stats": stats_map}
    except PyMongoError as e:
        raise HTTPException(status_code=500, detail=str(e))

# ---------------- Upload ZIP Endpoints ----------------
@app.post("/api/upload_batch")
async def upload_batch(file: UploadFile = File(...), user=Depends(verify_token)):
    dest = BATCH_DIR / file.filename
    contents = await file.read()
    with open(dest, "wb") as f:
        f.write(contents)
    print(f"Received batch ZIP: {file.filename} -> {dest}")
    return {"ok": True, "filename": file.filename}

@app.post("/api/upload_non_labeled")
async def upload_non_labeled(file: UploadFile = File(...), user=Depends(verify_token)):
    dest = NON_LABELED_DIR / file.filename
    contents = await file.read()
    with open(dest, "wb") as f:
        f.write(contents)
    print(f"Received non-labeled ZIP: {file.filename} -> {dest}")
    return {"ok": True, "filename": file.filename}
