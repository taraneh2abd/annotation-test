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
except Exception as e:
    print(f"[error] MongoDB connection failed: {e}")
    raise e

def ensure_indexes():
    try:
        labels_col.create_index([("query_image", ASCENDING)], unique=True)
        labels_col.create_index([("ts", ASCENDING)])
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
    if not ping_mongo():
        raise HTTPException(status_code=503, detail="MongoDB not reachable")
    doc = {"positives": body.positives, "negatives": body.negatives, "ts": int(time.time())}
    try:
        result = labels_col.update_one({"query_image": body.queryImage}, {"$set": doc}, upsert=True)
        return {"ok": True, "matched": result.matched_count, "modified": result.modified_count}
    except PyMongoError as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/batch/{index}", response_model=SessionResponse)
def get_batch(index: int = PathParam(..., ge=0), user=Depends(verify_token)):
    if index >= len(BATCHES):
        raise HTTPException(status_code=404, detail="Batch index out of range")
    return BATCHES[index]

@app.get("/api/batches/count")
def get_batches_count(user=Depends(verify_token)):
    return {"total": len(BATCHES)}

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
