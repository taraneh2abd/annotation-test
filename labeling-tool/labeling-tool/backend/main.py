import os
import time
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Query, Path as PathParam
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from pymongo import MongoClient, ASCENDING
from pymongo.errors import PyMongoError, ServerSelectionTimeoutError

# ---------------- Env ----------------
MONGO_HOST = os.getenv("MONGO_HOST", "mongo")
MONGO_PORT = int(os.getenv("MONGO_PORT", "27017"))
MONGO_USER = os.getenv("MONGO_INITDB_ROOT_USERNAME", "root")
MONGO_PASS = os.getenv("MONGO_INITDB_ROOT_PASSWORD", "example")
MONGO_DB = os.getenv("MONGO_DB", "labeldb")
MONGO_COLLECTION = os.getenv("MONGO_COLLECTION", "labels")
IMAGE_ROOT = os.getenv("IMAGE_ROOT", "/data/images")
PAGE_SIZE_DEFAULT = int(os.getenv("PAGE_SIZE", "20"))
BACKEND_PORT = int(os.getenv("BACKEND_PORT", "8000"))
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*")

MONGO_URI = f"mongodb://{MONGO_USER}:{MONGO_PASS}@{MONGO_HOST}:{MONGO_PORT}/"

# ---------------- App ----------------
app = FastAPI(title="Image Labeling API")

# ---------------- CORS ----------------
allow_origins = ["*"] if CORS_ORIGINS == "*" else [o.strip() for o in CORS_ORIGINS.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- Static Images ----------------
Path(IMAGE_ROOT).mkdir(parents=True, exist_ok=True)
app.mount("/images", StaticFiles(directory=IMAGE_ROOT), name="images")

# ---------------- Database ----------------
client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=2000)
db = client[MONGO_DB]
labels_col = db[MONGO_COLLECTION]

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

# ---------------- Image scan ----------------
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
ensure_indexes()

# ---------------- Schemas ----------------
class SessionResponse(BaseModel):
    queryImage: str
    images: List[str]

class SaveLabelsBody(BaseModel):
    queryImage: str
    positives: List[str] = []
    negatives: List[str] = []

# ---------------- Batches ----------------
BATCHES = [
    {
        "queryImage": "/images/img_69.png",
        "images": ["/images/img_70.png", "/images/img_71.png", "/images/img_72.png"]
    },
    {
        "queryImage": "/images/img_72.png",
        "images": ["/images/img_73.png", "/images/img_74.png", "/images/img_75.png", "/images/img_76.png"]
    },
    {
        "queryImage": "/images/img_77.png",
        "images": ["/images/img_78.png", "/images/img_79.png"]
    },
    {
        "queryImage": "/images/img_73.png",
        "images": ["/images/img_940.png", "/images/img_941.png", "/images/img_942.png", "/images/img_953.png","/images/img_954.png", "/images/img_965.png", "/images/img_966.png", "/images/img_977.png",
                   "/images/img_948.png", "/images/img_949.png", "/images/img_940.png", "/images/img_951.png","/images/img_952.png", "/images/img_963.png", "/images/img_964.png", "/images/img_975.png",
                   "/images/img_940.png", "/images/img_941.png", "/images/img_942.png", "/images/img_953.png","/images/img_954.png", "/images/img_965.png", "/images/img_966.png", "/images/img_977.png",
                   "/images/img_948.png", "/images/img_949.png", "/images/img_940.png", "/images/img_951.png","/images/img_952.png", "/images/img_963.png", "/images/img_964.png", "/images/img_975.png"
                   ]
    },
    {
        "queryImage": "/images/img_82.png",
        "images": ["/images/img_83.png", "/images/img_84.png", "/images/img_80.png", "/images/img_85.png"]
    },
]

# ---------------- Endpoints ----------------
@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "images": len(IMAGE_LIST),
        "mongo": ping_mongo(),
        "image_root": IMAGE_ROOT,
    }

@app.post("/api/refresh")
def refresh():
    global IMAGE_LIST
    IMAGE_LIST = scan_images(IMAGE_ROOT)
    return {"ok": True, "count": len(IMAGE_LIST)}

@app.get("/api/session", response_model=SessionResponse)
def get_session(
    offset: int = Query(0, ge=0),
    limit: int = Query(PAGE_SIZE_DEFAULT, ge=1, le=200),
):
    if not IMAGE_LIST:
        raise HTTPException(status_code=404, detail=f"No images found in {IMAGE_ROOT}")
    query_rel = IMAGE_LIST[offset % len(IMAGE_LIST)]
    start = (offset + 1) % len(IMAGE_LIST)
    page = IMAGE_LIST[start:start + limit]
    if len(page) < limit and len(IMAGE_LIST) > 0:
        page += IMAGE_LIST[0:limit - len(page)]
    page = [p for p in page if p != query_rel]
    return {
        "queryImage": f"/images/{query_rel}",
        "images": [f"/images/{p}" for p in page],
    }

# ---------------- Labels save with upsert ----------------
@app.post("/api/labels/save")
def save_labels(body: SaveLabelsBody):
    if not ping_mongo():
        raise HTTPException(status_code=503, detail="MongoDB not reachable")
    doc = {
        "positives": body.positives,
        "negatives": body.negatives,
        "ts": int(time.time()),
    }
    try:
        # Upsert: find by query_image, update if exists, insert if not
        result = labels_col.update_one(
            {"query_image": body.queryImage},
            {"$set": doc},
            upsert=True
        )
        return {"ok": True, "matched": result.matched_count, "modified": result.modified_count}
    except PyMongoError as e:
        raise HTTPException(status_code=500, detail=str(e))

# ---------------- Batch APIs ----------------
@app.get("/api/batch/{index}", response_model=SessionResponse, tags=["Batches"])
def get_batch(index: int = PathParam(..., ge=0, description="Batch index")):
    if index >= len(BATCHES):
        raise HTTPException(status_code=404, detail="Batch index out of range")
    return BATCHES[index]

@app.get("/api/batches/count", tags=["Batches"])
def get_batches_count():
    """Return total number of batches"""
    return {"total": len(BATCHES)}
