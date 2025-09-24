# my main

# main.py
import os
import time
from pathlib import Path
from typing import List, Dict
from fastapi import FastAPI, HTTPException, Query, Path as PathParam, Depends, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from neo4j import GraphDatabase

from auth import login_user, LoginRequest, LoginResponse, verify_token
from batches import BATCHES

# ---------------- Env ----------------
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://neo4j:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASS = os.getenv("NEO4J_PASS", "example")

IMAGE_ROOT = os.getenv("IMAGE_ROOT", "/data/images")
UPLOAD_ROOT = os.getenv("UPLOAD_ROOT", "/data/uploads")
PAGE_SIZE_DEFAULT = int(os.getenv("PAGE_SIZE", "20"))

BATCH_DIR = Path(UPLOAD_ROOT) / "batches"
NON_LABELED_DIR = Path(UPLOAD_ROOT) / "non_labeled"
BATCH_DIR.mkdir(parents=True, exist_ok=True)
NON_LABELED_DIR.mkdir(parents=True, exist_ok=True)

# ---------------- App ----------------
app = FastAPI(title="Image Labeling API (Neo4j, single :Image label)")

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
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))

def run_query(query: str, params: Dict = {}):
    with driver.session() as session:
        return list(session.run(query, params))

def ensure_indexes():
    # Only one label: :Image
    run_query("CREATE CONSTRAINT IF NOT EXISTS FOR (i:Image) REQUIRE i.path IS UNIQUE")

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

def merge_all_images_as_nodes(images: List[str]):
    # MERGE a node for each image under /images
    if not images:
        return
    # Use UNWIND for efficiency
    run_query("""
        UNWIND $paths AS p
        MERGE (:Image {path: p})
    """, {"paths": [f"/images/{p}" for p in images]})

IMAGE_LIST: List[str] = scan_images(IMAGE_ROOT)
merge_all_images_as_nodes(IMAGE_LIST)
print(f"[startup] images found: {len(IMAGE_LIST)} in {IMAGE_ROOT} (nodes merged)")

# ---------------- Schemas ----------------
class SessionResponse(BaseModel):
    queryImage: str
    images: List[str]

class SaveLabelsBody(BaseModel):
    queryImage: str
    positives: List[str] = []
    negatives: List[str] = []

class StatsBulkRequest(BaseModel):
    images: List[str]

class StatsBulkResponse(BaseModel):
    stats: dict

class ProjectStatsResponse(BaseModel):
    image_count: int
    total_positive_matches: float
    total_negative_matches: float
    mean_positive_matches_per_image: float
    mean_negative_matches_per_image: float
    sum_positive_count: int
    sum_negative_count: int

# ---------------- Auth Endpoint ----------------
@app.post("/api/login", response_model=LoginResponse)
def login(req: LoginRequest):
    return login_user(req)

# ---------------- Health ----------------
@app.get("/api/health")
def health():
    try:
        run_query("RETURN 1 as ok")
        alive = True
    except Exception:
        alive = False
    return {"status": "ok", "images": len(IMAGE_LIST), "neo4j": alive, "image_root": IMAGE_ROOT}

@app.post("/api/refresh")
def refresh():
    """
    Re-scan /images and MERGE missing :Image nodes (idempotent).
    """
    global IMAGE_LIST
    IMAGE_LIST = scan_images(IMAGE_ROOT)
    merge_all_images_as_nodes(IMAGE_LIST)
    return {"ok": True, "count": len(IMAGE_LIST)}

# ---------------- Session & Labels ----------------
@app.get("/api/session", response_model=SessionResponse)
def get_session(offset: int = Query(0, ge=0),
                limit: int = Query(PAGE_SIZE_DEFAULT, ge=1, le=200),
                user=Depends(verify_token)):
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
    Save labels by:
      1) Ensuring both the query image and all candidate images exist as :Image nodes
      2) Deleting any previous POSITIVE/NEGATIVE relationships from the query image to others
      3) Creating new POSITIVE and NEGATIVE relationships (directed from query -> candidate)

    NOTE: All nodes are labeled :Image. No :Query label is used.
    """
    q = body.queryImage
    positives = list(dict.fromkeys(body.positives or []))  # de-dupe, keep order
    negatives = list(dict.fromkeys(body.negatives or []))

    # Ensure the query and candidate nodes exist
    run_query("MERGE (:Image {path:$q})", {"q": q})
    if positives:
        run_query("""
            UNWIND $paths AS p
            MERGE (:Image {path: p})
        """, {"paths": positives})
    if negatives:
        run_query("""
            UNWIND $paths AS p
            MERGE (:Image {path: p})
        """, {"paths": negatives})

    # Remove old edges from the query image
    run_query("""
        MATCH (q:Image {path:$q})-[r:POSITIVE|NEGATIVE]->(:Image)
        DELETE r
    """, {"q": q})

    # Add new POSITIVE edges
    if positives:
        run_query("""
            MATCH (q:Image {path:$q})
            UNWIND $paths AS p
            MATCH (i:Image {path:p})
            MERGE (q)-[:POSITIVE]->(i)
        """, {"q": q, "paths": positives})

    # Add new NEGATIVE edges
    if negatives:
        run_query("""
            MATCH (q:Image {path:$q})
            UNWIND $paths AS p
            MATCH (i:Image {path:p})
            MERGE (q)-[:NEGATIVE]->(i)
        """, {"q": q, "paths": negatives})

    return {"ok": True, "saved": True}

@app.get("/api/labels/get")
def get_labels(queryImage: str, user=Depends(verify_token)):
    """
    Read labels for a query image from relationships:
      (query:Image)-[:POSITIVE]->(i:Image)
      (query:Image)-[:NEGATIVE]->(i:Image)
    """
    res_pos = run_query("""
        MATCH (q:Image {path:$q})-[:POSITIVE]->(i:Image)
        RETURN i.path as path
    """, {"q": queryImage})
    res_neg = run_query("""
        MATCH (q:Image {path:$q})-[:NEGATIVE]->(i:Image)
        RETURN i.path as path
    """, {"q": queryImage})
    return {
        "queryImage": queryImage,
        "positives": [r["path"] for r in res_pos],
        "negatives": [r["path"] for r in res_neg]
    }

@app.get("/api/batch/{index}", response_model=SessionResponse)
def get_batch(index: int = PathParam(..., ge=0), user=Depends(verify_token)):
    if index >= len(BATCHES):
        raise HTTPException(status_code=404, detail="Batch index out of range")
    return BATCHES[index]

@app.get("/api/batches/count")
def get_batches_count(user=Depends(verify_token)):
    return {"total": len(BATCHES)}

# ---------------- Stats ----------------
@app.post("/api/image_stats/bulk", response_model=StatsBulkResponse)
def image_stats_bulk(req: StatsBulkRequest, user=Depends(verify_token)):
    """
    For each requested image path:
      positive_count = number of POSITIVE edges connected to the node
      negative_count = number of NEGATIVE edges connected to the node
    """
    stats_map = {img: {"positive_count": 0, "negative_count": 0} for img in (req.images or [])}
    if not req.images:
        return {"stats": stats_map}

    # One query to count POSITIVE connections
    rows_pos = run_query("""
        UNWIND $targets AS t
        MATCH (i:Image {path:t})-[r:POSITIVE]-()
        RETURN t AS path, count(r) AS c
    """, {"targets": req.images})
    for r in rows_pos:
        stats_map[r["path"]]["positive_count"] = r["c"]

    # One query to count NEGATIVE connections
    rows_neg = run_query("""
        UNWIND $targets AS t
        MATCH (i:Image {path:t})-[r:NEGATIVE]-()
        RETURN t AS path, count(r) AS c
    """, {"targets": req.images})
    for r in rows_neg:
        stats_map[r["path"]]["negative_count"] = r["c"]

    return {"stats": stats_map}


@app.get("/api/stats/summary", response_model=ProjectStatsResponse)
def get_project_stats(user=Depends(verify_token)):
    """
    Summaries to match prior app behavior:
      - sum_positive_count: count of POSITIVE relationships (we keep this as "sum", same as before)
      - total_positive_matches: sum_positive_count / 2.0   (kept for backward-compat)
      - same for negative
      - means computed as totals / image_count
    """
    image_count = len(IMAGE_LIST)

    res_pos = run_query("MATCH (:Image)-[:POSITIVE]->(:Image) RETURN count(*) as c")
    res_neg = run_query("MATCH (:Image)-[:NEGATIVE]->(:Image) RETURN count(*) as c")
    sum_pos = res_pos[0]["c"]
    sum_neg = res_neg[0]["c"]

    # Preserve previous semantics: divide by 2
    total_pos_matches = sum_pos 
    total_neg_matches = sum_neg 

    mean_pos = total_pos_matches / image_count if image_count else 0.0
    mean_neg = total_neg_matches / image_count if image_count else 0.0

    return ProjectStatsResponse(
        image_count=image_count,
        total_positive_matches=total_pos_matches,
        total_negative_matches=total_neg_matches,
        mean_positive_matches_per_image=mean_pos,
        mean_negative_matches_per_image=mean_neg,
        sum_positive_count=sum_pos,
        sum_negative_count=sum_neg,
    )

# ---------------- Upload ZIP Endpoints ----------------
@app.post("/api/upload_batch")
async def upload_batch(file: UploadFile = File(...), user=Depends(verify_token)):
    dest = BATCH_DIR / file.filename
    contents = await file.read()
    with open(dest, "wb") as f:
        f.write(contents)
    return {"ok": True, "filename": file.filename}

@app.post("/api/upload_non_labeled")
async def upload_non_labeled(file: UploadFile = File(...), user=Depends(verify_token)):
    dest = NON_LABELED_DIR / file.filename
    contents = await file.read()
    with open(dest, "wb") as f:
        f.write(contents)
    return {"ok": True, "filename": file.filename}
