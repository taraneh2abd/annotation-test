# my main

# main.py
import zipfile
import re
import shutil
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

BATCH_DIR = Path(UPLOAD_ROOT) / "labeled"
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
PENDING_NON_LABELED: List[str] = []

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

class ImageStatItem(BaseModel):
    path: str
    positive_count: int
    negative_count: int
    sort_key: int

class ProjectStatsResponse(BaseModel):
    image_count: int
    total_positive_matches: float
    total_negative_matches: float
    mean_positive_matches_per_image: float
    mean_negative_matches_per_image: float
    sum_positive_count: int
    sum_negative_count: int
    total_connected_matches: int
    mean_connected_matches_per_image: float
    pending_non_labeled_count: int
    image_stats_sorted: List[ImageStatItem]   # ✅ new field


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
    Summaries to match prior app behavior + include CONNECTED relationships + 
    per-image positive/negative stats sorted by a custom key.
    """
    image_count = len(IMAGE_LIST) or 1

    # --- Global aggregates ---
    res_pos = run_query("MATCH (:Image)-[:POSITIVE]->(:Image) RETURN count(*) as c")
    res_neg = run_query("MATCH (:Image)-[:NEGATIVE]->(:Image) RETURN count(*) as c")
    res_con = run_query("MATCH (:Image)-[:CONNECTED]-(:Image) RETURN count(*) as c")

    sum_pos = res_pos[0]["c"] if res_pos else 0
    sum_neg = res_neg[0]["c"] if res_neg else 0
    sum_con = res_con[0]["c"] if res_con else 0

    total_pos_matches = sum_pos
    total_neg_matches = sum_neg
    total_con_matches = sum_con

    mean_pos = total_pos_matches / image_count
    mean_neg = total_neg_matches / image_count
    mean_con = total_con_matches / image_count

    global PENDING_NON_LABELED
    pending_count = len(PENDING_NON_LABELED)

    # --- Per-image stats ---
    rows = run_query("""
        MATCH (i:Image)
        OPTIONAL MATCH (i)-[p:POSITIVE]-()
        WITH i, count(p) AS pos
        OPTIONAL MATCH (i)-[n:NEGATIVE]-()
        RETURN i.path AS path, pos, count(n) AS neg
    """)

    image_stats = []
    for r in rows:
        pos = r.get("pos", 0)
        neg = r.get("neg", 0)
        path = r.get("path")
        sort_key = min(pos, neg) + (pos + neg)
        image_stats.append({
            "path": path,
            "positive_count": pos,
            "negative_count": neg,
            "sort_key": sort_key
        })

    # sort ascending
    image_stats.sort(key=lambda x: x["sort_key"])

    return {
        "image_count": image_count,
        "total_positive_matches": total_pos_matches,
        "total_negative_matches": total_neg_matches,
        "mean_positive_matches_per_image": mean_pos,
        "mean_negative_matches_per_image": mean_neg,
        "sum_positive_count": sum_pos,
        "sum_negative_count": sum_neg,
        "total_connected_matches": total_con_matches,
        "mean_connected_matches_per_image": mean_con,
        "pending_non_labeled_count": pending_count,
        "image_stats_sorted": image_stats,   # ✅ added field
    }


# ######################


def get_next_image_index(root: Path) -> int:
    """
    Scan IMAGE_ROOT for files named img_### and return next available number.
    """
    max_num = 0
    for n in os.listdir(root):
        m = re.match(r"img_(\d+)", Path(n).stem)
        if m:
            max_num = max(max_num, int(m.group(1)))
    return max_num + 1


def safe_extract_zip(zip_path: Path, extract_to: Path) -> List[Path]:
    """
    Extracts a ZIP file safely into extract_to directory, ignoring directories inside.
    Returns list of extracted file paths.
    """
    extracted_files = []
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        for member in zip_ref.infolist():
            if member.is_dir():
                continue
            # Sanitize filename
            filename = os.path.basename(member.filename)
            if not filename:
                continue
            dest_path = extract_to / filename
            with zip_ref.open(member) as src, open(dest_path, 'wb') as out_f:
                shutil.copyfileobj(src, out_f)
            extracted_files.append(dest_path)
    return extracted_files


def connect_all_images_in_graph(image_paths: List[str]):
    """
    Given a list of image paths (like /images/img_888.jpg),
    ensure all exist as :Image nodes and fully connect them
    with undirected :CONNECTED relationships.
    """
    if not image_paths or len(image_paths) < 2:
        return

    # Merge nodes first
    run_query("""
        UNWIND $paths AS p
        MERGE (:Image {path: p})
    """, {"paths": image_paths})

    # Connect all pairs (undirected, only once)
    run_query("""
        UNWIND $paths AS p1
        UNWIND $paths AS p2
        WITH p1, p2
        WHERE p1 < p2
        MATCH (a:Image {path:p1}), (b:Image {path:p2})
        MERGE (a)-[:POSITIVE]-(b)
    """, {"paths": image_paths})



@app.post("/api/upload_batch")
async def upload_batch(file: UploadFile = File(...), user=Depends(verify_token)):
    """
    Upload a labeled batch ZIP:
      1. Save to /uploads/labeled/
      2. Unzip into /images/
      3. Rename extracted files as img_### starting after highest index
      4. Create :Image nodes and fully connect them in Neo4j
    """
    dest_zip = BATCH_DIR / file.filename
    contents = await file.read()
    with open(dest_zip, "wb") as f:
        f.write(contents)

    try:
        # Step 1: Extract safely
        extracted = safe_extract_zip(dest_zip, Path(IMAGE_ROOT))
        if not extracted:
            raise HTTPException(status_code=400, detail="No images found in zip")

        # Step 2: Find next number in existing images
        next_idx = get_next_image_index(Path(IMAGE_ROOT))

        renamed_paths = []
        for old_path in extracted:
            ext = old_path.suffix.lower()
            new_name = f"img_{next_idx:03d}{ext}"
            new_path = Path(IMAGE_ROOT) / new_name
            os.rename(old_path, new_path)
            renamed_paths.append(f"/images/{new_name}")
            next_idx += 1

        # Step 3: Merge as :Image nodes and connect all
        connect_all_images_in_graph(renamed_paths)

        # Step 4: Refresh global image list (so UI gets updated)
        global IMAGE_LIST
        IMAGE_LIST = scan_images(IMAGE_ROOT)

        return {
            "ok": True,
            "imported": len(renamed_paths),
            "new_images": renamed_paths
        }

    except zipfile.BadZipFile:
        raise HTTPException(status_code=400, detail="Invalid ZIP file")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {e}")


@app.post("/api/upload_non_labeled")
async def upload_non_labeled(file: UploadFile = File(...), user=Depends(verify_token)):
    """
    Upload a ZIP of non-labeled images:
      1. Save to /uploads/non_labeled/
      2. Unzip into /images/
      3. Rename sequentially (img_###)
      4. Do not connect in Neo4j yet
      5. Keep list of new image paths in memory (PENDING_NON_LABELED)
    """
    dest_zip = NON_LABELED_DIR / file.filename
    contents = await file.read()
    with open(dest_zip, "wb") as f:
        f.write(contents)

    try:
        # Step 1: Extract safely
        extracted = safe_extract_zip(dest_zip, Path(IMAGE_ROOT))
        if not extracted:
            raise HTTPException(status_code=400, detail="No images found in zip")

        # Step 2: Find next number in existing images
        next_idx = get_next_image_index(Path(IMAGE_ROOT))

        renamed_paths = []
        for old_path in extracted:
            ext = old_path.suffix.lower()
            new_name = f"img_{next_idx:03d}{ext}"
            new_path = Path(IMAGE_ROOT) / new_name
            os.rename(old_path, new_path)
            renamed_paths.append(f"/images/{new_name}")
            next_idx += 1

        # Step 3: Save pending list for later processing
        global PENDING_NON_LABELED
        PENDING_NON_LABELED.extend(renamed_paths)

        # Step 4: Refresh global IMAGE_LIST (for /api/session)
        global IMAGE_LIST
        IMAGE_LIST = scan_images(IMAGE_ROOT)
        print(len(PENDING_NON_LABELED))
        return {
            "ok": True,
            "imported": len(renamed_paths),
            "pending_to_process": len(PENDING_NON_LABELED),
            "new_images": renamed_paths
        }

    except zipfile.BadZipFile:
        raise HTTPException(status_code=400, detail="Invalid ZIP file")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {e}")

@app.post("/api/process_pending")
def process_pending(user=Depends(verify_token)):
    import json, importlib, time, traceback
    from pathlib import Path
    import retrieval  # ✅ use retrieval instead of random

    print("\n[PROCESS_PENDING] Starting pending processing...")
    start_time = time.time()

    global PENDING_NON_LABELED, BATCHES, IMAGE_LIST

    try:
        project_root = Path(__file__).resolve().parent
        BATCH_FILE = project_root / "batches.py"
        print(f"[PROCESS_PENDING] Using batch file: {BATCH_FILE}")

        if not BATCH_FILE.exists():
            raise HTTPException(status_code=500, detail=f"batches.py not found at {BATCH_FILE}")

        print(f"[PROCESS_PENDING] Pending count: {len(PENDING_NON_LABELED)}")

        import batches as batches_module
        importlib.invalidate_caches()
        importlib.reload(batches_module)

        if not PENDING_NON_LABELED:
            print("[PROCESS_PENDING] No pending images. Exiting early.")
            return {"ok": True, "message": "No pending images"}

        # --- Step 1: Merge in Neo4j ---
        print("[PROCESS_PENDING] Merging nodes in Neo4j...")
        merge_all_images_as_nodes([p.replace("/images/", "") for p in PENDING_NON_LABELED])

        # --- Step 2: Refresh available images ---
        IMAGE_LIST = scan_images(IMAGE_ROOT)
        all_web_paths = [f"/images/{rel}" for rel in IMAGE_LIST]
        print(f"[PROCESS_PENDING] Total available images: {len(all_web_paths)}")

        # --- Step 3: Retrieval ---
        K = 4
        new_entries = []
        for idx, web_q in enumerate(PENDING_NON_LABELED, 1):
            print(f"[PROCESS_PENDING] [{idx}/{len(PENDING_NON_LABELED)}] Retrieving for {web_q}...")
            try:
                retrieved = retrieval.top_k_similar(web_q, all_web_paths, k=K)
                candidates = [r["path"] for r in retrieved]
                print(f"  ↳ got {len(candidates)} candidates")
            except Exception as e:
                print(f"[WARN] retrieval failed for {web_q}: {e}")
                traceback.print_exc()
                candidates = [p for p in all_web_paths if p != web_q][:K]

            new_entries.append({"queryImage": web_q, "images": candidates})

        # --- Step 4: Write to batches.py ---
        print(f"[PROCESS_PENDING] Writing {len(new_entries)} entries to batches.py")
        existing = list(getattr(batches_module, "BATCHES", []))
        index_by_query = {e["queryImage"]: i for i, e in enumerate(existing) if isinstance(e, dict)}
        added = 0
        for entry in new_entries:
            q = entry["queryImage"]
            if q in index_by_query:
                existing[index_by_query[q]] = entry
            else:
                existing.append(entry)
                added += 1

        with open(BATCH_FILE, "w", encoding="utf-8") as f:
            f.write("BATCHES = ")
            json.dump(existing, f, indent=2)
            f.write("\n")

        # --- Step 5: Reload batches ---
        importlib.invalidate_caches()
        importlib.reload(batches_module)
        BATCHES = batches_module.BATCHES
        print(f"[PROCESS_PENDING] Reloaded batches: {len(BATCHES)} total")

        processed = len(PENDING_NON_LABELED)
        PENDING_NON_LABELED.clear()
        took = round(time.time() - start_time, 2)
        print(f"[PROCESS_PENDING] Done! processed={processed}, added={added}, took={took}s")

        return {
            "ok": True,
            "processed": processed,
            "batches_added": added,
            "total_batches": len(BATCHES),
            "batches_file": str(BATCH_FILE),
            "took_seconds": took,
        }

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Process failed: {e}")


