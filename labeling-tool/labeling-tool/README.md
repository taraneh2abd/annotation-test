about dockerfile of backend:
i had created image previously so i used my local image base to reduce downloading parts like "torch". for using actual docker ->
use "backup-back-docker" -> maybe need a change in the openclip downloding part.
also i had my model predownload here:"C:\Users\T.Abdellahi\.cache\huggingface\hub\models--laion--CLIP-ViT-B-16-laion2B-s34B-b88K" 
and used that in my container the format of it is safetensors instead of bin and the folder is like this:
C:.
├───blobs
├───refs
└───snapshots
    └───7288da5a0d6f0b51c4a2b27c624837a9236d0112

to check if it has copied in to your container or not just run this in your backend container:
# du -sh /root/.cache/huggingface/hub/models--laion--CLIP-ViT-B-16-laion2B-s34B-b88K



warning: the folder should be full of images. all other files will consider as image

- pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121

- pip install open-clip-torch==2.24.0 --no-deps
pip install numpy==1.26.4 scikit-learn==1.5.2 pillow==10.4.0 tqdm==4.66.5 ftfy==6.3.1 regex==2025.9.18 sentencepiece==0.2.1 timm==1.0.20 torchvision==0.23.0 protobuf==6.32.1 huggingface-hub==0.35.3

be aware : my requirements.txt is not full

# Labeling Tool (MongoDB + REST API + Simple Frontend)

One command brings up the whole stack (MongoDB, Mongo Express, Backend API, and Frontend). The frontend shows a **query image** and a grid of **20 candidate images** from a local folder. Left-click marks a **positive** (green border). Right-click marks a **negative** (red border). Clicking again on a selected image clears it. When you click **Save Selections**, your choices are stored in MongoDB for fast retrieval.

## Quick Start

1. Put your images (jpg, jpeg, png, gif, webp, bmp) into the local `images/` folder (create it next to `docker-compose.yml`). Subfolders are supported.
2. In this directory, run:

name =user 
password =supersecret123

best way to export the db:
```bash
MATCH (n)-[r]->(m)
RETURN n,r,m
```

best way to delete all the db:
```bash
MATCH (n)
DETACH DELETE n;
```

run the code:
- you may need vpn
```bash
docker login

docker compose up --build -d
```

3. Open:
   - Frontend: http://localhost:8080
   - Backend API (docs): http://localhost:8000/docs
   - Mongo Express: http://localhost:8081

## Environment Variables (optional)

You can override defaults by creating a `.env` file or setting environment variables before `docker compose up`:

```env
# Mongo
MONGO_INITDB_ROOT_USERNAME=root
MONGO_INITDB_ROOT_PASSWORD=example
MONGO_DB=labeldb
MONGO_COLLECTION=labels

# Ports
BACKEND_PORT=8000
FRONTEND_PORT=8080
MONGO_PORT=27017
MONGO_EXPRESS_PORT=8081

# Frontend <-> Backend
API_BASE_URL=http://localhost:8000

# Backend CORS
CORS_ORIGINS=*
# Page size (# of candidates)
PAGE_SIZE=20
```

## How it Works

- **backend** (FastAPI) serves:
  - `GET /api/session` → returns a random query image and a page (default 20) of candidate images from `images/`.
  - `POST /api/labels/save` → stores `{queryImage, positives[], negatives[]}` in MongoDB.
  - Static file mount at `/images/*` to serve your local files (from the container's `/data/images`, which is volume-mounted from `./images`).

- **frontend** (Nginx + vanilla JS) calls the API and handles interactions:
  - Left-click = positive (green), Right-click = negative (red).
  - Clicking a selected tile again clears selection.

- **mongo** stores labels, and **mongo-express** makes it easy to browse the DB in your browser.

## Notes

- The backend indexes `(query_image, ts)` for fast inserts and lookups.
- If you add or remove images at runtime, you can call `POST /api/refresh` (or just restart the backend) to rescan the image folder.
- The `/api/session` tries to avoid including the current query image in the 20 candidates and wraps around if the list is short.

## Project Layout

```
.
├─ docker-compose.yml
├─ backend/
│  ├─ Dockerfile
│  ├─ requirements.txt
│  └─ main.py
├─ frontend/
│  ├─ Dockerfile
│  ├─ nginx.conf
│  └─ public/
│     └─ index.html
└─ images/               # <-- put your images here
```

