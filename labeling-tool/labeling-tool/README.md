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

