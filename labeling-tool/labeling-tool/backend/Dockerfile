FROM labeling-tool-backend-base

# Working directory
WORKDIR /app

# Prevent Python from writing .pyc files and force unbuffered logs
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Hugging Face / OpenCLIP cache paths inside the container
ENV HF_HOME=/root/.cache/huggingface
ENV OPENCLIP_CACHE_DIR=/root/.cache/huggingface
ENV TORCH_HOME=/root/.cache/torch

# Copy your app code
COPY main.py .
COPY retrieval.py .

# Expose FastAPI / Uvicorn port
EXPOSE 8000

# Default command to start the backend
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
