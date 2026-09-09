# Use Python 3.11 slim (Debian bookworm)
FROM python:3.11-slim

# Install only what's needed: libgl1 (replaces libgl1-mesa-glx in bookworm) + git-lfs
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    git \
    git-lfs \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first (Docker layer caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Handle Git LFS: if the model is a pointer, clone and check out its LFS object.
RUN if [ -f best_model.h5 ] && head -1 best_model.h5 | grep -q "version https://git-lfs"; then \
    echo "Model is an LFS pointer, fetching real file..." && \
    git lfs install --skip-repo && \
    git clone --depth 1 https://github.com/BenDawgz/crop-disease-detection-webapp-1.git /tmp/repo && \
    cd /tmp/repo && git lfs pull --include="best_model.h5" && \
    cp /tmp/repo/best_model.h5 /app/best_model.h5 && \
    rm -rf /tmp/repo && \
    echo "Model downloaded successfully"; \
    else echo "Model file is already present"; fi

# Verify model file exists and is not a pointer
RUN python -c "import os; size=os.path.getsize('best_model.h5'); print(f'Model size: {size} bytes'); assert size > 1000000, f'Model too small ({size}), likely LFS pointer'"

# Create upload directory
RUN mkdir -p static/shots

# Expose port
EXPOSE 5000

# Use single worker to stay within free tier memory
CMD gunicorn --bind 0.0.0.0:${PORT:-5000} --timeout 300 --workers 1 --threads 2 --preload app:app
