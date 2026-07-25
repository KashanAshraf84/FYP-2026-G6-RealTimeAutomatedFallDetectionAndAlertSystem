# GuardianAI - Cloud demonstration image (Hugging Face Spaces)
FROM python:3.11-slim

# Ultralytics, Matplotlib and the event logger all need writable directories.
# Only /tmp is reliably writable in the Spaces runtime.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    YOLO_CONFIG_DIR=/tmp/Ultralytics \
    MPLCONFIGDIR=/tmp/matplotlib \
    GUARDIANAI_LOG_DIR=/tmp/logs \
    PORT=7860

WORKDIR /app

# opencv-python-headless still links against libglib.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install the CPU-only PyTorch build explicitly. The default PyPI wheel bundles
# CUDA and pulls in several GB of NVIDIA libraries that are useless on a CPU
# Space and would blow the image size budget.
RUN pip install --no-cache-dir \
        torch torchvision \
        --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p /tmp/logs /tmp/Ultralytics /tmp/matplotlib \
    && chmod -R 777 /tmp/logs /tmp/Ultralytics /tmp/matplotlib

EXPOSE 7860

CMD ["python", "app.py"]
