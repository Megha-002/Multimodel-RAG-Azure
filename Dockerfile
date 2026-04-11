# Base image — Python 3.11 minimal
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
# ffmpeg — needed by Whisper for audio processing
# tesseract-ocr — needed by pytesseract for OCR
# libsndfile1 — needed by sounddevice for audio capture
RUN apt-get update && apt-get install -y \
    ffmpeg \
    tesseract-ocr \
    libsndfile1 \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for Docker layer caching
# If requirements don't change, this layer is reused
COPY requirements.txt .

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY . .

# Create necessary directories
RUN mkdir -p data chroma_db images

# Expose ports
# 8000 = FastAPI backend
# 8501 = Streamlit frontend
EXPOSE 8000
EXPOSE 8501

# Start both FastAPI and Streamlit together
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port 8000 & streamlit run ui.py --server.port 8501 --server.address 0.0.0.0"]