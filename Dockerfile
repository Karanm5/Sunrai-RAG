# Reproducible image. Includes the tesseract binary, which is the one
# system-level dependency a pip install cannot provide.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/src \
    HF_HOME=/app/.hf

RUN apt-get update && apt-get install -y --no-install-recommends \
        tesseract-ocr tesseract-ocr-eng libgl1 libglib2.0-0 make \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Verify the image can at least run the offline suite at build time.
RUN pytest tests/ -q

EXPOSE 8501
CMD ["streamlit", "run", "src/sunrai_rag/demo/app.py", \
     "--server.address=0.0.0.0", "--server.port=8501"]
