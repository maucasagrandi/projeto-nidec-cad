FROM python:3.11-slim

WORKDIR /app

# Ghostscript is used to recompress oversized report PDFs so they fit under the
# email attachment size limit.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ghostscript \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .

ENV PORT=8080
EXPOSE 8080

CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "1", "--threads", "2", "--timeout", "120", "main:app"]
