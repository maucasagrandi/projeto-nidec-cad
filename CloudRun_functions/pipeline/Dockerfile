FROM python:3.11-slim

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy pipeline function code
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the full project source needed by the pipeline
COPY main.py .
COPY src/ ./src/
COPY prompts.py .
COPY assets/ ./assets/

# Cloud Run uses PORT env var
ENV PORT=8080
EXPOSE 8080

CMD ["functions-framework", "--target=pipeline", "--port=8080"]
