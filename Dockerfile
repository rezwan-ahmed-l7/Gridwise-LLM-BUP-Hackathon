FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire refactored project
COPY app/ app/
COPY static/ static/
COPY sample_request.json ./
COPY Question/ Question/

ENV PORT=8000
EXPOSE 8000

# Launch the FastAPI app via the standard module path
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
