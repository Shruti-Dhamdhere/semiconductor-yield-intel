FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir torch==2.2.0 --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

COPY setup.py .
COPY src/ ./src/
COPY params.yaml .

RUN pip install --no-cache-dir -e .

EXPOSE 8000 8050
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
