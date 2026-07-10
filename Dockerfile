FROM python:3.11-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --timeout 120 -r requirements.txt

# Pre-download the embedding model so runtime doesn't need network
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"


FROM python:3.11-slim

WORKDIR /app

# Copy only what's needed from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /root/.cache /root/.cache

RUN apt-get update && apt-get install -y --no-install-recommends bash \
    && rm -rf /var/lib/apt/lists/*

COPY .streamlit/ .streamlit/
COPY app/ app/
COPY streamlit_app/ streamlit_app/
COPY data/ data/
COPY run.sh .

RUN mkdir -p /input /output

ENV CLASSIFIER_MODEL_PATH=/app/data/task_classifier.pkl
ENV FIREWORKS_BASE_URL=https://api.fireworks.ai/inference/v1

EXPOSE 8000 8501

CMD ["bash", "run.sh"]
