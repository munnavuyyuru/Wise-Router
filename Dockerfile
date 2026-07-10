FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends bash \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --timeout 120 -r requirements.txt \
    && rm -rf /root/.cache/pip

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
