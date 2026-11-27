FROM python:3.11-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 PYTHONPATH=/app/src

COPY training/requirements.txt training/requirements.txt
RUN pip install --no-cache-dir -r training/requirements.txt

COPY src src
COPY training training

# data/ is mounted by docker-compose so the image stays small
CMD ["python", "training/flow.py"]
