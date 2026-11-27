FROM python:3.11-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 PYTHONPATH=/app/src:/app/api

COPY api/requirements.txt api/requirements.txt
RUN pip install --no-cache-dir -r api/requirements.txt

COPY src src
COPY api api

EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=3s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
