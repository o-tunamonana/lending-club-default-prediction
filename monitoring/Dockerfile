# One image for both the Evidently monitor and the traffic simulator.
FROM python:3.11-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 PYTHONPATH=/app/src

COPY monitoring/requirements.txt monitoring/requirements.txt
RUN pip install --no-cache-dir -r monitoring/requirements.txt

COPY src src
COPY monitoring/monitor.py monitoring/simulate.py monitoring/

CMD ["python", "monitoring/monitor.py"]
