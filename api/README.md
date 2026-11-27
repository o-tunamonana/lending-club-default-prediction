# Lending Club Model API

The API takes one loan application, scores it with the MLflow champion, and writes a prediction to the existing PostgreSQL schema in `db/init.sql`. It checks for a new champion every 30 seconds, keeps serving the last successfully loaded model while refreshes fail, and uses the selected version's `approval_cutoff` tag. The risk bands are low at or below half that cutoff, medium up to the cutoff and high above it.

| Endpoint | Contract |
|---|---|
| `GET /health` | 200 when the model and predictions table are available, otherwise 503 |
| `POST /predict` | Accepts the eight fields in `lcdp.schema.FEATURES`. Returns `prediction_id`, `default_probability`, `decision`, `approval_cutoff`, `risk_level` and `model_version`. |
| `POST /feedback` | Accepts `prediction_id` and `actual` (0 or 1) to record the eventual outcome for monitoring. |
| `GET /metrics` | Exports `api_requests_total`, `api_request_latency_seconds`, `api_errors_total`, decision counts, probability distribution and loaded version. |

The simulator can set `X-Traffic-Scenario` to `normal` or `drift`. Requests without this header are recorded as `frontend`. Feedback is intended for the trusted classroom simulator on the internal Compose network.

## Team Compose integration

Use this service definition in the team's root Compose file. The existing Postgres service should mount `db/init.sql` as an entrypoint initialization script. Set `DATABASE_URL` to a PostgreSQL URL reachable by the API container and keep credentials in the team's ignored environment configuration.

```yaml
services:
  api:
    build:
      context: .
      dockerfile: api/Dockerfile
    environment:
      DATABASE_URL: ${DATABASE_URL:?set the API database URL}
      MLFLOW_TRACKING_URI: http://mlflow:5000
      MODEL_RELOAD_SECONDS: "30"
    ports:
      - "127.0.0.1:8000:8000"
```

The frontend, simulator and Prometheus can access the API at `http://api:8000` inside the Compose network. Open `http://localhost:8000/docs` for an interactive request form.

## Tests

```bash
pip install -r requirements-dev.txt
pytest tests/test_api.py
RUN_API_CONTAINER_TEST=1 pytest -m integration tests/integration/test_api_container.py
```

The integration test builds the API image and starts isolated API, MLflow and Postgres containers. It registers a deterministic fixture in MLflow, calls the HTTP API, verifies the PostgreSQL row and Prometheus metrics, and removes its test containers and volume. The test champion uses a fitted FICO model to make the expected direction deterministic. The team's gradient-boosting model can vary with other features and does not enforce global FICO monotonicity.
