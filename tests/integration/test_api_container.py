import os
import subprocess
import time
from pathlib import Path
from uuid import uuid4

import pytest
import requests

pytestmark = pytest.mark.integration
ROOT = Path(__file__).resolve().parents[2]
LOAN = {
    "revenue": 65000, "dti_n": 18.5, "loan_amnt": 12000, "fico_n": 700,
    "emp_length": "5 years", "purpose": "debt_consolidation",
    "home_ownership_n": "RENT", "addr_state": "NY",
}


def test_api_container():
    if os.getenv("RUN_API_CONTAINER_TEST") != "1":
        pytest.skip("Set RUN_API_CONTAINER_TEST=1 to start the isolated Docker stack")

    project = f"lcdp-api-{uuid4().hex[:10]}"
    command = ["docker", "compose", "-p", project, "-f", "tests/integration/compose.api.yml"]
    env = {**os.environ, "API_TEST_DB_PASSWORD": uuid4().hex}

    def compose(*args):
        return subprocess.run(
            [*command, *args], cwd=ROOT, env=env, check=True,
            capture_output=True, text=True, timeout=600,
        ).stdout.strip()

    try:
        compose("build", "api")
        compose("up", "-d", "api")
        compose("cp", "tests/integration/seed_api_registry.py", "api:/tmp/seed_api_registry.py")
        compose("exec", "-T", "api", "python", "/tmp/seed_api_registry.py")
        base = f"http://{compose('port', 'api', '8000')}"
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            try:
                if requests.get(f"{base}/health", timeout=3).status_code == 200:
                    break
            except requests.RequestException:
                pass
            time.sleep(1)
        else:
            pytest.fail(f"API did not become ready:\n{compose('logs', 'api', 'mlflow')}")

        response = requests.post(
            f"{base}/predict", json=LOAN, headers={"X-Traffic-Scenario": "normal"}, timeout=10,
        )
        assert response.status_code == 200, response.text
        result = response.json()
        assert 0 <= result["default_probability"] <= 1
        assert result["model_version"] == "1"
        prediction_id = int(result["prediction_id"])
        feedback = requests.post(
            f"{base}/feedback", json={"prediction_id": prediction_id, "actual": 0}, timeout=10,
        )
        assert feedback.status_code == 200, feedback.text
        row = compose(
            "exec", "-T", "postgres", "psql", "-U", "lcdp_test", "-d", "lcdp_test",
            "-Atc", f"SELECT model_version, scenario, actual, approval_cutoff FROM predictions WHERE id = {prediction_id}",
        )
        assert row == "1|normal|0|0.5"

        invalid = requests.post(f"{base}/predict", json={**LOAN, "fico_n": 999}, timeout=10)
        assert invalid.status_code == 422
        metrics = requests.get(f"{base}/metrics", timeout=10).text
        assert 'api_requests_total{endpoint="/predict",status="200"} 1.0' in metrics
        assert 'api_errors_total{endpoint="/predict",kind="validation",status="422"} 1.0' in metrics
        assert 'api_request_latency_seconds_count{endpoint="/predict"} 2.0' in metrics
    except subprocess.CalledProcessError as exc:
        pytest.fail(f"{exc}\n{exc.stdout}\n{exc.stderr}")
    finally:
        compose("down", "-v", "--remove-orphans")
