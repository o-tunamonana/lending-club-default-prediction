"""Traffic simulator: replays unseen 2016 loans against the API.

    python monitoring/simulate.py                  # normal traffic
    python monitoring/simulate.py --drift          # riskier applicants (drift)
    DRIFT=true docker compose up -d simulator      # switch the running simulator

After each prediction it sends the loan's real outcome to /feedback, the way
labels would arrive later in production, so Evidently can score the model.
A small share of deliberately invalid requests exercises error handling.
"""

import argparse
import logging
import math
import os
import random
import sys
import time
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lcdp.drift import inject_drift  # noqa: E402
from lcdp.schema import FEATURES, TARGET  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("simulator")


def env_flag(name: str) -> bool:
    return os.getenv(name, "false").lower() in {"1", "true", "yes"}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--api-url", default=os.getenv("API_URL", "http://localhost:8000"))
    p.add_argument("--data", default=os.getenv("SIM_DATA", "data/test.csv"))
    p.add_argument("--rate", type=float, default=float(os.getenv("SIM_RATE", "5")), help="requests per second")
    p.add_argument("--limit", type=int, default=int(os.getenv("SIM_LIMIT", "0")), help="0 = run forever")
    p.add_argument("--drift", action="store_true", default=env_flag("DRIFT"))
    p.add_argument("--error-rate", type=float, default=float(os.getenv("SIM_ERROR_RATE", "0.02")))
    p.add_argument("--seed", type=int, default=int(os.getenv("SIM_SEED", "7")))
    return p.parse_args()


def to_payload(row: dict) -> dict:
    payload = {k: row[k] for k in FEATURES}
    for k in ["revenue", "dti_n", "loan_amnt", "fico_n"]:
        value = float(payload[k])
        payload[k] = None if math.isnan(value) else value
    return payload


def wait_for_api(api_url: str) -> None:
    while True:
        try:
            if requests.get(f"{api_url}/health", timeout=3).json().get("model_loaded"):
                return
        except requests.RequestException:
            pass
        log.info("Waiting for the API to load a model...")
        time.sleep(5)


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    data = pd.read_csv(args.data, low_memory=False).sample(frac=1, random_state=args.seed)
    if args.drift:
        data = inject_drift(data, seed=args.seed)
    scenario = "drift" if args.drift else "normal"
    rows = data[[*FEATURES, TARGET]].to_dict("records")
    log.info("Sending %s traffic (%d loans) to %s at %.1f req/s", scenario, len(rows), args.api_url, args.rate)

    wait_for_api(args.api_url)
    session = requests.Session()
    session.headers["X-Traffic-Scenario"] = scenario
    sent = ok = failed = 0
    i = 0
    while args.limit == 0 or sent < args.limit:
        row = rows[i % len(rows)]
        i += 1
        payload = to_payload(row)
        if rng.random() < args.error_rate:
            payload["fico_n"] = 1200  # invalid on purpose
        try:
            r = session.post(f"{args.api_url}/predict", json=payload, timeout=10)
            if r.status_code == 200:
                ok += 1
                pid = r.json().get("prediction_id")
                if pid is not None:
                    session.post(f"{args.api_url}/feedback",
                                 json={"prediction_id": pid, "actual": int(row[TARGET])}, timeout=10)
            else:
                failed += 1
        except requests.RequestException as exc:
            failed += 1
            log.warning("Request failed: %s", exc)
        sent += 1
        if sent % 100 == 0:
            log.info("%s: sent=%d ok=%d failed=%d", scenario, sent, ok, failed)
        time.sleep(1 / args.rate if args.rate > 0 else 0)
    log.info("Done: sent=%d ok=%d failed=%d", sent, ok, failed)


if __name__ == "__main__":
    main()
