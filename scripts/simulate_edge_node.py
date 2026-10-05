"""
scripts/simulate_edge_node.py
------------------------------
SOFTWARE alternative to a physical ESP32 (set "edge_node.mode":
"software" in config.json). Runs the exact same trained pipeline
(scaler.joblib -> pca.joblib -> decision_tree.joblib, produced by
`python3 -m ml.train_export`) on your PC instead of on real hardware,
and sends alerts to the Ryu controller over the SAME REST API the
ESP32 sketch uses -- so the controller/mitigation side of the project
does not need to change at all between hardware and software modes.

Two feature sources:
  --mode replay   (default) replays real rows from the dataset used
                  for training (data/IoTID20.csv or the synthetic
                  stand-in), simulating a stream of live traffic.
  --mode stdin    reads NUM_RAW_FEATURES comma-separated floats per
                  line from stdin, for manual/scripted testing.

USAGE
  python3 scripts/simulate_edge_node.py                 # replay mode
  python3 scripts/simulate_edge_node.py --mode stdin
  python3 scripts/simulate_edge_node.py --limit 200 --delay 0.5
"""

import argparse
import json
import os
import sys
import time

import joblib
import numpy as np
import requests

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from ml.dataset import load_dataset  # noqa: E402

OUT_DIR = os.path.join(PROJECT_ROOT, "outputs")


def load_config():
    cfg_path = os.path.join(PROJECT_ROOT, "config.json")
    if not os.path.exists(cfg_path):
        cfg_path = os.path.join(PROJECT_ROOT, "config.example.json")
    with open(cfg_path) as fh:
        return json.load(fh)


def load_pipeline():
    scaler = joblib.load(os.path.join(OUT_DIR, "scaler.joblib"))
    pca = joblib.load(os.path.join(OUT_DIR, "pca.joblib"))
    clf = joblib.load(os.path.join(OUT_DIR, "decision_tree.joblib"))
    with open(os.path.join(OUT_DIR, "model_metadata.json")) as fh:
        meta = json.load(fh)
    return scaler, pca, clf, meta


def classify(raw_row: np.ndarray, scaler, pca, clf) -> int:
    scaled = scaler.transform(raw_row.reshape(1, -1))
    pc = pca.transform(scaled)
    return int(clf.predict(pc)[0])


def send_alert(alert_url: str, device_id: str, src_ip: str, predicted_class: int):
    payload = {
        "device_id": device_id,
        "src_ip": src_ip,
        "predicted_class": predicted_class,
        "timestamp_ms": int(time.time() * 1000),
    }
    try:
        resp = requests.post(alert_url, json=payload, timeout=3)
        print(f"[edge-sim] POST {alert_url} -> HTTP {resp.status_code} {resp.text[:200]}")
    except requests.exceptions.RequestException as e:
        print(f"[edge-sim] Could not reach controller ({alert_url}): {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["replay", "stdin"], default="replay")
    parser.add_argument("--limit", type=int, default=100,
                         help="Number of rows to replay (replay mode only)")
    parser.add_argument("--delay", type=float, default=1.0,
                         help="Seconds between samples")
    parser.add_argument("--src-ip", default=None,
                         help="Override the src_ip reported in alerts "
                              "(default: config.json network.attacker_ip)")
    args = parser.parse_args()

    cfg = load_config()
    alert_url = cfg["controller"]["alert_url"]
    device_id = cfg["edge_node"]["device_id"]
    src_ip = args.src_ip or cfg.get("network", {}).get("attacker_ip", "10.0.0.5")

    scaler, pca, clf, meta = load_pipeline()
    print(f"[edge-sim] Loaded pipeline: {meta['num_raw_features']} raw features "
          f"-> {meta['num_pca_components']} PCA components -> DecisionTree")
    print(f"[edge-sim] Alerts will be POSTed to: {alert_url}")

    if args.mode == "stdin":
        print("[edge-sim] Reading comma-separated feature rows from stdin "
              f"({meta['num_raw_features']} values per line)...")
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            values = [float(v) for v in line.split(",")]
            if len(values) != meta["num_raw_features"]:
                print(f"[edge-sim] Skipping malformed row (got {len(values)} values, "
                      f"expected {meta['num_raw_features']})")
                continue
            raw_row = np.array(values)
            predicted = classify(raw_row, scaler, pca, clf)
            print(f"[edge-sim] predicted={'ATTACK' if predicted == 1 else 'BENIGN'}")
            if predicted == 1:
                send_alert(alert_url, device_id, src_ip, predicted)
            time.sleep(args.delay)
        return

    # replay mode: pull real rows from the dataset used to train the model
    X, y, feature_names, source = load_dataset()
    assert list(X.columns) == meta["raw_feature_names"], (
        "Feature order mismatch between dataset.py and model_metadata.json -- "
        "re-run `python3 -m ml.train_export` after changing the dataset.")

    n = min(args.limit, len(X))
    sample_idx = np.random.default_rng(42).choice(len(X), size=n, replace=False)

    n_attacks_detected = 0
    for i, idx in enumerate(sample_idx):
        raw_row = X.iloc[idx].values
        true_label = int(y.iloc[idx])
        predicted = classify(raw_row, scaler, pca, clf)

        tag = "ATTACK" if predicted == 1 else "BENIGN"
        truth = "attack" if true_label == 1 else "benign"
        print(f"[edge-sim] sample {i+1}/{n} (true={truth}) -> predicted={tag}")

        if predicted == 1:
            n_attacks_detected += 1
            send_alert(alert_url, device_id, src_ip, predicted)

        time.sleep(args.delay)

    print(f"\n[edge-sim] Done. {n_attacks_detected}/{n} samples classified as "
          f"attacks and alerted to the controller.")


if __name__ == "__main__":
    main()
