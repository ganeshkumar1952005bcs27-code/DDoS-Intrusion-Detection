"""
config_loader.py
-----------------
Small shared helper so every part of the project (ml/, controller/,
network/, scripts/) reads the SAME configuration the SAME way.

Resolution order:
    1. Explicit path passed by the caller.
    2. <project_root>/config.json      (your local, edited copy)
    3. <project_root>/config.example.json  (fallback defaults)

Usage:
    from config_loader import load_config
    cfg = load_config()
    cfg["controller"]["alert_url"]
"""

import json
import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


def load_config(path: str = None) -> dict:
    candidates = []
    if path:
        candidates.append(path)
    candidates.append(os.path.join(PROJECT_ROOT, "config.json"))
    candidates.append(os.path.join(PROJECT_ROOT, "config.example.json"))

    for p in candidates:
        if p and os.path.exists(p):
            with open(p) as fh:
                cfg = json.load(fh)
            cfg["_loaded_from"] = p
            return cfg

    raise FileNotFoundError(
        "No config.json or config.example.json found next to config_loader.py. "
        "Copy config.example.json to config.json and edit it."
    )
