from __future__ import annotations

import csv
import random
from pathlib import Path

import numpy as np

from ..config import ROOT
from ..schemas import Alert, FlowMeta
from .flow_to_alert import flow_to_alert

CSV_FILES = [
    ROOT / "data" / "raw" / "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
    ROOT / "data" / "raw" / "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
    ROOT / "data" / "raw" / "Friday-WorkingHours-Morning.pcap_ISCX.csv",
]

RESERVOIR_PER_FILE = 1000
N_FEATURES = 78
PROTOCOLS = ["TCP", "UDP", "ICMP"]


def _row_to_features(row: list[str]) -> np.ndarray | None:
    if len(row) <N_FEATURES:
        return None
    try:
        vals = [float(x) for x in row[:N_FEATURES]]
    except ValueError:
        return None
    arr = np.asarray(vals, dtype=np.float32)
    if not np.all(np.isfinite(arr)):
        return None
    return arr


def _reservoir_sample(path: Path, k: int) -> list[np.ndarray]:
    rng = random.Random(42)
    reservoir = []
    with open(path,newline="", encoding="utf-8", errors="replace") as f:
        