import csv
import random
import numpy as np

from ..config import ROOT
from ..schemas import Alert, FlowMeta
from .flow_to_alert import flow_to_alert
from .model import FEATURE_NAMES


def _find_index(*candidates: str) -> int:
    lowered = [n.strip().lower() for n in FEATURE_NAMES]
    for c in candidates:
        c = c.lower()
        for i, n in enumerate(lowered):
            if n == c:
                return i
    for c in candidates:
        c = c.lower()
        for i, n in enumerate(lowered):
            if c in n:
                return i
    return -1


FLOW_DURATION_IDX = _find_index("flow duration", "flow_duration")
FWD_PACKETS_IDX = _find_index("total fwd packets", "tot fwd pkts", "fwd_packets")

CSV_FILES = [
    ROOT / "data" / "raw" / "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
    ROOT / "data" / "raw" / "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
    ROOT / "data" / "raw" / "Friday-WorkingHours-Morning.pcap_ISCX.csv",
]

SAMPLES_PER_FILE = 200
NUM_FEATURES = 78
PROTOCOLS = ["TCP", "UDP", "ICMP"]


def parse_row(row):
    if len(row) < NUM_FEATURES:
        return None
    values = []
    for x in row[:NUM_FEATURES]:
        try:
            values.append(float(x))
        except ValueError:
            return None
    arr = np.array(values, dtype=np.float32)
    for v in arr:
        if not np.isfinite(v):
            return None
    return arr


def sample_from_file(path, k):
    rng = random.Random(42)
    reservoir = []
    i = 0
    f = open(path, newline="", encoding="utf-8", errors="replace")
    reader = csv.reader(f)
    next(reader, None)
    for row in reader:
        features = parse_row(row)
        if features is None:
            continue
        if len(reservoir) < k:
            reservoir.append(features)
        else:
            j = rng.randint(0, i)
            if j < k:
                reservoir[j] = features
        i += 1
    f.close()
    return reservoir


all_flows = []
for path in CSV_FILES:
    if path.exists():
        rows = sample_from_file(path, SAMPLES_PER_FILE)
        all_flows.extend(rows)
    else:
        print("warning: csv file not found:", path)

if len(all_flows) == 0:
    raise RuntimeError("no flows were loaded, check the csv files")

print("loaded", len(all_flows), "flows for the demo")


def make_random_public_ip():
    a = random.randint(1, 223)
    b = random.randint(0, 255)
    c = random.randint(0, 255)
    d = random.randint(0, 255)
    return str(a) + "." + str(b) + "." + str(c) + "." + str(d)


def make_random_internal_ip():
    return "10.0.0." + str(random.randint(1, 254))


def next_demo_alert():
    features = random.choice(all_flows)
    meta = FlowMeta(
        src_ip=make_random_public_ip(),
        dst_ip=make_random_internal_ip(),
        protocol=random.choice(PROTOCOLS),
        flow_duration=int(features[FLOW_DURATION_IDX]) if FLOW_DURATION_IDX >= 0 else 0,
        fwd_packets=int(features[FWD_PACKETS_IDX]) if FWD_PACKETS_IDX >= 0 else 0,
    )
    alert = flow_to_alert(features, meta, include_shap=False, already_scaled=False)
    return alert
