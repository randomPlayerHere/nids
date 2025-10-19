#!/usr/bin/env python3
"""
nids_flow_server.py

- Capture packets with scapy
- Aggregate into flows (5-tuple)
- On flow timeout or TCP FIN/RST, compute features (KDD-approx)
- Load a saved KNN model + preprocessors (optional) and classify each flow
- Print/log results

Run as root:
    sudo python3 nids_flow_server.py

Dependencies:
    pip install scapy scapy-http  # scapy-http optional, but scapy alone is usually enough
"""

import time
import pickle
import threading
from collections import defaultdict, deque
from dataclasses import dataclass, field
import numpy as np
import os
import json
import argparse
from typing import Tuple

# scapy import (heavy); import inside try for nicer error message
try:
    from scapy.all import sniff, IP, TCP, UDP, Raw
except Exception as e:
    raise SystemExit("scapy required. Install with: pip install scapy\nException: " + str(e))


# ----------------------------
# Config
# ----------------------------
MODEL_PATH = "knn_model.pkl"     # your trained model (optional)
SCALER_PATH = "scaler.pkl"       # optional scaler used during training
ENCODERS_PATH = "encoders.pkl"   # optional dict for categorical encoders (e.g., flag mapping)
LOG_PATH = "nids_flow_log.jsonl" # append-only logs of flow features + prediction

# Flow timeout (seconds): if no packets for this flow in X seconds -> finalize flow
FLOW_TIMEOUT = 2.0

# Sliding window durations (seconds) used to compute aggregate features similar to KDD's short-term counts.
CONN_WINDOW_SHORT = 2.0    # for "count" type features (connections from same src to same dst host)
CONN_WINDOW_LONG = 100.0   # example for other aggregates (adjust as needed)

# Feature order that will be fed to model. You MUST ensure this matches the training feature order.
# I propose a reduced, practical set of features we can compute from packets:
FEATURE_ORDER = [
    'duration',          # in seconds
    'protocol_type',     # 0=tcp,1=udp,2=icmp (numeric mapping)
    'src_bytes',
    'dst_bytes',
    'pkts',              # total packets in flow
    'avg_pkt_size',
    'flag',              # numeric mapping of connection flag (see mapping below)
    'count_src_dst_2s',  # number of connections from same src->dst in last CONN_WINDOW_SHORT seconds
    'count_src_srv_2s',  # number of connections from same src to same service (dst port) in last CONN_WINDOW_SHORT seconds
    'dst_host_count_100s' # number of connections to this dst host in last CONN_WINDOW_LONG seconds
]
# If your model used a different feature set: replace FEATURE_ORDER and adjust preprocess accordingly.


# ----------------------------
# Helpers and state
# ----------------------------
# numeric mappings
PROTO_MAP = {'tcp': 0, 'udp': 1, 'icmp': 2}
# map basic TCP flag patterns to small set (SYN, SYN-ACK, FIN, RST, PSH, ACK-only, other)
FLAG_MAP = {
    'S': 1,    # SYN
    'SA': 2,   # SYN-ACK
    'F': 3,    # FIN
    'R': 4,    # RST
    'P': 5,    # PSH (push)
    'A': 6,    # ACK-only
    'O': 0     # other/unknown
}

@dataclass
class FlowInfo:
    first_ts: float
    last_ts: float
    src: str
    sport: int
    dst: str
    dport: int
    proto: int
    pkts: int = 0
    src_bytes: int = 0
    dst_bytes: int = 0
    tcp_flags: set = field(default_factory=set)

    def key(self) -> Tuple[str,int,str,int,int]:
        return (self.src, self.sport, self.dst, self.dport, self.proto)

# Live state
flows = {}  # key -> FlowInfo
# Recent finished flows for sliding-window aggregates: deque of (finish_ts, FlowInfo)
finished_flows_window = deque()  # used to compute counts like "how many connections to dst host in last N seconds"

state_lock = threading.Lock()

# Model + preprocessors (optional)
knn = None
scaler = None
encoders = None

# ----------------------------
# Load model if exists
# ----------------------------
def load_model():
    global knn, scaler, encoders
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH,'rb') as f:
            knn = pickle.load(f)
        print(f"[model] Loaded model from {MODEL_PATH}")
    else:
        print("[model] No model file found; server will still compute and print features. Place your knn_model.pkl to enable prediction.")

    if os.path.exists(SCALER_PATH):
        with open(SCALER_PATH,'rb') as f:
            scaler = pickle.load(f)
        print(f"[model] Loaded scaler from {SCALER_PATH}")

    if os.path.exists(ENCODERS_PATH):
        with open(ENCODERS_PATH,'rb') as f:
            encoders = pickle.load(f)
        print(f"[model] Loaded encoders from {ENCODERS_PATH}")

# ----------------------------
# Flow bookkeeping & timeouts
# ----------------------------
def make_flow_key(src, sport, dst, dport, proto):
    return (src, int(sport), dst, int(dport), int(proto))

def update_or_create_flow(pkt_time, src, sport, dst, dport, proto, plen, flags_str):
    key = make_flow_key(src,sport,dst,dport,proto)
    with state_lock:
        fi = flows.get(key)
        if fi is None:
            fi = FlowInfo(first_ts=pkt_time, last_ts=pkt_time, src=src, sport=int(sport), dst=dst, dport=int(dport), proto=int(proto), pkts=1)
            # set bytes heuristically: we don't know direction for first pkt until compare IPs on later pkts; but we'll set based on src==fi.src
            if src == fi.src:
                fi.src_bytes += plen
            else:
                fi.dst_bytes += plen
            if flags_str:
                fi.tcp_flags.add(flags_str)
            flows[key] = fi
        else:
            fi.pkts += 1
            fi.last_ts = pkt_time
            # approximate src/dst bytes by comparing pkt src to flow.src
            if src == fi.src:
                fi.src_bytes += plen
            else:
                fi.dst_bytes += plen
            if flags_str:
                fi.tcp_flags.add(flags_str)
    return key

def finish_flow(key):
    # remove from active flows, compute features and classify
    with state_lock:
        fi = flows.pop(key, None)
    if not fi:
        return
    finish_ts = time.time()
    # append to finished window
    with state_lock:
        finished_flows_window.append((finish_ts, fi))
        # prune old entries beyond the longest window we use
        cutoff = finish_ts - max(CONN_WINDOW_LONG, CONN_WINDOW_SHORT)
        while finished_flows_window and finished_flows_window[0][0] < cutoff:
            finished_flows_window.popleft()

    # compute features for the flow (see compute_features below)
    feat = compute_features(fi, finish_ts)
    # classify (if model present)
    prediction = None
    prob = None
    if knn is not None:
        try:
            x = np.array([feat[f] for f in FEATURE_ORDER], dtype=float).reshape(1, -1)
            if scaler is not None:
                x = scaler.transform(x)
            pred = knn.predict(x)[0]
            prediction = str(pred)
            if hasattr(knn, "predict_proba"):
                prob = float(knn.predict_proba(x).max())
        except Exception as e:
            prediction = f"error:{e}"

    # log result
    out = {
        'timestamp': time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(finish_ts)) + "Z",
        'flow_key': key,
        'features': feat,
        'prediction': prediction,
        'prob': prob
    }
    print(json.dumps(out))
    # append to log file
    try:
        with open(LOG_PATH, "a") as f:
            f.write(json.dumps(out) + "\n")
    except Exception as e:
        print("[log] write error:", e)

# called periodically to expire idle flows
def timeout_checker():
    while True:
        now = time.time()
        to_finish = []
        with state_lock:
            for key, fi in list(flows.items()):
                if now - fi.last_ts > FLOW_TIMEOUT:
                    to_finish.append(key)
        for key in to_finish:
            finish_flow(key)
        time.sleep(0.5)

# ----------------------------
# Feature computation
# ----------------------------
def map_proto(proto_int):
    # expecting 6 for TCP, 17 for UDP, 1 for ICMP (scapy's ip proto numbers),
    # but we store our own small enums in FlowInfo.proto (0=tcp,1=udp,2=icmp)
    return proto_int  # already mapped

def flag_to_numeric(flags_set):
    # flags_set is a set of strings seen (e.g., 'S', 'SA', 'F', 'R', 'A')
    # choose highest-severity or first seen mapping
    if not flags_set:
        return FLAG_MAP['O']
    # map common: if SYN and ACK seen => SA
    if 'SA' in flags_set:
        return FLAG_MAP['SA']
    if 'S' in flags_set and 'A' in flags_set:
        return FLAG_MAP['SA']
    if 'S' in flags_set:
        return FLAG_MAP['S']
    if 'F' in flags_set:
        return FLAG_MAP['F']
    if 'R' in flags_set:
        return FLAG_MAP['R']
    if 'P' in flags_set:
        return FLAG_MAP['P']
    if 'A' in flags_set:
        return FLAG_MAP['A']
    return FLAG_MAP['O']

def compute_features(fi: FlowInfo, finish_ts: float):
    """
    Returns dict mapping feature name -> value, matching FEATURE_ORDER
    Includes sliding window aggregates based on finished_flows_window.
    """
    # base flow-level features
    duration = max(0.0, fi.last_ts - fi.first_ts)
    src_bytes = fi.src_bytes
    dst_bytes = fi.dst_bytes
    pkts = fi.pkts
    avg_pkt_size = (src_bytes + dst_bytes) / pkts if pkts > 0 else 0.0
    proto = fi.proto  # already numeric small mapping
    flag_num = flag_to_numeric(fi.tcp_flags)

    # sliding window aggregates
    # count_src_dst_2s: number of flows from same src -> same dst within last CONN_WINDOW_SHORT seconds
    # count_src_srv_2s: number of flows from same src to same "service" i.e., same dst port in last CONN_WINDOW_SHORT
    # dst_host_count_100s: number of flows to same dst host in last CONN_WINDOW_LONG seconds

    tcut_short = finish_ts - CONN_WINDOW_SHORT
    tcut_long = finish_ts - CONN_WINDOW_LONG

    count_src_dst = 0
    count_src_srv = 0
    dst_host_count = 0

    with state_lock:
        for ts, other in finished_flows_window:
            if ts < tcut_short and ts < tcut_long:
                # this and older entries will be pruned elsewhere; skip
                continue
            # same src->dst
            if other.src == fi.src and other.dst == fi.dst and ts >= tcut_short:
                count_src_dst += 1
            # same src->same service (dst port) short window
            if other.src == fi.src and other.dport == fi.dport and ts >= tcut_short:
                count_src_srv += 1
            # kth: flows to same destination host in long window
            if other.dst == fi.dst and ts >= tcut_long:
                dst_host_count += 1

    feat = {
        'duration': duration,
        'protocol_type': proto,
        'src_bytes': src_bytes,
        'dst_bytes': dst_bytes,
        'pkts': pkts,
        'avg_pkt_size': avg_pkt_size,
        'flag': flag_num,
        'count_src_dst_2s': count_src_dst,
        'count_src_srv_2s': count_src_srv,
        'dst_host_count_100s': dst_host_count
    }
    return feat

# ----------------------------
# Packet callback
# ----------------------------
def pkt_callback(pkt):
    # called for each sniffed packet
    # ignore packets without IP layer
    if not pkt.haslayer(IP):
        return
    ip = pkt[IP]
    pkt_time = pkt.time
    src = ip.src
    dst = ip.dst
    proto = None
    sport = 0
    dport = 0
    flags_str = None

    # determine protocol and ports
    if pkt.haslayer(TCP):
        proto = PROTO_MAP['tcp']
        sport = pkt[TCP].sport
        dport = pkt[TCP].dport
        # TCP flags mapping
        raw_flags = pkt[TCP].flags
        # scapy represents flags as characters; translate to simple tokens
        # build simple set: 'S', 'A', 'F', 'R', 'P' when present
        flags_set = set()
        f = str(raw_flags)
        if 'S' in f: flags_set.add('S')
        if 'A' in f: flags_set.add('A')
        if 'F' in f: flags_set.add('F')
        if 'R' in f: flags_set.add('R')
        if 'P' in f: flags_set.add('P')
        # interpret SYN+ACK as SA
        if 'S' in f and 'A' in f:
            flags_str = 'SA'
        elif 'S' in f:
            flags_str = 'S'
        elif 'F' in f:
            flags_str = 'F'
        elif 'R' in f:
            flags_str = 'R'
        elif 'P' in f:
            flags_str = 'P'
        elif 'A' in f:
            flags_str = 'A'
    elif pkt.haslayer(UDP):
        proto = PROTO_MAP['udp']
        sport = pkt[UDP].sport
        dport = pkt[UDP].dport
    else:
        # treat as ICMP or other
        proto = PROTO_MAP['icmp'] if ip.proto == 1 else PROTO_MAP.get('icmp',2)
        sport = 0
        dport = 0

    # packet length
    try:
        plen = len(pkt)
    except Exception:
        plen = 0

    # update/create flow
    key = update_or_create_flow(pkt_time, src, sport, dst, dport, proto, plen, flags_str)

    # if TCP FIN/RST seen, finish flow immediately
    if pkt.haslayer(TCP):
        f = str(pkt[TCP].flags)
        if 'F' in f or 'R' in f:
            # schedule finishing
            finish_flow(key)


# ----------------------------
# Utility: optional routine to convert saved capture to features for retraining
# ----------------------------
def export_feature_records_for_retraining(output_csv_path="flow_features.csv", limit=None):
    """
    Optionally: sniff for some time and print raw features to CSV so you can build
    a retraining dataset that maps these features to labels using KDD labels.
    (Labeling must be done manually / by using known attack flows.)
    """
    raise NotImplementedError("Use the live log file (nids_flow_log.jsonl) to collect flow features for retraining.")


# ----------------------------
# Main
# ----------------------------
def main():
    parser = argparse.ArgumentParser(description="NIDS flow capture + classification (KDD-approx features)")
    parser.add_argument("--iface", default=None, help="Interface to sniff (default: all)")
    parser.add_argument("--model", default=MODEL_PATH, help="Path to knn model pickle")
    parser.add_argument("--scaler", default=SCALER_PATH, help="Path to scaler pickle (optional)")
    parser.add_argument("--encoders", default=ENCODERS_PATH, help="Path to encoders pickle (optional)")
    parser.add_argument("--log", default=LOG_PATH, help="Path to output log (jsonl)")
    args = parser.parse_args()

    global MODEL_PATH, SCALER_PATH, ENCODERS_PATH, LOG_PATH
    MODEL_PATH = args.model
    SCALER_PATH = args.scaler
    ENCODERS_PATH = args.encoders
    LOG_PATH = args.log

    load_model()

    # start timeout checker
    t = threading.Thread(target=timeout_checker, daemon=True)
    t.start()

    print("[nids] starting sniffing... (CTRL-C to stop)")
    try:
        sniff(iface=args.iface, prn=pkt_callback, store=False)
    except KeyboardInterrupt:
        print("[nids] interrupted by user; finishing active flows...")
        # finish all active flows
        keys = list(flows.keys())
        for k in keys:
            finish_flow(k)
        print("[nids] done.")
        return

if __name__ == "__main__":
    main()