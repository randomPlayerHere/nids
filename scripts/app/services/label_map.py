from __future__ import annotations

from typing import Literal

Severity = Literal["critical", "high", "medium", "low"]

SEVERITY_MAP: dict[str, Severity] = {
    "BENIGN": "low",
    "DDoS": "critical",
    "DoS GoldenEye": "critical",
    "DoS Hulk": "critical",
    "DoS Slowhttptest": "high",
    "DoS slowloris": "high",
    "FTP-Patator": "high",
    "SSH-Patator": "high",
    "Botnet": "medium",
    "PortScan": "medium",
    "Web Attacks": "medium",
}


def to_severity(label: str) -> Severity:
    return SEVERITY_MAP.get(label, "low")
