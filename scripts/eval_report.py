#!/usr/bin/env python3
"""Reproduce the held-out test split and produce a full classification report.

Mirrors the training notebook's split exactly: SEED=42, test_size=0.2, stratified.
Writes scripts/EVAL_REPORT.md and prints the report.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
SEED = 42

X = np.load(ROOT / "data/processed/X_dcnn.npy")
y = np.load(ROOT / "data/processed/y_labels.npy")
with open(ROOT / "models/new/class_mapping.json") as f:
    class_mapping = json.load(f)
labels = [class_mapping[str(i)] for i in range(len(class_mapping))]

# same first split as the notebook -> recovers the identical held-out test set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEED
)
print(f"X: {X.shape}  test: {X_test.shape[0]} rows")

model = tf.keras.models.load_model(ROOT / "models/new/nids_model.h5", compile=False)
probs = model.predict(X_test, batch_size=512, verbose=1)
y_pred = probs.argmax(axis=1)

acc = accuracy_score(y_test, y_pred)
macro_f1 = f1_score(y_test, y_pred, average="macro")
weighted_f1 = f1_score(y_test, y_pred, average="weighted")
macro_p = precision_score(y_test, y_pred, average="macro", zero_division=0)
macro_r = recall_score(y_test, y_pred, average="macro", zero_division=0)

report = classification_report(
    y_test, y_pred, target_names=labels, digits=4, zero_division=0
)
cm = confusion_matrix(y_test, y_pred)

# per-class table for markdown
prec = precision_score(y_test, y_pred, average=None, zero_division=0)
rec = recall_score(y_test, y_pred, average=None, zero_division=0)
f1c = f1_score(y_test, y_pred, average=None, zero_division=0)
support = np.bincount(y_test, minlength=len(labels))

lines = []
lines.append("# NIDS — Model Evaluation Report\n")
lines.append(f"Held-out test set: **{len(y_test):,} flows** | classes: **{len(labels)}** | "
             f"features: **{X.shape[1]}** (1D CNN)\n")
lines.append("## Headline metrics\n")
lines.append("| Metric | Value |")
lines.append("|---|---|")
lines.append(f"| Accuracy | **{acc*100:.2f}%** |")
lines.append(f"| Macro F1 | **{macro_f1*100:.2f}%** |")
lines.append(f"| Weighted F1 | **{weighted_f1*100:.2f}%** |")
lines.append(f"| Macro Precision | {macro_p*100:.2f}% |")
lines.append(f"| Macro Recall | {macro_r*100:.2f}% |\n")

lines.append("## Per-class metrics\n")
lines.append("| Class | Precision | Recall | F1 | Support |")
lines.append("|---|---|---|---|---|")
for i, name in enumerate(labels):
    lines.append(f"| {name} | {prec[i]*100:.2f}% | {rec[i]*100:.2f}% | "
                 f"{f1c[i]*100:.2f}% | {int(support[i]):,} |")

lines.append("\n## sklearn classification_report\n")
lines.append("```\n" + report + "\n```\n")

lines.append("## Confusion matrix (rows = true, cols = pred)\n")
header = "| true\\pred | " + " | ".join(str(i) for i in range(len(labels))) + " |"
lines.append(header)
lines.append("|" + "---|" * (len(labels) + 1))
for i, name in enumerate(labels):
    row = " | ".join(f"{cm[i, j]:,}" for j in range(len(labels)))
    lines.append(f"| {i} {name} | {row} |")
lines.append("\nClass index → name: " + ", ".join(f"{i}={n}" for i, n in enumerate(labels)) + "\n")

out = ROOT / "scripts/EVAL_REPORT.md"
out.write_text("\n".join(lines))
print("\n" + report)
print(f"\nAccuracy={acc:.4f}  MacroF1={macro_f1:.4f}  WeightedF1={weighted_f1:.4f}")
print(f"Wrote {out}")
