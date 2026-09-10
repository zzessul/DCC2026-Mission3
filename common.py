from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

LABELS = ["고열", "구토", "두통", "복통", "어지러움", "열상", "오심", "전신쇠약", "호흡곤란"]


def read_csv(path: str, require_labels: bool = True) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"label_file", "text"} | ({"symptom_str"} if require_labels else set())
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"필수 컬럼 누락: {sorted(missing)}")
    return frame


def encode(values) -> np.ndarray:
    output = []
    for value in values:
        present = set() if pd.isna(value) or value == "" else set(str(value).split("|"))
        unknown = present - set(LABELS)
        if unknown:
            raise ValueError(f"알 수 없는 증상: {sorted(unknown)}")
        output.append([int(label in present) for label in LABELS])
    return np.asarray(output, dtype=np.float32)


def decode(binary: np.ndarray) -> list[str]:
    return ["|".join(label for label, flag in zip(LABELS, row) if flag) for row in binary]


def tune_thresholds(probability: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, float]:
    grid = np.arange(0.08, 0.91, 0.01)
    thresholds = []
    for index, label in enumerate(LABELS):
        scores = [f1_score(target[:, index], probability[:, index] >= t, zero_division=0) for t in grid]
        best = int(np.argmax(scores))
        thresholds.append(float(grid[best]))
        print(f"{label}: threshold={grid[best]:.2f}, F1={scores[best]:.4f}")
    thresholds = np.asarray(thresholds)
    macro = f1_score(target, probability >= thresholds, average="macro", zero_division=0)
    return thresholds, float(macro)
