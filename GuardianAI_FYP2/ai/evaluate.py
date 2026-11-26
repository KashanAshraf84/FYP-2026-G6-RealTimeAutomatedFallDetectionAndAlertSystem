"""Evaluate labelled predictions, without ML dependencies.
python -m ai.evaluate predictions.csv --output metrics.json
Required columns: clip_id, subject_id, split, y_true, y_pred
One row per independent clip/sequence; test split only.
"""

import argparse, csv, json

LABELS = ["normal", "warning", "fall"]


def evaluate(rows):
    matrix = [[0] * 3 for _ in range(3)]
    seen = set()
    for row in rows:
        if row.get("split") != "test":
            raise ValueError("Only held-out test rows may be evaluated")
        if not row.get("clip_id") or not row.get("subject_id"):
            raise ValueError("clip_id and subject_id are required")
        key = (row["clip_id"], row["subject_id"])
        if key in seen:
            raise ValueError("Duplicate clip/subject row")
        seen.add(key)
        if row["y_true"] not in LABELS or row["y_pred"] not in LABELS:
            raise ValueError("Unknown label")
        matrix[LABELS.index(row["y_true"])][LABELS.index(row["y_pred"])] += 1
    n = sum(map(sum, matrix))
    if not n:
        raise ValueError("No predictions provided")
    metrics = {}
    for i, label in enumerate(LABELS):
        tp = matrix[i][i]
        actual = sum(matrix[i])
        predicted = sum(row[i] for row in matrix)
        precision = tp / predicted if predicted else None
        recall = tp / actual if actual else None
        f1 = 2 * tp / (actual + predicted) if actual + predicted else None
        metrics[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": actual,
        }
    return {
        "samples": n,
        "accuracy": sum(matrix[i][i] for i in range(3)) / n,
        "labels": LABELS,
        "confusion_matrix_true_rows_predicted_columns": matrix,
        "per_class": metrics,
        "limitations": "Sequence-level metrics only. Evaluate event recall, false alarms per hour and detection latency separately. This tool cannot verify train/test subject separation.",
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("csv")
    p.add_argument("--output", default="metrics.json")
    args = p.parse_args()
    try:
        with open(args.csv, newline="", encoding="utf-8-sig") as f:
            result = evaluate(csv.DictReader(f))
    except (ValueError, KeyError) as exc:
        p.error(str(exc))
    from pathlib import Path

    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
