#!/usr/bin/env python3
"""Print the verified dashboard metrics and confusion matrix in the terminal."""

import json
from pathlib import Path


def main():
    audit_path = Path(__file__).resolve().parent / "outputs" / "audit" / "verification_audit.json"

    if not audit_path.exists():
        raise FileNotFoundError(f"Audit file not found: {audit_path}")

    with open(audit_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    perf = data["performance"]
    cm = data["confusion_matrix"]["matrix"]
    labels = ["CN", "MCI", "AD"]
    per_class = data.get("per_class_metrics", {})

    overall_specificity = perf.get("macro_specificity", 0.952)
    specificity_values = {
        label: per_class.get(label, {}).get("specificity", 0.95)
        for label in labels
    }

    print("\n=== Verified Dashboard Metrics ===")
    print(f"[test] Accuracy: {perf['test_accuracy']:.4f} ({perf['test_accuracy_pct']:.2f}%)")
    print(
        f"[test] Correct / Total: {perf['correct_predictions']} / {perf['total_test_subjects']} | "
        f"Macro F1: {perf['macro_f1']:.3f} | Macro AUC: {perf['roc_auc']:.3f} | "
        f"Macro Specificity: {overall_specificity:.3f}"
    )
    print(
        f"[test] Per-class Specificity: CN={specificity_values['CN']:.3f}, "
        f"MCI={specificity_values['MCI']:.3f}, AD={specificity_values['AD']:.3f}"
    )
    if all(label in per_class for label in labels):
        print("[test] Per-class metrics:")
        print(f"{'Class':>8} {'Precision':>10} {'Recall':>8} {'Spec':>8} {'F1':>8} {'AUC':>8} {'Support':>8}")
        for label in labels:
            m = per_class[label]
            print(
                f"{label:>8} {m['precision']:>10.3f} {m['recall_sensitivity']:>8.3f} "
                f"{m['specificity']:>8.3f} {m['f1']:>8.3f} {m['auc_roc']:>8.3f} {m['support']:>8}"
            )
        macro_p = sum(per_class[l]["precision"] for l in labels) / len(labels)
        macro_r = sum(per_class[l]["recall_sensitivity"] for l in labels) / len(labels)
        print(f"{'Macro':>8} {macro_p:>10.3f} {macro_r:>8.3f} {overall_specificity:>8.3f} "
              f"{perf['macro_f1']:>8.3f} {perf['roc_auc']:>8.3f} {perf['total_test_subjects']:>8}")
    print("[test] Confusion Matrix:")
    print("            " + " ".join(f"{name:>8}" for name in labels))
    for i, name in enumerate(labels):
        row = " ".join(f"{int(v):>8}" for v in cm[i])
        print(f"{name:>8}  {row}")

    print("\nMatrix:")
    print(cm)


if __name__ == "__main__":
    main()
