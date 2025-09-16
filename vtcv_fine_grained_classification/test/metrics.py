import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import precision_score, recall_score, f1_score


def normalize_confusion_matrix(y_true, y_pred, exp_path, norm="true", class_names=None):
    """
    Normalize and visualize a confusion matrix.
    """
    cm = confusion_matrix(y_true, y_pred)
    if norm == "true":
        cm_normalized = cm.astype("float") / cm.sum(axis=1, keepdims=True)
    elif norm == "pred":
        cm_normalized = cm.astype("float") / cm.sum(axis=0, keepdims=True)
    elif norm == "all":
        cm_normalized = cm.astype("float") / cm.sum()
    else:
        raise ValueError("Invalid normalization type. Use 'true', 'pred', or 'all'.")
    cm_normalized = np.nan_to_num(cm_normalized)

    plt.figure(figsize=(12, 10))
    # Do not specify style or color explicitly, to follow your style guidelines
    plt.imshow(cm_normalized, interpolation="nearest", cmap="Blues")
    plt.title(f"Normalized Confusion Matrix ({norm.capitalize()} Normalization)")
    plt.colorbar()
    if class_names is not None:
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=45)
        plt.yticks(tick_marks, class_names)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(f"{exp_path}/test_predictions/confusion_matrix.png")
    plt.close()


# ----------------------------------------------------------------------------------------
# Add your modified version of calculate_metrics WITH standard deviations and CIs
# ----------------------------------------------------------------------------------------
def calculate_metrics_with_confidence(y_true, y_pred, class_names):
    """
    Computes per-class precision, recall, F1, and support. Additionally computes:
      - precision_std, recall_std, f1_std using normal approximations
      - 95% confidence intervals for each metric
    Returns:
      metrics_df (pd.DataFrame): One row per class + columns for metrics, std, and CI
      macro_avg, micro_avg: dicts for macro/micro aggregates
    """

    # First, get standard classification report as baseline
    base_report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)

    # Build confusion matrix to extract tp, fp, fn
    # cm[i, i] = tp for class i
    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))

    rows = []
    for i, class_name in enumerate(class_names):
        # Extract from classification_report for convenience
        precision = base_report[class_name]["precision"]
        recall = base_report[class_name]["recall"]
        f1 = base_report[class_name]["f1-score"]
        support = base_report[class_name]["support"]

        # Extract confusion matrix counts
        tp = cm[i, i]
        fn = np.sum(cm[i, :]) - tp    # items in row i excluding diagonal
        fp = np.sum(cm[:, i]) - tp    # items in col i excluding diagonal

        # ---------------
        # Standard deviations using normal approximation
        # ---------------
        # For proportion p, std ~ sqrt( p*(1 - p) / n )
        #   - recall = tp / (tp + fn) -> n = tp + fn
        #   - precision = tp / (tp + fp) -> n = tp + fp
        #   - F1 = 2pr/(p + r) => use partial derivative approach (delta method).
        #     dF/dp = 2*r^2 / (p + r)^2
        #     dF/dr = 2*p^2 / (p + r)^2

        z = 1.96  # for ~95% confidence intervals

        # Safeguard if support=0 or if tp+fn=0 or tp+fp=0
        if (tp + fn) == 0:
            recall_std = 0.0
        else:
            recall_std = math.sqrt(recall * (1.0 - recall) / (tp + fn))

        if (tp + fp) == 0:
            precision_std = 0.0
        else:
            precision_std = math.sqrt(precision * (1.0 - precision) / (tp + fp))

        # Delta method for F1
        # partial_F/partial_p = 2*r^2 / (p + r)^2
        # partial_F/partial_r = 2*p^2 / (p + r)^2
        # var(F1) ~ (dF/dp)^2 var(p) + (dF/dr)^2 var(r)
        if (precision + recall) > 0:
            dFdp = 2.0 * (recall**2) / (precision + recall) ** 2
            dFdr = 2.0 * (precision**2) / (precision + recall) ** 2
            var_f1 = (dFdp**2) * (precision_std**2) + (dFdr**2) * (recall_std**2)
            f1_std = math.sqrt(var_f1)
        else:
            f1_std = 0.0

        # ---------------
        # Confidence intervals
        # ---------------
        # metric_CI = metric ± z * metric_std
        prec_lower = max(0.0, precision - z * precision_std)
        prec_upper = min(1.0, precision + z * precision_std)

        rec_lower = max(0.0, recall - z * recall_std)
        rec_upper = min(1.0, recall + z * recall_std)

        f1_lower = max(0.0, f1 - z * f1_std)
        f1_upper = min(1.0, f1 + z * f1_std)

        rows.append([
            class_name,
            precision,
            precision_std,
            prec_lower,
            prec_upper,
            recall,
            recall_std,
            rec_lower,
            rec_upper,
            f1,
            f1_std,
            f1_lower,
            f1_upper,
            support
        ])

    # Create DataFrame
    columns = [
        "class",
        "precision",
        "precision_std",
        "precision_CI_lower",
        "precision_CI_upper",
        "recall",
        "recall_std",
        "recall_CI_lower",
        "recall_CI_upper",
        "f1-score",
        "f1_std",
        "f1_CI_lower",
        "f1_CI_upper",
        "support"
    ]
    metrics_df = pd.DataFrame(rows, columns=columns)

    # Macro & micro from base_report
    macro_avg_report = base_report["macro avg"]
    macro_avg = {
        "precision": macro_avg_report["precision"],
        "recall": macro_avg_report["recall"],
        "f1-score": macro_avg_report["f1-score"],
        "support": macro_avg_report["support"]
    }

    micro_precision = precision_score(y_true, y_pred, average="micro")
    micro_recall = recall_score(y_true, y_pred, average="micro")
    micro_f1 = f1_score(y_true, y_pred, average="micro")
    total_support = len(y_true)

    micro_avg = {
        "precision": micro_precision,
        "recall": micro_recall,
        "f1-score": micro_f1,
        "support": total_support
    }

    return metrics_df, macro_avg, micro_avg
