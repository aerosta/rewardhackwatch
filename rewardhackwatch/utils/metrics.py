"""Metrics calculation utilities."""


def precision(tp: int, fp: int) -> float:
    """Calculate precision."""
    if tp + fp == 0:
        return 0.0
    return tp / (tp + fp)


def recall(tp: int, fn: int) -> float:
    """Calculate recall."""
    if tp + fn == 0:
        return 0.0
    return tp / (tp + fn)


def f1_score(precision_val: float, recall_val: float) -> float:
    """Calculate F1 score."""
    if precision_val + recall_val == 0:
        return 0.0
    return 2 * (precision_val * recall_val) / (precision_val + recall_val)


def accuracy(tp: int, tn: int, fp: int, fn: int) -> float:
    """Calculate accuracy."""
    total = tp + tn + fp + fn
    if total == 0:
        return 0.0
    return (tp + tn) / total


def confusion_matrix_from_predictions(
    predictions: list[bool], labels: list[bool]
) -> dict[str, int]:
    """Calculate confusion matrix from predictions."""
    tp = sum(1 for p, l in zip(predictions, labels) if p and l)
    tn = sum(1 for p, l in zip(predictions, labels) if not p and not l)
    fp = sum(1 for p, l in zip(predictions, labels) if p and not l)
    fn = sum(1 for p, l in zip(predictions, labels) if not p and l)
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}


def calculate_all_metrics(predictions: list[bool], labels: list[bool]) -> dict[str, float]:
    """Calculate all standard metrics."""
    cm = confusion_matrix_from_predictions(predictions, labels)
    prec = precision(cm["tp"], cm["fp"])
    rec = recall(cm["tp"], cm["fn"])

    return {
        "precision": prec,
        "recall": rec,
        "f1": f1_score(prec, rec),
        "accuracy": accuracy(cm["tp"], cm["tn"], cm["fp"], cm["fn"]),
        **cm,
    }


def auc_roc_approx(scores: list[float], labels: list[bool]) -> float:
    """Approximate AUC-ROC calculation."""
    if not scores or not labels:
        return 0.5

    # Simple trapezoidal approximation
    pairs = list(zip(scores, labels))
    pairs.sort(key=lambda x: -x[0])  # Sort by score descending

    total_pos = sum(labels)
    total_neg = len(labels) - total_pos

    if total_pos == 0 or total_neg == 0:
        return 0.5

    auc = 0.0
    tp = 0

    for score, label in pairs:
        if label:
            tp += 1
        else:
            auc += tp

    return auc / (total_pos * total_neg)
