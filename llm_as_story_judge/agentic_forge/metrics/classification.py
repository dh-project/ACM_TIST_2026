from typing import Sequence, Any
from sklearn.metrics import (
    accuracy_score as _accuracy_score, precision_score as _precision_score, recall_score as _recall_score,
    f1_score as _f1_score, confusion_matrix as _confusion_matrix, matthews_corrcoef as _matthews_corrcoef
)


def accuracy(y_true: Sequence[Any], y_pred: Sequence[Any]) -> float:
    """
    Accuracy score.

    Description
    -----------
    Fraction of predictions matching the true labels. Simple and intuitive,
    but sensitive to class imbalance.

    When to use
    -----------
    - Balanced label distributions
    - Quick baseline metric for judge comparison

    Inputs
    ------
    y_true, y_pred: sequences of categorical labels.

    Interpretation
    --------------
    [0, 1]
    - 1.0: perfect agreement
    - ~0.5: random (binary)
    - <0.5: worse than chance

    """
    return float(_accuracy_score(y_true, y_pred))


def precision_macro(y_true, y_pred, **kwargs):
    """
    Macro-averaged precision.

    Description
    -----------
    Measures how often predicted labels are correct, averaged equally across classes.
    Penalizes false positives.

    When to use
    -----------
    - Multi-class judge evaluations
    - Avoiding overprediction of certain labels

    Interpretation
    --------------
    [0, 1]
    - High = few false positives

    """
    return float(_precision_score(y_true, y_pred, average="macro", **kwargs))


def recall_macro(y_true, y_pred, **kwargs):
    """
    Macro-averaged recall.

    Description
    -----------
    Measures judge sensitivity across classes, averaged equally.
    Penalizes false negatives.

    When to use
    -----------
    - Avoid missing certain labels
    - Balanced per-class detection

    Interpretation
    --------------
    [0, 1]
    - High = few false negatives

    """
    return float(_recall_score(y_true, y_pred, average="macro", **kwargs))


def f1_macro(y_true, y_pred,**kwargs):
    """
    Macro-averaged F1 score.

    Description
    -----------
    Harmonic mean of macro precision and recall. Good for class-imbalanced settings.

    When to use
    -----------
    - Multi-class tasks with imbalance
    - Balanced combination of precision and recall

    Interpretation
    --------------
    [0, 1]
    - High = both precision and recall are strong

    """
    return float(_f1_score(y_true, y_pred, average="macro",**kwargs))


def mcc(y_true, y_pred):
    """
    Matthews Correlation Coefficient (MCC).

    Description
    -----------
    Balanced and reliable classification metric. Considers all parts of the
    confusion matrix and remains stable even with severe class imbalance.

    When to use
    -----------
    - Binary judge evaluations
    - Highly imbalanced class distributions
    - Stronger alternative to accuracy

    Inputs
    ------
    y_true, y_pred: sequences of labels.

    Interpretation
    --------------
    MCC ∈ [-1, 1]
    - 1   : perfect prediction
    - 0   : random prediction
    - -1  : perfectly wrong (inverse labeling)

    """
    return float(_matthews_corrcoef(y_true, y_pred))


def confmat(y_true, y_pred):
    """
    Confusion matrix.

    Description
    -----------
    Tabulates true labels vs predicted labels for diagnostic analysis.
    Useful to inspect systematic judge bias and error patterns.

    Interpretation
    --------------
    Array where:
    - Rows   = true labels
    - Columns= predicted labels
    High diagonal dominance indicates strong correctness.

    """
    return _confusion_matrix(y_true, y_pred)


if __name__=="__main__":
    
    def demo_classification():
        print("\n=== CLASSIFICATION METRICS (discrete labels) ===")
        y_true = ["A", "B", "A", "C", "B"]
        y_pred = ["A", "A", "A", "C", "B"]

        acc = accuracy(y_true, y_pred)
        print(f"\nAccuracy: {acc:.3f} → {acc*100:.1f}% correct predictions.")

        pre = precision_macro(y_true, y_pred)
        print(f"\nPrecision (macro): {pre:.3f} → Focus on false positives.")

        rec = recall_macro(y_true, y_pred)
        print(f"Recall (macro): {rec:.3f} → Focus on false negatives.")

        f1 = f1_macro(y_true, y_pred)
        print(f"F1 (macro): {f1:.3f} → Balanced between precision & recall.")

        m = mcc(y_true, y_pred)
        print(f"\nMCC: {m:.3f} → Robust correlation-like score for classification.")

        print("\nConfusion Matrix:")
        print(confmat(y_true, y_pred))

    demo_classification()