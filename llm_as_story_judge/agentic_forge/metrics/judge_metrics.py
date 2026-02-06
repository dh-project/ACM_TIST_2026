from typing import Dict, List, Literal, Any
import numpy as np

# Agreement metrics
from .agreement import cohen_kappa, krippendorff_alpha

# Correlation (ordinal consistency)
from .correlations import spearman_corr

# Classification-style metrics
from .classification import (
    accuracy,
    f1_macro,
    precision_macro,
    recall_macro,
    mcc,
    confmat,
)

FieldType = Literal["nominal", "ordinal"]


def compute_pairwise_metrics(
    judge_a: Dict[tuple[str,str], List[Any]],
    judge_b: Dict[tuple[str,str], List[Any]],
) -> Dict[str, Dict[str, float | None]]:
    """
    Compute pairwise agreement metrics between two label sources (A ↔ B).

    Input
    -----
    judge_a, judge_b:
        Dict[(field_name, field_type) -> list di valori]
        dove field_type ∈ {"nominal", "ordinal"}.

    Output
    ------
    Dict[field_name -> {"kappa": float, "spearman": float|None}]
    """
    results: Dict[str, Dict[str, float | None]] = {}

    for (field_name, ftype), a_values in judge_a.items():
        # assume che judge_b abbia esattamente le stesse chiavi
        if (field_name, ftype) not in judge_b:
            # se vuoi essere paranoico puoi alzare un errore
            # raise KeyError(f"Field {(field_name, ftype)} missing in judge_b")
            continue

        b_values = judge_b[(field_name, ftype)]

        # Cohen's kappa sempre (nominal / ordinal)
        kappa = cohen_kappa(a_values, b_values)

        # Spearman solo per ordinal
        if ftype == "ordinal":
            rho = spearman_corr(a_values, b_values)
        else:
            rho = None

        results[field_name] = {
            "kappa": float(kappa),
            "spearman": None if rho is None else float(rho),
        }

    return results


# ============================================================
# MULTI-JUDGE RELIABILITY (Krippendorff α)
# ============================================================
def compute_multi_judge_alpha(
    judge_matrix: Dict[str, List[List[Any]]],
    field_types: Dict[str, FieldType],
):
    """
    Compute Krippendorff's alpha for multiple label sources on each field.

    What α tells us
    ---------------
    α = *group reliability* among all judges (A + B + C + ...)

    Importantly:
        α measures reliability of the whole panel of judges
        It does NOT measure how close B is to A.

    Inputs
    ------
    judge_matrix : dict[field -> matrix (units × judges)]
    field_types  : dict[field -> nominal/ordinal]

    Returns
    -------
    Dict[field -> α value]
    """
    alpha_results = {}

    for field, matrix in judge_matrix.items():
        level = field_types[field]
        alpha_val = krippendorff_alpha(matrix, level=level)
        alpha_results[field] = float(alpha_val)

    return alpha_results


# ============================================================
# GLOBAL AGREEMENT SCORE (Macro κ)
# ============================================================
def compute_macro_kappa(
    pairwise_results: Dict[str, Dict[str, float]] | None = None,
    judgeA: Dict[str, list] | None = None,
    judgeB: Dict[str, list] | None = None,
    field_types: Dict[str, FieldType] | None = None,
) -> float:
    """
    Compute macro-average Cohen's κ.

    Parameters
    ----------
    pairwise_results : dict | None
        Precomputed per-field metrics from compute_pairwise_metrics().
        If provided, judgeA/judgeB/field_types MUST NOT be passed.

    judgeA : dict | None
        Mapping field -> values from judge A.
        Used **only if pairwise_results is None**.

    judgeB : dict | None
        Mapping field -> values from judge B.
        Used **only if pairwise_results is None**.

    field_types : dict | None
        Mapping field -> FieldType ("nominal" | "ordinal").
        Used **only if pairwise_results is None**.

    Returns
    -------
    float
        Macro-average Cohen's κ across fields.
    """

    # ---- Validation: prevent ambiguous API usage ----
    if pairwise_results is not None:
        if judgeA or judgeB or field_types:
            raise ValueError(
                "Provide EITHER pairwise_results OR (judgeA, judgeB, field_types), not both."
            )
        pr = pairwise_results

    else:
        # Must calculate pairwise
        if judgeA is None or judgeB is None or field_types is None:
            raise ValueError(
                "To compute pairwise results, you must pass judgeA, judgeB, and field_types."
            )

        from agentic_forge.metrics.judge_metrics import compute_pairwise_metrics
        pr = compute_pairwise_metrics(judgeA, judgeB, field_types)

    # ---- Aggregate ----
    kappas = [v["kappa"] for v in pr.values()]
    return float(sum(kappas) / len(kappas))



def compute_macro_spearman(
    pairwise_results: Dict[str, Dict[str, float]] | None = None,
    judgeA: Dict[str, list] | None = None,
    judgeB: Dict[str, list] | None = None,
    field_types: Dict[str, FieldType] | None = None,
) -> float:
    """
    Compute macro-average Spearman ρ (only ordinal fields).

    Same calling logic as compute_macro_kappa():

    - If `pairwise_results` is provided → use it directly.
    - If None → compute pairwise from judgeA/judgeB/field_types.
    - If both provided → error.

    Returns
    -------
    float
        Macro-average ρ across ordinal fields.
    """

    # ---- Validation ----
    if pairwise_results is not None:
        if judgeA or judgeB or field_types:
            raise ValueError(
                "Provide EITHER pairwise_results OR (judgeA, judgeB, field_types), not both."
            )
        pr = pairwise_results

    else:
        if judgeA is None or judgeB is None or field_types is None:
            raise ValueError(
                "To compute Spearman, pass either pairwise_results OR judgeA/judgeB/field_types."
            )

        from agentic_forge.metrics.judge_metrics import compute_pairwise_metrics
        pr = compute_pairwise_metrics(judgeA, judgeB, field_types)

    # ---- Aggregate only valid Spearman ----
    rhos = [v["spearman"] for v in pr.values() if v["spearman"] is not None]

    if not rhos:
        raise ValueError("No ordinal fields with defined Spearman correlation.")

    return float(sum(rhos) / len(rhos))




# ============================================================
# OPTIONAL: CLASSIFICATION-STYLE COMPARISON
# ============================================================
def compute_classification_metrics(judge_a: List[Any], judge_b: List[Any]):
    """
    Compute classification-style comparison metrics between two judges.

    These metrics treat Judge A as “ground truth” and Judge B as a classifier.

    What they measure
    ------------------
    - accuracy       : % matching labels
    - precision      : per-class correctness (macro)
    - recall         : per-class coverage (macro)
    - f1_macro       : balanced harmonic score
    - mcc            : correlation-like measure (binary / multi-class)
    - confusion matrix

    Note
    ----
    For classes with no predicted samples, precision/recall/F1 are set to 0.0
    (zero_division=0) and no warnings are raised. This is expected behavior
    on small or imbalanced label sets.

    Returns
    -------
    Dict[str, float or ndarray]
    """
    return {
        "accuracy": float(accuracy(judge_a, judge_b)),
        "precision_macro": float(precision_macro(judge_a, judge_b, zero_division=0)),
        "recall_macro": float(recall_macro(judge_a, judge_b, zero_division=0)),
        "f1_macro": float(f1_macro(judge_a, judge_b, zero_division=0)),
        "mcc": float(mcc(judge_a, judge_b)),
        "confusion_matrix": confmat(judge_a, judge_b),
    }



# ============================================================
# DEMO / MAIN
# ============================================================
if __name__ == "__main__":
    print("\n=== PAIRWISE AGREEMENT (κ + ρ) ===")
    print("Evaluating how closely Judge B reproduces Judge A.\n")

    judge_a = {
        "storytelling": [1, 2, 3],
        "inciting_present": [True, False, True],
        "scene": [2, 5, 3],
    }

    judge_b = {
        "storytelling": [1, 3, 3],
        "inciting_present": [True, True, True],
        "scene": [2, 4, 3],
    }

    field_types = {
        "storytelling": "ordinal",
        "inciting_present": "nominal",
        "scene": "ordinal",
    }

    pair = compute_pairwise_metrics(judge_a, judge_b, field_types)
    for f, vals in pair.items():
        print(f" - {f}: κ={vals['kappa']:.3f}, ρ={vals['spearman']}")

    print("\n=== MULTI-JUDGE RELIABILITY (α) ===")
    print("Evaluating consistency among ALL judges.\n")

    judge_matrix = {
        "storytelling": [
            [1, 1, 2],
            [2, 3, 3],
            [3, 3, 3],
        ],
        "inciting_present": [
            [True, True, True],
            [False, True, False],
            [True, True, True],
        ],
        "scene": [
            [2, 2, 3],
            [5, 4, 5],
            [3, 3, 3],
        ],
    }

    alpha = compute_multi_judge_alpha(judge_matrix, field_types)
    for f, val in alpha.items():
        print(f" - {f}: α={val:.3f}")

    print("\n=== FINAL OVERALL AGREEMENT (Macro κ) ===\n")
    macro = compute_macro_kappa(pair)
    print("Macro κ Score:", round(macro, 3))

    print("\n=== GLOBAL ORDINAL CONSISTENCY (Macro ρ) ===\n")
    macro_rho = compute_macro_spearman(pair)
    print("Macro Spearman ρ:", round(macro_rho, 3))

    print("\n=== OPTIONAL: CLASSIFICATION METRICS ===\n")
    clf = compute_classification_metrics(judge_a["storytelling"], judge_b["storytelling"])
    for k, v in clf.items():
        print(f"{k}: {v}")
