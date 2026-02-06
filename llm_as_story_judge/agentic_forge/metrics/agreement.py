from typing import Sequence, Any,List
import numpy as np
from sklearn.metrics import cohen_kappa_score as _cohen_kappa_score
from statsmodels.stats.inter_rater import fleiss_kappa as _fleiss_kappa
import pingouin as pg
import krippendorff as _krippendorff



def cohen_kappa(y1: Sequence[Any], y2: Sequence[Any]) -> float:
    """
    Cohen's Kappa.

    Description
    -----------
    Chance-corrected agreement measure for two raters. More informative than
    accuracy because it accounts for the expected agreement due to class frequencies.

    When to use
    -----------
    - Two judges or LLMs producing categorical labels.
    - Class-imbalanced datasets.

    Inputs
    ------
    y1, y2: sequences of categorical labels of equal length.

    Interpretation
    --------------
    κ ∈ [-1, 1]
    - <0       : worse than chance
    - 0–0.40   : slight to fair agreement
    - 0.41–0.60: moderate
    - 0.61–0.80: substantial
    - >0.80    : near-perfect agreement

    Convenzione locale:
    - se entrambe le serie sono costanti e identiche  -> κ = 1.0
    - se entrambe costanti ma su label diverse        -> κ = -1.0
    - altrimenti: κ standard sklearn
    """
    y1 = list(y1)
    y2 = list(y2)

    if len(y1) == 0 or len(y2) == 0:
        return float("nan")

    labels1 = set(y1)
    labels2 = set(y2)

    if len(labels1) == 1 and len(labels2) == 1:
        # entrambe costanti
        v1 = next(iter(labels1))
        v2 = next(iter(labels2))
        if v1 == v2:
            return 1.0
        else:
            # disaccordo sistematico assoluto
            return -1.0

    # tutti gli altri casi -> definizione standard
    return float(_cohen_kappa_score(y1, y2))


def fleiss_kappa_counts(counts: Sequence[Sequence[int]]) -> float:
    """
    Fleiss' Kappa.

    Description
    -----------
    Chance-corrected agreement for multiple raters. Generalization of Cohen’s κ
    using rater counts per category for each item.

    When to use
    -----------
    - Three or more judges/LLMs.
    - Label counts are available instead of individual labels.

    Inputs
    ------
    counts: matrix (items × categories), each cell = number of raters choosing the category.

    Interpretation
    --------------
    κ ∈ [-1, 1] (same scale as Cohen's κ)
    Higher values indicate stronger collective agreement across judges.
    """
    arr = np.asarray(counts, dtype=int)
    return float(_fleiss_kappa(arr))


def krippendorff_alpha(data: List[List[Any]], level: str = "nominal") -> float:
    """
    Robust Krippendorff's Alpha that supports:
    - nominal data (strings or booleans)
    - ordinal data (converted to ordered integers)
    - missing values as None

    The function normalizes all inputs to appropriate numeric/string domains.

    Parameters
    ----------
    data : list[list[Any]]
        Matrix (units × judges). Use None for missing values.
    level : str
        "nominal" or "ordinal".

    Returns
    -------
    float
        Krippendorff's alpha.
    """
    # NOMINAL: everything as string, missing = "nan"
    if level == "nominal":
        norm = []
        for row in data:
            norm.append([
                "nan" if v is None else str(v)
                for v in row
            ])
        arr = np.array(norm, dtype=str)
        return float(_krippendorff.alpha(arr, level_of_measurement="nominal"))

    # ORDINAL: map distinct values to ordered integers, missing = np.nan
    elif level == "ordinal":
        # flatten ignoring None
        flat = [v for row in data for v in row if v is not None]
        # sorted unique values define the ordinal domain
        unique = sorted(set(flat))
        mapping = {val: i for i, val in enumerate(unique)}

        mapped = []
        for row in data:
            mapped.append([
                np.nan if v is None else float(mapping[v])
                for v in row
            ])

        mapped_arr = np.array(mapped, dtype=float)
        return float(_krippendorff.alpha(mapped_arr, level_of_measurement="ordinal"))

    else:
        raise ValueError(f"Unsupported level: {level}")




if __name__=="__main__":
    from utils import explain
    def demo_agreement():
        print("\n=== AGREEMENT METRICS (categorical inter-rater reliability) ===")

        y1 = ["pos", "neg", "pos", "pos", "neg"]
        y2 = ["pos", "neg", "neg", "pos", "neg"]
        ck = cohen_kappa(y1, y2)
        print(f"\nCohen's Kappa: {ck:.3f} {explain(ck, 'cohen')}")
        print("Meaning: Agreement between 2 judges beyond chance.")

        counts = [
            [2, 1, 0],
            [0, 3, 0],
            [1, 1, 1],
        ]
        fk = fleiss_kappa_counts(counts)
        print(f"\nFleiss' Kappa: {fk:.3f} {explain(fk, 'fleiss')}")
        print("Meaning: Agreement among multiple judges.")

        data = [
            ["A", "A", "A"],
            ["B", "B", None],
            ["A", "C", "C"],
        ]
        alpha_val = krippendorff_alpha(data, level="nominal")
        print(f"\nKrippendorff Alpha: {alpha_val:.3f} {explain(alpha_val, 'alpha')}")
        print("Meaning: General reliability, supports missing data.")

    demo_agreement()