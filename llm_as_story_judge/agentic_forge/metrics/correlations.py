from typing import Sequence
import numpy as np
from scipy.stats import pearsonr as _pearsonr, spearmanr as _spearmanr, kendalltau as _kendalltau
import pingouin as pg
import pandas as pd


def pearson_corr(x: Sequence[float], y: Sequence[float]) -> float:
    """
    Pearson correlation coefficient.

    Description
    -----------
    Measures linear association between two numerical score vectors. Suitable for
    continuous or quasi-continuous ratings and widely used in evaluation tasks
    where you expect or want to detect linear consistency between judges or models.

    When to use
    -----------
    - Ratings on continuous scales (e.g., 1–10 quality scores).
    - When you want to quantify linear agreement between two judges/models.
    - When differences in magnitude matter.

    Inputs
    ------
    x, y: sequences of floats or ints of equal length.
        Pairs containing NaN are automatically removed.

    Interpretation
    --------------
    r ∈ [-1, 1]
    - +1   : perfect linear agreement
    - 0    : no linear relationship
    - -1   : perfect inverse linear relationship
    Values around ±0.7 or higher generally indicate strong linear consistency.

    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = ~np.isnan(x) & ~np.isnan(y)
    r, _ = _pearsonr(x[mask], y[mask])
    return float(r)


def spearman_corr(x: Sequence[float], y: Sequence[float]) -> float:
    """
    Spearman rank correlation coefficient.

    Description
    -----------
    Rank-based correlation measuring monotonic relationships. It evaluates how well
    the ordering of items is preserved between judges, irrespective of absolute values.

    When to use
    -----------
    - Ordinal ratings (Likert scales, discrete score levels).
    - Comparing judges with different scoring scales.
    - When ordering is more important than exact score magnitude.

    Inputs
    ------
    x, y: sequences of numeric values of equal length.

    Interpretation
    --------------
    ρ ∈ [-1, 1]
    - +1   : identical ranking
    - 0    : no monotonic association
    - -1   : perfectly reversed ranking
    Typically, ρ > 0.5 indicates meaningful ordinal agreement.

    Convenzione:
    - se entrambe le serie sono costanti e identiche -> 1.0
    - se una è costante e l'altra no, o costanti ma diverse -> NaN
    - altrimenti: spearman standard
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = ~np.isnan(x) & ~np.isnan(y)
    x = x[mask]
    y = y[mask]

    if len(x) == 0:
        return float("nan")

    x_const = np.all(x == x[0])
    y_const = np.all(y == y[0])

    if x_const and y_const:
        return 1.0 if np.all(x == y) else float("nan")
    if x_const or y_const:
        return float("nan")

    rho, _ = _spearmanr(x, y)
    return float(rho)



def kendall_corr(x: Sequence[float], y: Sequence[float]) -> float:
    """
    Kendall's tau correlation coefficient.

    Description
    -----------
    Non-parametric rank correlation based on concordant/discordant item pairs.
    More robust than Spearman for small samples and ties.

    When to use
    -----------
    - Ordinal comparison of judges.
    - Small datasets.
    - As a robust alternative to Spearman for ranking stability.

    Inputs
    ------
    x, y: sequences of numeric or ordinal values.

    Interpretation
    --------------
    τ ∈ [-1, 1]
    - +1   : perfect ordering agreement
    - 0    : half concordant, half discordant pairs (random)
    - -1   : completely reversed ordering

    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = ~np.isnan(x) & ~np.isnan(y)
    x = x[mask]
    y = y[mask]

    if len(x) == 0:
        return float("nan")

    x_const = np.all(x == x[0])
    y_const = np.all(y == y[0])
    if x_const and y_const:
        return 1.0 if np.all(x == y) else float("nan")
    if x_const or y_const:
        return float("nan")

    tau, _ = _kendalltau(x, y)
    return float(tau)


def icc(data: Sequence[Sequence[float]], icc_type: str = "ICC2") -> float:
    """
    Intraclass Correlation Coefficient (ICC).

    Description
    -----------
    Measures inter-rater reliability for two or more judges. ICC quantifies how
    much of the score variance is due to differences between items versus random
    noise or judge inconsistency. Supports multiple ICC formulations via Pingouin.

    When to use
    -----------
    - Evaluating consistency among 2+ judges or LLMs.
    - Measuring absolute agreement or consistency in multi-rater settings.
    - Psychometric-quality reliability assessment.

    Inputs
    ------
    data: Sequence[Sequence[float]]
        Matrix-like structure shaped (subjects × judges).
    icc_type: str
        One of "ICC1", "ICC2", "ICC3", "ICC1k", "ICC2k", "ICC3k".

    Interpretation
    --------------
    ICC ∈ [-1, 1]
    - < 0.0   : worse than chance, unreliable raters
    - 0.0–0.5 : poor reliability
    - 0.5–0.75: moderate
    - 0.75–0.9: good
    - > 0.9   : excellent reliability

    """
    data = np.asarray(data, dtype=float)
    n_subjects, n_judges = data.shape

    df = pd.DataFrame({
        "subject": np.repeat(np.arange(n_subjects), n_judges),
        "judge": np.tile(np.arange(n_judges), n_subjects),
        "rating": data.flatten(),
    })

    icc_df = pg.intraclass_corr(
        data=df, targets="subject", raters="judge", ratings="rating"
    )
    row = icc_df[icc_df["Type"] == icc_type].iloc[0]
    return float(row["ICC"])



if __name__=="__main__":
    from utils import explain
    
    def demo_correlations():
        print("\n=== CORRELATION METRICS (continuous or ordinal scores) ===")
        x = [1, 2, 3, 4, 5]
        y = [1.2, 2.1, 3.1, 3.9, 5.2]  # nearly linear

        p = pearson_corr(x, y)
        print(f"\nPearson Correlation: {p:.3f} {explain(p, 'pearson')}")
        print("Meaning: Measures linear consistency between two judges.")

        s = spearman_corr(x, y)
        print(f"\nSpearman Rank Correlation: {s:.3f} {explain(s, 'spearman')}")
        print("Meaning: Measures monotonic (rank) consistency.")

        k = kendall_corr(x, y)
        print(f"\nKendall Tau: {k:.3f} {explain(k, 'kendall')}")
        print("Meaning: Measures concordance/discordance of rankings.")

        ratings = [
            [3.2, 3.0, 3.1],
            [4.0, 4.1, 3.9],
            [2.5, 2.7, 2.6],
        ]
        icc_value = icc(ratings, icc_type="ICC2")
        print(f"\nICC(ICC2): {icc_value:.3f} {explain(icc_value, 'icc')}")
        print("Meaning: Measures multi-judge reliability over continuous scores.")

    demo_correlations()

