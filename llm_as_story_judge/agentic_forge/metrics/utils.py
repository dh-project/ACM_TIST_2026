def explain(value: float, name: str):
    """Quick qualitative interpretation."""
    if name in ["pearson", "spearman", "kendall", "icc"]:
        if value > 0.75: note = "→ strong agreement"
        elif value > 0.5: note = "→ moderate agreement"
        elif value > 0.3: note = "→ weak agreement"
        else: note = "→ very weak / none"
    elif name in ["cohen", "fleiss", "alpha"]:
        if value > 0.8: note = "→ excellent"
        elif value > 0.6: note = "→ substantial"
        elif value > 0.4: note = "→ moderate"
        else: note = "→ poor"
    else:
        note = ""
    return note