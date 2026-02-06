from __future__ import annotations
from pathlib import Path
import json
import csv


def combine_runs(runs_dir: Path, out_csv: Path, out_meta: Path):
    """
    Combina tutte le run sotto runs_dir.
    Aggiunge automaticamente la colonna 'judge' usando il nome della folder.
    """

    if not runs_dir.exists():
        raise ValueError(f"runs-dir does not exist: {runs_dir}")

    run_folders = [d for d in runs_dir.iterdir() if d.is_dir()]

    combined_rows = []
    combined_meta = {}

    for run_dir in run_folders:

        judge_name = run_dir.name  # <-- QUI la magia

        csv_path = run_dir / "out.csv"
        meta_path = run_dir / "fields_column_meta.json"

        if not csv_path.exists():
            print(f"[SKIP] No out.csv in {run_dir}")
            continue

        # ---- Load rows from this run ----
        with csv_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        # Inject judge name in every row
        for r in rows:
            r["judge"] = judge_name

        combined_rows.extend(rows)

        # ---- Merge metadata ----
        if meta_path.exists():
            with meta_path.open("r", encoding="utf-8") as f:
                meta = json.load(f)
            combined_meta.update(meta)

    # -----------------------------------------------------
    # WRITE COMBINED CSV
    # -----------------------------------------------------
    all_cols = set()
    for r in combined_rows:
        all_cols.update(r.keys())

    # Make judge & id first
    all_cols = ["id", "judge"] + sorted([c for c in all_cols if c not in ("id", "judge")])

    out_csv.parent.mkdir(parents=True, exist_ok=True)

    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_cols)
        writer.writeheader()
        for r in combined_rows:
            writer.writerow({c: r.get(c, "") for c in all_cols})

    # -----------------------------------------------------
    # WRITE COMBINED META
    # -----------------------------------------------------
    out_meta.parent.mkdir(parents=True, exist_ok=True)
    with out_meta.open("w", encoding="utf-8") as f:
        json.dump(combined_meta, f, indent=2, ensure_ascii=False)

    print(f"[DONE] Combined CSV  → {out_csv}")
    print(f"[DONE] Combined META → {out_meta}")
