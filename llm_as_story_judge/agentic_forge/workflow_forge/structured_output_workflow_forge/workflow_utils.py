from __future__ import annotations
import json
import re
import socket
import getpass
import subprocess
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List, Mapping

import csv

# -------------------------------------------------------------------------
# Git utilities
# -------------------------------------------------------------------------

def get_git_commit_or_none() -> Optional[str]:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
        )
        return out.decode("utf-8").strip()
    except Exception:
        return None


# -------------------------------------------------------------------------
# RunConfig + salvataggio
# -------------------------------------------------------------------------

@dataclass
class RunConfig:
    exp_name: str
    run_name: str
    model_key: str
    model_name: str

    dataset_path: str
    schemas_path: str
    prompts_path: str

    batch_size: int
    max_retries: int
    reasoning_enabled: bool
    verbose_llm: bool
    base_output_dir: str

    created_at: str
    user: str
    host: str
    git_commit: Optional[str] = None


def build_run_config(
    exp_name: str,
    run_name: str,
    model_key: str,
    model_name: str,
    dataset_path: Path,
    schemas_path: Path,
    prompts_path: Path,
    batch_size: int,
    max_retries: int,
    reasoning_enabled: bool,
    verbose_llm: bool,
    base_output_dir: Path,
) -> RunConfig:

    return RunConfig(
        exp_name=exp_name,
        run_name=run_name,
        model_key=model_key,
        model_name=model_name,
        dataset_path=str(dataset_path),
        schemas_path=str(schemas_path),
        prompts_path=str(prompts_path),
        batch_size=batch_size,
        max_retries=max_retries,
        reasoning_enabled=reasoning_enabled,
        verbose_llm=verbose_llm,
        base_output_dir=str(base_output_dir.resolve()),
        created_at=datetime.utcnow().isoformat(),
        user=getpass.getuser(),
        host=socket.gethostname(),
        git_commit=get_git_commit_or_none(),
    )


def save_run_config(cfg: RunConfig, run_dir: Path) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = run_dir / "run_config.json"
    with cfg_path.open("w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, ensure_ascii=False, indent=2)
    return cfg_path


# -------------------------------------------------------------------------
# Run directory creation + path management
# -------------------------------------------------------------------------

def resolve_run_dir(
    base_output_dir: Path,
    exp_name: str,
    model_key: str,
    run_name: Optional[str],
) -> tuple[Path, str]:
    """
    Determina run_dir evitando collisioni:
    <base_output_dir>/<exp_name>/<model_key>_<runX>/
    """
    exp_dir = base_output_dir / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    if run_name:
        final_run_name = run_name
    else:
        # auto generazione run0, run1, run2...
        pattern = re.compile(rf"{re.escape(model_key)}_run(\d+)$")
        max_idx = -1

        for d in exp_dir.iterdir():
            if d.is_dir():
                m = pattern.match(d.name)
                if m:
                    try:
                        idx = int(m.group(1))
                        max_idx = max(max_idx, idx)
                    except ValueError:
                        pass

        final_run_name = f"run{max_idx + 1}"

    run_dir = exp_dir / f"{model_key}_{final_run_name}"
    run_dir.mkdir(parents=True, exist_ok=True)

    return run_dir, final_run_name


def build_output_paths(run_dir: Path) -> Dict[str, Path]:
    return {
        "out_json": run_dir / "out.json",
        "out_csv": run_dir / "out.csv",
        "meta_json": run_dir / "fields_column_meta.json",
        "run_config": run_dir / "run_config.json",
    }


# -------------------------------------------------------------------------
# FLATTEN + META UTILITIES (completamente integrati)
# -------------------------------------------------------------------------

def _to_camel_case(s: str) -> str:
    parts = s.split("_")
    if not parts:
        return s
    first = parts[0].lower()
    rest = [p.capitalize() for p in parts[1:]]
    return first + "".join(rest)


def _walk_fields(fields_data, prefix=None):
    """
    Esplora recursivamente spec.fields_spec e costruisce una mappa:
    (path_tuple) -> FieldSpec
    """
    from agentic_forge.structured_output_forge.field_spec import FieldSpec

    mapping = {}
    prefix = prefix or []

    for fd in fields_data:
        f = fd if isinstance(fd, FieldSpec) else FieldSpec(**fd)

        path = prefix + [f.name]
        mapping[tuple(path)] = f

        if f.kind == "object" and f.fields:
            mapping.update(_walk_fields(f.fields, path))

    return mapping


def _infer_dtype_from_schema(spec, field_path: List[str]) -> str:
    """
    Restituisce:
        'categorical_nominal', 'categorical_ordinal',
        'numeric_discrete', 'numeric_continuous',
        'free_text', 'other'
    """
    fields_data = spec.fields_spec
    field_map = _walk_fields(fields_data)

    path_tuple = tuple(field_path)
    f = field_map.get(path_tuple)

    return f.measurement_type if f else "other"


def flatten_results_to_rows_and_meta(
    results: List[Dict[str, Any]],
    specs_by_name: Mapping[str, Any],
    judge_name: str,
) -> tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:

    rows = []
    col_meta: Dict[str, Dict[str, Any]] = {}

    def register_col(col, *, role, dtype, spec=None, field_path=None):
        if col not in col_meta:
            col_meta[col] = {
                "role": role,
                "dtype": dtype,
                "spec": spec,
                "field_path": field_path,
            }

    # colonne fisse
    register_col("id", role="item_id", dtype="nominal")
    register_col("judge", role="judge_id", dtype="nominal")
    register_col("error", role="error", dtype="text")

    for item in results:
        row = {}
        row["id"] = item.get("id")
        row["judge"] = judge_name
        row["error"] = item.get("error")

        results_by_spec = item.get("results", {}) or {}
        errors_by_spec = item.get("errors", {}) or {}

        for spec_name, spec_result in results_by_spec.items():
            spec = specs_by_name.get(spec_name)

            def recurse(prefix, value):
                if isinstance(value, dict):
                    for k, v in value.items():
                        recurse(prefix + [k], v)
                    return

                field_path_list = prefix
                path_snake = "_".join(field_path_list)
                path_camel = _to_camel_case(path_snake)
                col_name = f"{spec_name}__{path_camel}" if path_camel else spec_name
                row[col_name] = value

                dtype = _infer_dtype_from_schema(spec, field_path_list) if spec else "unknown"

                register_col(
                    col_name,
                    role="metric",
                    dtype=dtype,
                    spec=spec_name,
                    field_path=".".join(field_path_list),
                )

            if isinstance(spec_result, dict):
                recurse([], spec_result)
            else:
                col_name = f"{spec_name}__value"
                row[col_name] = spec_result
                register_col(col_name, role="metric", dtype="unknown", spec=spec_name)

            # errors
            err = errors_by_spec.get(spec_name)
            if err is not None:
                err_col = f"{spec_name}__error"
                row[err_col] = err
                register_col(err_col, role="error", dtype="text", spec=spec_name)

        rows.append(row)

    return rows, col_meta


def save_csv_and_meta(
    rows: List[Dict[str, Any]],
    col_meta: Dict[str, Dict[str, Any]],
    csv_path: Path,
    meta_path: Path,
):
    if not rows:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "judge"])

        with meta_path.open("w", encoding="utf-8") as f:
            json.dump(col_meta, f, ensure_ascii=False, indent=2)
        return

    base_cols = ["id", "judge"]
    all_cols = list(base_cols)

    all_set = set(base_cols)
    for r in rows:
        for k in r.keys():
            if k not in all_set:
                all_set.add(k)
                all_cols.append(k)

    other_cols = sorted(c for c in all_cols if c not in base_cols)
    fieldnames = base_cols + other_cols

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({col: r.get(col, "") for col in fieldnames})

    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(col_meta, f, ensure_ascii=False, indent=2)
