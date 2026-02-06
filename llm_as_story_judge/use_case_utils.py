
from agentic_forge.dataset_forge.item_dataset_forge import forge_item_dataset
from agentic_forge.workflow_forge.structured_output_workflow_forge.reasoning_policy import ReasoningPolicy

import csv
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any, Union

def build_historical_context(raw_input: dict) -> str:
    """
    Costruisce il valore di `historical_context` per il prompt HistoricalContext,
    a partire dalle informazioni di fase_3 (storia di finzione).

    Usa i blocchi logici:
      - **Periodo dichiarato**
      - **Luogo dichiarato**
      - **Scene della storia**
    """
    fase3 = raw_input.get("fase_3", {}) or {}

    periodo = (fase3.get("tempo") or "").strip()
    luogo = (fase3.get("luogo") or "").strip()
    scene = fase3.get("scene") or []

    parts: list[str] = []

    # Periodo dichiarato
    parts.append("**Periodo dichiarato**")
    if periodo:
        parts.append(periodo)
    else:
        parts.append("(non specificato)")
    parts.append("")

    # Luogo dichiarato
    parts.append("**Luogo dichiarato**")
    if luogo:
        parts.append(luogo)
    else:
        parts.append("(non specificato)")
    parts.append("")

    # Scene della storia
    parts.append("**Scene della storia**")
    if scene:
        for s in scene:
            num = s.get("num")
            titolo = (s.get("titolo") or "").strip()
            contenuto = (s.get("contenuto") or "").strip()

            header_parts = []
            if num is not None:
                header_parts.append(f"Scena {num}")
            if titolo:
                if header_parts:
                    header_parts[-1] = header_parts[-1] + f" - {titolo}"
                else:
                    header_parts.append(titolo)

            if header_parts:
                parts.append(header_parts[0])
            if contenuto:
                parts.append(contenuto)

            parts.append("")  # separatore tra scene
    else:
        parts.append("(nessuna scena disponibile)")

    return "\n".join(parts).strip()


def build_adherence_context(raw_input: dict) -> str:
    """
    Costruisce il valore di `adherence_context` per il prompt
    HistoricalAdherenceOriginalEvent, a partire da:

      - reperto.descrizione
      - reperto.descrizione_curatoriale (se non è un doppione)
      - fase_3.scene[*].{num, titolo, contenuto}

    I blocchi logici prodotti sono:

      - **Descrizione del reperto (fonte storica)**
      - **Descrizione curatoriale (approfondimento)**  [se utile]
      - **Scene della storia**
    """
    reperto = raw_input.get("reperto", {}) or {}
    fase3 = raw_input.get("fase_3", {}) or {}

    desc = (reperto.get("descrizione") or "").strip()
    desc_cur = (reperto.get("descrizione_curatoriale") or "").strip()
    scene = fase3.get("scene") or []

    parts: list[str] = []

    # Descrizione del reperto
    parts.append("**Descrizione del reperto**")
    if desc:
        parts.append(desc)
    else:
        parts.append("(non disponibile)")

    # Descrizione curatoriale, solo se non è un doppione
    if desc_cur:
        include_cur = False
        if not desc:
            include_cur = True
        else:
            # normalizzo un minimo per ridurre i falsi duplicati
            def _norm(s: str) -> str:
                return " ".join(s.lower().split())

            if _norm(desc_cur) != _norm(desc):
                include_cur = True

        if include_cur:
            parts.append("")
            parts.append("**Descrizione curatoriale)**")
            parts.append(desc_cur)

    parts.append("")
    parts.append("**Scene della storia**")

    if scene:
        for s in scene:
            num = s.get("num")
            titolo = (s.get("titolo") or "").strip()
            contenuto = (s.get("contenuto") or "").strip()

            header_parts = []
            if num is not None:
                header_parts.append(f"Scena {num}")
            if titolo:
                if header_parts:
                    header_parts[-1] = header_parts[-1] + f" - {titolo}"
                else:
                    header_parts.append(titolo)

            if header_parts:
                parts.append(header_parts[0])
            if contenuto:
                parts.append(contenuto)

            parts.append("")  # separatore tra scene
    else:
        parts.append("(nessuna scena disponibile)")

    return "\n".join(parts).strip()


def build_storytelling_context(raw_input: dict) -> str:
    """
    Costruisce il valore di `storytelling_context` per il prompt
    DramaStorytellingPotential, a partire da:
      - fase_1.evento_centrale
      - fase_1.descrizione
    """
    fase1_evento = raw_input["fase_1"]["evento_centrale"]
    fase1_descrizione = raw_input["fase_1"]["descrizione"]

    out = (
        "**Evento centrale**\n"
        f"{fase1_evento.strip()}\n\n"
        "**Descrizione**\n"
        f"{fase1_descrizione.strip()}"
    )

    return out


def build_daramatic_situation_context(raw_input: dict) -> str:
    """
    Costruisce il valore di `dramatic_situation_context` per il prompt
    sulla situazione drammatica, a partire da:
      - fase_1.evento_centrale
      - fase_1.descrizione
      - fase_2.titolo
      - fase_2.elementi_dinamici[*].{elemento,valore}
      - fase_3.titolo/luogo/tempo/scene
    """
    fase1 = raw_input.get("fase_1", {}) or {}
    fase2 = raw_input.get("fase_2", {}) or {}
    fase3 = raw_input.get("fase_3", {}) or {}

    evento_centrale = (fase1.get("evento_centrale") or "").strip()
    descrizione_evento = (fase1.get("descrizione") or "").strip()
    titolo_situazione = (fase2.get("titolo") or "").strip()

    elementi_dinamici = fase2.get("elementi_dinamici") or []

    titolo_storia = (fase3.get("titolo") or "").strip()
    luogo_storia = (fase3.get("luogo") or "").strip()
    tempo_storia = (fase3.get("tempo") or "").strip()
    scene = fase3.get("scene") or []

    # Blocchi testuali con nomi SIGNIFICATIVI, indipendenti dai nomi raw
    parts = []

    parts.append("**Evento centrale**")
    if evento_centrale:
        parts.append(evento_centrale)
    else:
        parts.append("(non specificato)")

    parts.append("")  # riga vuota

    parts.append("**Descrizione dell'evento**")
    if descrizione_evento:
        parts.append(descrizione_evento)
    else:
        parts.append("(non specificata)")

    parts.append("")

    parts.append("**Situazione drammatica scelta per realizzare l'evento**")
    if titolo_situazione:
        parts.append(titolo_situazione)
    else:
        parts.append("(non specificato)")

    parts.append("")

    parts.append("**Elementi dinamici scelti per realizzare l'evento**")
    if elementi_dinamici:
        for elem in elementi_dinamici:
            nome = (elem.get("elemento") or "").strip()
            valore = (elem.get("valore") or "").strip()

            if nome and valore:
                parts.append(f"- **{nome}**: {valore}")
            elif nome:
                parts.append(f"- **{nome}**")
            elif valore:
                parts.append(f"- {valore}")
    else:
        parts.append("(nessun elemento dinamico specificato)")

    # Dettagli della situazione drammatica (dal dataset arricchito)
    situaz_details = raw_input.get("situazione_drammatica_details") or {}
    if situaz_details:
        parts.append("")
        parts.append(
            "Questa situazione drammatica fa riferimento alla situazione "
            "drammatica della teoria di Polti di seguito descritta."
        )
        parts.append("")
        parts.append("**Situazione drammatica (teoria di Polti)**")

        description = (situaz_details.get("description") or "").strip()
        dynamic_elements = situaz_details.get("dynamic_elements") or []
        examples = situaz_details.get("examples") or []

        if description:
            parts.append(description)

        if dynamic_elements:
            parts.append("")
            parts.append("Gli elementi dinamici tipicamente sono:")
            for elem in dynamic_elements:
                if isinstance(elem, dict):
                    nome = (
                        (elem.get("elemento") or elem.get("element") or "").strip()
                    )
                    valore = (
                        (elem.get("valore") or elem.get("element_description") or "").strip()
                    )
                    if nome and valore:
                        parts.append(f"- {nome}: {valore}")
                    elif nome:
                        parts.append(f"- {nome}")
                    elif valore:
                        parts.append(f"- {valore}")
                else:
                    parts.append(f"- {elem}")

        if examples:
            parts.append("")
            parts.append("Ecco alcuni esempi:")
            for ex in examples:
                parts.append(f"- {ex}")

    parts.append("")

    parts.append("**Titolo della storia**")
    if titolo_storia:
        parts.append(titolo_storia)
    else:
        parts.append("(non specificato)")

    parts.append("")

    parts.append("**Periodo dichiarato**")
    if tempo_storia:
        parts.append(tempo_storia)
    else:
        parts.append("(non specificato)")

    parts.append("")

    parts.append("**Luogo dichiarato**")
    if luogo_storia:
        parts.append(luogo_storia)
    else:
        parts.append("(non specificato)")

    parts.append("")

    parts.append("**Scene della storia**")
    if scene:
        for s in scene:
            num = s.get("num")
            titolo = (s.get("titolo") or "").strip()
            contenuto = (s.get("contenuto") or "").strip()

            header_parts = []
            if num is not None:
                header_parts.append(f"Scena {num}")
            if titolo:
                if header_parts:
                    header_parts[-1] = header_parts[-1] + f" - {titolo}"
                else:
                    header_parts.append(titolo)

            if header_parts:
                parts.append(header_parts[0])
            if contenuto:
                parts.append(contenuto)

            parts.append("")
    else:
        parts.append("(nessuna scena disponibile)")

    return "\n".join(parts).strip()

def build_dramatic_situation_context(raw_input: dict) -> str:
    return build_daramatic_situation_context(raw_input)


def build_dynamic_elements_context(raw_input: dict) -> str:
    """
    Costruisce il valore di `dynamic_elements_context` per il prompt
    DramaDynamicElementsRealization, a partire da:

      - fase_2.titolo
      - fase_2.elementi_dinamici[*].{elemento, valore}
      - fase_3.scene[*].{num, titolo, contenuto}
    """
    fase2 = raw_input.get("fase_2", {}) or {}
    fase3 = raw_input.get("fase_3", {}) or {}

    titolo_situazione = (fase2.get("titolo") or "").strip()
    elementi_dinamici = fase2.get("elementi_dinamici") or []
    scene = fase3.get("scene") or []

    parts: list[str] = []

    # Titolo situazione
    parts.append("**Titolo della situazione drammatica**")
    if titolo_situazione:
        parts.append(titolo_situazione)
    else:
        parts.append("(non specificato)")
    parts.append("")

    # Elementi dinamici
    parts.append("**Elementi dinamici della situazione**")
    if elementi_dinamici:
        for elem in elementi_dinamici:
            nome = (elem.get("elemento") or "").strip()
            valore = (elem.get("valore") or "").strip()

            if nome and valore:
                parts.append(f"- **{nome}**: {valore}")
            elif nome:
                parts.append(f"- **{nome}**")
            elif valore:
                parts.append(f"- {valore}")
    else:
        parts.append("(nessun elemento dinamico specificato)")
    parts.append("")

    # Scene della storia
    parts.append("**Scene della storia**")
    if scene:
        for s in scene:
            num = s.get("num")
            titolo = (s.get("titolo") or "").strip()
            contenuto = (s.get("contenuto") or "").strip()

            header_parts = []
            if num is not None:
                header_parts.append(f"Scena {num}")
            if titolo:
                if header_parts:
                    header_parts[-1] = header_parts[-1] + f" - {titolo}"
                else:
                    header_parts.append(titolo)

            if header_parts:
                parts.append(header_parts[0])
            if contenuto:
                parts.append(contenuto)

            parts.append("")  # separatore tra scene
    else:
        parts.append("(nessuna scena disponibile)")

    return "\n".join(parts).strip()


def build_turning_points_context(raw_input: dict) -> str:
    """
    Costruisce il valore di `turning_points_context` per il prompt
    DramaTurningPoints, a partire dalle scene della storia.

    Usa i campi logici:
      - Titolo della storia (se presente)
      - Scene della storia (Scena N - Titolo + contenuto)
    """
    fase3 = raw_input.get("fase_3", {}) or {}
    titolo_storia = (fase3.get("titolo") or "").strip()
    scene = fase3.get("scene") or []

    parts: list[str] = []

    if titolo_storia:
        parts.append("**Titolo della storia**")
        parts.append(titolo_storia)
        parts.append("")  # riga vuota

    parts.append("**Scene della storia**")

    if scene:
        for s in scene:
            num = s.get("num")
            titolo = (s.get("titolo") or "").strip()
            contenuto = (s.get("contenuto") or "").strip()

            header_parts = []
            if num is not None:
                header_parts.append(f"Scena {num}")
            if titolo:
                if header_parts:
                    header_parts[-1] = header_parts[-1] + f" - {titolo}"
                else:
                    header_parts.append(titolo)

            if header_parts:
                parts.append(header_parts[0])
            if contenuto:
                parts.append(contenuto)

            parts.append("")  # separatore tra scene
    else:
        parts.append("(nessuna scena disponibile)")

    return "\n".join(parts).strip()



DATASET_FIELD_BUILDERS={
    "storytelling":build_storytelling_context,
    "dramatic_situation":build_daramatic_situation_context,
    "turning_points":build_turning_points_context,
    "dynamic_elements":build_dynamic_elements_context,
    "historical":build_historical_context,
    "adherence":build_adherence_context
}


def concat_csv_files(csv_paths: list[Union[str, Path]], output_csv: Union[str, Path]) -> Path:
    """
    Concatena verticalmente una lista di file CSV e salva il risultato in un nuovo CSV.
    Controlla che le colonne siano identiche; altrimenti solleva errore.
    """
    if not csv_paths:
        raise ValueError("csv_paths must contain at least one file path.")

    dataframes: list[pd.DataFrame] = []
    expected_cols: list[str] | None = None

    for csv_path in csv_paths:
        path = Path(csv_path)
        if not path.exists():
            raise FileNotFoundError(f"CSV not found: {path}")

        df = pd.read_csv(path)
        cols = list(df.columns)

        if expected_cols is None:
            expected_cols = cols
        elif cols != expected_cols:
            raise ValueError(
                f"CSV columns mismatch for {path}. "
                f"Expected: {expected_cols} | Found: {cols}"
            )

        dataframes.append(df)

    if expected_cols is None:
        raise ValueError("No columns found across input CSV files.")

    combined = pd.concat(dataframes, axis=0, ignore_index=True)

    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out_path, index=False)

    return out_path


def concat_panel(
    csv_map: dict[Union[str, Path], dict[str, Any | None] | None]
) -> pd.DataFrame:
    """
    Concatena CSV con colonne potenzialmente diverse e forza un valore per colonna.
    csv_map mappa path -> {colonna: valore}; valore None non modifica.
    Le colonne mancanti nei singoli CSV restano come valori nulli.
    """
    if not csv_map:
        raise ValueError("csv_map must contain at least one file path.")

    dataframes: list[pd.DataFrame] = []
    combined_cols: list[str] = []

    for csv_path, rename_map in csv_map.items():
        path = Path(csv_path)
        if not path.exists():
            raise FileNotFoundError(f"CSV not found: {path}")

        df = pd.read_csv(path, dtype=str, keep_default_na=True)

        if rename_map:
            for col_name, fill_value in rename_map.items():
                if fill_value is None:
                    continue
                if col_name in df.columns:
                    df[col_name] = fill_value

        for col in df.columns:
            if col not in combined_cols:
                combined_cols.append(col)

        dataframes.append(df)

    aligned = [df.reindex(columns=combined_cols) for df in dataframes]
    return pd.concat(aligned, axis=0, ignore_index=True)


def infer_metric_columns(
    panel_df: pd.DataFrame,
    *,
    id_col: str = "id",
    judge_col: str = "judge",
    error_col: str = "error",
    historical_prefix: str = "historical_",
    dramatic_prefix: str = "dramatic_",
    turning_prefix: str = "dramatic_turningPoints__",
) -> dict[str, list[str]]:
    """
    Inferisce colonne numeriche, turning points e metadati.
    - numeriche: prefix historical_/dramatic_ (escludendo turning points e spiegazioni)
    - turning points: dramatic_turningPoints__ (escludendo spiegazioni)
    - metadati: tutto il resto (incluse spiegazioni)
    """
    cols = list(panel_df.columns)
    metadata = {id_col, judge_col, error_col}

    def _is_explanation(name: str) -> bool:
        return "spiegazione" in name.lower()

    turning_cols = [
        c for c in cols
        if c.startswith(turning_prefix) and not _is_explanation(c)
    ]
    numeric_cols = [
        c for c in cols
        if (c.startswith(historical_prefix) or c.startswith(dramatic_prefix))
        and not c.startswith(turning_prefix)
        and not _is_explanation(c)
    ]
    metadata_cols = [
        c for c in cols if c not in set(numeric_cols + turning_cols) and c not in metadata
    ]

    return {
        "numeric": numeric_cols,
        "turning": turning_cols,
        "metadata": metadata_cols,
    }


def apply_partial_judge_mask(
    panel_df: pd.DataFrame,
    *,
    judge_col: str = "judge",
    historical_prefix: str = "historical_",
    dramatic_prefix: str = "dramatic_",
    turning_prefix: str = "dramatic_turningPoints__",
    judge_domain_map: dict[str, set[str]] | None = None,
) -> pd.DataFrame:
    """
    Impone NA sui domini NON valutati da specifici giudici.
    Evita che i mancanti vengano interpretati come etichette reali.
    Domini validi: {"historical", "dramatic", "turning"}.
    """
    if panel_df.empty:
        return panel_df

    if judge_domain_map is None:
        judge_domain_map = {
            "humanHistorical": {"historical"},
            "humanDrama": {"dramatic", "turning"},
        }

    present = set(panel_df[judge_col].dropna().astype(str).unique().tolist())
    active = {j: domains for j, domains in judge_domain_map.items() if j in present}
    if not active:
        return panel_df

    cols = list(panel_df.columns)
    historical_cols = [c for c in cols if c.startswith(historical_prefix)]
    dramatic_cols = [c for c in cols if c.startswith(dramatic_prefix)]
    turning_cols = [c for c in cols if c.startswith(turning_prefix)]

    out = panel_df.copy()
    for judge, domains in active.items():
        mask = out[judge_col] == judge
        if not mask.any():
            continue
        if "historical" not in domains and historical_cols:
            out.loc[mask, historical_cols] = pd.NA
        if "dramatic" not in domains and dramatic_cols:
            out.loc[mask, dramatic_cols] = pd.NA
        if "turning" not in domains and turning_cols:
            out.loc[mask, turning_cols] = pd.NA

    return out


def build_judge_comparison_table(
    panel_df: pd.DataFrame,
    judge_pairs: list[tuple[str, str]] | None = None,
    *,
    id_col: str = "id",
    judge_col: str = "judge",
    error_col: str = "error",
    historical_prefix: str = "historical_",
    dramatic_prefix: str = "dramatic_",
    turning_prefix: str = "dramatic_turningPoints__",
    numeric_cols: list[str] | None = None,
    turning_cols: list[str] | None = None,
    partial_judge_map: dict[str, set[str]] | None = None,
) -> pd.DataFrame:
    """
    Costruisce una tabella di confronto tra coppie di giudici.
    - numeriche: Spearman rho + Kendall tau
    - turning points: Accuracy + F1 macro
    - aggregazioni: storico/drammatico (correlazioni) e turning overall
    """
    from agentic_forge.metrics.classification import accuracy, f1_macro
    from agentic_forge.metrics.correlations import spearman_corr, kendall_corr

    panel_df = apply_partial_judge_mask(
        panel_df,
        judge_col=judge_col,
        historical_prefix=historical_prefix,
        dramatic_prefix=dramatic_prefix,
        turning_prefix=turning_prefix,
        judge_domain_map=partial_judge_map,
    )

    def _parse_pairs(pairs: list[tuple[str, str]]) -> list[tuple[str, str]]:
        import warnings

        parsed: list[tuple[str, str]] = []
        seen: set[tuple[str, str]] = set()
        for item in pairs:
            if not isinstance(item, (list, tuple)) or len(item) != 2:
                raise ValueError(f"Invalid judge pair entry: {item}")
            a, b = item
            a_str = str(a)
            b_str = str(b)
            key = (a_str, b_str)
            rev_key = (b_str, a_str)
            if rev_key in seen:
                warnings.warn(
                    f"Duplicate reversed pair skipped: ({a_str}, {b_str})",
                    RuntimeWarning,
                )
                continue
            if key in seen:
                warnings.warn(
                    f"Duplicate pair skipped: ({a_str}, {b_str})",
                    RuntimeWarning,
                )
                continue
            seen.add(key)
            parsed.append(key)
        return parsed

    def _safe_spearman(x: pd.Series, y: pd.Series) -> float | None:
        x_num = pd.to_numeric(x, errors="coerce")
        y_num = pd.to_numeric(y, errors="coerce")
        mask = x_num.notna() & y_num.notna()
        if mask.sum() < 2:
            return None
        rho = spearman_corr(x_num[mask].tolist(), y_num[mask].tolist())
        if np.isnan(rho):
            return None
        return float(rho)

    def _safe_kendall(x: pd.Series, y: pd.Series) -> float | None:
        x_num = pd.to_numeric(x, errors="coerce")
        y_num = pd.to_numeric(y, errors="coerce")
        mask = x_num.notna() & y_num.notna()
        if mask.sum() < 2:
            return None
        tau = kendall_corr(x_num[mask].tolist(), y_num[mask].tolist())
        if np.isnan(tau):
            return None
        return float(tau)

    def _safe_accuracy(x: pd.Series, y: pd.Series) -> float | None:
        mask = x.notna() & y.notna()
        if mask.sum() == 0:
            return None
        return float(accuracy(x[mask].tolist(), y[mask].tolist()))

    def _safe_f1(x: pd.Series, y: pd.Series) -> float | None:
        mask = x.notna() & y.notna()
        if mask.sum() == 0:
            return None
        return float(f1_macro(x[mask].tolist(), y[mask].tolist(), zero_division=0))

    if numeric_cols is None or turning_cols is None:
        inferred = infer_metric_columns(
            panel_df,
            id_col=id_col,
            judge_col=judge_col,
            error_col=error_col,
            historical_prefix=historical_prefix,
            dramatic_prefix=dramatic_prefix,
            turning_prefix=turning_prefix,
        )
        if numeric_cols is None:
            numeric_cols = inferred["numeric"]
        if turning_cols is None:
            turning_cols = inferred["turning"]

    if judge_pairs is None:
        from itertools import combinations

        judges = [str(j) for j in panel_df[judge_col].dropna().unique().tolist()]
        pairs = [(a, b) for a, b in combinations(judges, 2)]
    else:
        pairs = _parse_pairs(judge_pairs)
    result_rows: list[dict[str, Any]] = []

    for judge_a, judge_b in pairs:
        df_a = panel_df[panel_df[judge_col] == judge_a]
        df_b = panel_df[panel_df[judge_col] == judge_b]

        cols = list(dict.fromkeys([id_col] + numeric_cols + turning_cols))
        df_a = df_a.reindex(columns=cols)
        df_b = df_b.reindex(columns=cols)

        merged = df_a.merge(df_b, on=id_col, suffixes=("_a", "_b"))

        for col in numeric_cols:
            result_rows.append(
                {
                    "judgeA": judge_a,
                    "judgeB": judge_b,
                    "field": col,
                    "spearman_rho": _safe_spearman(merged[f"{col}_a"], merged[f"{col}_b"]),
                    "kendall_tau": _safe_kendall(merged[f"{col}_a"], merged[f"{col}_b"]),
                    "accuracy": None,
                    "f1_macro": None,
                }
            )

        for col in turning_cols:
            result_rows.append(
                {
                    "judgeA": judge_a,
                    "judgeB": judge_b,
                    "field": col,
                    "spearman_rho": None,
                    "kendall_tau": None,
                    "accuracy": _safe_accuracy(merged[f"{col}_a"], merged[f"{col}_b"]),
                    "f1_macro": _safe_f1(merged[f"{col}_a"], merged[f"{col}_b"]),
                }
            )

        historical_cols = [c for c in numeric_cols if c.startswith(historical_prefix)]
        dramatic_cols = [
            c for c in numeric_cols if c.startswith(dramatic_prefix)
        ]

        def _agg_corr(cols: list[str]) -> tuple[float | None, float | None]:
            rhos = []
            taus = []
            for c in cols:
                rho = _safe_spearman(merged[f"{c}_a"], merged[f"{c}_b"])
                tau = _safe_kendall(merged[f"{c}_a"], merged[f"{c}_b"])
                if rho is not None:
                    rhos.append(rho)
                if tau is not None:
                    taus.append(tau)
            rho_out = float(sum(rhos) / len(rhos)) if rhos else None
            tau_out = float(sum(taus) / len(taus)) if taus else None
            return rho_out, tau_out

        for label_name, cols in (
            ("historical__aggregate", historical_cols),
            ("dramatic__aggregate", dramatic_cols),
        ):
            if cols:
                rho, tau = _agg_corr(cols)
                result_rows.append(
                    {
                        "judgeA": judge_a,
                        "judgeB": judge_b,
                        "field": label_name,
                        "spearman_rho": rho,
                        "kendall_tau": tau,
                        "accuracy": None,
                        "f1_macro": None,
                    }
                )

        if turning_cols:
            acc_vals = []
            f1_vals = []
            for c in turning_cols:
                acc = _safe_accuracy(merged[f"{c}_a"], merged[f"{c}_b"])
                f1 = _safe_f1(merged[f"{c}_a"], merged[f"{c}_b"])
                if acc is not None:
                    acc_vals.append(acc)
                if f1 is not None:
                    f1_vals.append(f1)

            result_rows.append(
                {
                    "judgeA": judge_a,
                    "judgeB": judge_b,
                    "field": "dramatic_turningPoints__accuracy_overall",
                    "spearman_rho": None,
                    "kendall_tau": None,
                    "accuracy": float(sum(acc_vals) / len(acc_vals)) if acc_vals else None,
                    "f1_macro": None,
                }
            )
            result_rows.append(
                {
                    "judgeA": judge_a,
                    "judgeB": judge_b,
                    "field": "dramatic_turningPoints__f1_macro_overall",
                    "spearman_rho": None,
                    "kendall_tau": None,
                    "accuracy": None,
                    "f1_macro": float(sum(f1_vals) / len(f1_vals)) if f1_vals else None,
                }
            )

    return pd.DataFrame(result_rows)


def plot_judge_confusion_matrices(
    metrics_df: pd.DataFrame,
    *,
    id_col: str = "id",
    judge_col: str = "judge",
    error_col: str = "error",
    numeric_cols: list[str] | None = None,
    turning_cols: list[str] | None = None,
    historical_prefix: str = "historical_",
    dramatic_prefix: str = "dramatic_",
    turning_prefix: str = "dramatic_turningPoints__",
    numeric_metrics: list[str] | None = None,
    turning_metrics: list[str] | None = None,
):
    """
    Plotta una matrice (judge x judge) per ogni campo, con valori di metrica.
    - numeriche: Spearman e Kendall (default)
    - turning points: Accuracy e F1 macro (default)
    Ritorna un dizionario field -> {"judges": [...], "metric": str, "matrix": ndarray}.
    """
    import matplotlib.pyplot as plt

    if numeric_metrics is None:
        numeric_metrics = ["spearman_rho", "kendall_tau"]
    if turning_metrics is None:
        turning_metrics = ["accuracy", "f1_macro"]

    table = metrics_df.copy()

    judges = sorted(pd.unique(table[["judgeA", "judgeB"]].values.ravel("K")).tolist())
    judge_idx = {j: i for i, j in enumerate(judges)}

    results: dict[str, dict[str, Any]] = {}
    fields = sorted(table["field"].dropna().unique().tolist())

    for field in fields:
        has_corr = table.loc[table["field"] == field, "spearman_rho"].notna().any()
        field_metrics = numeric_metrics if has_corr else turning_metrics
        for metric_name in field_metrics:
            mat = np.full((len(judges), len(judges)), np.nan)

            subset = table[table["field"] == field]
            for _, row in subset.iterrows():
                a = row["judgeA"]
                b = row["judgeB"]
                val = row.get(metric_name)
                if a in judge_idx and b in judge_idx:
                    i = judge_idx[a]
                    j = judge_idx[b]
                    mat[i, j] = val
                    mat[j, i] = val

            results[f"{field}__{metric_name}"] = {
                "judges": judges,
                "metric": metric_name,
                "matrix": mat,
            }

            if metric_name in ("spearman_rho", "kendall_tau"):
                vmin, vmax = -1, 1
            else:
                vmin, vmax = 0, 1

            fig, ax = plt.subplots(figsize=(4.5, 4))
            im = ax.imshow(mat, cmap="Blues", vmin=vmin, vmax=vmax)
            ax.set_title(f"{field} ({metric_name})")
            ax.set_xlabel("Judge")
            ax.set_ylabel("Judge")
            ax.set_xticks(range(len(judges)))
            ax.set_yticks(range(len(judges)))
            ax.set_xticklabels(judges, rotation=45, ha="right")
            ax.set_yticklabels(judges)

            for (i, j), val in np.ndenumerate(mat):
                if np.isnan(val):
                    continue
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8)

            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            fig.tight_layout()

    return results


def plot_judge_score_distributions(
    panel_df: pd.DataFrame,
    *,
    id_col: str = "id",
    judge_col: str = "judge",
    error_col: str = "error",
    numeric_cols: list[str] | None = None,
    turning_cols: list[str] | None = None,
    historical_prefix: str = "historical_",
    dramatic_prefix: str = "dramatic_",
    turning_prefix: str = "dramatic_turningPoints__",
    bins: int = 10,
    normalize: bool = True,
    partial_judge_map: dict[str, set[str]] | None = None,
):
    """
    Plotta la distribuzione dei voti per ogni giudice nello stesso plot.
    - Per le colonne numeriche: istogramma sovrapposto.
    - Per turning points: barre per classe.
    Include anche aggregazioni numeriche historical/dramatic.
    """
    import matplotlib.pyplot as plt

    panel_df = apply_partial_judge_mask(
        panel_df,
        judge_col=judge_col,
        historical_prefix=historical_prefix,
        dramatic_prefix=dramatic_prefix,
        turning_prefix=turning_prefix,
        judge_domain_map=partial_judge_map,
    )

    if numeric_cols is None or turning_cols is None:
        inferred = infer_metric_columns(
            panel_df,
            id_col=id_col,
            judge_col=judge_col,
            error_col=error_col,
            historical_prefix=historical_prefix,
            dramatic_prefix=dramatic_prefix,
            turning_prefix=turning_prefix,
        )
        if numeric_cols is None:
            numeric_cols = inferred["numeric"]
        if turning_cols is None:
            turning_cols = inferred["turning"]

    judges = sorted(panel_df[judge_col].dropna().unique().tolist())

    historical_cols = [c for c in numeric_cols if c.startswith(historical_prefix)]
    dramatic_cols = [c for c in numeric_cols if c.startswith(dramatic_prefix)]

    def _plot_numeric(field: str, series_by_judge: dict[str, pd.Series]):
        fig, ax = plt.subplots(figsize=(5, 4))
        for j, s in series_by_judge.items():
            vals = pd.to_numeric(s, errors="coerce").dropna()
            if vals.empty:
                continue
            ax.hist(
                vals,
                bins=bins,
                alpha=0.35,
                label=j,
                density=normalize,
                histtype="stepfilled",
            )
        ax.set_title(field)
        ax.set_xlabel("score")
        ax.set_ylabel("density" if normalize else "count")
        ax.legend()
        fig.tight_layout()

    def _plot_categorical(field: str, series_by_judge: dict[str, pd.Series]):
        labels = []
        for s in series_by_judge.values():
            labels.extend([v for v in s.dropna().unique().tolist()])
        labels = sorted(pd.unique(labels).tolist())
        if not labels:
            return
        x = np.arange(len(labels))
        width = 0.8 / max(1, len(series_by_judge))

        fig, ax = plt.subplots(figsize=(5.5, 4))
        for idx, (j, s) in enumerate(series_by_judge.items()):
            counts = s.value_counts(dropna=True)
            vals = [counts.get(l, 0) for l in labels]
            if normalize:
                total = sum(vals) or 1
                vals = [v / total for v in vals]
            ax.bar(x + idx * width, vals, width=width, label=j)
        ax.set_title(field)
        ax.set_xlabel("label")
        ax.set_ylabel("share" if normalize else "count")
        ax.set_xticks(x + width * (len(series_by_judge) - 1) / 2)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.legend()
        fig.tight_layout()

    def _plot_box(field: str, series_by_judge: dict[str, pd.Series]):
        fig, ax = plt.subplots(figsize=(5.5, 4))
        data = []
        labels = []
        for j, s in series_by_judge.items():
            vals = pd.to_numeric(s, errors="coerce").dropna()
            if vals.empty:
                continue
            data.append(vals)
            labels.append(j)
        if not data:
            return
        ax.boxplot(data, labels=labels, showfliers=False)
        ax.set_title(field)
        ax.set_ylabel("score")
        ax.tick_params(axis="x", rotation=45)
        fig.tight_layout()

    def _is_discrete(series_by_judge: dict[str, pd.Series]) -> bool:
        vals = pd.concat(
            [pd.to_numeric(s, errors="coerce") for s in series_by_judge.values()],
            ignore_index=True,
        ).dropna()
        if vals.empty:
            return False
        unique_vals = vals.unique()
        if len(unique_vals) <= 5 and np.all(np.isclose(unique_vals, np.round(unique_vals))):
            return True
        return False

    # Per-field distributions
    for field in numeric_cols:
        series_by_judge = {
            j: panel_df.loc[panel_df[judge_col] == j, field]
            for j in judges
        }
        if _is_discrete(series_by_judge):
            _plot_categorical(field, series_by_judge)
        else:
            _plot_numeric(field, series_by_judge)

    for field in turning_cols:
        series_by_judge = {
            j: panel_df.loc[panel_df[judge_col] == j, field]
            for j in judges
        }
        _plot_categorical(field, series_by_judge)

    # Aggregazioni numeriche per giudice (media per riga)
    if historical_cols:
        series_by_judge = {}
        for j in judges:
            df_j = panel_df.loc[panel_df[judge_col] == j, historical_cols]
            series_by_judge[j] = df_j.apply(pd.to_numeric, errors="coerce").mean(axis=1)
        _plot_box("historical__aggregate", series_by_judge)

    if dramatic_cols:
        series_by_judge = {}
        for j in judges:
            df_j = panel_df.loc[panel_df[judge_col] == j, dramatic_cols]
            series_by_judge[j] = df_j.apply(pd.to_numeric, errors="coerce").mean(axis=1)
        _plot_box("dramatic__aggregate", series_by_judge)


def compare_panel_distribution(
    panel_df: pd.DataFrame,
    *,
    id_col: str = "id",
    judge_col: str = "judge",
    error_col: str = "error",
    numeric_cols: list[str] | None = None,
    turning_cols: list[str] | None = None,
    historical_prefix: str = "historical_",
    dramatic_prefix: str = "dramatic_",
    turning_prefix: str = "dramatic_turningPoints__",
    aggregation_style: str = "violin",
    partial_judge_map: dict[str, set[str]] | None = None,
    judges_filter: list[str] | None = None,
    judge_rename_map: dict[str, str] | None = None,
):
    """
    Confronta le distribuzioni dei giudici per capire chi e' piu' "generoso".
    - Discrete (es. 1-3): barre normalizzate per giudice.
    - Continue: violin/box per giudice (default violin).
    - Aggregazioni: per giudice su historical/dramatic.
    - Duplica i plot usando solo il supporto valutato dai giudici umani.
    - Rinomina eventuali giudici nei label/legend.
    """
    import matplotlib.pyplot as plt

    panel_df = apply_partial_judge_mask(
        panel_df,
        judge_col=judge_col,
        historical_prefix=historical_prefix,
        dramatic_prefix=dramatic_prefix,
        turning_prefix=turning_prefix,
        judge_domain_map=partial_judge_map,
    )
    if judges_filter:
        panel_df = panel_df.loc[panel_df[judge_col].isin(judges_filter)].copy()

    def _display_judge(judge: str) -> str:
        if judge_rename_map and judge in judge_rename_map:
            return judge_rename_map[judge]
        return judge

    if numeric_cols is None or turning_cols is None:
        inferred = infer_metric_columns(
            panel_df,
            id_col=id_col,
            judge_col=judge_col,
            error_col=error_col,
            historical_prefix=historical_prefix,
            dramatic_prefix=dramatic_prefix,
            turning_prefix=turning_prefix,
        )
        if numeric_cols is None:
            numeric_cols = inferred["numeric"]
        if turning_cols is None:
            turning_cols = inferred["turning"]

    historical_cols = [c for c in numeric_cols if c.startswith(historical_prefix)]
    dramatic_cols = [c for c in numeric_cols if c.startswith(dramatic_prefix)]

    def _is_discrete(series_by_judge: dict[str, pd.Series]) -> bool:
        vals = pd.concat(
            [pd.to_numeric(s, errors="coerce") for s in series_by_judge.values()],
            ignore_index=True,
        ).dropna()
        if vals.empty:
            return False
        unique_vals = vals.unique()
        if len(unique_vals) <= 5 and np.all(np.isclose(unique_vals, np.round(unique_vals))):
            return True
        return False

    def _title_for(field: str) -> str:
        if field.startswith("historical__aggregate"):
            suffix = field[len("historical__aggregate") :]
            return f"Historical Scores Mean Distribution{suffix}"
        if field.startswith("dramatic__aggregate"):
            suffix = field[len("dramatic__aggregate") :]
            return f"Dramatic Scores Mean Distribution{suffix}"
        return field

    def _plot_discrete(field: str, series_by_judge: dict[str, pd.Series]):
        labels = []
        for s in series_by_judge.values():
            labels.extend([v for v in s.dropna().unique().tolist()])
        labels = sorted(pd.unique(labels).tolist())
        if not labels:
            return
        x = np.arange(len(labels))
        width = 0.8 / max(1, len(series_by_judge))

        fig, ax = plt.subplots(figsize=(5.5, 4))
        for idx, (j, s) in enumerate(series_by_judge.items()):
            counts = s.value_counts(dropna=True)
            vals = [counts.get(l, 0) for l in labels]
            total = sum(vals) or 1
            vals = [v / total for v in vals]
            ax.bar(x + idx * width, vals, width=width, label=_display_judge(j))
        ax.set_title(_title_for(field))
        ax.set_xlabel("label")
        ax.set_ylabel("share")
        ax.set_xticks(x + width * (len(series_by_judge) - 1) / 2)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.legend()
        fig.tight_layout()

    def _plot_continuous(field: str, series_by_judge: dict[str, pd.Series]):
        data = []
        labels = []
        means = []
        counts = []
        for j, s in series_by_judge.items():
            vals = pd.to_numeric(s, errors="coerce").dropna()
            if vals.empty:
                continue
            data.append(vals)
            labels.append(_display_judge(j))
            means.append(float(vals.mean()))
            counts.append(int(vals.shape[0]))
        if not data:
            return
        fig, ax = plt.subplots(figsize=(5.5, 4))
        if aggregation_style == "box":
            ax.boxplot(data, labels=labels, showfliers=False)
        else:
            parts = ax.violinplot(data, showmedians=True)
            for pc in parts.get("bodies", []):
                pc.set_alpha(0.6)
            ax.set_xticks(range(1, len(labels) + 1))
            ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_title(_title_for(field))
        ax.set_ylabel("score")
        # Media per giudice (punto)
        ax.scatter(
            np.arange(1, len(means) + 1),
            means,
            color="black",
            s=20,
            zorder=3,
            label="mean",
        )
        # Valore della media vicino al pallino
        x_offset = 0.08
        for i, m in enumerate(means, start=1):
            ax.text(
                i + x_offset,
                m,
                f"{m:.2f}",
                ha="left",
                va="center",
                fontsize=8,
            )
        ax.legend()
        fig.tight_layout()

    def _plot_panel(panel_subset: pd.DataFrame, *, title_suffix: str = ""):
        judges = sorted(panel_subset[judge_col].dropna().unique().tolist())

        # Per-field distributions
        for field in numeric_cols:
            label = f"{field}{title_suffix}"
            series_by_judge = {
                j: panel_subset.loc[panel_subset[judge_col] == j, field]
                for j in judges
            }
            if _is_discrete(series_by_judge):
                _plot_discrete(label, series_by_judge)
            else:
                _plot_continuous(label, series_by_judge)

        for field in turning_cols:
            label = f"{field}{title_suffix}"
            series_by_judge = {
                j: panel_subset.loc[panel_subset[judge_col] == j, field]
                for j in judges
            }
            _plot_discrete(label, series_by_judge)

        # Aggregazioni numeriche per giudice (media per riga)
        if historical_cols:
            series_by_judge = {}
            for j in judges:
                df_j = panel_subset.loc[panel_subset[judge_col] == j, historical_cols]
                series_by_judge[j] = df_j.apply(pd.to_numeric, errors="coerce").mean(axis=1)
            _plot_continuous(f"historical__aggregate{title_suffix}", series_by_judge)

        if dramatic_cols:
            series_by_judge = {}
            for j in judges:
                df_j = panel_subset.loc[panel_subset[judge_col] == j, dramatic_cols]
                series_by_judge[j] = df_j.apply(pd.to_numeric, errors="coerce").mean(axis=1)
            _plot_continuous(f"dramatic__aggregate{title_suffix}", series_by_judge)

    _plot_panel(panel_df)

    # Duplica i plot sul supporto umano (stesse storie valutate dagli umani)
    human_mask = panel_df[judge_col].astype(str).str.lower().str.startswith("human")
    if human_mask.any():
        human_ids = (
            panel_df.loc[human_mask, id_col]
            .dropna()
            .unique()
            .tolist()
        )
        if human_ids:
            panel_human_support = panel_df.loc[panel_df[id_col].isin(human_ids)].copy()
            _plot_panel(panel_human_support, title_suffix=" (same stories)")


class DefaultReasoningPolicy(ReasoningPolicy):
    """
    Implementazione predefinita della reasoning policy.
    Mantiene la logica esistente, ma ora è modulare e sostituibile.

    Gestisce i casi:
    - Modello con reasoning built-in (Claude 3.7 → sempre attivo)
    - Modello con reasoning controllabile (DeepSeek / Kimi)
    - Modello senza reasoning (raise se enabled=True)
    """

    def resolve(self, llm, model_name: str, enabled: bool):
        """
        Ritorna il payload reasoning corretto (None, dict, str, bool).
        """

        # ------------------------------------------------------------
        # 1) Modelli con reasoning BUILT-IN
        # ------------------------------------------------------------
        # Es: Claude 3.7 ha reasoning integrato non disattivabile
        if hasattr(llm, "mode") and getattr(llm, "mode") == "builtin":
            if not enabled:
                raise ValueError(
                    f"Model '{model_name}' has built-in reasoning that cannot be disabled."
                )
            return {}  # payload standard built-in

        # ------------------------------------------------------------
        # 2) Modelli con reasoning CONTROLLABILE
        # ------------------------------------------------------------
        # Es: DeepSeek, Kimi, GPT-OSS-LW ecc.
        if hasattr(llm, "mode") and getattr(llm, "mode") == "controllable":
            if not enabled:
                return None  # reasoning disattivato
            # euristiche personalizzate
            if model_name.startswith("deepseek"):
                return True
            if "kimi" in model_name.lower():
                return {"enabled": True}
            if "gpt-oss" in model_name.lower():
                return "medium"
            # fallback generico
            return True

        # ------------------------------------------------------------
        # 3) Modelli SENZA reasoning
        # ------------------------------------------------------------
        if enabled:
            raise ValueError(
                f"Model '{model_name}' does NOT support reasoning. Disable it."
            )

        return None


from itertools import combinations

# panel già letto
# panel = pd.read_csv(PATH_PANEL)

def build_pair_panel(
    panel_df,
    judge_pairs=None,
    id_col="id",
    judge_col="judge",
    partial_judge_map: dict[str, set[str]] | None = None,
):
    panel_df = apply_partial_judge_mask(
        panel_df,
        judge_col=judge_col,
        judge_domain_map=partial_judge_map,
    )
    inferred = infer_metric_columns(panel_df)
    metric_cols = inferred["numeric"] + inferred["turning"]

    if judge_pairs is None:
        judges = panel_df[judge_col].dropna().astype(str).unique().tolist()
        judge_pairs = list(combinations(judges, 2))
    else:
        judge_pairs = [(str(a), str(b)) for a, b in judge_pairs]

    out = []
    for judgeA, judgeB in judge_pairs:
        df_a = panel_df[panel_df[judge_col] == judgeA][[id_col] + metric_cols]
        df_b = panel_df[panel_df[judge_col] == judgeB][[id_col] + metric_cols]

        merged = df_a.merge(df_b, on=id_col, suffixes=("_a", "_b"))

        tup_df = merged[[id_col]].copy()
        tup_df.insert(0, "judgeA", judgeA)
        tup_df.insert(1, "judgeB", judgeB)

        for c in metric_cols:
            tup_df[c] = list(zip(merged[f"{c}_a"], merged[f"{c}_b"]))

        out.append(tup_df)

    return pd.concat(out, ignore_index=True)
