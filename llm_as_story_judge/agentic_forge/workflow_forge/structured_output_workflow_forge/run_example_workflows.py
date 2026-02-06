from __future__ import annotations
import sys
from pathlib import Path
import yaml

# ------------------------------------------------------------
# Fix import path
# ------------------------------------------------------------
CURRENT = Path(__file__).resolve()
ROOT = CURRENT.parents[3]
sys.path.insert(0, str(ROOT))

print("[DEBUG] Project root:", ROOT)

# ------------------------------------------------------------
# Imports (after path fix)
# ------------------------------------------------------------
from agentic_forge.workflow_forge.structured_output_workflow_forge.combine_runs import combine_runs
from agentic_forge.workflow_forge.structured_output_workflow_forge.review_case_utils import (
    DATASET_FIELD_BUILDERS,
    DefaultReasoningPolicy,
)
from agentic_forge.workflow_forge.structured_output_workflow_forge.structured_output_workflow_forge import (
    StructuredOutputWorkflowForge,
)

# ------------------------------------------------------------
# Config paths
# ------------------------------------------------------------
CONFIG_DIR = ROOT / "agentic_forge" / "workflow_forge" / "structured_output_workflow_forge" / "configs.example"

LLM_MANAGER_YAML = CONFIG_DIR / "llm_manager_config.yaml"
MODEL_KEYS_YAML  = CONFIG_DIR / "model_key.yaml"
WORKFLOW_GLOB    = "*_workflow.yaml"


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------
def main():

    # Find workflows
    WORKFLOW_FILES = sorted(CONFIG_DIR.glob(WORKFLOW_GLOB))
    if not WORKFLOW_FILES:
        raise RuntimeError("❌ No workflow configs found in configs.example/*.yaml")

    print("\n[INFO] Workflow trovati:")
    for wf in WORKFLOW_FILES:
        print("   →", wf)

    # Forge globale
    forge = StructuredOutputWorkflowForge.forge(
        llm_manager_yaml=LLM_MANAGER_YAML,
        model_keys_yaml=MODEL_KEYS_YAML,
        reasoning_policy=DefaultReasoningPolicy(),
    )

    exp_to_base_dir: dict[str, Path] = {}   # exp_name → base_output_dir
    exp_to_run_dirs: dict[str, list[Path]] = {}

    # ------------------------------------------------------------
    # 1️⃣ Execute all workflows
    # ------------------------------------------------------------
    for wf_yaml in WORKFLOW_FILES:

        print("\n-----------------------------------------")
        print("[RUN] Workflow:", wf_yaml)
        print("-----------------------------------------")

        with open(wf_yaml, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)["workflow"]

        exp_name        = cfg["exp_name"]
        base_output_dir = Path(cfg["base_output_dir"])

        exp_to_base_dir[exp_name] = base_output_dir

        # Init workflow
        workflow = forge.forge_structured_output_workflow(
            workflow_yaml_path=wf_yaml,
            field_builders=DATASET_FIELD_BUILDERS,
        )

        # Run
        try:
            result = workflow.run()
        except Exception as e:
            print(f"❌ ERRORE durante il workflow {wf_yaml}: {e}")
            continue

        print(f"[DONE] Risultati generati in {result['run_dir']}")

        exp_to_run_dirs.setdefault(exp_name, []).append(Path(result["run_dir"]))

    # ------------------------------------------------------------
    # 2️⃣ Combine runs per experiment
    # ------------------------------------------------------------
    print("\n====================================")
    print("   🔄 COMBINAZIONE RISULTATI")
    print("====================================\n")

    for exp_name, base_dir in exp_to_base_dir.items():

        exp_dir = base_dir / exp_name

        print(f"[COMBINE] exp_name={exp_name}")
        print(f"          exp_dir={exp_dir}")

        if not exp_dir.exists():
            print(f"[WARN] Nessuna directory trovata per exp_name={exp_name}")
            continue

        out_csv  = exp_dir / "combined_panel.csv"
        out_meta = exp_dir / "combined_panel_meta.json"

        try:
            combine_runs(exp_dir, out_csv, out_meta)
        except Exception as e:
            print(f"❌ ERRORE durante combine per {exp_name}: {e}")
            continue

        print(f"✔ Combined CSV  → {out_csv}")
        print(f"✔ Combined META → {out_meta}")

    print("\n====================================")
    print("       ✔ TUTTI I WORKFLOW COMPLETATI")
    print("====================================\n")


if __name__ == "__main__":
    main()
