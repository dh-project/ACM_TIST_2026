from __future__ import annotations
import sys
import json
from pathlib import Path
import traceback
import yaml


# ============================================================
# 🔧 IMPORT DOPO AVER SISTEMATO IL PATH
# ============================================================

from agentic_forge.workflow_forge.structured_output_workflow_forge.structured_output_workflow_forge import (
    StructuredOutputWorkflowForge,
)
from agentic_forge.workflow_forge.structured_output_workflow_forge.combine_runs import (
    combine_runs,
)
from use_case_utils import (
    DATASET_FIELD_BUILDERS,
    DefaultReasoningPolicy,
)

# ============================================================
# 🔧 PATH A CONFIGURAZIONI
# ============================================================

CONFIG_DIR = Path("configs")
LLM_MANAGER_YAML = CONFIG_DIR / "llm_manager_config.yaml"
MODEL_KEYS_YAML  = CONFIG_DIR / "model_key.yaml"
WORKFLOW_GLOB     = "*_workflow.yaml"   # tutti i workflow YAML


# ============================================================
# 🚀 MAIN SCRIPT
# ============================================================

def main():

    print("\n===============================================")
    print("### BATCH STRUCTURED-OUTPUT WORKFLOW LAUNCHER ###")
    print("===============================================\n")

    workflow_files = sorted(CONFIG_DIR.glob(WORKFLOW_GLOB))
    if not workflow_files:
        raise RuntimeError("❌ No workflow configs found in configs.example/*_workflow.yaml")

    print("[INFO] Workflow configs found:")
    for wf in workflow_files:
        print("  →", wf)
    print()

    # Forge globale (model_forge + reasoning policy)
    forge = StructuredOutputWorkflowForge.forge(
        llm_manager_yaml=LLM_MANAGER_YAML,
        model_keys_yaml=MODEL_KEYS_YAML,
        reasoning_policy=DefaultReasoningPolicy(),
    )

    # Per combinare correttamente alla fine
    exp_to_base_dir = {}   # exp_name → base_output_dir
    exp_names = set()

    # ============================================================
    # 1️⃣ ESECUTIAMO TUTTI I WORKFLOW
    # ============================================================

    for wf_yaml in workflow_files:

        print("\n-----------------------------------------")
        print("[RUN] Workflow:", wf_yaml)
        print("-----------------------------------------")

        with open(wf_yaml, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)["workflow"]

        exp_name = cfg["exp_name"]
        base_output_dir = cfg.get("base_output_dir", "experiments")

        exp_names.add(exp_name)
        exp_to_base_dir[exp_name] = Path(base_output_dir)

        # Instanzia workflow
        workflow = forge.forge_structured_output_workflow(
            workflow_yaml_path=wf_yaml,
            field_builders=DATASET_FIELD_BUILDERS,
        )

        # Esegui workflow
        try:
            result = workflow.run()
        except Exception as e:
            print(f"❌ ERROR running workflow {wf_yaml}: {e}")
            tb = "".join(traceback.format_exception(e))
            print("[ERROR] Traceback:")
            print(tb)
            try:
                error_log = Path(base_output_dir) / "workflow_error.log"
                error_log.parent.mkdir(parents=True, exist_ok=True)
                with error_log.open("a", encoding="utf-8") as logf:
                    logf.write(f"\n[workflow={wf_yaml}]\n")
                    logf.write(tb)
                print(f"[ERROR] Full traceback saved to {error_log}")
            except Exception as log_exc:
                print(f"[WARN] Could not write error log: {log_exc}")
            continue

        print(f"[DONE] Results generated in {result['run_dir']}\n")

    # ============================================================
    # 2️⃣ COMBINAZIONE RISULTATI PER OGNI EXPERIMENT
    # ============================================================

    print("\n====================================")
    print("   🔄 COMBINAZIONE RISULTATI")
    print("====================================\n")

    for exp_name in exp_names:

        base_dir = exp_to_base_dir[exp_name]
        exp_dir  = base_dir / exp_name   # <--- CORRETTO

        if not exp_dir.exists():
            print(f"[WARN] No run directory exists for exp_name={exp_name} at {exp_dir}")
            continue

        out_csv  = exp_dir / "combined_panel.csv"
        out_meta = exp_dir / "combined_panel_meta.json"

        print(f"[COMBINE] exp_name={exp_name}")
        print(f"          runs_dir={exp_dir}")

        try:
            combine_runs(exp_dir, out_csv, out_meta)
        except Exception as e:
            print(f"❌ ERROR during combine for {exp_name}: {e}")
            continue

        print(f"✔ Combined CSV  → {out_csv}")
        print(f"✔ Combined META → {out_meta}\n")

    print("===============================================")
    print("✔ ALL WORKFLOWS COMPLETED AND COMBINED!")
    print("===============================================\n")


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()
