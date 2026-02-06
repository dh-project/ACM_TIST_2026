from __future__ import annotations
from pathlib import Path
from typing import Optional, Dict, Any

from agentic_forge.workflow_forge.structured_output_workflow_forge.workflow_utils import (
    resolve_run_dir,
    build_output_paths,
    build_run_config,
    save_run_config,
    flatten_results_to_rows_and_meta,
    save_csv_and_meta,
)

from agentic_forge.structured_output_forge.structured_output_spec_forge import (
    forge_structured_output_specs,
)

from agentic_forge.structured_output_forge.structured_output_model_runner import (
    run_structured_output_over_item_dataset,
    _default_fields_builder,
)


class StructuredOutputWorkflow():
    def __init__(self, model_forge, reasoning_policy, item_field_builders, fields_builder=None):
        """
        item_field_builders = builder per il DATASET
        fields_builder      = builder per fields_by_name (spec → variabili prompt)
        """
        self.model_forge = model_forge
        self.reasoning_policy = reasoning_policy

        self.item_field_builders = item_field_builders
        self.fields_builder = fields_builder or _default_fields_builder

        self.cfg: Dict[str, Any] = {}

    # -------------------------------------------------------
    def configure_from_dict(self, cfg: Dict[str, Any]) -> None:
        """Salva configurazione del workflow"""
        self.cfg = cfg

    # -------------------------------------------------------
    def run(self, fields_builder=None) -> Dict[str, Any]:
        """
        fields_builder: permette di sovrascrivere
                        il builder per spec → variabili del prompt.
        """
        if not self.cfg:
            raise RuntimeError("Workflow not configured.")

        cfg = self.cfg

        exp_name = cfg["exp_name"]
        logical_model_key = cfg["logical_model_key"]

        dataset_path = Path(cfg["dataset_path"])
        schemas_path = Path(cfg["schemas_path"])
        prompts_path = Path(cfg["prompts_path"])

        run_name = cfg.get("run_name")
        batch_size = cfg.get("batch_size", 5)
        max_retries = cfg.get("max_retries", 1)
        reasoning_enabled = cfg.get("reasoning_enabled", False)
        verbose_llm = cfg.get("verbose_llm", False)
        base_output_dir = Path(cfg.get("base_output_dir", "experiments"))

        # MODEL
        llm = self.model_forge.forge_large_model(logical_model_key)
        model_name = self.model_forge.key_forge.get(logical_model_key)["model"]

        reasoning_param = self.reasoning_policy.resolve(llm, model_name, reasoning_enabled)

        # DIRS
        run_dir, resolved_run_name = resolve_run_dir(
            base_output_dir, exp_name, logical_model_key, run_name
        )
        output_paths = build_output_paths(run_dir)

        # SAVE CONFIG
        run_cfg = build_run_config(
            exp_name, resolved_run_name,
            logical_model_key, model_name,
            dataset_path, schemas_path, prompts_path,
            batch_size, max_retries,
            reasoning_enabled, verbose_llm,
            base_output_dir,
        )
        save_run_config(run_cfg, run_dir)

        # LOAD SPECS
        specs = forge_structured_output_specs(
            schemas_path, prompts_path,
            pattern="*.yaml",
            with_format_instruction=True
        )

        # seleziona il fields_builder effettivo
        effective_fields_builder = fields_builder or self.fields_builder

        # EXEC
        results = run_structured_output_over_item_dataset(
            llm=llm,
            json_path=dataset_path,
            item_field_builders=self.item_field_builders,   # dataset builder
            bake_data=False,
            is_processed=False,
            specs=specs,
            batch_size=batch_size,
            max_retries=max_retries,
            reasoning=reasoning_param,
            verbose_llm=verbose_llm,
            fields_builder=effective_fields_builder,        # builder per spec
            error_dir=run_dir,
            output_path=output_paths["out_json"],
        )

        # FLATTEN
        rows, col_meta = flatten_results_to_rows_and_meta(
            results, specs_by_name=specs, judge_name=resolved_run_name
        )

        save_csv_and_meta(
            rows, col_meta,
            csv_path=output_paths["out_csv"],
            meta_path=output_paths["meta_json"]
        )

        return {
            "run_dir": run_dir,
            "run_name": resolved_run_name,
            "out_json": output_paths["out_json"],
            "out_csv": output_paths["out_csv"],
            "meta_json": output_paths["meta_json"],
        }
