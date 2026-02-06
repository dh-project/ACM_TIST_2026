from __future__ import annotations
from pathlib import Path
from typing import Dict, Any
import yaml

from agentic_forge.model_forge.large_model_forge import LargeModelForge
from agentic_forge.workflow_forge.structured_output_workflow_forge.reasoning_policy import ReasoningPolicy
from agentic_forge.workflow_forge.structured_output_workflow_forge.structured_output_workflow import StructuredOutputWorkflow


class StructuredOutputWorkflowForge:
    """
    Factory specifica per creare workflow di Structured Output.

    - Carica LargeModelForge tramite YAML (llm_manager + model_keys)
    - Mantiene una ReasoningPolicy condivisa
    - Permette di istanziare workflow multipli:
        workflow = forge.build_workflow_from_yaml("my_workflow.yaml")

    Ogni workflow è indipendente, configurabile, ed esegue .run().
    """

    def __init__(self, model_forge: LargeModelForge, reasoning_policy: ReasoningPolicy):
        self.model_forge = model_forge
        self.reasoning_policy = reasoning_policy

    # ------------------------------------------------------------
    # STATIC FACTORY: costruisce la forge caricando YAML del model forge
    # ------------------------------------------------------------
    @classmethod
    def forge(
        cls,
        llm_manager_yaml: str | Path,
        model_keys_yaml: str | Path,
        reasoning_policy: ReasoningPolicy
    ) -> "StructuredOutputWorkflowForge":
        """
        Crea la factory caricando tutto ciò che serve ai modelli:
          - LLM Manager YAML
          - Model Key YAML
        """

        model_forge = LargeModelForge.forge(
            llm_config=llm_manager_yaml,
            model_key_config=model_keys_yaml,
        )

        return cls(model_forge=model_forge, reasoning_policy=reasoning_policy)

    # ------------------------------------------------------------
    # CREA UN WORKFLOW DA YAML (non esegue nulla!)
    # ------------------------------------------------------------
    def forge_structured_output_workflow(self, workflow_yaml_path: str | Path,field_builders) -> StructuredOutputWorkflow:
        """
        Carica la configurazione di UN workflow, istanzia un StructuredOutputWorkflow,
        lo configura e lo ritorna.
        """

        with open(workflow_yaml_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        if "workflow" not in cfg:
            raise ValueError("The YAML config must contain a top-level `workflow:` section.")

        wf_cfg: Dict[str, Any] = cfg["workflow"]

        workflow = StructuredOutputWorkflow(
            model_forge=self.model_forge,
            reasoning_policy=self.reasoning_policy,
            item_field_builders=field_builders   # <-- Nome corretto
        )


        workflow.configure_from_dict(wf_cfg)

        return workflow
