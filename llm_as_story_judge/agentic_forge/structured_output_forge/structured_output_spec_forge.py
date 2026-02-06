from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union

from pydantic import BaseModel
from langchain_core.prompts import ChatPromptTemplate
import yaml

from agentic_forge.structured_output_forge.structured_output_schema_forge import (
    forge_structured_output_schema,
    DEFAULT_JSON_INSTRUCTIONS,
    FieldSpec,
)

from agentic_forge.prompt_forge.chat_prompt_template_forge import (
    forge_chat_prompt_template,
)


@dataclass
class StructuredOutputSpec:
    name: str
    schema: type[BaseModel]
    format_instruction: Optional[str]
    prompt: ChatPromptTemplate
    schema_path: Path
    prompt_path: Path
    fields_spec: List[FieldSpec]


def _collect_yaml_paths(base: Path, pattern="*.yaml") -> List[Path]:
    if base.is_file():
        return [base]
    if base.is_dir():
        return sorted(base.glob(pattern))
    raise FileNotFoundError(f"Path non valido: {base}")


# =====================================================================
#   🚀 NUOVA VERSIONE — FULL SPEC×PROMPT COMBINATOR
# =====================================================================

def forge_structured_output_specs(
    schemas_path: Union[str, Path],
    prompts_path: Union[str, Path],
    pattern: str = "*.yaml",
    with_format_instruction: bool = True,
) -> Dict[str, StructuredOutputSpec]:

    schemas_path = Path(schemas_path)
    prompts_path = Path(prompts_path)

    schema_files = _collect_yaml_paths(schemas_path, pattern)
    prompt_files = _collect_yaml_paths(prompts_path, pattern)

    # ------------------------------------------------------------
    # 1) Raggruppa SCHEMI per spec_name
    # ------------------------------------------------------------
    schemas_by_spec: Dict[str, List[tuple]] = {}

    for sp in schema_files:
        raw = yaml.safe_load(sp.read_text("utf-8"))

        spec_name = raw.get("spec_name", raw["model_name"])
        model_name = raw["model_name"]

        fields_spec = [FieldSpec(**fd) for fd in raw["fields"]]

        model_cls, fmt = forge_structured_output_schema(
            model_name=model_name,
            fields_spec=fields_spec,
            json_instructions=DEFAULT_JSON_INSTRUCTIONS,
        )

        schemas_by_spec.setdefault(spec_name, []).append(
            (model_name, model_cls, fmt, sp, fields_spec)
        )

    # ------------------------------------------------------------
    # 2) Raggruppa PROMPT per spec_name
    # ------------------------------------------------------------
    prompts_by_spec: Dict[str, List[tuple]] = {}

    for pp in prompt_files:
        prompt = forge_chat_prompt_template(pp)

        raw = yaml.safe_load(pp.read_text("utf-8"))
        spec_name = raw.get("spec_name", getattr(prompt, "prompt_name"))

        prompts_by_spec.setdefault(spec_name, []).append(
            (prompt, pp, getattr(prompt, "prompt_name"))
        )

    # ------------------------------------------------------------
    # 3) Genera tutte le combinazioni schema × prompt
    # ------------------------------------------------------------
    specs: Dict[str, StructuredOutputSpec] = {}

    for spec_name, schema_list in schemas_by_spec.items():

        if spec_name not in prompts_by_spec:
            print(f"[WARNING] Nessun prompt per spec_name='{spec_name}'")
            continue

        for (model_name, model_cls, fmt, sp, fields_spec) in schema_list:
            for (prompt, pp, prompt_name) in prompts_by_spec[spec_name]:

                # Nome univoco della spec
                spec_id = f"{model_name}__{prompt_name}"

                specs[spec_id] = StructuredOutputSpec(
                    name=spec_id,
                    schema=model_cls,
                    prompt=prompt,
                    format_instruction=fmt if with_format_instruction else None,
                    schema_path=sp,
                    prompt_path=pp,
                    fields_spec=fields_spec,
                )

    if not specs:
        print("[forge_structured_output_specs] WARNING: nessuna spec generata.")

    return specs
