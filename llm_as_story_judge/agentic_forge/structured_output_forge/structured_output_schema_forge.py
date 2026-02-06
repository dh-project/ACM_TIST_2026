# agentic_forge/structured_output/model_forge.py

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Literal, Type

import yaml
from pydantic import BaseModel, Field, create_model
from .format_instruction_forge import DEFAULT_JSON_INSTRUCTIONS,forge_json_format_instruction
from .field_spec import FieldSpec




    


def _literal_from_choices(choices: List[Any]):
    """
    Crea un Literal dinamico a partire da una lista di valori (stringhe o int).

    Esempi:
        ["low", "medium", "high"] -> Literal["low", "medium", "high"]
        [1, 2, 3]                 -> Literal[1, 2, 3]
    """
    from typing import Literal as _Literal

    return _Literal.__getitem__(tuple(choices))


def forge_structured_output_schema(
    model_name: str,
    fields_spec: List[FieldSpec],
    json_instructions: str = DEFAULT_JSON_INSTRUCTIONS,
) -> tuple[Type[BaseModel], str]:
    """
    Costruisce dinamicamente:
      - un BaseModel Pydantic per output strutturato LLM
      - la JSON format instruction associata

    Ritorna (ModelClass, FormatInstruction).
    """

    fields: dict[str, tuple[type, Field]] = {}

    for c in fields_spec:
        field_name = c.name
        title = c.label or c.name.replace("_", " ").title()
        desc = c.description or ""

        if c.kind == "float":
            fields[field_name] = (
                float,
                Field(..., ge=c.min, le=c.max, title=title, description=desc),
            )

        elif c.kind == "int":
            fields[field_name] = (
                int,
                Field(..., ge=c.min, le=c.max, title=title, description=desc),
            )

        elif c.kind == "bool":
            fields[field_name] = (
                bool,
                Field(..., title=title, description=desc),
            )

        elif c.kind == "text":
            fields[field_name] = (
                str,
                Field(..., title=title, description=desc),
            )

        elif c.kind == "enum":
            if not c.choices:
                raise ValueError(
                    f"Field '{c.name}' ha kind='enum' ma 'choices' è vuoto o mancante."
                )
            lit_type = _literal_from_choices(c.choices)
            enum_desc = desc or (
                "Allowed values: " + ", ".join(str(v) for v in c.choices)
            )
            fields[field_name] = (
                lit_type,
                Field(..., title=title, description=enum_desc),
            )

        elif c.kind == "object":
            if not c.fields:
                raise ValueError(
                    f"Field '{c.name}' ha kind='object' ma 'fields' è vuoto o mancante."
                )
            sub_model_name = f"{model_name}{field_name.capitalize()}"
            sub_model, _ = forge_structured_output_schema(
                sub_model_name,
                c.fields,
                json_instructions=json_instructions,
            )
            fields[field_name] = (
                sub_model,
                Field(..., title=title, description=desc),
            )

        else:
            raise ValueError(f"Unsupported kind '{c.kind}' for field '{c.name}'")

    # 🟢 1. Creiamo il modello
    StructuredModel: Type[BaseModel] = create_model(model_name, **fields)  # type: ignore[arg-type]

    # 🟢 2. Generiamo la format-instruction
    format_instruction = forge_json_format_instruction(
        fields_spec=fields_spec,
        json_instructions=json_instructions,
    )

    return StructuredModel, format_instruction

def forge_structured_output_schema_from_yaml(
    path: str | Path,
    json_instructions: str = DEFAULT_JSON_INSTRUCTIONS,
) -> tuple[Type[BaseModel], str]:
    """
    Legge uno YAML e produce:
       - il modello Pydantic
       - le JSON format instructions

    Ritorna (ModelClass, FormatInstruction).
    """

    path = Path(path)
    data = yaml.safe_load(path.read_text(encoding="utf-8"))

    model_name = data["model_name"]
    fields_data = data["fields"]

    fields_spec = [FieldSpec(**fd) for fd in fields_data]

    return forge_structured_output_schema(
        model_name=model_name,
        fields_spec=fields_spec,
        json_instructions=json_instructions,
    )
