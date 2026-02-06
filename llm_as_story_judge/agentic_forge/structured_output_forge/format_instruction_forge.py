import json
import yaml
from typing import Any
from .field_spec import FieldSpec
from pathlib import Path

DEFAULT_JSON_INSTRUCTIONS = """\
Output only a valid JSON object.
Do not include any text before or after the JSON.
Do not add fields not defined in the schema.
Use exactly the field names provided.
Your output must start with '{{' and end with '}}'.

Follow this JSON structure (replace placeholders with real values):
<example name=json_formatting>
{example_json}
</example>
"""



def _placeholder_value_for_field(field: FieldSpec) -> Any:
    """
    Genera placeholder descrittivi usando SOLO il nome del campo (`name`),
    che è l’unica chiave effettiva nel JSON. Niente `label`.
    """
    fn = field.name  # più leggibile

    if field.kind == "float":
        if field.min is not None or field.max is not None:
            return (
                f"<float value for '{fn}'"
                f"{' min=' + str(field.min) if field.min is not None else ''}"
                f"{' max=' + str(field.max) if field.max is not None else ''}>"
            )
        return f"<float value for '{fn}'>"

    if field.kind == "int":
        if field.min is not None or field.max is not None:
            return (
                f"<integer value for '{fn}'"
                f"{' min=' + str(field.min) if field.min is not None else ''}"
                f"{' max=' + str(field.max) if field.max is not None else ''}>"
            )
        return f"<integer value for '{fn}'>"

    if field.kind == "bool":
        return f"<true or false for '{fn}'>"

    if field.kind == "enum":
        if field.choices:
            return (
                f"<one of the allowed values for '{fn}': "
                f"{', '.join(map(str, field.choices))}>"
            )
        return f"<enum value for '{fn}'>"

    if field.kind == "text":
        return f"<text for '{fn}'>"

    if field.kind == "object":
        if not field.fields:
            return {}
        return _placeholder_dict_from_fields(field.fields)

    return f"<value for '{fn}'>"


def _placeholder_dict_from_fields(fields_spec: list[FieldSpec]) -> dict[str, Any]:
    """
    Costruisce un dizionario di esempio a partire da una lista di FieldSpec,
    usando SOLO placeholder descrittivi.
    """
    example: dict[str, Any] = {}
    for f in fields_spec:
        example[f.name] = _placeholder_value_for_field(f)
    return example



def forge_json_format_instruction(
    fields_spec: list[FieldSpec],
    json_instructions: str =DEFAULT_JSON_INSTRUCTIONS,
) -> str:
    """
    Genera una stringa di format instructions per un LLM-judge che deve
    produrre SOLO un JSON conforme al modello Pydantic generato da `fields_spec`.

    - `model_name`: nome del modello (classe Pydantic)
    - `fields_spec`: specifiche dei campi
    - `instructions`: testo naturale (inglese) con le regole del compito deve contenere la variabile dinamica {example_json}; se None,
      viene usato DEFAULT_JSON_INSTRUCTIONS.
    """
    example_payload = _placeholder_dict_from_fields(fields_spec)
    example_json = json.dumps(example_payload, indent=2, ensure_ascii=False)

    # Tabellina dei campi con tipo, descrizione, vincoli
    field_lines: list[str] = []
    for f in fields_spec:
        title = f.label or f.name
        desc = f.description or ""
        kind = f.kind
        line = f"- `{f.name}` ({kind}) — {title}"
        if desc:
            line += f": {desc}"
        if f.measurement_type:
            line += f"  [measurement_type={f.measurement_type}]"
        if kind == "enum" and f.choices:
            line += f"  Allowed values: {', '.join(map(str, f.choices))}."
        if kind in {"float", "int"} and (f.min is not None or f.max is not None):
            line += (
                f"  Allowed range:"
                f"{' min=' + str(f.min) if f.min is not None else ''}"
                f"{' max=' + str(f.max) if f.max is not None else ''}."
            )
        field_lines.append(line)

    format_instruction=json_instructions.format(example_json=example_json)
    return format_instruction


def forge_json_format_instruction_from_yaml(
    path: str | Path,
    json_instructions: str = DEFAULT_JSON_INSTRUCTIONS,
) -> str:
    """
    Legge uno YAML con:
      - model_name: ...
      - fields: [ ... FieldSpec ... ]

    Estrae model_name e fields, costruisce i FieldSpec e
    genera la stringa di JSON format instructions usando
    `forge_json_format_instruction`.

    Lo YAML deve avere almeno:
      model_name: SomeModelName
      fields:
        - name: ...
          kind: ...
          ...
    """
    path = Path(path)
    data = yaml.safe_load(path.read_text(encoding="utf-8"))

    fields_data = data["fields"]

    fields_spec = [FieldSpec(**fd) for fd in fields_data]

    return forge_json_format_instruction(
        fields_spec=fields_spec,
        json_instructions=json_instructions,
    )
