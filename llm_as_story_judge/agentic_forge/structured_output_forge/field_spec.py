# agentic_forge/structured_output/field_spec.py

from pydantic import BaseModel, Field
from typing import Literal, List, Any, Optional


class FieldSpec(BaseModel):
    """
    Specifica di un singolo campo di output strutturato.
    Usata dal builder per generare i campi del BaseModel.

    Nota:
      - `kind` descrive il tipo di dato JSON / Pydantic (struttura).
      - `measurement_type` (alias YAML: `type`) descrive la scala di misura
        o la natura statistica del campo (nominale, ordinale, numerico, testo...).
    """

    name: str                          # nome del campo nel modello (es. "correctness")
    label: Optional[str] = None        # titolo umano (opzionale)
    description: Optional[str] = None  # descrizione per il prompt / JSON schema

    kind: Literal[
        "float",
        "int",
        "bool",
        "enum",     # tipo discreto con choices esplicite
        "text",
        "object",
    ] = "float"

    # scala di misura / natura statistica (opzionale)
    # YAML key: `type`
    measurement_type: Optional[
        Literal[
            "categorical_nominal",
            "categorical_ordinal",
            "numeric_discrete",
            "numeric_continuous",
            "free_text",
            "other",
        ]
    ] = Field(default=None, alias="type")

    # per float/int
    min: float | None = None
    max: float | None = None

    # per enum: possono essere stringhe o interi nello YAML
    choices: List[Any] | None = None

    # per oggetti annidati
    fields: List["FieldSpec"] | None = None  # usata solo se kind == "object"

    class Config:
        # permette di usare sia il nome del campo Python (`measurement_type`)
        # sia l'alias YAML (`type`)
        allow_population_by_field_name = True
        populate_by_name = True
