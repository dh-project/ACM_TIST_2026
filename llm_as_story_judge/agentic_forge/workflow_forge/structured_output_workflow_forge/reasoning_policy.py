"""
Modulo reasoning policy per workflow.

Contiene:

- ReasoningPolicy         → classe base astratta
- DefaultReasoningPolicy  → implementazione attuale (euristiche)

Il workflow usa queste classi per determinare il payload "reasoning"
da passare al modello strutturato.

Il comportamento può essere completamente sostituito creando nuove policy.
"""

from __future__ import annotations
from typing import Any


class ReasoningPolicy:
    """
    Classe base astratta per una reasoning policy.

    Ogni reasoning policy deve implementare:

        resolve(llm, model_name: str, enabled: bool) -> Any

    Parametri:
    - llm:       istanza del modello (ChatModel, ecc.)
    - model_name: nome reale del modello (es. "deepseek-r1")
    - enabled:  flag globale dell'utente (--reasoning-enabled)
    """

    def resolve(self, llm, model_name: str, enabled: bool) -> Any:
        raise NotImplementedError(
            "Override `resolve()` nella tua policy reasoning."
        )
