from pathlib import Path
from typing import Dict, Union


class LargeModelKeyForge:
    """
    Carica e valida una semplice struttura:
    
        {
            "kimi":     {"provider": "openrouter", "model": "qwen/qwq-32b"},
            "deepseek": {"provider": "openai",     "model": "gpt-4.1"}
        }

    Non crea enum.
    Non trasforma i dati.
    Li ritorna così come sono.

    Scopo: fornire uno strato uniforme per fornire ModelKey → {provider, model}.
    """

    def __init__(self, mapping: Dict[str, Dict[str, str]]):
        self.mapping = mapping

    # ------------------------------------------------------------
    # LOAD YAML OR DICT
    # ------------------------------------------------------------
    @staticmethod
    def _load(config: Union[str, Path, dict]) -> Dict[str, Dict[str, str]]:
        """
        Carica YAML o dict. NON richiede 'models:'.
        Deve essere del formato:
        
            {"key": {"provider": "...", "model": "..."}}
        """
        if isinstance(config, (str, Path)):
            import yaml
            with open(config, "r") as f:
                config = yaml.safe_load(f)

        if not isinstance(config, dict):
            raise ValueError("ModelKey config must be a dict or YAML mapping.")

        # validazione minima
        for k, v in config.items():
            if not isinstance(v, dict):
                raise ValueError(f"Entry `{k}` must map to a dict.")
            if "provider" not in v or "model" not in v:
                raise ValueError(
                    f"Entry `{k}` must contain keys 'provider' and 'model'."
                )

        return config

    # ------------------------------------------------------------
    # STATIC FACTORY
    # ------------------------------------------------------------
    @classmethod
    def forge(cls, config: Union[str, Path, dict]):
        mapping = cls._load(config)
        return cls(mapping)

    # ------------------------------------------------------------
    # ACCESSOR
    # ------------------------------------------------------------
    def get(self, key: str) -> Dict[str, str]:
        """
        Ritorna {"provider": ..., "model": ...} per la chiave logica richiesta.
        """
        if key not in self.mapping:
            raise KeyError(f"ModelKey `{key}` non definito.")
        return self.mapping[key]
