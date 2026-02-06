from reasoning_policy import ReasoningPolicy

def build_review(raw):
    return raw["review"]

def build_product_info(raw):
    return raw["product_info"]

# ⚠️ QUESTO VA USATO SOLO PER forge_item_dataset
DATASET_FIELD_BUILDERS = {
    "review": build_review,
    "product_info": build_product_info,
}




class DefaultReasoningPolicy(ReasoningPolicy):
    """
    Implementazione predefinita della reasoning policy.
    Mantiene la logica esistente, ma ora è modulare e sostituibile.

    Gestisce i casi:
    - Modello con reasoning built-in (Claude 3.7 → sempre attivo)
    - Modello con reasoning controllabile (DeepSeek / Kimi)
    - Modello senza reasoning (raise se enabled=True)
    """

    def resolve(self, llm, model_name: str, enabled: bool):
        """
        Ritorna il payload reasoning corretto (None, dict, str, bool).
        """

        # ------------------------------------------------------------
        # 1) Modelli con reasoning BUILT-IN
        # ------------------------------------------------------------
        # Es: Claude 3.7 ha reasoning integrato non disattivabile
        if hasattr(llm, "mode") and getattr(llm, "mode") == "builtin":
            if not enabled:
                raise ValueError(
                    f"Model '{model_name}' has built-in reasoning that cannot be disabled."
                )
            return {}  # payload standard built-in

        # ------------------------------------------------------------
        # 2) Modelli con reasoning CONTROLLABILE
        # ------------------------------------------------------------
        # Es: DeepSeek, Kimi, GPT-OSS-LW ecc.
        if hasattr(llm, "mode") and getattr(llm, "mode") == "controllable":
            if not enabled:
                return None  # reasoning disattivato
            # euristiche personalizzate
            if model_name.startswith("deepseek"):
                return True
            if "kimi" in model_name.lower():
                return {"enabled": True}
            if "gpt-oss" in model_name.lower():
                return "medium"
            # fallback generico
            return True

        # ------------------------------------------------------------
        # 3) Modelli SENZA reasoning
        # ------------------------------------------------------------
        if enabled:
            raise ValueError(
                f"Model '{model_name}' does NOT support reasoning. Disable it."
            )

        return None
