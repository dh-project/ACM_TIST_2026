import os
import yaml
from pathlib import Path
from typing import Dict, Literal, Optional, Union,Any
from pydantic import model_validator, ValidationError

from langchain_openai.chat_models import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_ollama.chat_models import ChatOllama
from langchain_ollama.embeddings import OllamaEmbeddings

from agentic_forge.configs.base_config import BaseConfig
import warnings
import jsonschema






# ------------------------------------------------------------
# ENUMS
# ------------------------------------------------------------
from enum import Enum

class ProviderType(str, Enum):
    OPENAI = "openai"
    OLLAMA = "ollama"
    OPENROUTER= "openrouter"

class ModelType(str, Enum):
    CHAT = "chat"
    EMBEDDING = "embedding"


# ------------------------------------------------------------
# MODEL CONFIGS
# ------------------------------------------------------------
from typing import Optional, Dict, Any, Union

class ChatModelConfig(BaseConfig):
    name: str
    max_tokens: Optional[int] = None
    temperature: float = 0.0
    top_p: float = 1.0
    support_tools_parallel: bool = False
    support_tool_choice: Union[bool,str] = False
    tool_choice: Union[dict, str, bool] = False


    router: Optional[str] = None     
    context: Optional[int] = None     

    support_reasoning: Optional[Dict[str, Any]] = None

    support_structured_output: Union[bool,str]="n/d"



class EmbeddingModelConfig(BaseConfig):
    name: str
    embedding_chunk_size: Optional[int] = 1

class ModelConfig(BaseConfig):
    """
    ModelConfig definisce un singolo modello, di tipo 'chat' o 'embedding',
    e incapsula la relativa config (ChatModelConfig o EmbeddingModelConfig).
    """
    type: ModelType
    config: Union[ChatModelConfig, EmbeddingModelConfig]

    @model_validator(mode="before")
    @classmethod
    def coerce_config(cls, data):
        type_ = data.get("type")
        config_data = data.get("config")
        if config_data is None:
            raise ValueError(f"Missing 'config' for model type '{type_}'.")

        constructor_map = {
            ModelType.CHAT.value: ChatModelConfig,
            ModelType.EMBEDDING.value: EmbeddingModelConfig,
        }
        constructor = constructor_map.get(type_)
        if constructor is None:
            raise ValueError(f"Unsupported model type '{type_}'.")

        try:
            data["config"] = constructor(**config_data)
        except ValidationError as e:
            details = "\n".join(f"- {'.'.join(map(str, err['loc']))}: {err['msg']}"
                                for err in e.errors())
            raise ValueError(
                f"Invalid fields in 'config' for model type '{type_}':\n{details}"
            ) from e

        return data


# ------------------------------------------------------------
# PROVIDER CONFIGS
# ------------------------------------------------------------
class OpenAIProviderConfig(BaseConfig):
    api_key_file: str
    models: Dict[str, ModelConfig]

class OllamaProviderConfig(BaseConfig):
    base_url: str
    keep_alive: Optional[str] = None
    models: Dict[str, ModelConfig]

class OpenRouterProviderConfig(BaseConfig):  # <-- NEW
    base_url: str = "https://openrouter.ai/api/v1"
    api_key_file: str
    # campi opzionali per header raccomandati da OpenRouter
    site_url: Optional[str] = None   # es. "https://example.com"
    app_name: Optional[str] = None   # es. "A.D.A.M.O."
    models: Dict[str, ModelConfig]

class ProviderConfig(BaseConfig):
    """
    ProviderConfig incapsula:
      - type: 'openai' o 'ollama' o 'openrouter'
      - config: a sua volta OpenAIProviderConfig o OllamaProviderConfig o OpenRouterProviderConfig
    """
    type: ProviderType
    config: Union[OpenAIProviderConfig, OllamaProviderConfig, OpenRouterProviderConfig]

    @model_validator(mode="before")
    @classmethod
    def coerce_config(cls, data):
        type_ = data.get("type")
        config_data = data.get("config")
        if config_data is None:
            raise ValueError(f"Missing 'config' for provider type '{type_}'.")

        constructor_map = {
            ProviderType.OPENAI.value: OpenAIProviderConfig,
            ProviderType.OLLAMA.value: OllamaProviderConfig,
            ProviderType.OPENROUTER.value: OpenRouterProviderConfig
        }
        constructor = constructor_map.get(type_)
        if constructor is None:
            raise ValueError(f"Unsupported provider type '{type_}'.")

        try:
            data["config"] = constructor(**config_data)
        except ValidationError as e:
            details = "\n".join(f"- {'.'.join(map(str, err['loc']))}: {err['msg']}"
                                for err in e.errors())
            raise ValueError(
                f"Invalid fields in 'config' for provider type '{type_}':\n{details}"
            ) from e

        return data


# ------------------------------------------------------------
# LLM MANAGER CONFIG
# ------------------------------------------------------------
class LlmManagerConfig(BaseConfig):


    providers: Dict[str, ProviderConfig]

# ------------------------------------------------------------
# REASONING CHAT MODEL
# ------------------------------------------------------------


class ReasoningChatModel:
    def __init__(self, base_model, reasoning_config):
        self.base_model = base_model
        self.controllable = reasoning_config.get("controllable", True)
        self.api_param = reasoning_config.get("api_param", "reasoning")
        self.schema = reasoning_config.get("schema") or {}
        self.mode = "controllable" if self.controllable else "builtin"

    def _validate(self, r):
        if self.schema:
            jsonschema.validate(r, self.schema)
        return r

    def _default_builtin(self):
        if self.schema.get("type") == "object":
            return {}
        return None

    def invoke(self, messages, **kwargs):

        user_r = kwargs.pop("reasoning", None)

        # -------- CONTROLLABLE --------
        if self.mode == "controllable":
            if user_r is None:
                return self.base_model.invoke(messages, **kwargs)
            payload = self._validate(user_r)

        # -------- BUILTIN (Claude 3.7) --------
        else:
            if user_r is not None:
                raise ValueError("Built-in reasoning: do NOT pass `reasoning`.")

            payload = self._default_builtin()
            if payload is None:      # e.g. future models
                return self.base_model.invoke(messages, **kwargs)

        # INIETTA nel modo IDENTICO al base model
        kwargs[self.api_param] = payload
        return self.base_model.invoke(messages, **kwargs)

    def __getattr__(self, n):
        return getattr(self.base_model, n)
    
class NoReasoningGuard:
    def __init__(self, base_model, model_name):
        self.base_model = base_model
        self.model_name = model_name

    def invoke(self, messages, **kwargs):
        if "reasoning" in kwargs:
            raise ValueError(
                f"Model '{self.model_name}' does NOT support reasoning. "
                f"Do NOT pass `reasoning`."
            )
        return self.base_model.invoke(messages, **kwargs)

    def __getattr__(self, name):
        return getattr(self.base_model, name)

# ------------------------------------------------------------
# LLM MANAGER (Main class)
# ------------------------------------------------------------
class LlmManager:
    """
    Manager unificato per modelli OpenAI e Ollama.
    Ora utilizza un dizionario di provider, ciascuno con chiave arbitraria.
    """

    def __init__(self, config: LlmManagerConfig):
        self.config = config
        self.providers: Dict[str, ProviderConfig] = self.config.providers

        self._initialize_env()
        self._log_default_warnings()  # NUOVO: warning con contesto

    def _log_default_warnings(self) -> None:
        """
        Scansiona provider e modelli, e logga warning SOLO per i campi
        che NON sono stati specificati nell'input (YAML) e quindi stanno
        usando il default di classe.

        Usa `__fields_set__` di Pydantic:
        - __fields__: tutti i field del modello
        - __fields_set__: field effettivamente passati in input

        Se un field è in __fields__ ma NON in __fields_set__,
        vuol dire che non era nel file e il valore viene dal default.
        """
        for provider_key, provider in self.providers.items():
            models = getattr(provider.config, "models", {}) or {}
            for model_key, model_entry in models.items():
                cfg = model_entry.config

                # Pydantic v1: __fields_set__ esiste
                fields_set = getattr(cfg, "__fields_set__", set())

                for name, field in getattr(cfg, "__fields__", {}).items():
                    # Se il field era nel dict di input, NON warnare
                    if name in fields_set:
                        continue

                    default = field.default
                    # Se proprio non ha default (es. required puro), skippa
                    if default is None:
                        continue

                    warnings.warn(
                        f"[Provider={provider_key}][Model={model_key}][{cfg.__class__.__name__}] "
                        f"field '{name}' not specified in config; using default: {default!r}",
                        UserWarning,
                    )

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "LlmManager":
        config = LlmManagerConfig.from_yaml(path)
        return cls(config)

    def _initialize_env(self):
        for provider_key, provider in self.providers.items():

            if provider.type == ProviderType.OPENAI:
                # Se il percorso non è assoluto, lo trasformiamo
                api_path = provider.config.api_key_file
                if not os.path.isabs(api_path):
                    api_path = os.path.abspath(api_path)
                with open(api_path, 'r') as f:
                    api = yaml.safe_load(f)
                    os.environ["OPENAI_API_KEY"] = api["openai_api_key"]

            elif provider.type == ProviderType.OLLAMA:
                if provider.config.keep_alive:
                    os.environ["OLLAMA_KEEP_ALIVE"] = provider.config.keep_alive

            elif provider.type == ProviderType.OPENROUTER:
                api_path = provider.config.api_key_file
                if not os.path.isabs(api_path):
                    api_path = os.path.abspath(api_path)
                with open(api_path, 'r') as f:
                    api = yaml.safe_load(f)
                    # Manteniamo separata la key per chiarezza
                    os.environ["OPENROUTER_API_KEY"] = api.get("openrouter_api_key") or api.get("api_key") or ""
                # Nessuna variabile d'ambiente "obbligatoria" sugli header: li passiamo al build

    def get_model(self, provider_key: str, model_key: str, **kwargs):
        """
        Restituisce un modello (Chat o Embedding) per il provider identificato da `provider_key`
        (che è la chiave del dizionario `providers`), per il modello interno `model_key`.

        Esempio di chiamata:
            manager.get_model(provider_key="ollama_default", model_key="default_chat", stream=True)
        """
        if provider_key not in self.providers:
            raise ValueError(f"Provider `{provider_key}` non trovato fra: {list(self.providers.keys())}")

        provider_config = self.providers[provider_key]
        model_entry = provider_config.config.models.get(model_key)
        if model_entry is None:
            raise ValueError(
                f"Modello `{model_key}` non trovato sotto il provider `{provider_key}`. "
                f"Chiavi disponibili: {list(provider_config.config.models.keys())}"
            )

        model_type = model_entry.type
        model_cfg = model_entry.config

        if model_type == ModelType.CHAT:
            return self._build_chat(provider_config, model_cfg, **kwargs)
        elif model_type == ModelType.EMBEDDING:
            return self._build_embedding(provider_config, model_cfg, **kwargs)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    def _build_chat(
        self,
        provider_config: ProviderConfig,
        model_cfg: ChatModelConfig,
        json_mode: bool = False,
        stream: bool = False,
        **kwargs: Any,
    ):
        """
        Costruisce un ChatOpenAI o ChatOllama a seconda del provider.

        Nuova logica:
        - support_reasoning = None → modello NON wrappato ma con guardia anti-reasoning
        - support_reasoning ≠ None → ReasoningChatModel
        """

        params: Dict[str, Any] = {
            "model_name" if provider_config.type == ProviderType.OPENAI else "model": model_cfg.name,
            **kwargs,
        }

        if model_cfg.max_tokens is not None:
            params["max_tokens"] = model_cfg.max_tokens

        # Debug log of final params (once per build)
        try:
            print(
                f"[LLM BUILD] provider={provider_config.type} model={model_cfg.name} "
                f"json_mode={json_mode} stream={stream} params={params}"
            )
        except Exception:
            pass

        # -------- OPENAI --------
        if provider_config.type == ProviderType.OPENAI:
            if not model_cfg.name.startswith("gpt-5"):
                params["temperature"] = model_cfg.temperature
                params["top_p"] = model_cfg.top_p

            model_kwargs: Dict[str, Any] = {}
            if json_mode:
                model_kwargs["response_format"] = {"type": "json_object"}

            base_model = ChatOpenAI(
                **params,
                openai_api_key=os.getenv("OPENAI_API_KEY"),
                streaming=stream,
                model_kwargs=model_kwargs,
            )

        # -------- OLLAMA --------
        elif provider_config.type == ProviderType.OLLAMA:
            base_model = ChatOllama(
                **params,
                base_url=provider_config.config.base_url,
                format="json" if json_mode else "",
                stream=stream,
            )

        # -------- OPENROUTER --------
        elif provider_config.type == ProviderType.OPENROUTER:
            default_headers: Dict[str, str] = {}
            site_url = getattr(provider_config.config, "site_url", None)
            app_name = getattr(provider_config.config, "app_name", None)
            if site_url:
                default_headers["HTTP-Referer"] = site_url
            if app_name:
                default_headers["X-Title"] = app_name

            model_kwargs: Dict[str, Any] = {}
            if json_mode:
                model_kwargs["response_format"] = {"type": "json_object"}

            base_model = ChatOpenAI(
                **params,
                base_url=provider_config.config.base_url,
                openai_api_key=os.getenv("OPENROUTER_API_KEY"),
                streaming=stream,
                default_headers=default_headers,
                model_kwargs=model_kwargs,
            )

        else:
            raise ValueError(f"Unsupported provider type: {provider_config.type}")

        # -------- REASONING HANDLING --------
        reasoning_cfg = model_cfg.support_reasoning

        if reasoning_cfg is None:
            # Modello SENZA reasoning → installiamo la guardia
            return NoReasoningGuard(base_model, model_cfg.name)

        # Modello CON reasoning → usiamo ReasoningChatModel
        return ReasoningChatModel(
            base_model=base_model,
            reasoning_config=reasoning_cfg,
        )


    def _build_embedding(
        self,
        provider_config: ProviderConfig,
        model_cfg: EmbeddingModelConfig,
        **kwargs
    ):
        """
        Costruisce un OpenAIEmbeddings o OllamaEmbeddings a seconda di `provider_config.type`.
        """
        if provider_config.type == ProviderType.OPENAI:
            return OpenAIEmbeddings(
                model=model_cfg.name,
                chunk_size=model_cfg.embedding_chunk_size or 1,
                openai_api_key=os.getenv("OPENAI_API_KEY"),
                **kwargs
            )
        elif provider_config.type == ProviderType.OLLAMA:
            return OllamaEmbeddings(
                model=model_cfg.name,
                base_url=provider_config.config.base_url,
                **kwargs
            )
        elif provider_config.type == ProviderType.OPENROUTER:
            # API compatibile OpenAI per embeddings
            return OpenAIEmbeddings(
                model=model_cfg.name,
                chunk_size=model_cfg.embedding_chunk_size or 1,
                openai_api_key=os.getenv("OPENROUTER_API_KEY"),
                base_url=provider_config.config.base_url,
                **kwargs
            )
        else:
            raise ValueError(f"Unsupported provider type: {provider_config.type}")
        
    @classmethod
    def forge(cls, config_or_path):
        if isinstance(config_or_path, (str, Path)):
            config = LlmManagerConfig.from_yaml(config_or_path)
        elif isinstance(config_or_path, dict):
            config = LlmManagerConfig.from_dict(config_or_path)
        elif isinstance(config_or_path, LlmManagerConfig):
            config = config_or_path
        else:
            raise ValueError("Invalid LLM Manager configuration")

        return cls(config)


from typing import Union, Type, Dict, Tuple, Any
from pathlib import Path
from enum import Enum
from agentic_forge.model_forge.large_model_key_forge import LargeModelKeyForge


from agentic_forge.model_forge.large_model_key_forge import LargeModelKeyForge


class LargeModelForge:
    """
    Punto di accesso unificato per istanziare modelli (LLM/VLM/Embedding).

    Ha due modalità di utilizzo:

    1) INIT MANUALE:
        forge = LargeModelForge(manager, key_forge)

    2) INIT AUTOMATICA VIA STATIC FORGE:
        forge = LargeModelForge.forge(llm_config, model_key_config)

    Runtime API:
        forge_large_model("kimi") → istanza modello
    """

    def __init__(
        self,
        manager: LlmManager,
        key_forge: LargeModelKeyForge,
    ):
        """
        Inizializzazione manuale:
        - richiede LlmManager già costruito
        - richiede LargeModelKeyForge già costruito
        - usa direttamente key_forge.mapping senza copiarla
        """
        self.manager = manager
        self.key_forge = key_forge  # ← usato per leggere provider/model

    # ------------------------------------------------------------
    # STATIC FACTORY: costruisce tutto automaticamente
    # ------------------------------------------------------------
    @classmethod
    def forge(
        cls,
        llm_config: Union[str, Path, dict],
        model_key_config: Union[str, Path, dict],
    ):
        """
        Costruisce:
            - LlmManager tramite LlmManager.forge(...)
            - LargeModelKeyForge tramite LargeModelKeyForge.forge(...)
        E ritorna una istanza LargeModelForge completa.
        """

        manager = LlmManager.forge(llm_config)
        key_forge = LargeModelKeyForge.forge(model_key_config)

        return cls(manager=manager, key_forge=key_forge)

    # ------------------------------------------------------------
    # API RUNTIME
    # ------------------------------------------------------------
    def forge_large_model(self, key: str, **kwargs: Any):
        """
        Restituisce l'istanza del modello indicato dalla chiave logica.

        La chiave deve essere presente nel LargeModelKeyForge:
            key_forge.mapping[key] = {"provider": ..., "model": ...}

        Esempio:
            model = forge.forge_large_model("kimi")
        """
        info = self.key_forge.get(key)

        provider_key = info["provider"]
        model_key = info["model"]

        return self.manager.get_model(
            provider_key=provider_key,
            model_key=model_key,
            **kwargs
        )

