from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Union

import yaml
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    AIMessagePromptTemplate,
)
from pydantic import BaseModel


def _build_message_template(role: str, content: str):
    """
    Traduce il role dello YAML in un message template LangChain.
    Supporta: system, human/user, ai/assistant.
    """
    r = role.lower()
    if r == "system":
        return SystemMessagePromptTemplate.from_template(content)
    if r in ("human", "user"):
        return HumanMessagePromptTemplate.from_template(content)
    if r in ("ai", "assistant"):
        return AIMessagePromptTemplate.from_template(content)

    # Se vuoi gestire ruoli custom, puoi usare ChatMessagePromptTemplate:
    # from langchain.prompts.chat import ChatMessagePromptTemplate
    # return ChatMessagePromptTemplate.from_template(role=role, template=content)
    raise ValueError(f"Unsupported role in YAML messages: {role!r}")


class ForgeChatPromptTemplate(ChatPromptTemplate):
    """
    Estensione di ChatPromptTemplate che include metadati di forge:

    - prompt_name: nome logico del prompt (dallo YAML)
    - description: descrizione testuale
    - template_fields: lista di placeholder attesi nel template
    """

    prompt_name: str = ""
    description: str = ""
    template_fields: List[str] = []


def forge_chat_prompt_template(
    yaml_path: Union[str, Path],
) -> ForgeChatPromptTemplate:
    """
    Carica uno YAML di descrizione prompt nel formato:

        prompt_name: ...
        description: ...
        template_fields: [...]
        messages:
          - role: system
            content: "..."
          - role: human
            content: "..."

    e restituisce un ForgeChatPromptTemplate (sottoclasse di ChatPromptTemplate)
    pronto da .format / .format_messages.
    """
    path = Path(yaml_path)
    with path.open("r", encoding="utf-8") as f:
        data: Dict[str, Any] = yaml.safe_load(f)

    prompt_name: str = data.get("prompt_name", "")
    description: str = data.get("description", "")
    template_fields: List[str] = data.get("template_fields", []) or []
    messages_cfg = data.get("messages", [])

    if not isinstance(messages_cfg, list) or not messages_cfg:
        raise ValueError(f"{yaml_path}: campo 'messages' mancante o vuoto")

    # Costruisci i message template di LangChain
    msg_templates = []
    for m in messages_cfg:
        try:
            role = m["role"]
            content = m["content"]
        except KeyError as e:
            raise ValueError(
                f"Ogni voce in 'messages' deve avere 'role' e 'content'. "
                f"Errore: {e}"
            )
        msg_templates.append(_build_message_template(role=role, content=content))

    # Prima costruiamo un ChatPromptTemplate "normale"
    base_prompt = ChatPromptTemplate.from_messages(msg_templates)

    # Poi creiamo la nostra sottoclasse con gli stessi dati + metadati
    prompt = ForgeChatPromptTemplate(
        messages=base_prompt.messages,
        input_variables=base_prompt.input_variables,
        prompt_name=prompt_name,
        description=description,
        template_fields=template_fields,
    )

        # (Facoltativo) sanity check: tutte le template_fields devono comparire nel template
    missing: List[str] = []
    for field in template_fields:
        needle = "{" + field + "}"
        found = False
        for m in msg_templates:
            # SystemMessagePromptTemplate / HumanMessagePromptTemplate / AIMessagePromptTemplate
            # hanno un attributo .prompt che è un PromptTemplate con .template (stringa)
            prompt_obj = getattr(m, "prompt", None)
            template_str = getattr(prompt_obj, "template", "") if prompt_obj is not None else ""
            if needle in template_str:
                found = True
                break
        if not found:
            missing.append(field)

    if missing:
        # qui puoi decidere se loggare, warnare o alzare
        raise ValueError(f"template_fields non trovati nei messaggi: {missing}")
    
    return prompt

