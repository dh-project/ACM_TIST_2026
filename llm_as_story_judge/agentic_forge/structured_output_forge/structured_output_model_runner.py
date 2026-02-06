# structured_output_model_runner.py

from __future__ import annotations

from pathlib import Path
import re
import traceback
from typing import (
    Any,
    Dict,
    Optional,
    Tuple,
    Union,
    Type,
    List,
    Mapping,
    Sequence,
    Callable,
)

from collections.abc import Mapping as ABCMapping, Sequence as ABCSequence

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage
from pydantic import BaseModel, ValidationError
from tqdm.auto import tqdm  # <-- progresso carino

from agentic_forge.structured_output_forge.structured_output_spec_forge import (
    StructuredOutputSpec,
    forge_structured_output_specs,
)

from agentic_forge.graph_forge.workflow_forge.parallel_map_workflow_forge import (
    forge_parallel_map_workflow,
    run_parallel_map_workflow,
)

from agentic_forge.dataset_forge.item_dataset_forge import (
    forge_item_dataset,
    FieldBuilders,
)


# ============================================================================
# CORE: UN SOLO JOB (schema + prompt + fields)
# ============================================================================


def _run_single_structured_output_job(
    llm: BaseChatModel,
    schema: Type[BaseModel],
    prompt: Union[ChatPromptTemplate, str],
    fields: Optional[Dict[str, Any]] = None,
    reasoning: Optional[Union[bool, Dict[str, Any]]] = None,
    max_retries: int = 0,
    verbose: bool = False,
    job_tag: Optional[str] = None,
) -> Tuple[Dict[str, Any], AIMessage, Optional[Any]]:
    """
    Esegue UN singolo job di output strutturato (schema + prompt + fields).

    - Usa llm.with_structured_output(schema, include_raw=True)
    - Costruisce i messaggi da ChatPromptTemplate o da stringa
    - Ritenta fino a max_retries volte solo su errori di parsing/validazione
    - A ogni retry aggiunge:
        * il raw precedente (se disponibile)
        * un HumanMessage con la descrizione dell'errore
    """
    if fields is None:
        fields = {}

    # Modello con structured output
    llm_structured = llm.with_structured_output(schema, include_raw=True)

    # Costruzione dei messaggi base
    if isinstance(prompt, ChatPromptTemplate):
        base_messages: List[BaseMessage] = prompt.format_messages(**fields)
        if verbose:
            tag = f"[{job_tag}] " if job_tag else ""
            print(f"\n=== {tag}Initial formatted prompt (messages) ===")
            for m in base_messages:
                print(f"{tag}[{m.type.upper()}] {m.content}\n")
    else:
        text = prompt.format(**fields) if fields else prompt
        base_messages = [HumanMessage(content=text)]
        if verbose:
            tag = f"[{job_tag}] " if job_tag else ""
            print(f"\n=== {tag}Initial formatted prompt (text) ===")
            print(tag + text)
            print()

    raw_msg: Optional[AIMessage] = None
    parsed: Optional[Any] = None
    parsing_error: Optional[Any] = None

    def _format_messages_for_log(msgs: List[BaseMessage]) -> str:
        return "\n".join(f"[{m.type.upper()}] {m.content}" for m in msgs)

    # Loop tentativi: al massimo max_retries + 1
    for attempt in range(max_retries + 1):
        tag = f"[{job_tag}][attempt={attempt}] " if job_tag else f"[attempt={attempt}] "

        if attempt == 0:
            messages = list(base_messages)
            if verbose:
                print(f"{tag}--- initial ---")
        else:
            error_text = str(parsing_error) if parsing_error is not None else "Unknown parsing error."
            correction_message = HumanMessage(
                content=(
                    "Your previous response could not be parsed into the required JSON "
                    "schema. Here is the parsing error:\n\n"
                    f"{error_text}\n\n"
                    "Please return ONLY a JSON object that strictly follows the schema, "
                    "with no explanations or extra text."
                )
            )
            messages = list(base_messages)
            if raw_msg is not None:
                messages.append(raw_msg)
            else:
                messages.append(
                    AIMessage(
                        content=(
                            "I had some problem generating the correct JSON schema. "
                            "Please pay close attention to the error description."
                        )
                )
            )
            messages.append(correction_message)

            if verbose:
                print(f"{tag}--- retry ---")
                for m in messages:
                    print(f"{tag}[{m.type.upper()}] {m.content}\n")

        prompt_preview = _format_messages_for_log(messages)
        parsing_failed = False
        resp = None

        try:
            if reasoning is not None:
                resp = llm_structured.invoke(messages, reasoning=reasoning)
            else:
                resp = llm_structured.invoke(messages)
            if verbose:
                print(f"{tag}=== RAW PROVIDER FULL ===")
                print(resp)
        except ValidationError as e:
            parsing_error = e
            raw_msg = None
            parsed = None
            parsing_failed = True
        except Exception as e:
            raise RuntimeError(
                f"LLM call failed (job_tag={job_tag}, attempt={attempt}): {e}\n"
                f"Formatted prompt:\n{prompt_preview}"
            ) from e
        else:
            raw_msg = resp.get("raw")
            parsed = resp.get("parsed")
            parsing_error = resp.get("parsing_error", None)
            parsing_failed = parsing_error is not None

        if verbose:
            print(f"{tag}parsing_error = {parsing_error}")

        if not parsing_failed:
            break

    # Normalizzazione del parsed in dict
    if isinstance(parsed, BaseModel):
        parsed_dict: Dict[str, Any] = parsed.model_dump()
    elif isinstance(parsed, dict):
        parsed_dict = parsed
    elif parsed is None:
        parsed_dict = {}
    else:
        parsed_dict = dict(parsed)

    if raw_msg is None:
        raw_msg = AIMessage(content="No raw message available.")

    return parsed_dict, raw_msg, parsing_error


# ============================================================================
# RUNNER PER UNO O PIÙ StructuredOutputSpec (SERIALE SULLE SPEC)
# ============================================================================


def run_structured_output_specs(
    llm: BaseChatModel,
    specs: Union[
        StructuredOutputSpec,
        Sequence[StructuredOutputSpec],
        Mapping[str, StructuredOutputSpec],
    ],
    fields_by_name: Optional[Mapping[str, Dict[str, Any]]] = None,
    reasoning: Optional[Union[bool, Dict[str, Any]]] = None,
    max_retries: int = 0,
    verbose: bool = False,
) -> Dict[str, Tuple[Dict[str, Any], AIMessage, Optional[Any]]]:
    """
    Esegue uno o più task strutturati a partire da StructuredOutputSpec su UN item.

    `specs` può essere:
      - una singola StructuredOutputSpec
      - una Sequence[StructuredOutputSpec]
      - un Mapping[str, StructuredOutputSpec] (es. dict name -> spec)

    `fields_by_name` ha la forma:
      {
        spec_name_1: { field1: ..., field2: ... },
        spec_name_2: { ... },
      }

    Ritorna:
      dict name -> (parsed_dict, raw_msg, parsing_error)
    """
    # Normalizza specs in lista
    if isinstance(specs, StructuredOutputSpec):
        specs_list: List[StructuredOutputSpec] = [specs]
    elif isinstance(specs, ABCMapping):
        specs_list = list(specs.values())
    elif isinstance(specs, ABCSequence):
        specs_list = list(specs)
    else:
        raise TypeError(
            f"'specs' deve essere StructuredOutputSpec, Sequence[StructuredOutputSpec] "
            f"o Mapping[str, StructuredOutputSpec], non {type(specs)!r}"
        )

    fields_by_name = fields_by_name or {}

    results: Dict[str, Tuple[Dict[str, Any], AIMessage, Optional[Any]]] = {}

    for spec in specs_list:
        # fields per questa spec
        spec_fields = dict(fields_by_name.get(spec.name, {}))

        # se la spec ha una format_instruction e l’utente non l’ha messa,
        # la iniettiamo noi
        if spec.format_instruction is not None and "format_instruction" not in spec_fields:
            spec_fields["format_instruction"] = spec.format_instruction

        parsed_dict, raw_msg, parsing_error = _run_single_structured_output_job(
            llm=llm,
            schema=spec.schema,
            prompt=spec.prompt,
            fields=spec_fields,
            reasoning=reasoning,
            max_retries=max_retries,
            verbose=verbose,
            job_tag=spec.name,
        )

        results[spec.name] = (parsed_dict, raw_msg, parsing_error)

    return results


# ============================================================================
# BUILDER DI DEFAULT PER fields_by_name + estrazione id
# ============================================================================

def _default_fields_builder(
    item: Mapping[str, Any],
    specs_by_name: Mapping[str, StructuredOutputSpec],
) -> Dict[str, Dict[str, Any]]:
    """
    Per ogni spec:
      - legge spec.prompt.input_variables
      - costruisce fields_by_name[spec_name] = {var → item[var]}
    """

    out = {}

    for spec_name, spec in specs_by_name.items():
        needed_vars = spec.prompt.input_variables  # ← già fornito dal prompt YAML
        spec_fields = {}

        for var in needed_vars:
            if var == "format_instruction":
                # viene aggiunta dopo in run_structured_output_specs
                continue

            if var not in item:
                raise KeyError(
                    f"Il dataset non contiene la variabile '{var}' "
                    f"richiesta dal prompt della spec '{spec_name}'."
                )

            spec_fields[var] = item[var]

        out[spec_name] = spec_fields

    return out



def _extract_item_id(item: Any) -> Optional[str]:
    """
    Heuristica per estrarre un id dall'item.

    - se è un mapping, prova 'id' o 'story_id'
    - altrimenti prova attributi .id o .story_id
    - altrimenti None
    """
    if isinstance(item, Mapping):
        if "id" in item:
            return str(item["id"])
        if "story_id" in item:
            return str(item["story_id"])

    for attr in ("id", "story_id"):
        if hasattr(item, attr):
            try:
                return str(getattr(item, attr))
            except Exception:
                continue

    return None


def _safe_item_id_for_filename(item_id: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", item_id.strip())
    return cleaned or "unknown"


def _write_item_error_file(error_dir: Path, item_id: str, error_text: str) -> Path:
    error_dir.mkdir(parents=True, exist_ok=True)
    safe_id = _safe_item_id_for_filename(item_id)
    error_path = error_dir / f"workflow_{safe_id}_error.txt"
    with error_path.open("w", encoding="utf-8") as f:
        f.write(error_text)
    return error_path


# ============================================================================
# WORKER PER UNA SINGOLA STORIA (TUTTE LE SPEC)
# ============================================================================


def _multi_spec_worker(
    item: Any,
    llm: BaseChatModel,
    specs_by_name: Mapping[str, StructuredOutputSpec],
    max_retries: int = 0,
    reasoning: Optional[Union[bool, Dict[str, Any]]] = None,
    verbose_llm: bool = False,
    error_dir: Optional[Path] = None,
    fields_builder: Callable[
        [Mapping[str, Any], Mapping[str, StructuredOutputSpec]],
        Dict[str, Dict[str, Any]],
    ] = _default_fields_builder,
) -> Dict[str, Any]:
    """
    Applica run_structured_output_specs a UN item del dataset, su TUTTE le spec.

    Ritorna un record:

        {
          "id": ...,
          "results": { spec_name -> parsed_dict },
          "errors":  { spec_name -> error_str or None }
        }
    """
    # normalizza item in mapping
    if isinstance(item, Mapping):
        item_map: Mapping[str, Any] = item
    else:
        if hasattr(item, "dict"):
            item_map = item.dict()  # type: ignore[assignment]
        else:
            item_map = getattr(item, "__dict__", {})

    item_id = _extract_item_id(item_map) or "<unknown>"

    try:
        # 1) fields_by_name per questo item
        fields_by_name = fields_builder(item_map, specs_by_name)

        # 2) esecuzione sulle spec
        out = run_structured_output_specs(
            llm=llm,
            specs=specs_by_name,
            fields_by_name=fields_by_name,
            reasoning=reasoning,
            max_retries=max_retries,
            verbose=verbose_llm,
        )
    except Exception as e:
        error_text = "".join(traceback.format_exception(e))
        error_path: Optional[Path] = None
        if error_dir is not None:
            error_path = _write_item_error_file(error_dir, item_id, error_text)

        return {
            "id": _extract_item_id(item),
            "results": {},
            "errors": {},
            "error": str(error_path) if error_path is not None else None,
        }

    # 3) spacchetta risultati/errori
    results_by_spec: Dict[str, Dict[str, Any]] = {}
    errors_by_spec: Dict[str, Optional[str]] = {}

    for name, (parsed_dict, _raw_msg, parsing_error) in out.items():
        results_by_spec[name] = parsed_dict
        errors_by_spec[name] = str(parsing_error) if parsing_error is not None else None

    record: Dict[str, Any] = {
        "id": _extract_item_id(item),
        "results": results_by_spec,
        "errors": errors_by_spec,
        "error": None,
    }
    return record


# ============================================================================
# FUNZIONE AD ALTO LIVELLO: UNICA API PER L'ESPERIMENTO
# ============================================================================


def run_structured_output_over_item_dataset(
    llm: BaseChatModel,
    *,
    # dataset: o lo passi già pronto...
    dataset: Optional[Sequence[Any]] = None,
    # ...oppure lo forgiamo noi da JSON
    json_path: Optional[Union[str, Path]] = None,
    item_field_builders: Optional[FieldBuilders] = None,
    bake_data: bool = False,
    is_processed: bool = False,
    # specs: direttamente o via schemas/prompts
    specs: Union[
        StructuredOutputSpec,
        Sequence[StructuredOutputSpec],
        Mapping[str, StructuredOutputSpec],
        None,
    ] = None,
    schemas_path: Optional[Union[str, Path]] = None,
    prompts_path: Optional[Union[str, Path]] = None,
    with_format_instruction: bool = True,
    batch_size: int = 8,
    max_retries: int = 0,
    reasoning: Optional[Union[bool, Dict[str, Any]]] = None,
    verbose_llm: bool = False,
    fields_builder: Callable[
        [Mapping[str, Any], Mapping[str, StructuredOutputSpec]],
        Dict[str, Dict[str, Any]],
    ] = _default_fields_builder,
    error_dir: Optional[Union[str, Path]] = None,
    output_path: Optional[Union[str, Path]] = None,
) -> List[Dict[str, Any]]:
    """
    High-level API per applicare N StructuredOutputSpec a un intero ItemDataset,
    sfruttando il parallel map di LangGraph (parallelismo sui dati).

    Puoi:
      - passare direttamente `dataset` (ItemDataset o qualunque Sequence),
      - oppure farlo forgiare automaticamente da un JSON usando forge_item_dataset.

    Parametri
    ---------
    llm:
        Modello chat (BaseChatModel).

    dataset:
        Dataset già caricato (ItemDataset o qualunque Sequence di item).
        Se è None, viene usato json_path + forge_item_dataset.

    json_path:
        Path al JSON da cui forgiare l'ItemDataset se `dataset` è None.

    item_field_builders:
        Dict {field_name -> callable(raw_item) -> value} passato a forge_item_dataset.
        Usato solo se json_path è non None e is_processed=False.

    bake_data:
        Parametro di forge_item_dataset (precalcolo o meno dei campi).

    is_processed:
        Parametro di forge_item_dataset (JSON già processato o raw).

    specs:
        Se non None:
          - una singola StructuredOutputSpec
          - una Sequence di StructuredOutputSpec
          - un Mapping name -> StructuredOutputSpec
        Se None: DEVI passare schemas_path e prompts_path.

    schemas_path / prompts_path:
        Usati solo se specs è None. Possono essere file singoli o directory.

    with_format_instruction:
        Se True, usa le format_instruction generate dagli schema YAML.
        Se False, nelle spec .format_instruction sarà None.

    batch_size:
        max_concurrency per il parallel map (limita le chiamate LLM in parallelo).

    max_retries:
        Numero di retry per errori di parsing per ogni spec e per ogni item.

    reasoning:
        Payload reasoning passato alle chiamate LLM strutturate
        (es. {"enabled": True}) oppure None.

    verbose_llm:
        Se True, stampa prompt e retry dal runner strutturato.

    fields_builder:
        Funzione che costruisce fields_by_name per un item, a partire da:
          - item_map (Mapping)
          - specs_by_name (dict name -> StructuredOutputSpec)
        Default = _default_fields_builder (usa chiave == nome spec).

    error_dir:
        Se non None, scrive un file di errore per item fallito in questa directory.

    output_path:
        Se non None: salva i risultati finali (lista di record) in JSON.

    Ritorna
    -------
    results:
        Lista di dict per item:

          {
            "id": ...,
            "results": { spec_name -> parsed_dict },
            "errors":  { spec_name -> error_str or None }
          }
    """
    # ------------------------------------------------------------------
    # 0) Dataset: o lo passi già pronto, o lo forgiamo da JSON
    # ------------------------------------------------------------------
    if dataset is not None and json_path is not None:
        raise ValueError(
            "Passa *o* 'dataset' *o* 'json_path', non entrambi. "
            "Se hai già un ItemDataset, usa solo 'dataset'."
        )

    if dataset is None:
        if json_path is None:
            raise ValueError(
                "Devi fornire almeno uno tra 'dataset' e 'json_path'. "
                "Per forgiare automaticamente il dataset serve json_path."
            )
        dataset_obj = forge_item_dataset(
            json_path=json_path,
            field_builders=item_field_builders,
            bake_data=bake_data,
            is_processed=is_processed,
        )
    else:
        dataset_obj = dataset

    # ------------------------------------------------------------------
    # 1) Normalizza / costruisci specs_by_name
    # ------------------------------------------------------------------
    if specs is None:
        if schemas_path is None or prompts_path is None:
            raise ValueError(
                "Se 'specs' è None devi passare sia 'schemas_path' che 'prompts_path'."
            )
        specs_dict: Dict[str, StructuredOutputSpec] = forge_structured_output_specs(
            schemas_path,
            prompts_path,
            pattern="*.yaml",
            with_format_instruction=with_format_instruction,
        )
    else:
        if isinstance(specs, StructuredOutputSpec):
            specs_list: List[StructuredOutputSpec] = [specs]
        elif isinstance(specs, ABCMapping):
            specs_list = list(specs.values())
        elif isinstance(specs, ABCSequence):
            specs_list = list(specs)
        else:
            raise TypeError(
                f"'specs' deve essere StructuredOutputSpec, Sequence[StructuredOutputSpec] "
                f"o Mapping[str, StructuredOutputSpec], non {type(specs)!r}"
            )
        specs_dict = {s.name: s for s in specs_list}

    if not specs_dict:
        raise ValueError("Nessuna StructuredOutputSpec disponibile (specs_dict è vuoto).")

    # ------------------------------------------------------------------
    # 2) Costruisci lista di item per il worker, preservando gli id
    # ------------------------------------------------------------------
    # Se il dataset ha un metodo .keys() (come ItemDataset), usiamo quello
    # per prendere la chiave come id; altrimenti accettiamo la Sequence così com'è.
    items_list: List[Any] = []

    if hasattr(dataset_obj, "keys") and callable(getattr(dataset_obj, "keys")):
        # Dataset "keyed" tipo ItemDataset: {item_id -> raw_item}
        # Vogliamo che ogni item passato al worker abbia un 'id' ricavato dalla chiave.
        keys = list(dataset_obj.keys())  # type: ignore[attr-defined]
        for key in keys:
            item_data = dataset_obj[key]  # type: ignore[index]
            if isinstance(item_data, Mapping):
                merged = dict(item_data)
                # se l'item non ha già un id esplicito, usiamo la chiave del dataset
                merged.setdefault("id", key)
            else:
                # caso di fallback: impacchettiamo tutto
                merged = {"id": key, "data": item_data}
            items_list.append(merged)
    else:
        # Qualunque Sequence "normale": ci fidiamo che contenga già id dentro.
        items_list = list(dataset_obj)

    num_items = len(items_list)
    num_specs = len(specs_dict)

    print(
        f"[RUN] Structured output over dataset — items={num_items}, specs={num_specs}, "
        f"batch_size={batch_size}"
    )

    # ------------------------------------------------------------------
    # 3) Costruisci workflow parallelo
    # ------------------------------------------------------------------
    wf = forge_parallel_map_workflow(
        _multi_spec_worker,
        name="structured_output_over_item_dataset",
        llm=llm,
        specs_by_name=specs_dict,
        max_retries=max_retries,
        reasoning=reasoning,
        verbose_llm=verbose_llm,
        error_dir=Path(error_dir) if error_dir is not None else None,
        fields_builder=fields_builder,
    )

    # ------------------------------------------------------------------
    # 4) Esegui parallel map sul dataset con tqdm a livello di batch
    # ------------------------------------------------------------------
    all_results: List[Dict[str, Any]] = []

    if num_items == 0:
        print("[RUN] Dataset vuoto: nessun item da processare.")
    else:
        with tqdm(
            total=num_items,
            desc="Structured outputs",
            unit="item",
        ) as pbar:
            for start in range(0, num_items, batch_size):
                end = min(start + batch_size, num_items)
                batch = items_list[start:end]

                # fan-out/fan-in parallelo sul batch corrente
                batch_results: List[Dict[str, Any]] = run_parallel_map_workflow(
                    wf,
                    items=batch,
                    max_concurrency=batch_size,
                )

                all_results.extend(batch_results)
                pbar.update(len(batch))
                pbar.set_postfix_str(f"processed={len(all_results)}/{num_items}")

    # ------------------------------------------------------------------
    # 5) Salva JSON se richiesto
    # ------------------------------------------------------------------
    if output_path is not None:
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        import json

        with out_path.open("w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)

        print(f"[RUN] Risultati salvati in: {out_path}")

    return all_results
