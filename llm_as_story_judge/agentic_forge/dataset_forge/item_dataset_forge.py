# agentic_forge/forge_item_dataset.py

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Union

from torch.utils.data import Dataset


TransformFn = Callable[[Mapping[str, Any]], Any]
FieldBuilders = Mapping[str, TransformFn]


class ItemDataset(Dataset):
    """
    Dataset generico per strutture "keyed" tipo:

        {
            "item_id_1": { ... raw item ... },
            "item_id_2": { ... },
            ...
        }

    Può lavorare in due modalità:

    1) RAW + field_builders (online o bake):
       - data        = {item_id -> raw_item}
       - field_builders = {field_name -> callable(raw_item) -> value}
       - se bake_data = False: le trasformazioni vengono applicate ad ogni __getitem__
       - se bake_data = True: tutte le trasformazioni vengono applicate una volta sola
         in fase di init e si lavora solo col dataset processato.

    2) DATASET GIÀ PROCESSATO:
       - is_processed = True
       - data = {item_id -> {field_name -> value}}
       - nessun field_builder usato, __getitem__ ritorna direttamente i dict processati.
    """

    def __init__(
        self,
        data: Mapping[str, Any],
        field_builders: Optional[FieldBuilders] = None,
        bake_data: bool = False,
        is_processed: bool = False,
    ) -> None:
        if not isinstance(data, Mapping):
            raise ValueError(
                "ItemDataset expects a mapping {item_id -> item_data}."
            )

        self._keys: List[str] = list(data.keys())

        # Se is_processed=True, trattiamo data come dataset già processato
        if is_processed:
            # item_id -> dict(field_name -> value)
            self._processed_data: Dict[str, Dict[str, Any]] = dict(data)  # type: ignore[assignment]
            self._raw_data: Optional[Dict[str, Any]] = None
            self._field_builders: Dict[str, TransformFn] = {}
            return

        # Caso normale: abbiamo raw_data e (opzionalmente) field_builders
        self._raw_data: Optional[Dict[str, Any]] = dict(data)
        self._field_builders: Dict[str, TransformFn] = dict(field_builders or {})
        self._processed_data: Optional[Dict[str, Dict[str, Any]]] = None

        if bake_data:
            # Pre-calcola tutti i field e lavora solo sul dataset processato
            self._processed_data = self._bake_all()
            self._raw_data = None  # opzionale: liberiamo i raw se non servono più

    # ------------------------------------------------------------------
    # API base
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._keys)

    def keys(self) -> List[str]:
        """Lista ordinata degli item_id."""
        return list(self._keys)

    def resolve_key(self, idx: Union[int, str]) -> str:
        """
        Converte un indice (posizione int o item_id str) in una chiave stringa.
        """
        if isinstance(idx, int):
            try:
                return self._keys[idx]
            except IndexError:
                raise IndexError(f"Index {idx} out of range (len={len(self._keys)})")

        if isinstance(idx, str):
            if idx not in self._keys:
                raise KeyError(f"Item id '{idx}' not found in dataset")
            return idx

        raise TypeError("Index must be an int (position) or str (item_id).")

    def get_raw_item(self, key: str) -> Mapping[str, Any]:
        """
        Ritorna il raw item (se ancora disponibile).
        """
        if self._raw_data is None:
            raise RuntimeError("Raw data non disponibile (dataset già baked o caricato processato).")
        return self._raw_data[key]

    def to_dict_raw(self) -> Optional[Dict[str, Any]]:
        """
        Ritorna il dizionario raw {item_id -> raw_item}, se disponibile.
        """
        if self._raw_data is None:
            return None
        return dict(self._raw_data)

    def to_dict_processed(self) -> Optional[Dict[str, Dict[str, Any]]]:
        """
        Ritorna il dizionario processato {item_id -> {field_name -> value}}, se disponibile.
        Se non è mai stato baked, ritorna None.
        """
        if self._processed_data is None:
            return None
        return dict(self._processed_data)

    # ------------------------------------------------------------------
    # Core: trasformazione e __getitem__
    # ------------------------------------------------------------------
    def __getitem__(self, idx: Union[int, str]) -> Dict[str, Any]:
        key = self.resolve_key(idx)

        # 1) Se abbiamo dati processati, li usiamo direttamente
        if self._processed_data is not None:
            return self._processed_data[key]

        # 2) Altrimenti lavoriamo sul raw + field_builders
        if self._raw_data is None:
            raise RuntimeError(
                "Né raw_data né processed_data disponibili: stato inconsistente."
            )

        raw_item = self._raw_data[key]

        # Se non ci sono field_builders, ritorniamo il raw item così com'è
        if not self._field_builders:
            # type: ignore[return-value]
            return raw_item  # Dict[str, Any]

        # Applichiamo le trasformazioni dinamicamente
        out: Dict[str, Any] = {
            field_name: builder(raw_item)
            for field_name, builder in self._field_builders.items()
        }
        return out

    # ------------------------------------------------------------------
    # Baking interno
    # ------------------------------------------------------------------
    def _bake_all(self) -> Dict[str, Dict[str, Any]]:
        if self._raw_data is None:
            raise RuntimeError("Impossibile fare bake: raw_data mancante.")
        if not self._field_builders:
            # Se non ci sono field_builders, interpretiamo raw_data come già processato
            return {
                key: dict(self._raw_data[key])  # type: ignore[dict-item]
                for key in self._keys
            }

        processed: Dict[str, Dict[str, Any]] = {}
        for key in self._keys:
            raw_item = self._raw_data[key]
            processed[key] = {
                field_name: builder(raw_item)
                for field_name, builder in self._field_builders.items()
            }
        return processed

    # ------------------------------------------------------------------
    # Persistenza del dataset processato
    # ------------------------------------------------------------------
    def save_processed(self, path: Union[str, Path], indent: int = 2) -> None:
        """
        Salva su disco il dataset processato in JSON:

            { item_id -> {field_name -> value} }

        Se il dataset non è ancora stato baked, lo bake-a al volo
        (senza modificare lo stato interno, a parte eventualmente _processed_data).
        """
        if self._processed_data is None:
            # Se non è baked, calcoliamo ora i processati e li memorizziamo
            self._processed_data = self._bake_all()

        out_data = self._processed_data
        path = Path(path)
        with path.open("w", encoding="utf-8") as f:
            json.dump(out_data, f, ensure_ascii=False, indent=indent)

    @classmethod
    def load_processed(cls, path: Union[str, Path]) -> "ItemDataset":
        """
        Carica un dataset GIÀ processato da un JSON del tipo:

            { item_id -> {field_name -> value} }

        e ritorna un ItemDataset con is_processed=True.
        """
        path = Path(path)
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, dict):
            raise ValueError(
                "ItemDataset.load_processed si aspetta un JSON con top-level object "
                "{item_id: {field_name: value}}."
            )

        return cls(data=data, field_builders=None, bake_data=False, is_processed=True)


# ----------------------------------------------------------------------
# Helper: inizializzare da path JSON
# ----------------------------------------------------------------------
def forge_item_dataset(
    json_path: Union[str, Path],
    field_builders: Optional[FieldBuilders] = None,
    bake_data: bool = False,
    is_processed: bool = False,
) -> ItemDataset:
    """
    Crea un ItemDataset a partire da un file JSON.

    - json_path: path al file JSON.
      Se is_processed=False, il formato atteso è:
          { item_id -> raw_item }
      Se is_processed=True, il formato atteso è:
          { item_id -> {field_name -> value} }

    - field_builders: dict {field_name -> callable(raw_item) -> value}
      usato solo se is_processed=False.

    - bake_data:
      * False -> applica i field_builders on-the-fly in __getitem__
      * True  -> pre-calcola tutti i field in init e usa solo il dataset processato

    - is_processed:
      * False -> JSON considerato raw_data
      * True  -> JSON considerato già dataset processato
    """
    path = Path(json_path)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError(
            "forge_item_dataset si aspetta un JSON con top-level object {item_id: ...}."
        )

    return ItemDataset(
        data=data,
        field_builders=field_builders,
        bake_data=bake_data,
        is_processed=is_processed,
    )
