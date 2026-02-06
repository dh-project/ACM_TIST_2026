import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union,Callable

from torch.utils.data import Dataset, DataLoader 
from torch.utils.data import Subset

class StoryItem:
    """
    Wrapper per un singolo item del dataset (es. 'gem_1875_rivalita').

    Espone metodi di formattazione per:
    - reperto
    - fase_1
    - fase_2
    - storia
    - tabella_riassunto
    """

    def __init__(self, story_id: str, data: Dict[str, Any]) -> None:
        self.story_id = story_id
        self.data = data

    # -------------------------------
    # REPERTO
    # -------------------------------
    def format_reperto(
        self,
        include_catalogo: bool = True,
        include_descrizione: bool = True,
        include_descrizione_curatoriale: bool = True,
    ) -> str:
        reperto = self.data.get("reperto", {})
        parts: List[str] = []

        if include_catalogo and "id_catalogo" in reperto:
            parts.append(f"ID catalogo: {reperto['id_catalogo']}")

        if include_descrizione and "descrizione" in reperto:
            parts.append(f"Descrizione: {reperto['descrizione']}")

        if include_descrizione_curatoriale and "descrizione_curatoriale" in reperto:
            parts.append(f"Descrizione curatoriale: {reperto['descrizione_curatoriale']}")

        return "\n".join(parts).strip()

    # -------------------------------
    # FASE 1
    # -------------------------------
    def format_fase1(
        self,
        include_evento_centrale: bool = True,
        include_descrizione: bool = True,
        include_potenziale_narrativo: bool = True,
    ) -> str:
        fase_1 = self.data.get("fase_1", {})
        parts: List[str] = []

        if include_evento_centrale and "evento_centrale" in fase_1:
            parts.append(f"Evento centrale: {fase_1['evento_centrale']}")

        if include_descrizione and "descrizione" in fase_1:
            parts.append(f"Descrizione evento: {fase_1['descrizione']}")

        if include_potenziale_narrativo and "potenziale_narrativo" in fase_1:
            parts.append(f"Potenziale narrativo: {fase_1['potenziale_narrativo']}")

        return "\n".join(parts).strip()

    # -------------------------------
    # FASE 2
    # -------------------------------
    def format_fase2(
        self,
        include_titolo: bool = True,
        include_elementi_dinamici: bool = True,
        include_aderenza: bool = True,
    ) -> str:
        fase_2 = self.data.get("fase_2", {})
        parts: List[str] = []

        if include_titolo and "titolo" in fase_2:
            parts.append(f"Titolo fase 2: {fase_2['titolo']}")

        if include_elementi_dinamici and "elementi_dinamici" in fase_2:
            elementi = fase_2["elementi_dinamici"]
            if isinstance(elementi, list) and elementi:
                elem_lines = []
                for e in elementi:
                    elemento = e.get("elemento")
                    valore = e.get("valore")
                    if elemento is not None and valore is not None:
                        elem_lines.append(f"- {elemento}: {valore}")
                if elem_lines:
                    parts.append("Elementi dinamici:\n" + "\n".join(elem_lines))

        if include_aderenza and "aderenza" in fase_2:
            parts.append(f"Aderenza (evento originale): {fase_2['aderenza']}")

        return "\n".join(parts).strip()

    # -------------------------------
    # STORIA
    # -------------------------------
    def format_storia(
        self,
        include_titolo: bool = True,
        include_luogo: bool = True,
        include_tempo: bool = True,
        include_scene: bool = True,
        include_scene_numero: bool = True,
        include_scene_titolo: bool = True,
        include_scene_contenuto: bool = True,
    ) -> str:
        """
        Restituisce una stringa con tutte le componenti della storia
        (campo 'storia') in un formato leggibile, pensato per il valutatore.
        """
        storia = self.data.get("storia", {})
        parts: List[str] = []

        titolo = storia.get("titolo")
        if include_titolo and titolo:
            parts.append(f"Titolo della storia: {titolo}")

        luogo = storia.get("luogo")
        if include_luogo and luogo:
            parts.append(f"Luogo: {luogo}")

        tempo = storia.get("tempo")
        if include_tempo and tempo:
            parts.append(f"Tempo: {tempo}")

        if include_scene and "scene" in storia:
            parts.append("Scene:")
            scene_lines: List[str] = []
            for scene in storia["scene"]:
                line_parts: List[str] = []
                if include_scene_numero and "num" in scene:
                    line_parts.append(f"Scena {scene['num']}")
                if include_scene_titolo and "titolo" in scene:
                    # se abbiamo già messo "Scena X", aggiungo " - Titolo"
                    if line_parts:
                        line_parts[-1] = line_parts[-1] + f" - {scene['titolo']}"
                    else:
                        line_parts.append(scene["titolo"])
                if include_scene_contenuto and "contenuto" in scene:
                    scene_text = scene["contenuto"]
                    if line_parts:
                        scene_lines.append("".join(line_parts) + f":\n{scene_text}")
                    else:
                        scene_lines.append(scene_text)
                else:
                    if line_parts:
                        scene_lines.append("".join(line_parts))

            if scene_lines:
                parts.append("\n".join(scene_lines))

        return "\n\n".join(parts).strip()

    # -------------------------------
    # TABELLA RIASSUNTO (OPZIONALE)
    # -------------------------------
    def format_tabella_riassunto(
        self,
        include_header: bool = True,
    ) -> str:
        """
        Converte 'tabella_riassunto' (lista di liste) in una tabella testuale.
        Utile se vuoi passare anche questa al giudice oppure loggarla.
        """
        table = self.data.get("tabella_riassunto")
        if not table:
            return ""

        lines: List[str] = []
        rows = table

        start_idx = 0
        if not include_header and rows:
            start_idx = 1

        for i, row in enumerate(rows):
            if i < start_idx:
                continue
            # semplice join con |, ma puoi cambiarlo in formato markdown
            lines.append(" | ".join(str(cell) for cell in row))

        return "\n".join(lines).strip()
    
    def format_original_event(
        self,
        include_catalogo: bool = False,
        include_reperto_descrizione: bool = False,
        include_reperto_descrizione_curatoriale: bool = True,
        include_fase1_evento_centrale: bool = True,
        include_fase1_descrizione: bool = True,
        include_fase1_potenziale_narrativo: bool = False,
        include_periodo: bool = True,
        include_luogo: bool = True,
    ) -> str:
        """
        Restituisce il testo da inserire nel blocco <original_event> ... </original_event>
        per la rubrica storica.

        Deve contenere tutte le informazioni STORICHE utili a valutare:
        - plausibilità di eventi, personaggi, luoghi/ambientazioni;
        - aderenza all'evento originale.

        NON deve contenere la storia di finzione (che va in <story>)
        né metadati interni puramente narratologici (es. potenziale narrativo),
        se non esplicitamente richiesti.
        """
        reperto = self.data.get("reperto", {})
        fase_1 = self.data.get("fase_1", {})

        parts: List[str] = []

        # --- METADATA / CONTESTO STORICO (se disponibili come dati dell'evento reale) ---
        meta_lines: List[str] = []

        if include_catalogo and "id_catalogo" in reperto:
            meta_lines.append(f"ID catalogo: {reperto['id_catalogo']}")

        # Periodo / luogo dell'EVENTO STORICO (non della storia di finzione!)
        if include_periodo:
            periodo = (
                reperto.get("periodo")
                or fase_1.get("periodo")
                or self.data.get("periodo_evento")
            )
            if periodo:
                meta_lines.append(f"Periodo storico dell'evento: {periodo}")

        if include_luogo:
            luogo = (
                reperto.get("luogo")
                or fase_1.get("luogo")
                or self.data.get("luogo_evento")
            )
            if luogo:
                meta_lines.append(f"Luogo dell'evento: {luogo}")

        if meta_lines:
            parts.append("[METADATA_EVENTO_STORICO]")
            parts.append("\n".join(meta_lines))

        # --- DESCRIZIONI DEL REPERTO / EVENTO ORIGINALE ---
        descr_lines: List[str] = []
        if include_reperto_descrizione and "descrizione" in reperto:
            descr_lines.append(f"Descrizione: {reperto['descrizione']}")
        if include_reperto_descrizione_curatoriale and "descrizione_curatoriale" in reperto:
            descr_lines.append(f"Descrizione curatoriale: {reperto['descrizione_curatoriale']}")

        if descr_lines:
            parts.append("[DESCRIZIONE_REPERTO]")
            parts.append("\n".join(descr_lines))

        # --- RIASSUNTO STRUTTURATO DELL'EVENTO (fase_1) ---
        fase1_lines: List[str] = []
        if include_fase1_evento_centrale and "evento_centrale" in fase_1:
            fase1_lines.append(f"Evento centrale: {fase_1['evento_centrale']}")
        if include_fase1_descrizione and "descrizione" in fase_1:
            fase1_lines.append(f"Descrizione evento: {fase_1['descrizione']}")
        if include_fase1_potenziale_narrativo and "potenziale_narrativo" in fase_1:
            fase1_lines.append(f"Potenziale narrativo (dato interno): {fase_1['potenziale_narrativo']}")

        if fase1_lines:
            parts.append("[RIASSUNTO_EVENTO_ORIGINALE]")
            parts.append("\n".join(fase1_lines))

        return "\n\n".join(parts).strip()
    
    def __str__(self) -> str:
        """
        Rappresentazione testuale completa dello StoryItem.

        Include:
        - reperto
        - fase_1
        - fase_2
        - storia
        - tabella_riassunto (se presente)

        Pensata per logging / debug / stampa a video.
        """
        parts: List[str] = [f"Story ID: {self.story_id}"]

        reperto_str = self.format_reperto()
        if reperto_str:
            parts.append("=== REPERTO ===")
            parts.append(reperto_str)

        fase1_str = self.format_fase1()
        if fase1_str:
            parts.append("=== FASE 1 ===")
            parts.append(fase1_str)

        fase2_str = self.format_fase2()
        if fase2_str:
            parts.append("=== FASE 2 ===")
            parts.append(fase2_str)

        storia_str = self.format_storia()
        if storia_str:
            parts.append("=== STORIA ===")
            parts.append(storia_str)

        tabella_str = self.format_tabella_riassunto()
        if tabella_str:
            parts.append("=== TABELLA RIASSUNTO ===")
            parts.append(tabella_str)

        return "\n\n".join(parts).strip()



class DatasetStories(Dataset):
    """
    Dataset di storie storiche/drammatiche nel formato del JSON che hai fornito.

    - Caricamento da:
      - dizionario Python (già caricato)
      - path a file JSON

    - Indicizzazione:
      - dataset[i] -> StoryItem (i-esima storia, ordinata per chiave)
      - dataset["gem_1875_rivalita"] -> StoryItem corrispondente

    """

    def __init__(self, source: Union[Dict[str, Any], str, Path]) -> None:
        if isinstance(source, (str, Path)):
            path = Path(source)
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            data = source

        if not isinstance(data, dict):
            raise ValueError("StoryDataset expects a dict or a JSON file with a top-level object.")

        self._raw_data: Dict[str, Any] = data
        # manteniamo un ordine stabile per le chiavi
        self._keys: List[str] = list(data.keys())
        self._items: Dict[str, StoryItem] = {
            key: StoryItem(key, data[key]) for key in self._keys
        }

    def __len__(self) -> int:
        return len(self._keys)

    def __getitem__(self, idx: Union[int, str]) -> StoryItem:
        if isinstance(idx, int):
            key = self._keys[idx]
            return self._items[key]
        elif isinstance(idx, str):
            return self._items[idx]
        else:
            raise TypeError("Index must be an int (posizione) or str (story_id).")

    def keys(self) -> List[str]:
        return list(self._keys)

    def get(self, story_id: str) -> Optional[StoryItem]:
        return self._items.get(story_id)

    def to_dict(self) -> Dict[str, Any]:
        """Ritorna il dizionario grezzo (tutto il dataset)."""
        return self._raw_data

    def save_subset(self, keys: List[str], output_path: Union[str, Path]) -> None:
        """
        Salva un nuovo dataset JSON contenente solo le chiavi richieste.
        """
        output = {}
        for key in keys:
            if key not in self._raw_data:
                raise KeyError(f"Key not found in dataset: {key}")
            output[key] = self._raw_data[key]

        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
    
def _limit_dataloader(dataloader, limit: int):
    
    subset = Subset(dataloader.dataset, range(limit))
    return DataLoader(
        subset,
        batch_size=dataloader.batch_size,
        shuffle=False,
        num_workers=dataloader.num_workers,
        collate_fn=dataloader.collate_fn,
        pin_memory=dataloader.pin_memory,
    )
    

def forge_dataloader_stories(
    json_path: Union[str, Path],
    batch_size: Union[int,None] = 4,
    num_workers: int = 0,
    shuffle: bool = False,
    pin_memory: bool = False,
    drop_last: bool = False,
    collate_fn: Optional[Callable[[List[StoryItem]], Any]] = None,
    limit: Optional[int] = None,
) -> DataLoader:
    """
    Crea un DataLoader PyTorch a partire da un file JSON di storie.

    Parametri
    ---------
    json_path:
        Path al file JSON del dataset. Verrà caricato in un DatasetStories.

    batch_size:
        Numero di StoryItem per batch.

    num_workers:
        Numero di worker per il caricamento:
        - 0  -> tutto nel processo principale
        - >0 -> multiprocess (su Windows: ricordati il guard `if __name__ == "__main__":`)

    shuffle:
        Se True, mescola l'ordine degli indici a ogni epoch.
        Default: False (come richiesto).

    pin_memory:
        Se True, usa pinned memory (utile se poi sposti su GPU).

    drop_last:
        Se True, droppa l'ultimo batch se incompleto.

    collate_fn:
        Funzione di collate personalizzata.
        - Se None, usa `_default_stories_collate_fn`, che ritorna `List[StoryItem]` così com'è.
        - Se la passi tu, riceve `List[StoryItem]` e ritorna quello che ti serve
          (es. dict, prompt string, tensori, ecc.).

    Ritorna
    -------
    DataLoader
        DataLoader che produce batch determinati da `collate_fn`.

        
    """

        

    
    def _default_stories_collate_fn(batch: List[StoryItem]) -> List[StoryItem]:
        """
        Collate function di default per il DataLoader.

        Lascia il batch così com'è:
        batch = List[StoryItem]
        """
        return batch
    dataset = DatasetStories(json_path)

    if limit is not None:
        dataset = Subset(dataset, range(limit))

    effective_collate = collate_fn or _default_stories_collate_fn
    return DataLoader(
        dataset,
        batch_size=batch_size or len(dataset),
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        collate_fn=effective_collate,
    ),dataset
