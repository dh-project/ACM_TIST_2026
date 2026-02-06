"""
workflow_forge_parallel.py

Modulo che espone una funzione per costruire un grafo LangGraph che implementa
un pattern fan-out / fan-in (parallel map):

    - input:  state["items"] è una lista di elementi
    - fan-out: per ogni elemento viene creato un "worker" via Send(...)
    - worker: applica la stessa funzione fn (con eventuali args/kwargs fissi)
    - fan-in: i risultati vengono raccolti e ordinati per indice originale

Uso tipico:

    graph = forge_parallel_map_graph(fn_elaborazione, fn_arg1, k=42)
    results = run_parallel_map(graph, items=[...], max_concurrency=8)
"""

from __future__ import annotations

import operator
from typing import Any, Callable, Dict, List, TypedDict, Annotated, Optional

from langgraph.graph import StateGraph, START, END
from langgraph.types import Send  # oggetto usato per il fan-out dinamico


# ---------------------------------------------------------------------------
# Definizione dello state del grafo
# ---------------------------------------------------------------------------

class ParallelMapState(TypedDict, total=False):
    """
    Stato condiviso del grafo di parallel map.

    Chiavi:

    - items: lista di input globali da processare
    - task:  sub-state per un singolo worker, contiene:
        - index: posizione dell'elemento nella lista originale
        - value: valore da passare a fn
    - partial_results: lista "append-only" di risultati parziali prodotti dai worker
        Ogni elemento è un dict con chiavi:
            - index: indice originale
            - value: risultato di fn(item, *fn_args, **fn_kwargs)
      Annotated con operator.add per permettere merge da rami paralleli.
    - results: lista finale dei risultati, ordinata per index originale
    """

    items: List[Any]

    task: Dict[str, Any]

    partial_results: Annotated[
        List[Dict[str, Any]],
        operator.add,  # reducer: concatena liste da rami paralleli
    ]

    results: List[Any]


# ---------------------------------------------------------------------------
# Costruttore del grafo: fan-out / worker / fan-in
# ---------------------------------------------------------------------------

def forge_parallel_map_workflow(
    fn: Callable[..., Any],
    *fn_args: Any,
    name: str = "parallel_map",
    **fn_kwargs: Any,
):
    """
    Costruisce e compila un grafo di tipo "parallel map" utilizzando LangGraph.

    Parametri
    ---------
    fn:
        Funzione da applicare a ciascun elemento di `items`.
        La firma effettiva sarà:
            fn(item, *fn_args, **fn_kwargs)

    *fn_args, **fn_kwargs:
        Argomenti fissi che verranno passati a `fn` per ogni worker.

    name:
        Nome logico del grafo (solo cosmetico, utile per logging / tracing).

    Ritorna
    -------
    app:
        Istanza compilata del grafo (oggetto compatibile con .invoke / .ainvoke).

    Pattern implementato
    --------------------
    - Nodo "fan_out":
        nodo "dummy" che non modifica lo stato ma è il punto da cui parte il fan-out.
    - Funzione di conditional edges `continue_to_workers`:
        legge state["items"] (lista) e ritorna una lista di Send("worker", sub_state)
    - Nodo "worker":
        legge state["task"]["index"] e state["task"]["value"],
        applica fn(value, *fn_args, **fn_kwargs),
        restituisce un update sulla chiave "partial_results"
    - Nodo "gather":
        legge state["partial_results"], ordina per index e costruisce state["results"]
    """

    graph = StateGraph(ParallelMapState, name=name)

    # ----------------------- fan-out NODE -----------------------
    def fan_out(state: ParallelMapState) -> ParallelMapState:
        """
        Nodo di "preparazione" per il fan-out.

        Non deve restituire Send (altrimenti LangGraph si incazza):
        i Send vanno usati SOLO nelle funzioni di conditional edges.

        Qui possiamo semplicemente ritornare {} per non modificare lo stato.
        """
        return {}

    # ----------------------- conditional edges: Send(...) -----------------------
    def continue_to_workers(state: ParallelMapState) -> List[Send]:
        """
        Funzione di conditional edges associata al nodo "fan_out".

        Qui è dove usiamo la Send API per creare N esecuzioni concorrenti
        del nodo "worker", una per ogni elemento di `items`.
        """
        items = state.get("items") or []
        sends: List[Send] = []

        for idx, item in enumerate(items):
            sends.append(
                Send(
                    "worker",
                    {
                        "task": {
                            "index": idx,
                            "value": item,
                        }
                    },
                )
            )

        return sends

    # ----------------------- worker ------------------------
    def worker_node(state: ParallelMapState) -> ParallelMapState:
        """
        Nodo worker.

        Riceve in input un sotto-stato `task` con:
            - index: indice originale dell'item
            - value: valore da passare alla funzione `fn`

        Esegue:
            result = fn(value, *fn_args, **fn_kwargs)

        e aggiorna `partial_results` aggiungendo un elemento:
            {"index": index, "value": result}

        Grazie all'annotazione `partial_results: Annotated[..., operator.add]`,
        LangGraph concatenerà in modo thread-safe i risultati prodotti da tutti
        i worker.
        """
        task = state["task"]
        idx = task["index"]
        value = task["value"]

        result = fn(value, *fn_args, **fn_kwargs)

        return {
            "partial_results": [
                {
                    "index": idx,
                    "value": result,
                }
            ]
        }

    # ----------------------- gather ------------------------
    def gather(state: ParallelMapState) -> ParallelMapState:
        """
        Nodo di fan-in.

        - Legge `partial_results`, che è una lista di dict
          [{"index": i, "value": out_i}, ...] prodotta dai worker.
        - Ordina per `index` per ripristinare lo stesso ordine della lista
          `items` originale.
        - Scrive `results` come lista di soli valori.

        Questo è il punto in cui ricomponiamo la mappa parallela in un'unica
        lista ordinata.
        """
        partial = state.get("partial_results") or []

        # Ordina per indice per preservare l'ordine degli input originali.
        ordered = sorted(partial, key=lambda r: r["index"])

        return {
            "results": [r["value"] for r in ordered],
        }

    # ----------------------- wiring del grafo ------------------------

    # Registrazione nodi
    graph.add_node("fan_out", fan_out)
    graph.add_node("worker", worker_node)
    graph.add_node("gather", gather)

    # Edges:
    # - START -> fan_out: il grafo parte dal nodo fan_out
    # - conditional edges da "fan_out" -> "worker" tramite continue_to_workers:
    #   qui è dove ritorniamo List[Send(...)] e LangGraph fa il fan-out parallelo.
    # - worker -> gather: ogni worker, al termine, porta lo state verso gather
    # - gather -> END: chiusura del grafo
    graph.add_edge(START, "fan_out")
    graph.add_conditional_edges("fan_out", continue_to_workers, ["worker"])
    graph.add_edge("worker", "gather")
    graph.add_edge("gather", END)

    # Compilazione del grafo (senza checkpointer per semplicità)
    app = graph.compile()
    return app


# ---------------------------------------------------------------------------
# Helper di esecuzione: max_concurrency via max_concurrency
# ---------------------------------------------------------------------------

def run_parallel_map_workflow(
    graph_app,
    items: List[Any],
    max_concurrency: Optional[int] = None,
) -> List[Any]:
    """
    Esegue un grafo creato da `forge_parallel_map_graph` su una lista di `items`.

    Parametri
    ---------
    graph_app:
        Oggetto compilato ritornato da `forge_parallel_map_graph`.

    items:
        Lista di input da processare. Ogni elemento verrà passato come primo
        argomento posizionale a `fn` (dentro il worker).

    max_concurrency:
        Limite massimo di worker concorrenti. Viene mappato su `max_concurrency`.
        - Se None: usa len(items) (quindi "no limit" dal punto di vista logico).
        - Se > 0: limita il numero di worker in parallelo.

        Nota: il numero di Send generati è comunque len(items), ma il runtime
        potrà eseguirli al massimo `max_concurrency` alla volta.

        Questo è esattamente il batching automatico di LangGraph:
        se hai 100 items e max_concurrency=8, avrai sempre al massimo 8 worker
        attivi, gli altri in coda.

    Ritorna
    -------
    results:
        Lista dei risultati `fn(item, ...)`, ordinata come `items`.
    """
    if max_concurrency is None:
        # "threshold" naturale: massimo un worker per elemento
        max_concurrency = max(len(items), 1)

    config = {"max_concurrency": max_concurrency}

    # invochiamo il grafo passando solo la chiave "items" nello state iniziale
    final_state: ParallelMapState = graph_app.invoke(
        {"items": items},
        config=config,
    )
    return final_state.get("results", [])


# ---------------------------------------------------------------------------
# Esempio d'uso (main) per verificare il parallelismo
# ---------------------------------------------------------------------------

def main():
    import time
    import threading


    def slow_worker(x: int, delay: float = 1.0) -> str:
        
        """
        Worker di esempio che simula un lavoro lento.

        - Stampa quando inizia e quando termina, con:
            - timestamp relativo
            - id del thread corrente
            - valore processato
        - Attende `delay` secondi per simulare il carico.

        Ritorna una stringa per permettere di verificare l'ordine dei risultati.
        """
        thread_name = threading.current_thread().name
        t_start = time.perf_counter()
        print(f"[{t_start:.3f}s] [THREAD {thread_name}] START x={x}")

        time.sleep(delay)

        t_end = time.perf_counter()
        print(f"[{t_end:.3f}s] [THREAD {thread_name}] END   x={x}")
        return f"done_{x}"


    def demo_parallelism() -> None:
        """
        Dimostrazione del comportamento di parallel map:

        - Costruisce un grafo di parallel map su `slow_worker`
        - Esegue su una lista di interi [0..7]
        - Confronta:
            - esecuzione sequenziale (max_concurrency=1)
            - esecuzione parallela (max_concurrency=4)
        """
        items = list(range(8))

        # Costruzione del grafo con il nostro worker lento.
        graph_app = forge_parallel_map_workflow(
            slow_worker,
            name="demo_parallel_slow_worker",
        )

        print("\n=== ESECUZIONE PARALLELISMO MASSIMO DEI DATI (max_concurrency=None) ===")
        t0 = time.perf_counter()
        results_seq = run_parallel_map_workflow(
            graph_app,
            items=items,
            max_concurrency=None,
        )
        t1 = time.perf_counter()
        print(f"Risultati parallelismo massimo dei dati: {results_seq}")
        print(f"Tempo totale parallelismo massimo dei dati: {t1 - t0:.3f}s\n")

        print("\n=== ESECUZIONE SEQUENZIALE (max_concurrency=1) ===")
        t0 = time.perf_counter()
        results_seq = run_parallel_map_workflow(
            graph_app,
            items=items,
            max_concurrency=1,
        )
        t1 = time.perf_counter()
        print(f"Risultati sequenziale: {results_seq}")
        print(f"Tempo totale sequenziale: {t1 - t0:.3f}s\n")

        print("\n=== ESECUZIONE PARALLELA (max_concurrency=4) ===")
        t2 = time.perf_counter()
        results_par = run_parallel_map_workflow(
            graph_app,
            items=items,
            max_concurrency=4,
        )
        t3 = time.perf_counter()
        print(f"Risultati parallelo: {results_par}")
        print(f"Tempo totale parallelo: {t3 - t2:.3f}s\n")
    
    demo_parallelism()


if __name__ == "__main__":
    main()

