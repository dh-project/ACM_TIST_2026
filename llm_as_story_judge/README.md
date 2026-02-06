# llm-as-story-judge

![Teaser](teaser_img.png)

Repository for *From a True Story: Leveraging Museum Catalogue Data for LLM-Driven Narrative Generation*.

This work targets cultural heritage storytelling, where historical accuracy, narrative structure, and audience engagement must be balanced. Museum catalogues are rich but loosely structured, making it hard to turn them into coherent stories without losing fidelity. The paper presents a structured, interpretable pipeline that separates event extraction, narratological structuring, and constrained generation, anchored by an explicit narrative scaffold that enforces conflict roles and causal progression. It also introduces a rubric-based evaluation protocol used by both human experts and LLM-based judges, enabling scalable assessment and analysis of agreement, bias, and reliability.

**What it does**
Evaluates LLM-generated narratives with structured prompts and schemas, producing tabular outputs and a combined panel for comparison with human ground truth.

**Repo structure**
- `configs/`: workflow configs, model key mapping, and LLM manager config.
- `baseline_assets/prompts/`: baseline judging prompts.
- `baseline_assets/schemas/`: YAML schemas for structured outputs.
- `judge_data/`: datasets, ground truth, and the analysis notebook.
- `run_workflows.py`: launcher that runs all workflows and combines results.

**Quick setup**
1. Create `agentic_forge/configs/providers_api_key.yaml` from `agentic_forge/configs/providers_api_key.example.yaml` and add your keys. OpenRouter and OpenAI require API keys; Ollama does not.
2. Install dependencies: `pip install -r requirements.txt`.
3. Configure your providers and models in `configs/llm_manager_config.yaml`, and update `configs/model_key.yaml` accordingly.

**How to run**
1. Configure models in `configs/model_key.yaml`.
2. Configure providers and available models in `configs/llm_manager_config.yaml`.
3. Configure one or more workflows in `configs/*_workflow.yaml`.
4. Run from the project root: `python run_workflows.py`.

The launcher:
- Finds all `configs/*_workflow.yaml` files.
- Runs each workflow sequentially.
- Saves per-run outputs and produces a combined panel per experiment.

**Workflow config (`configs/*_workflow.yaml`)**

| Parameter | Meaning |
| --- | --- |
| `exp_name` | Experiment name. Used for the output folder. |
| `logical_model_key` | Logical model key, must exist in `configs/model_key.yaml`. |
| `dataset_path` | Input dataset path (JSON). |
| `schemas_path` | Directory with YAML schemas for structured outputs. |
| `prompts_path` | Directory with YAML prompts used by judges. |
| `run_name` | Optional run name. If `null`, it auto-generates `run0`, `run1`, ... |
| `batch_size` | Parallel batch size used during dataset processing. |
| `max_retries` | Max retries for failed items. |
| `reasoning_enabled` | Enable/disable reasoning when supported by the model. |
| `verbose_llm` | Verbose logging of LLM calls. |
| `base_output_dir` | Base directory for outputs (default: `experiments`). |

**Model key (`configs/model_key.yaml`)**

| Parameter | Meaning |
| --- | --- |
| `provider` | Provider identifier defined in `configs/llm_manager_config.yaml`. |
| `model` | Provider model name. |

**Baseline assets**
- `baseline_assets/prompts/`: prompts for judges (historical, dramatic, turning points).
- `baseline_assets/schemas/`: YAML schemas defining fields and types.

**Outputs and panel**
Each run produces:
- `experiments/<exp_name>/<model_key>_<run>/out.json`
- `experiments/<exp_name>/<model_key>_<run>/out.csv`
- `experiments/<exp_name>/<model_key>_<run>/fields_column_meta.json`
- `experiments/<exp_name>/<model_key>_<run>/run_config.json`

At the end, the launcher creates:
- `experiments/<exp_name>/combined_panel.csv`
- `experiments/<exp_name>/combined_panel_meta.json`

The `combined_panel.csv` includes all runs, with the `judge` column set to the run folder name.

**Alignment with ground truth and notebook**
Ground truth files are in:
- `judge_data/processed/processed_historical_gt.csv`
- `judge_data/processed/processed_drama_gt.csv`

For analysis:
1. Align the combined panel with ground truth (typically on `id`).
2. Open `judge_data/compute.ipynb` and set `PATH_PANEL` to your combined panel path.
3. Run the notebook to compute metrics and comparisons.
