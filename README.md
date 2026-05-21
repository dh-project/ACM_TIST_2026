# From a True Story: Leveraging Museum Catalogue Data for LLM-driven Narrative Generation

![Teaser image](teaser_image.png)

This repository contains the code for the paper "From a True Story: Leveraging Museum Catalogue Data for LLM-driven Narrative Generation."

## Structure

- `story_generation/`: code for the story generation pipeline.
- `llm_as_story_judge/`: code for the LLM-based evaluation (the "story judge").
- `datasets/`: contains the stories produced by running the story generation pipeline.
- `visualization_tool/`: allows users to explore the generated stories and dramatic situations descriptions.

## Datasets

The `datasets/` folder contains four datasets, each representing the results obtained by running the full pipeline or one of its variants. Each dataset includes 300 structured stories, 100 for each model used (`deepseek-v3.2`, `gemini-2.5-flash`, `gpt-oss-120b`).

- `full_pipeline_structured_narratives.json`: stories generated with the complete pipeline.

- `direct_prompting_structured_narratives.json`: stories generated directly from the cleaned catalogue entries used in the full-pipeline condition.

- `no_scaffold_structured_narratives.json`: stories generated from the same narrative units as the full pipeline, but without the explicit dramatic-situation scaffold.

- `story_potential_structured_narratives.json`: stories generated from narrative units selected according to storytelling potential, without the explicit Polti-based mapping.


## License

This project is licensed under the Creative Commons Attribution 4.0 International License (CC BY 4.0).
See: https://creativecommons.org/licenses/by/4.0/