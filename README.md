# Tracing the Birth of a Fact

Code and results for the NLP final project: *Tracing the Birth of a Fact: Exposure-Level Dynamics of Knowledge Acquisition and Hallucination in LMEnt* (Marah Assad, Tel Aviv University).

Uses [LMEnt](https://huggingface.co/collections/dhgottesman/lment-68a9dd370e1f746cacd8ce58) (Gottesman et al., 2025) and its `popqa-kas` companion dataset to recover, for individual facts, the exact training step at which a language model was exposed to them, and to track how that exposure relates to the model's answer confidence across training.

## Repo layout

```
scripts/   pipeline code + raw result CSVs
report/    ACL-format paper source (paper.tex, references.bib, figures/)
```

## Pipeline, in order

1. **`exposure_lookup.py`** — core module. Builds a chunk-ID → training-step index from LMEnt's `batch_indices_epoch_1.npy`, and joins it against `popqa-kas`'s per-fact chunk lists to recover a full exposure timeline for any fact.
2. **`full_pilot_v2.py`** / **`full_pilot_v2_600M.py`** — the main experiment. Samples 102 facts stratified by exposure frequency, probes the LMEnt-170M-1E / LMEnt-600M-1E checkpoints (12 each) with a same-relation-decoy multiple-choice protocol, and saves `full_pilot_v2_results.csv` / `full_pilot_v2_600M_results.csv`. These two scripts are identical except for the model ID and output filename — the 600M run was executed on the TAU CS Slurm cluster (`studentkillable` partition, RTX 2080 Ti GPU) since it needs more memory/compute than a laptop comfortably handles.
3. **`decoy_artifact_check.py`** — reruns the pipeline while also logging which decoy "won" at each checkpoint, to test whether the forgetting effect reported in the paper is a decoy-comparison artifact rather than genuine forgetting. Saves `decoy_artifact_check.csv`.
4. **`recency_analysis.py`** — pure data analysis (no model inference) testing whether exposure *recency*, not just count, predicts forgetting. Reads `full_pilot_v2_results.csv`, saves `recency_merged.csv`.
5. **`make_figures.py`** — generates the three report figures (as PDF) from the result CSVs, saved into `report/figures/`.

## Reproducing

Requires `torch`, `transformers`, `huggingface_hub`, `pandas`, `pyarrow`. All data (LMEnt checkpoints, `popqa-kas`) downloads automatically from the Hugging Face Hub on first run — no manual data setup needed, and no access to LMEnt's full 127GB retrieval index or 212GB annotated corpus is required.

## Report

The paper itself (`report/paper.tex`) compiles in the [ACL template](https://www.overleaf.com/latex/templates/association-for-computational-linguistics-acl-conference/jvxskxpnznfj) — `acl.sty` and `acl_natbib.bst` are included in `report/` for convenience.
