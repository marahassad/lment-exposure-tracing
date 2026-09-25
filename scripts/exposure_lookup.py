"""
Proof of concept: for a given entity (subject) from the popqa-kas dataset,
build its full training-exposure timeline using only two lightweight files:
  - popqa-kas: entity -> list of corpus chunk_ids that mention it
  - batch_indices_epoch_1.npy: training step -> list of chunk_ids seen at that step

No need to touch the 127GB Elasticsearch index or the 212GB raw metadata.
"""
import numpy as np
import pandas as pd
from collections import defaultdict
from huggingface_hub import hf_hub_download

CHECKPOINT_STEPS = [0, 10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000, 109672]


def load_chunk_to_steps():
    """Build a reverse index: chunk_id -> list of training steps it appeared in."""
    path = hf_hub_download("dhgottesman/LMEnt-Dataset", "dataset-cache/batch_indices_epoch_1.npy", repo_type="dataset")
    batch_indices = np.load(path, allow_pickle=True)
    chunk_to_steps = defaultdict(list)
    for step, chunk_ids in enumerate(batch_indices):
        for cid in chunk_ids:
            chunk_to_steps[int(cid)].append(step)
    return chunk_to_steps


def exposure_timeline(subject_chunks, chunk_to_steps):
    """Every training step at which ANY chunk mentioning this subject was seen."""
    steps = []
    for cid in subject_chunks:
        steps.extend(chunk_to_steps.get(int(cid), []))
    return sorted(steps)


def cumulative_exposure_at_checkpoints(steps, checkpoints=CHECKPOINT_STEPS):
    """How many exposures had happened by the time each checkpoint was saved."""
    steps = np.array(steps)
    return {ckpt: int((steps <= ckpt).sum()) for ckpt in checkpoints}


def main():
    print("Loading chunk->step reverse index (from 90MB batch_indices file)...")
    chunk_to_steps = load_chunk_to_steps()
    print(f"  indexed {len(chunk_to_steps):,} distinct chunks across training.\n")

    print("Loading one popqa-kas shard for example facts...")
    path = hf_hub_download("dhgottesman/popqa-kas", "data/train-00003-of-00009.parquet", repo_type="dataset")
    df = pd.read_parquet(path)

    for _, row in df.head(3).iterrows():
        steps = exposure_timeline(row["subject_chunks"], chunk_to_steps)
        cum = cumulative_exposure_at_checkpoints(steps)
        print(f"Subject: {row['subj']}  (QID {row['s_uri'].split('/')[-1]}, pop={row['s_pop']})")
        print(f"  Question: {row['question']}")
        print(f"  Total mentions in corpus: {row['subject_num_chunks']}, actually observed in training batches: {len(steps)}")
        print(f"  Cumulative exposures by checkpoint: {cum}")
        print()


if __name__ == "__main__":
    main()
