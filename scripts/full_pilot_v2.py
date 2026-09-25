"""
v2 of the full pilot, fixing two issues found in v1:
  1. Decoys are now sampled from the SAME relation (e.g. other real directors
     for a director question), not from any relation - so the test measures
     fact-specific recall, not just answer-type matching (name vs. color).
  2. Extreme power-law outliers (e.g. "Sydney" with 61,516 mentions) are
     excluded before bucketing, so "high" means "clearly more exposure" not
     "a totally different scale of celebrity." Correlation is computed on
     log(1+exposures) instead of raw counts, which is standard for
     power-law-distributed frequency data.
"""
import random
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from exposure_lookup import load_chunk_to_steps, exposure_timeline, cumulative_exposure_at_checkpoints
from huggingface_hub import hf_hub_download

MODEL_ID = "dhgottesman/LMEnt-170M-1E"
CHECKPOINT_STEPS = [0, 10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000, 109672]
SHARDS = ["00001-of-00009", "00003-of-00009", "00004-of-00009", "00005-of-00009", "00006-of-00009"]
N_PER_BUCKET = 34
N_DECOYS = 2
SEED = 0
MAX_SUBJECT_CHUNKS = 2000  # drop mega-celebrity outliers before bucketing

TEMPLATES = {
    "producer": "The producer of {subj} was",
    "director": "The director of {subj} was",
    "screenwriter": "The screenwriter of {subj} was",
    "composer": "The composer of {subj} was",
    "genre": "The genre of {subj} is",
    "father": "The father of {subj} was",
    "religion": "The religion of {subj} is",
    "color": "The color of {subj} is",
    "capital of": "{subj} is the capital of",
}
DEFAULT_TEMPLATE = "The {prop} of {subj} was"


def make_prompt(row):
    tmpl = TEMPLATES.get(row["prop"], DEFAULT_TEMPLATE)
    return tmpl.format(subj=row["subj"], prop=row["prop"])


def load_pool():
    frames = []
    for s in SHARDS:
        path = hf_hub_download("dhgottesman/popqa-kas", f"data/train-{s}.parquet", repo_type="dataset")
        frames.append(pd.read_parquet(path))
    df = pd.concat(frames, ignore_index=True)
    df = df[df["subject_num_chunks"] > 0].drop_duplicates(subset=["subj", "prop"]).reset_index(drop=True)
    return df


def build_prop_answer_pools(df):
    """NOTE: sorted() here is load-bearing, not cosmetic - Python randomizes
    string-set iteration order per process, so a plain list(set(...)) makes
    rng.sample() non-reproducible across runs even with a fixed seed."""
    pools = {}
    for prop, grp in df.groupby("prop"):
        answers = set()
        for a in grp["possible_answers"]:
            answers.add(list(a)[0])
        pools[prop] = sorted(answers)
    return pools


def stratified_sample(df, seed=SEED):
    df = df[df["subject_num_chunks"] <= MAX_SUBJECT_CHUNKS].copy()
    df = df.sort_values("subject_num_chunks").reset_index(drop=True)
    n = len(df)
    low = df.iloc[: n // 3].sample(N_PER_BUCKET, random_state=seed)
    mid = df.iloc[n // 3 : 2 * n // 3].sample(N_PER_BUCKET, random_state=seed)
    high = df.iloc[2 * n // 3 :].sample(N_PER_BUCKET, random_state=seed)
    low["bucket"], mid["bucket"], high["bucket"] = "low", "mid", "high"
    print(f"  bucket ranges (subject_num_chunks): "
          f"low={low['subject_num_chunks'].min()}-{low['subject_num_chunks'].max()}  "
          f"mid={mid['subject_num_chunks'].min()}-{mid['subject_num_chunks'].max()}  "
          f"high={high['subject_num_chunks'].min()}-{high['subject_num_chunks'].max()}")
    return pd.concat([low, mid, high]).reset_index(drop=True)


def load_models(steps):
    models = {}
    for step in steps:
        tok = AutoTokenizer.from_pretrained(MODEL_ID, subfolder=f"step{step}")
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID, subfolder=f"step{step}")
        model.eval()
        models[step] = (tok, model)
        print(f"  loaded step{step}")
    return models


@torch.no_grad()
def answer_logprob(tok, model, prompt, answer):
    prompt_ids = tok(prompt, return_tensors="pt").input_ids
    answer_ids = tok(" " + answer.strip(), return_tensors="pt", add_special_tokens=False).input_ids
    full_ids = torch.cat([prompt_ids, answer_ids], dim=1)
    logits = model(full_ids).logits
    ans_start = prompt_ids.shape[1]
    ans_logits = logits[0, ans_start - 1 : ans_start - 1 + answer_ids.shape[1]]
    logprobs = F.log_softmax(ans_logits, dim=-1)
    token_logprobs = logprobs.gather(1, answer_ids[0].unsqueeze(1)).squeeze(1)
    return token_logprobs.mean().item()


def main():
    print("Loading exposure index + popqa-kas shards...")
    chunk_to_steps = load_chunk_to_steps()
    pool = load_pool()
    print(f"  pooled {len(pool)} unique (subj, prop) facts across {len(SHARDS)} shards")
    prop_pools = build_prop_answer_pools(pool)

    sample = stratified_sample(pool)
    rng = random.Random(SEED)

    print(f"Loading {len(CHECKPOINT_STEPS)} checkpoints of {MODEL_ID} ...")
    models = load_models(CHECKPOINT_STEPS)

    rows = []
    for i, (_, row) in enumerate(sample.iterrows()):
        steps = exposure_timeline(row["subject_chunks"], chunk_to_steps)
        cum = cumulative_exposure_at_checkpoints(steps, CHECKPOINT_STEPS)
        prompt = make_prompt(row)
        gold = list(row["possible_answers"])[0]

        same_prop_pool = [a for a in prop_pools[row["prop"]] if a != gold]
        if len(same_prop_pool) < N_DECOYS:
            continue  # not enough same-relation decoys, skip this fact
        decoys = rng.sample(same_prop_pool, N_DECOYS)

        transition_step = None
        for step in CHECKPOINT_STEPS:
            tok, model = models[step]
            gold_lp = answer_logprob(tok, model, prompt, gold)
            decoy_lps = [answer_logprob(tok, model, prompt, d) for d in decoys]
            correct = gold_lp > max(decoy_lps)
            if correct and transition_step is None:
                transition_step = step
            rows.append({
                "subj": row["subj"], "prop": row["prop"], "bucket": row["bucket"],
                "subject_num_chunks": row["subject_num_chunks"], "step": step,
                "exposures": cum[step], "gold_logprob": gold_lp,
                "max_decoy_logprob": max(decoy_lps), "correct": correct,
            })
        print(f"[{i+1}/{len(sample)}] {row['subj']!r} ({row['bucket']}, "
              f"{row['subject_num_chunks']} mentions, prop={row['prop']}) -> transition_step={transition_step}")

    res_df = pd.DataFrame(rows)
    res_df.to_csv("full_pilot_v2_results.csv", index=False)
    print("\nSaved full_pilot_v2_results.csv\n")

    print("=== Accuracy by bucket, per checkpoint ===")
    for step in CHECKPOINT_STEPS:
        sub = res_df[res_df["step"] == step]
        accs = sub.groupby("bucket")["correct"].mean()
        print(f"step={step:>7}  overall={sub['correct'].mean():.2f}  " +
              "  ".join(f"{b}={accs.get(b, float('nan')):.2f}" for b in ["low", "mid", "high"]))

    print("\n=== Mean transition step by bucket (only facts that crossed over) ===")
    # NOTE: group by (bucket, subj, prop), not just (bucket, subj) - some
    # subjects appear with more than one relation (e.g. "The Kid" as both
    # composer and producer), and grouping by subj alone silently merges
    # those into a single fact, corrupting the mean. Caught via make_figures.py
    # disagreeing with this live-printed number.
    trans = res_df[res_df["correct"]].groupby(["bucket", "subj", "prop"])["step"].min().reset_index()
    n_facts = res_df.groupby("bucket")[["subj", "prop"]].apply(lambda g: len(g.drop_duplicates()))
    n_crossed = trans.groupby("bucket")[["subj", "prop"]].apply(lambda g: len(g.drop_duplicates()))
    print(trans.groupby("bucket")["step"].mean())
    for b in ["low", "mid", "high"]:
        print(f"  {b}: {n_crossed.get(b, 0)}/{n_facts.get(b, 0)} facts ever crossed over")

    print("\n=== Correlation: log(1+exposures) vs. gold_logprob (pooled) ===")
    log_exp = np.log1p(res_df["exposures"])
    corr = log_exp.corr(res_df["gold_logprob"])
    print(f"Pearson r = {corr:.3f}  (n={len(res_df)})")


if __name__ == "__main__":
    main()
