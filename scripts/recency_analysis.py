"""
Recency analysis: uses ONLY data already collected (full_pilot_v2_results.csv)
plus the same exposure-timeline machinery, to test whether a fact's forgetting
(correct early, wrong later) is explained by NOT being re-exposed late in
training, rather than by total exposure count alone.

No new model inference - pure data analysis.
"""
import numpy as np
import pandas as pd
from exposure_lookup import load_chunk_to_steps, exposure_timeline
from full_pilot_v2 import load_pool, SHARDS

CHECKPOINT_STEPS = [0, 10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000, 109672]
RESULTS_CSV = "full_pilot_v2_results.csv"


def recency_gap(steps_seen, checkpoint):
    """Steps since the fact was last seen, as of `checkpoint`. NaN if never seen yet."""
    seen_before = [s for s in steps_seen if s <= checkpoint]
    if not seen_before:
        return np.nan
    return checkpoint - max(seen_before)


def main():
    print("Loading results + exposure index + fact pool...")
    res = pd.read_csv(RESULTS_CSV)
    chunk_to_steps = load_chunk_to_steps()
    pool = load_pool()

    facts = res[["subj", "prop"]].drop_duplicates()
    pool_idx = pool.set_index(["subj", "prop"])

    recency_rows = []
    for _, f in facts.iterrows():
        key = (f["subj"], f["prop"])
        if key not in pool_idx.index:
            continue
        prow = pool_idx.loc[key]
        if isinstance(prow, pd.DataFrame):  # duplicate keys, take first
            prow = prow.iloc[0]
        steps_seen = exposure_timeline(prow["subject_chunks"], chunk_to_steps)
        for ckpt in CHECKPOINT_STEPS:
            recency_rows.append({
                "subj": f["subj"], "prop": f["prop"], "step": ckpt,
                "recency_gap": recency_gap(steps_seen, ckpt),
            })
    recency_df = pd.DataFrame(recency_rows)

    merged = res.merge(recency_df, on=["subj", "prop", "step"])
    merged.to_csv("recency_merged.csv", index=False)
    print(f"Saved recency_merged.csv ({len(merged)} rows)\n")

    seen = merged.dropna(subset=["recency_gap"])
    print("=== Correlation: recency_gap vs. gold_logprob (only facts already seen at least once) ===")
    print(f"Pearson r = {seen['recency_gap'].corr(seen['gold_logprob']):.3f}  (n={len(seen)})")
    print("(expect negative: longer since last exposure -> lower confidence)\n")

    print("=== Peaked-then-forgotten analysis ===")
    peak_step = 20000  # where overall accuracy peaked in the v2 pilot
    final_step = CHECKPOINT_STEPS[-1]
    piv = merged.pivot_table(index=["subj", "prop"], columns="step",
                              values=["correct", "recency_gap"])
    correct_at_peak = piv[("correct", peak_step)].fillna(False).astype(bool)
    correct_at_final = piv[("correct", final_step)].fillna(False).astype(bool)
    forgotten = correct_at_peak & ~correct_at_final
    stayed_correct = correct_at_peak & correct_at_final

    forgotten_recency = piv.loc[forgotten, ("recency_gap", final_step)]
    stayed_recency = piv.loc[stayed_correct, ("recency_gap", final_step)]

    print(f"Facts correct at step {peak_step} but wrong by step {final_step} ('forgotten'): {forgotten.sum()}")
    print(f"  -> mean recency_gap at final step: {forgotten_recency.mean():.0f}  (median {forgotten_recency.median():.0f})")
    print(f"Facts correct at step {peak_step} AND still correct at step {final_step} ('retained'): {stayed_correct.sum()}")
    print(f"  -> mean recency_gap at final step: {stayed_recency.mean():.0f}  (median {stayed_recency.median():.0f})")


if __name__ == "__main__":
    main()
