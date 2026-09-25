"""
Diagnostic: is the "forgetting" pattern from full_pilot_v2 a genuine loss of
fact-specific knowledge, or a decoy-comparison artifact - i.e. the model
develops a generic preference for one particular wrong answer within a
relation (e.g. always favors one specific "director" name late in training),
which would beat the gold answer for MANY different facts sharing that
relation, regardless of which fact it actually is.

This is an exact rerun of full_pilot_v2's computation (same seed, same code
path, so decoys are reproduced identically) but this time logging which
decoy actually won at each checkpoint, not just the max logprob value.
"""
import random
import torch
import pandas as pd
from full_pilot_v2 import (
    MODEL_ID, CHECKPOINT_STEPS, N_DECOYS, SEED,
    load_pool, build_prop_answer_pools, stratified_sample, load_models,
    make_prompt, answer_logprob,
)
from exposure_lookup import load_chunk_to_steps, exposure_timeline, cumulative_exposure_at_checkpoints


def main():
    print("Loading exposure index + popqa-kas shards...")
    chunk_to_steps = load_chunk_to_steps()
    pool = load_pool()
    prop_pools = build_prop_answer_pools(pool)
    sample = stratified_sample(pool)
    rng = random.Random(SEED)

    print(f"Loading {len(CHECKPOINT_STEPS)} checkpoints...")
    models = load_models(CHECKPOINT_STEPS)

    rows = []
    for i, (_, row) in enumerate(sample.iterrows()):
        steps = exposure_timeline(row["subject_chunks"], chunk_to_steps)
        cum = cumulative_exposure_at_checkpoints(steps, CHECKPOINT_STEPS)
        prompt = make_prompt(row)
        gold = list(row["possible_answers"])[0]

        same_prop_pool = [a for a in prop_pools[row["prop"]] if a != gold]
        if len(same_prop_pool) < N_DECOYS:
            continue
        decoys = rng.sample(same_prop_pool, N_DECOYS)  # same rng draws as full_pilot_v2

        for step in CHECKPOINT_STEPS:
            tok, model = models[step]
            gold_lp = answer_logprob(tok, model, prompt, gold)
            decoy_lps = [answer_logprob(tok, model, prompt, d) for d in decoys]
            winner_idx = max(range(len(decoy_lps)), key=lambda j: decoy_lps[j])
            rows.append({
                "subj": row["subj"], "prop": row["prop"], "bucket": row["bucket"],
                "step": step, "exposures": cum[step],
                "gold": gold, "gold_logprob": gold_lp,
                "winning_decoy": decoys[winner_idx], "winning_decoy_logprob": decoy_lps[winner_idx],
                "correct": gold_lp > decoy_lps[winner_idx],
            })
        print(f"[{i+1}/{len(sample)}] {row['subj']!r} done")

    df = pd.DataFrame(rows)
    df.to_csv("decoy_artifact_check.csv", index=False)
    print("\nSaved decoy_artifact_check.csv\n")

    # Sanity: this should reproduce full_pilot_v2's numbers exactly
    print("=== Sanity check: accuracy by checkpoint (should match full_pilot_v2) ===")
    for step in CHECKPOINT_STEPS:
        print(f"step={step:>7}  acc={df[df['step']==step]['correct'].mean():.2f}")

    # The actual diagnostic: which facts flip from correct(peak) -> wrong(final)?
    peak_step, final_step = 20000, CHECKPOINT_STEPS[-1]
    piv_correct = df.pivot_table(index=["subj", "prop"], columns="step", values="correct")
    piv_winner = df.pivot_table(index=["subj", "prop"], columns="step", values="winning_decoy", aggfunc="first")

    forgotten_mask = piv_correct[peak_step].fillna(False) & ~piv_correct[final_step].fillna(False)
    forgotten = piv_correct[forgotten_mask].index

    print(f"\n=== 'Forgotten' facts (correct@{peak_step}, wrong@{final_step}): {len(forgotten)} ===")
    print("For each, what wrong answer beat the gold answer at the final checkpoint?\n")
    winner_counts = {}
    for subj, prop in forgotten:
        winner = piv_winner.loc[(subj, prop), final_step]
        print(f"  {subj!r} ({prop}) -> beaten by decoy: {winner!r}")
        winner_counts.setdefault((prop, winner), 0)
        winner_counts[(prop, winner)] += 1

    print("\n=== Does the SAME decoy repeatedly win within a relation? (artifact signature) ===")
    repeated = {k: v for k, v in winner_counts.items() if v > 1}
    if repeated:
        for (prop, winner), count in sorted(repeated.items(), key=lambda x: -x[1]):
            print(f"  prop={prop!r}  decoy={winner!r}  won {count} different facts' final-step comparison")
    else:
        print("  No decoy repeats across forgotten facts - each lost to a different, fact-specific decoy.")
        print("  This argues AGAINST the artifact explanation: looks like genuine per-fact forgetting.")


if __name__ == "__main__":
    main()
