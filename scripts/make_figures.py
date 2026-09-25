"""
Generate the three report figures from the two full-pilot result CSVs
(170M and 600M). Saves as PDF (not PNG/JPEG) with large fonts, per the
course guidelines.

Figures:
  1. fig_logprob_vs_exposure.pdf  - scatter, log(1+exposures) vs. gold logprob,
     170M and 600M side by side.
  2. fig_accuracy_vs_checkpoint.pdf - accuracy over training steps, both
     models, with a chance-level reference line.
  3. fig_transition_by_bucket.pdf - mean transition step by frequency
     bucket, grouped bars, both models.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 13,
    "axes.titlesize": 15,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "figure.dpi": 150,
})

CHECKPOINT_STEPS = [0, 10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000, 109672]
CHANCE_LEVEL = 1 / 3  # 1 gold vs 2 decoys

df170 = pd.read_csv("full_pilot_v2_results.csv")
df600 = pd.read_csv("full_pilot_v2_600M_results.csv")
df170["model"] = "170M"
df600["model"] = "600M"

MODEL_COLORS = {"170M": "#4C72B0", "600M": "#C44E52"}


# --- Figure 1: log-prob vs exposure scatter ---
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
for ax, df, name in zip(axes, [df170, df600], ["170M", "600M"]):
    x = np.log1p(df["exposures"])
    y = df["gold_logprob"]
    ax.scatter(x, y, s=8, alpha=0.25, color=MODEL_COLORS[name])
    r = x.corr(y)
    # simple linear trend line
    coef = np.polyfit(x, y, 1)
    xs = np.linspace(x.min(), x.max(), 100)
    ax.plot(xs, np.polyval(coef, xs), color="black", linewidth=2, linestyle="--")
    ax.set_title(f"LMEnt-{name}-1E  (r = {r:.3f})")
    ax.set_xlabel("log(1 + exposures)")
axes[0].set_ylabel("Gold-answer log-probability")
fig.suptitle("Exposure count predicts answer confidence, more strongly at scale", y=1.02)
fig.tight_layout()
fig.savefig("fig_logprob_vs_exposure.pdf", bbox_inches="tight")
plt.close(fig)


# --- Figure 2: accuracy vs checkpoint ---
fig, ax = plt.subplots(figsize=(7, 4.5))
for df, name in [(df170, "170M"), (df600, "600M")]:
    accs = df.groupby("step")["correct"].mean().reindex(CHECKPOINT_STEPS)
    ax.plot(CHECKPOINT_STEPS, accs.values, marker="o", label=f"LMEnt-{name}-1E",
            color=MODEL_COLORS[name], linewidth=2)
ax.axhline(CHANCE_LEVEL, color="gray", linestyle=":", linewidth=1.5, label="chance (1/3)")
ax.set_xlabel("Training step")
ax.set_ylabel("Accuracy (gold beats both decoys)")
ax.set_title("Accuracy rises fast, then partially decays with more training")
ax.legend()
fig.tight_layout()
fig.savefig("fig_accuracy_vs_checkpoint.pdf", bbox_inches="tight")
plt.close(fig)


# --- Figure 3: mean transition step by bucket ---
def transition_by_bucket(df):
    trans = df[df["correct"]].groupby(["bucket", "subj", "prop"])["step"].min().reset_index()
    return trans.groupby("bucket")["step"].mean()

t170 = transition_by_bucket(df170)
t600 = transition_by_bucket(df600)
buckets = ["low", "mid", "high"]
x = np.arange(len(buckets))
width = 0.35

fig, ax = plt.subplots(figsize=(6.5, 4.5))
ax.bar(x - width/2, [t170.get(b, 0) for b in buckets], width, label="170M", color=MODEL_COLORS["170M"])
ax.bar(x + width/2, [t600.get(b, 0) for b in buckets], width, label="600M", color=MODEL_COLORS["600M"])
ax.set_xticks(x)
ax.set_xticklabels([b.capitalize() for b in buckets])
ax.set_ylabel("Mean transition step\n(first checkpoint gold beats decoys)")
ax.set_xlabel("Exposure-frequency bucket")
ax.set_title("Transition step by exposure-frequency bucket")
ax.legend()
fig.tight_layout()
fig.savefig("fig_transition_by_bucket.pdf", bbox_inches="tight")
plt.close(fig)

print("Saved: fig_logprob_vs_exposure.pdf, fig_accuracy_vs_checkpoint.pdf, fig_transition_by_bucket.pdf")

# Print the numbers embedded in the figures, for cross-checking against NOTES.md
print()
print("r(170M) =", np.log1p(df170['exposures']).corr(df170['gold_logprob']))
print("r(600M) =", np.log1p(df600['exposures']).corr(df600['gold_logprob']))
print("transition steps 170M:", dict(t170))
print("transition steps 600M:", dict(t600))
