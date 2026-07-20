import numpy as np, pandas as pd, torch, torch.nn as nn
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, recall_score
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

SEEDS = [42, 43, 44]        
EPOCHS, BATCH, LR = 60, 1024, 1e-3
EVAL_BATCH = 8192           
DEV = "cuda" if torch.cuda.is_available() else "cpu"

X = torch.tensor(np.load("X.npy").astype(np.float32), device=DEV)
y = torch.tensor(np.load("y.npy").astype(np.int64), device=DEV)
meta = pd.read_csv("meta.csv")
meta["is_synthetic"] = meta.is_synthetic.astype(bool)
names = pd.read_csv("label_map.csv").sort_values("label").lecturer_name.tolist()
K = int(y.max()) + 1
assert len(X) == len(y) == len(meta), "X / y / meta not row-aligned"
print(f"{len(X):,} sentences | {K} classes | {meta.is_synthetic.mean():.0%} synthetic\n")

idx = np.arange(len(X))
y_np = y.cpu().numpy()
real = ~meta.is_synthetic.to_numpy()
T = lambda a: torch.as_tensor(a, device=DEV)


@torch.no_grad()  # we're only asking the model for predictions here, not training it, so don't bother tracking how to backpropagate (saves memory/time)
def logits(head, rows):
    head.eval()  # switch the model to "prediction mode" rather than "learning mode"
    return torch.cat([head(X[rows[i:i + EVAL_BATCH]])
                      for i in range(0, len(rows), EVAL_BATCH)])  # feed the rows through the model a chunk at a time and glue the outputs back together, so we don't run out of GPU memory doing it all at once


def train(tr, te, seed):
    """Fresh head trained on tr. Returns head, predictions on te, per-epoch curves."""
    torch.manual_seed(seed)  # fix the randomness (weight init + batch shuffling) so this run is reproducible
    head = nn.Sequential(nn.Linear(512, 2048), nn.ReLU(),   # the model: a stack of layers. In: a 512-number sentence "fingerprint".
                         nn.Linear(2048, 2048), nn.ReLU(),  # Middle: hidden layers that combine those numbers into learned patterns.
                         nn.Linear(2048, 2048), nn.ReLU(),  # ReLU just zeroes out negative values between layers, which is what lets
                         nn.Linear(2048, K)).to(DEV)        # the stack learn non-straight-line patterns. Out: one score per lecturer.
    opt = torch.optim.Adam(head.parameters(), lr=LR)          # optimiser, nudges weights in the right direction
    tr_t, te_t = T(tr), T(te)  # move the train/test row numbers onto the GPU, next to the data
    hist = []  # will collect [train_accuracy, test_accuracy] after every full pass over the data, so we can watch learning happen
    for ep in range(EPOCHS):  # repeat the whole training process 60 times ("epochs" = full passes over the training rows)
        head.train()  # switch the model to "learning mode" rather than "prediction mode"
        for b in torch.randperm(len(tr_t), device=DEV).split(BATCH):  # shuffle all training rows into random order, then chop that into groups of 1024 ("batches") to process one group at a time
            i = tr_t[b]  # the actual row numbers, in X/y, that make up this batch
            opt.zero_grad()  # clear out the leftover "how to change the weights" numbers from the previous batch
            nn.functional.cross_entropy(head(X[i]), y[i]).backward()  # run inference once (forward pass: head(X[i]) is the model's guess), score how well that guessed probability distribution matches the true class (cross_entropy), then work out how much each individual weight is to blame for that error (backward pass)
            opt.step()  # nudge every weight a little bit in the direction that would have made the error smaller
        hist.append([(logits(head, tr_t).argmax(1) == y[tr_t]).float().mean().item(),      # snapshot after this epoch: of the rows it TRAINED on, what fraction does it now guess correctly?
                     (logits(head, te_t).argmax(1) == y[te_t]).float().mean().item()])      # and of the rows it has NEVER trained on, what fraction does it guess correctly? A gap between these two is the sign of memorising rather than learning.
    return head, logits(head, te_t).argmax(1).cpu().numpy(), np.array(hist)  # hand back: the trained model itself, its final guesses on the held-out rows, and the epoch-by-epoch accuracy history for plotting


def score(name, te, pred, quiet=False):
    t = y_np[te]
    rec = recall_score(t, pred, average=None, labels=range(K), zero_division=0)
    # StratifiedGroupKFold does a best-effort balance of lecturer_id across the fold,
    # but it's still a GROUPED split -- if a lecturer's lectures/courses are too few or
    # too concentrated to split, they can still end up entirely in train and never
    # appear in test. Without this check you'd read "18 zero-recall lecturers" as "the
    # model failed on 18 people" when some of them were simply never tested.
    absent = K - len(np.unique(t))
    row = dict(eval=name,
               accuracy=(pred == t).mean(),
               macro_f1=f1_score(t, pred, labels=np.unique(t), average="macro", zero_division=0),  # only lecturers present in t, so absent classes don't force a 0 into the average
               weighted_f1=f1_score(t, pred, average="weighted", zero_division=0),
               zero_recall=int((rec == 0).sum()),
               classes_absent=absent)
    if not quiet:
        print(f"{name:<24} acc {row['accuracy']:.3f} | macro-F1 {row['macro_f1']:.3f} "
              f"| weighted-F1 {row['weighted_f1']:.3f} | zero-recall {row['zero_recall']}/{K}"
              + (f"  [!! {absent} classes NEVER TESTED -- not model failure]" if absent else ""))
    return row, rec



# A/B/C train seperate models. the training data itself differs, so the leakage can only be removed by retraining. 
# D reuses B's model and only changes the TEST


runs, per_class, curves = [], {}, {}

for seed in SEEDS:
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    first = seed == SEEDS[0]

    # A. random by sentence
    tr, te = train_test_split(idx, test_size=.2, stratify=y_np, random_state=seed)
    _, pred, hist = train(tr, te, seed)
    r, rec = score("A leaky (sentence)", te, pred, quiet=not first)
    runs.append({**r, "seed": seed})
    if first: per_class["A_leaky"], curves["leaky"] = rec, hist

    # B. grouped by lecture
    tr, te_g = next(sgkf.split(idx, y_np, groups=meta.lecture_id))
    head, pred, hist = train(tr, te_g, seed)
    r, rec = score("B grouped (lecture)", te_g, pred, quiet=not first)
    runs.append({**r, "seed": seed})
    if first: per_class["B_lecture"], curves["grouped"], head_b, te_b = rec, hist, head, te_g

    # C. grouped by course
    tr, te_c = next(sgkf.split(idx, y_np, groups=meta.course_id))
    _, pred, _ = train(tr, te_c, seed)
    r, rec = score("C grouped (course)", te_c, pred, quiet=not first)
    runs.append({**r, "seed": seed})
    if first: per_class["C_course"] = rec

    # D. Same weights as B, but eval only on real sentences
    te_r = te_g[real[te_g]]
    r, rec = score("D grouped, real only", te_r,
                   logits(head, T(te_r)).argmax(1).cpu().numpy(), quiet=not first)
    runs.append({**r, "seed": seed})
    if first: per_class["D_real_only"] = rec

# try 3 different seeds
res = pd.DataFrame(runs)
agg = res.groupby("eval").accuracy.agg(["mean", "std"]).round(3)
res.to_csv("results.csv", index=False)
print(f"\naccuracy across {len(SEEDS)} seeds (mean +/- std):")
print(agg.to_string())

# --- per-class: did the padded lecturers collapse on real speech? --------------
pc = pd.DataFrame(per_class, index=range(K))
pc.insert(0, "lecturer", names)
pc.insert(1, "synth_pct", (meta.groupby("label").is_synthetic.mean() * 100).round(1).values)
pc.sort_values("synth_pct", ascending=False).to_csv("per_class.csv", index=False)
p = pc[pc.synth_pct > 0]
print(f"\npadded lecturers ({len(p)}): mean recall {p.B_lecture.mean():.3f} "
      f"(fake in test) -> {p.D_real_only.mean():.3f} (real only)   <- fingerprint reveal")

# --- curves: converging (leaky) vs diverging (grouped). The overfit slide. ------
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
for ax, (h, t) in zip(axes, [(curves["leaky"], "A: random by sentence (LEAKY)"),
                             (curves["grouped"], "B: grouped by lecture")]):
    ax.plot(h[:, 0], label="train"); ax.plot(h[:, 1], label="validation")
    ax.set_title(f"{t}\nfinal gap = {h[-1, 0] - h[-1, 1]:.3f}")
    ax.set_xlabel("epoch"); ax.legend(); ax.grid(alpha=.3)
axes[0].set_ylabel("accuracy")
plt.tight_layout(); plt.savefig("curves.png", dpi=150); plt.close()

# --- aggregation: style is DISTRIBUTIONAL --------------------------------------
# pool more predictions together, add their log-probabilities, and see if the model can guess the lecturer better.
print()
rng = np.random.default_rng(SEEDS[0])
sizes, accs = [1, 2, 5, 10, 20, 50], []
logp = {lab: torch.log_softmax(logits(head_b, T(te_b[y_np[te_b] == lab])), 1).cpu().numpy()
        for lab in range(K) if (y_np[te_b] == lab).sum()}
for m in sizes:
    hit = tot = 0
    for lab, lp in logp.items():
        if len(lp) < m:
            continue
        for _ in range(200):
            pick = rng.choice(len(lp), m, replace=False)
            hit += int(lp[pick].sum(0).argmax() == lab)   # sum log-probs = combine evidence
            tot += 1
    accs.append(hit / max(tot, 1))
    print(f"  pooled {m:>2} sentences -> accuracy {accs[-1]:.3f}")

plt.figure(figsize=(6, 4)); plt.plot(sizes, accs, "o-"); plt.xscale("log")
plt.xlabel("sentences pooled (log scale)"); plt.ylabel("accuracy"); plt.grid(alpha=.3)
plt.title("style is distributional: one sentence is not enough")
plt.tight_layout(); plt.savefig("aggregation.png", dpi=150); plt.close()

torch.save(head_b.state_dict(), "head.pt")
print("\nwrote curves.png, aggregation.png, per_class.csv, results.csv, head.pt")
