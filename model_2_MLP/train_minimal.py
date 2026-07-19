#!/usr/bin/env python3
"""
Just the training, no split-comparison / analysis reveal (see train.py for that).

  in : X.npy (n,512), y.npy (n,)          -- precomputed embeddings + labels
  out: head.pt                             -- trained classifier head
"""
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split

SEED, EPOCHS, BATCH, LR = 42, 60, 1024, 1e-3
DEV = "cuda" if torch.cuda.is_available() else "cpu"

# 1. load embeddings + labels
X = torch.tensor(np.load("X.npy").astype(np.float32), device=DEV)
y = torch.tensor(np.load("y.npy").astype(np.int64), device=DEV)
K = int(y.max()) + 1

# 2. plain random split
idx = np.arange(len(X))
tr, te = train_test_split(idx, test_size=0.2, stratify=y.cpu(), random_state=SEED)
tr, te = torch.as_tensor(tr, device=DEV), torch.as_tensor(te, device=DEV)

# 3. model: 512 -> 2048 -> 2048 -> 2048 -> K
torch.manual_seed(SEED)
head = nn.Sequential(
    nn.Linear(512, 2048), nn.ReLU(),
    nn.Linear(2048, 2048), nn.ReLU(),
    nn.Linear(2048, 2048), nn.ReLU(),
    nn.Linear(2048, K),
).to(DEV)
opt = torch.optim.Adam(head.parameters(), lr=LR)

# 4. train
for epoch in range(EPOCHS):
    head.train()
    for batch in torch.randperm(len(tr), device=DEV).split(BATCH):
        i = tr[batch]
        opt.zero_grad()
        nn.functional.cross_entropy(head(X[i]), y[i]).backward()
        opt.step()

# 5. save
head.eval()
with torch.no_grad():
    test_acc = (head(X[te]).argmax(1) == y[te]).float().mean().item()
print(f"test accuracy: {test_acc:.3f}")

torch.save(head.state_dict(), "head.pt")
print("wrote head.pt")
