"""Behaviour cloning: learn the search's choices so training never starts blind.

Exploring from a random policy on a track means thousands of episodes spent
discovering that walls are bad. The segment search already knows how to get
round; cloning it is the cheap way to begin somewhere sensible, and it is what
the later expert-iteration loop refines rather than replaces.

Each head trains on its own bootstrap resample of the lap, which is what makes
the heads disagree honestly later — identical training data would give identical
heads and a confidence signal that is flat everywhere.
"""

import numpy as np
import torch


def train(model, obs, actions, epochs=400, lr=1e-3, batch=128, seed=0, verbose=True):
    obs = torch.as_tensor(np.asarray(obs, dtype=np.float32))
    actions = torch.as_tensor(np.asarray(actions, dtype=np.float32)).reshape(-1, 1)
    n = len(obs)
    g = torch.Generator().manual_seed(seed)
    opts = [torch.optim.Adam(h.parameters(), lr=lr) for h in model.heads]
    loss_fn = torch.nn.MSELoss()

    # One bootstrap resample per head: same lap, different view of it.
    idx = [torch.randint(0, n, (n,), generator=g) for _ in model.heads]

    for epoch in range(epochs):
        losses = []
        for h, opt, ix in zip(model.heads, opts, idx):
            perm = ix[torch.randperm(n, generator=g)]
            tot = 0.0
            for k in range(0, n, batch):
                b = perm[k:k + batch]
                opt.zero_grad()
                loss = loss_fn(h(obs[b]), actions[b])
                loss.backward()
                opt.step()
                tot += float(loss) * len(b)
            losses.append(tot / n)
        if verbose and (epoch % 100 == 0 or epoch == epochs - 1):
            print(f"  epoch {epoch:4d}  per-head MSE "
                  + " ".join(f"{l:.4f}" for l in losses), flush=True)
    return model
