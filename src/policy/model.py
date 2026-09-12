"""The policy, which is an ensemble on purpose.

SPEC.md asks the agent to show how sure it is while driving, and the rule there
is that confidence must be *earned*: it is disagreement between independently
trained heads, never `abs(steering)` and never a single network's softmax. So
the policy is an ensemble from the first line of code rather than a single net
that gets a confidence estimate bolted on later — you cannot retrofit
disagreement onto something that only ever had one opinion.

Each head sees the same observations in a different order and starts from a
different initialisation. Where the road is familiar they converge and the
spread is small; on geometry none of them has seen — a London roundabout, say —
they part company, and that spread is the signal.
"""

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


class Head(nn.Module):
    def __init__(self, n_obs, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_obs, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1), nn.Tanh(),      # steering in [-1, 1]
        )

    def forward(self, x):
        return self.net(x)


class Ensemble(nn.Module):
    def __init__(self, n_obs, n_heads=5, hidden=64, obs_version=1, world="mk64"):
        super().__init__()
        self.n_obs = n_obs
        self.obs_version = obs_version
        self.world = world
        self.heads = nn.ModuleList([Head(n_obs, hidden) for _ in range(n_heads)])

    def forward(self, x):
        return torch.stack([h(x) for h in self.heads])      # (H, B, 1)

    @torch.no_grad()
    def act(self, obs):
        """Return (steer, confidence, spread) for one observation.

        `spread` is the standard deviation of the heads' proposed steering, in
        steering units. `confidence` maps that onto 0..1 only for display; the
        spread is the honest quantity and is what gets calibrated against
        outcomes.
        """
        x = torch.as_tensor(np.asarray(obs, dtype=np.float32)).reshape(1, -1)
        out = self(x).reshape(-1)               # (H,)
        steer = float(out.mean())
        spread = float(out.std(unbiased=False))
        return steer, 1.0 / (1.0 + 8.0 * spread), spread

    # --- persistence -----------------------------------------------------
    def save(self, path, meta=None):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict": self.state_dict(),
            "n_obs": self.n_obs,
            "n_heads": len(self.heads),
            "obs_version": self.obs_version,
            "world": self.world,
            "meta": meta or {},
        }, path)
        return path

    @classmethod
    def load(cls, path, expect_obs_version=None, expect_n_obs=None):
        blob = torch.load(path, weights_only=False)
        # A policy names the world and observation version it was trained
        # against and refuses a mismatch. Silently reinterpreting input 7 is
        # not a failure anyone notices until the numbers are already wrong.
        if expect_obs_version is not None and blob["obs_version"] != expect_obs_version:
            raise ValueError(
                f"policy expects observation v{blob['obs_version']}, "
                f"caller has v{expect_obs_version}")
        if expect_n_obs is not None and blob["n_obs"] != expect_n_obs:
            raise ValueError(
                f"policy takes {blob['n_obs']} inputs, caller offers {expect_n_obs}")
        m = cls(blob["n_obs"], blob["n_heads"], obs_version=blob["obs_version"],
                world=blob["world"])
        m.load_state_dict(blob["state_dict"])
        m.eval()
        return m, blob.get("meta", {})
