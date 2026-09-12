"""Train a driving policy in the sim with evolution strategies.

The sim already steps N karts at once, so the population *is* the batch: give
every kart its own weights and one rollout evaluates the whole generation. No
gradients, no credit assignment, and it suits a long episode with delayed reward.

Five independent runs from different seeds give five policies that are the
ensemble — and because they are separately evolved rather than copies, their
disagreement is real. That is the confidence signal SPEC.md asks for.
"""

import numpy as np

H = 32          # hidden units


def n_params(n_obs, hidden=H):
    return n_obs * hidden + hidden + hidden + 1


def unpack(theta, n_obs, hidden=H):
    """(P, D) -> batched weights, so every kart runs its own policy."""
    p = 0
    w1 = theta[:, p:p + n_obs * hidden].reshape(-1, n_obs, hidden); p += n_obs * hidden
    b1 = theta[:, p:p + hidden]; p += hidden
    w2 = theta[:, p:p + hidden].reshape(-1, hidden, 1); p += hidden
    b2 = theta[:, p:p + 1]
    return w1, b1, w2, b2


def act(theta, obs, n_obs, hidden=H):
    w1, b1, w2, b2 = unpack(theta, n_obs, hidden)
    h = np.tanh(np.einsum("ni,nij->nj", obs, w1) + b1)
    return np.tanh(np.einsum("nj,njk->nk", h, w2) + b2)[:, 0]


def rollout(sim, theta, frames=900, jitter_lateral=35.0, jitter_yaw=0.12,
            start_cp=0, seed=None):
    """Score a whole population in one pass. Fitness = ground covered, honestly."""
    if seed is not None:
        sim.rng = np.random.default_rng(seed)
    obs = sim.reset(jitter_lateral=jitter_lateral, jitter_yaw=jitter_yaw,
                    start_cp=start_cp)
    n_obs = obs.shape[1]
    total = np.zeros(sim.n)
    offroad = np.zeros(sim.n)
    for _ in range(frames):
        steer = act(theta, obs, n_obs)
        obs = sim.step(steer)
        total += sim.progress()
        offroad += (np.abs(sim._project()[2]) > sim.p.half_width)
    # Progress is the reward. Time off the road is charged for separately so a
    # policy cannot win by cutting the corner across the grass.
    return total - 0.35 * offroad * sim.p.top_speed, total, offroad


def randomise(base, rng):
    """Jitter the physics each generation so the policy cannot overfit the model.

    The sim is a fit, not a port, and its residual error is exactly what broke
    the first transfer attempt. Training across a spread of plausible physics
    buys robustness to that error instead of a policy tuned to one wrong number.
    """
    from dataclasses import replace
    tt = np.asarray(base.turn_table, dtype=float).copy()
    tt[:, 1] *= rng.uniform(0.70, 1.45)
    at = np.asarray(base.accel_table, dtype=float).copy()
    at[:, 1] *= rng.uniform(0.80, 1.25)
    return replace(base,
                   turn_table=tuple(map(tuple, tt)),
                   accel_table=tuple(map(tuple, at)),
                   top_speed=base.top_speed * rng.uniform(0.9, 1.1),
                   half_width=base.half_width * rng.uniform(0.75, 1.15))


def evolve(sim, generations=60, elite_frac=0.125, sigma=0.35, decay=0.97,
           frames=900, seed=0, verbose=True, domain_random=True,
           jitter_lateral=55.0, jitter_yaw=0.30, random_start=True):
    rng = np.random.default_rng(seed)
    n_obs = sim.observe().shape[1]
    D = n_params(n_obs)
    P = sim.n
    mean = np.zeros(D)
    base_physics = sim.p
    n_elite = max(4, int(P * elite_frac))
    best = None

    for g in range(generations):
        theta = mean + sigma * rng.standard_normal((P, D))
        # Same start and same physics for everyone in a generation (fair),
        # different between generations (so nothing wins by memorising one
        # launch or one set of handling).
        if domain_random:
            sim.p = randomise(base_physics, rng)
        start = int(rng.integers(0, len(sim.track.centre))) if random_start else 0
        fit, prog, off = rollout(sim, theta, frames=frames, seed=1000 + g,
                                 jitter_lateral=jitter_lateral,
                                 jitter_yaw=jitter_yaw, start_cp=start)
        order = np.argsort(-fit)
        elite = theta[order[:n_elite]]
        mean = elite.mean(axis=0)
        sigma *= decay
        best = (float(fit[order[0]]), float(prog[order[0]]), mean.copy())
        if verbose and (g % 5 == 0 or g == generations - 1):
            frac = prog[order[0]] / sim.track.length
            print(f"  gen {g:3d}  best fitness {fit[order[0]]:8.0f}  "
                  f"progress {prog[order[0]]:7.0f} ({frac:4.2f} laps)  "
                  f"offroad {off[order[0]]:5.0f}f  sigma {sigma:.3f}", flush=True)
    sim.p = base_physics
    return mean, best
