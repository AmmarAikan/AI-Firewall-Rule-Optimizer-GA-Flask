from __future__ import annotations

import numpy as np


def tournament_select(rng: np.random.Generator, fitness: np.ndarray, k: int = 3) -> int:
    """Return index of the selected individual via tournament selection."""

    n = fitness.shape[0]
    idx = rng.integers(0, n, size=k)
    best = idx[np.argmax(fitness[idx])]
    return int(best)


def order_crossover_ox(rng: np.random.Generator, p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    """Order Crossover (OX) for permutations."""

    n = p1.shape[0]
    a, b = sorted(rng.choice(n, size=2, replace=False).tolist())
    child = np.full(n, -1, dtype=p1.dtype)
    child[a : b + 1] = p1[a : b + 1]

    used = set(child[a : b + 1].tolist())
    fill_positions = [i for i in range(n) if child[i] == -1]
    fill_vals = [x for x in p2.tolist() if x not in used]
    child[fill_positions] = np.array(fill_vals, dtype=p1.dtype)
    return child


def mutate_swap(rng: np.random.Generator, perm: np.ndarray, p: float) -> np.ndarray:
    """Swap mutation: with probability p swap two elements."""

    if rng.random() >= p:
        return perm
    n = perm.shape[0]
    i, j = rng.choice(n, size=2, replace=False)
    out = perm.copy()
    out[i], out[j] = out[j], out[i]
    return out


def mutate_inversion(rng: np.random.Generator, perm: np.ndarray, p: float) -> np.ndarray:
    """Inversion mutation: reverse a random segment."""

    if rng.random() >= p:
        return perm
    n = perm.shape[0]
    i, j = sorted(rng.choice(n, size=2, replace=False).tolist())
    out = perm.copy()
    out[i : j + 1] = out[i : j + 1][::-1]
    return out


def mutate(rng: np.random.Generator, perm: np.ndarray, p_swap: float, p_inv: float) -> np.ndarray:
    out = mutate_swap(rng, perm, p_swap)
    out = mutate_inversion(rng, out, p_inv)
    return out
