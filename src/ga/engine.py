from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..evaluation.fitness import FitnessConfig, evaluate_ordering
from ..evaluation.policy_reference import ReferencePolicy
from .operators import tournament_select, order_crossover_ox, mutate


@dataclass(frozen=True)
class GAConfig:
    population_size: int = 60
    generations: int = 60
    tournament_k: int = 3
    crossover_rate: float = 0.90
    p_swap: float = 0.20
    p_inversion: float = 0.10
    elitism: int = 2
    patience: int = 15  # early stop if no improvement


@dataclass(frozen=True)
class GAResult:
    best_ordering: np.ndarray
    best_fitness: float
    history_best: list[float]
    history_mean: list[float]


def _init_population(rng: np.random.Generator, seed_perm: np.ndarray, pop_size: int) -> np.ndarray:
    """Initialize with a strong seed + nearby permutations + random rest."""

    n_rules = seed_perm.shape[0]
    pop = np.empty((pop_size, n_rules), dtype=np.int32)
    pop[0] = seed_perm

    # Create a few "nearby" individuals by applying small perturbations.
    num_near = min(pop_size - 1, 8)
    for i in range(1, num_near + 1):
        perm = seed_perm.copy()
        # Apply 1-3 random swaps
        swaps = int(rng.integers(1, 4))
        for _ in range(swaps):
            a, b = rng.choice(n_rules, size=2, replace=False)
            perm[a], perm[b] = perm[b], perm[a]
        pop[i] = perm

    # Fill remainder with random permutations
    base = np.arange(n_rules, dtype=np.int32)
    for i in range(num_near + 1, pop_size):
        pop[i] = rng.permutation(base)
    return pop


def run_ga(
    ref: ReferencePolicy,
    ga_cfg: GAConfig,
    fit_cfg: FitnessConfig,
    seed: int = 0,
) -> GAResult:
    """Run the permutation GA to optimize firewall rule ordering."""

    rng = np.random.default_rng(seed)
    n_rules = len(ref.ref_ordering)

    pop = _init_population(rng, ref.ref_ordering.astype(np.int32), ga_cfg.population_size)
    fitness = np.empty(ga_cfg.population_size, dtype=np.float64)

    def eval_all() -> None:
        for i in range(ga_cfg.population_size):
            f, _ = evaluate_ordering(pop[i], ref, fit_cfg)
            fitness[i] = f

    eval_all()

    best_idx = int(np.argmax(fitness))
    best = pop[best_idx].copy()
    best_fit = float(fitness[best_idx])
    history_best = [best_fit]
    history_mean = [float(fitness.mean())]
    no_improve = 0

    for _gen in range(ga_cfg.generations):
        elite_idx = np.argsort(fitness)[-ga_cfg.elitism :]
        elites = pop[elite_idx].copy()

        new_pop = np.empty_like(pop)
        new_pop[: ga_cfg.elitism] = elites

        fill = ga_cfg.elitism
        while fill < ga_cfg.population_size:
            i1 = tournament_select(rng, fitness, k=ga_cfg.tournament_k)
            i2 = tournament_select(rng, fitness, k=ga_cfg.tournament_k)
            p1, p2 = pop[i1], pop[i2]

            if rng.random() < ga_cfg.crossover_rate:
                child = order_crossover_ox(rng, p1, p2)
            else:
                child = p1.copy()

            child = mutate(rng, child, p_swap=ga_cfg.p_swap, p_inv=ga_cfg.p_inversion)
            new_pop[fill] = child
            fill += 1

        pop = new_pop
        eval_all()

        gen_best_idx = int(np.argmax(fitness))
        gen_best_fit = float(fitness[gen_best_idx])
        history_best.append(gen_best_fit)
        history_mean.append(float(fitness.mean()))

        if gen_best_fit > best_fit:
            best_fit = gen_best_fit
            best = pop[gen_best_idx].copy()
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= ga_cfg.patience:
            break

    return GAResult(best_ordering=best, best_fitness=best_fit, history_best=history_best, history_mean=history_mean)
