from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .metrics import EvalMetrics, compute_metrics
from .policy_reference import apply_order, ReferencePolicy


@dataclass(frozen=True)
class FitnessConfig:
    """Fitness weights and constraints."""

    # Hard-ish constraint: if coverage falls below this threshold, penalize heavily.
    min_coverage: float = 0.990
    # Soft weights
    w_coverage: float = 0.70
    w_shadow: float = 0.15
    w_time: float = 0.15


def evaluate_ordering(
    ordering: np.ndarray,
    ref: ReferencePolicy,
    cfg: FitnessConfig,
    default_action: int = 0,
) -> tuple[float, EvalMetrics]:
    """Return (fitness, metrics) for an ordering."""

    decisions, first_pos, avg_checks = apply_order(ordering, ref.match_matrix, ref.actions, default_action)
    metrics = compute_metrics(decisions, ref.ref_decisions, first_pos, avg_checks, len(ordering))

    avg_checks_norm = metrics.avg_checks / max(1.0, len(ordering))
    shadow_norm = metrics.shadow_rate

    # Hard penalty if coverage drops too much.
    if metrics.coverage < cfg.min_coverage:
        deficit = cfg.min_coverage - metrics.coverage
        return -1000.0 * float(deficit) - 100.0 * float(avg_checks_norm), metrics

    fitness = (
        cfg.w_coverage * metrics.coverage
        - cfg.w_shadow * shadow_norm
        - cfg.w_time * avg_checks_norm
    )
    return float(fitness), metrics
