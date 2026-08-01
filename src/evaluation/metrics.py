from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

from ..preprocessing.anomaly_checks import traffic_based_anomalies


@dataclass(frozen=True)
class EvalMetrics:
    coverage: float
    avg_checks: float
    shadow_rate: float
    inactive_rules: int

    def to_dict(self) -> dict:
        return asdict(self)


def compute_metrics(
    decisions: np.ndarray,
    ref_decisions: np.ndarray,
    first_pos: np.ndarray,
    avg_checks: float,
    n_rules: int,
) -> EvalMetrics:
    coverage = float((decisions == ref_decisions).mean())
    an = traffic_based_anomalies(first_pos, n_rules)
    return EvalMetrics(
        coverage=coverage,
        avg_checks=avg_checks,
        shadow_rate=float(an.shadow_rate),
        inactive_rules=int(an.inactive_rules),
    )
