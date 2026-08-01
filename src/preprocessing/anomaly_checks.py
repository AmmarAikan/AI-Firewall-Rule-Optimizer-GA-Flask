from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..io.load_rules import Rule


@dataclass(frozen=True)
class AnomalySummary:
    """Lightweight anomaly indicators derived from traffic samples."""

    shadow_rate: float
    inactive_rules: int


def rule_specificity(rule: Rule) -> float:
    """Heuristic specificity score.

    Higher means narrower match (more specific). This is used for a strong
    baseline ordering ("specific-first").
    """

    src_span = max(1, rule.src_end - rule.src_start + 1)
    dst_span = max(1, rule.dst_end - rule.dst_start + 1)
    port_span = max(1, rule.port_end - rule.port_start + 1)

    # Log-scale to keep things stable.
    score = 0.0
    score += -np.log10(src_span)
    score += -np.log10(dst_span)
    score += -np.log10(port_span)
    score += 0.25 if rule.proto != 0 else 0.0
    return float(score)


def traffic_based_anomalies(first_match_positions: np.ndarray, n_rules: int) -> AnomalySummary:
    """Estimate shadowing as rules never hit by traffic samples."""

    used = np.bincount(first_match_positions[first_match_positions < n_rules], minlength=n_rules) > 0
    inactive = int((~used).sum())
    return AnomalySummary(shadow_rate=inactive / max(1, n_rules), inactive_rules=inactive)
