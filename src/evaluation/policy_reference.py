from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from ..io.load_rules import Rule
from .traffic import TrafficSamples


def build_match_matrix(samples: TrafficSamples, rules: list[Rule]) -> np.ndarray:
    """Build a boolean matrix M[packet, rule] = True if rule matches packet.

    This is precomputed once to make GA evaluations fast.
    """

    p = samples.src_ip.shape[0]
    n = len(rules)
    M = np.zeros((p, n), dtype=bool)

    src = samples.src_ip.astype(np.uint32)
    dst = samples.dst_ip.astype(np.uint32)
    port = samples.dst_port.astype(np.uint16)
    proto = samples.proto.astype(np.uint8)

    for j, r in enumerate(rules):
        src_ok = (src >= np.uint32(r.src_start)) & (src <= np.uint32(r.src_end))
        dst_ok = (dst >= np.uint32(r.dst_start)) & (dst <= np.uint32(r.dst_end))
        port_ok = (port >= np.uint16(r.port_start)) & (port <= np.uint16(r.port_end))
        if r.proto == 0:
            proto_ok = np.ones(p, dtype=bool)
        else:
            proto_ok = proto == np.uint8(r.proto)
        M[:, j] = src_ok & dst_ok & port_ok & proto_ok
    return M


def apply_order(
    ordering: np.ndarray,
    match_matrix: np.ndarray,
    actions: np.ndarray,
    default_action: int = 0,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Apply an ordering to traffic samples.

    Returns:
      - decisions: array of 0/1 per packet
      - first_pos: position in ordering of the first match (or N if none)
      - avg_checks: average number of rules checked per packet
    """

    n = ordering.shape[0]
    ordered_matches = match_matrix[:, ordering]
    any_match = ordered_matches.any(axis=1)
    first_pos = np.argmax(ordered_matches, axis=1)
    first_pos = np.where(any_match, first_pos, n).astype(np.int32)

    actions_perm = actions[ordering]
    decisions = np.full(first_pos.shape[0], default_action, dtype=np.int8)
    mask = first_pos < n
    if mask.any():
        decisions[mask] = actions_perm[first_pos[mask]]

    checks = np.where(first_pos < n, first_pos + 1, n)
    avg_checks = float(checks.mean())
    return decisions, first_pos, avg_checks


@dataclass(frozen=True)
class ReferencePolicy:
    match_matrix: np.ndarray
    actions: np.ndarray
    ref_ordering: np.ndarray
    ref_decisions: np.ndarray


def make_reference_policy(
    rules: list[Rule],
    samples: TrafficSamples,
    ref_ordering: Optional[np.ndarray] = None,
) -> ReferencePolicy:
    if ref_ordering is None:
        ref_ordering = np.arange(len(rules), dtype=np.int32)
    actions = np.array([r.action for r in rules], dtype=np.int8)
    M = build_match_matrix(samples, rules)
    ref_decisions, _, _ = apply_order(ref_ordering, M, actions)
    return ReferencePolicy(match_matrix=M, actions=actions, ref_ordering=ref_ordering, ref_decisions=ref_decisions)


def save_match_matrix(path: str | Path, match_matrix: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, match_matrix)


def load_match_matrix(path: str | Path) -> np.ndarray:
    return np.load(Path(path))
