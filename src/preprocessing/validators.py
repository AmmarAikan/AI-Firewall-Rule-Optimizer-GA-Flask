from __future__ import annotations

from ..io.load_rules import Rule


def validate_rules(rules: list[Rule]) -> None:
    """Basic validation checks.

    This is intentionally lightweight (fast) because the project focuses on
    optimization. Any fatal issues raise ValueError.
    """

    if not rules:
        raise ValueError("No rules provided")

    for r in rules:
        if r.src_start > r.src_end or r.dst_start > r.dst_end:
            raise ValueError(f"Invalid IP range in rule {r.rule_id}")
        if r.port_start > r.port_end:
            raise ValueError(f"Invalid port range in rule {r.rule_id}")
        if r.proto not in (0, 1, 2, 3):
            raise ValueError(f"Invalid proto in rule {r.rule_id}")
        if r.action not in (0, 1):
            raise ValueError(f"Invalid action in rule {r.rule_id}")
