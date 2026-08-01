from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def generate_rules(n_rules: int, seed: int = 0) -> list[dict]:
    """Generate a synthetic firewall ruleset.

    The generator purposely creates:
      - common service rules (HTTP/HTTPS/DNS/SSH...)
      - broader "macro" rules with the SAME action as their more specific rules,
        which creates shadowing/redundancy that can be reduced by re-ordering
      - a small amount of opposite-action overlap to make the optimization non-trivial
    """

    rng = np.random.default_rng(seed)

    internal_nets = [
        "10.0.0.0/8",
        "192.168.0.0/16",
        "172.16.0.0/12",
    ]
    external_nets = [
        "0.0.0.0/0",
        "8.0.0.0/8",
        "1.1.1.0/24",
        "52.0.0.0/8",
    ]

    services = [
        ("TCP", 80, "ALLOW"),
        ("TCP", 443, "ALLOW"),
        ("UDP", 53, "ALLOW"),
        ("TCP", 22, "ALLOW"),
        ("UDP", 123, "ALLOW"),
        ("TCP", 3389, "DENY"),
        ("TCP", 445, "DENY"),
    ]

    rules: list[dict] = []

    # 1) Create base service rules (specific)
    base_count = max(30, int(0.35 * n_rules))
    for _ in range(base_count):
        proto, port, action = services[rng.integers(0, len(services))]
        src = internal_nets[rng.integers(0, len(internal_nets))]
        # Half the time destination is external, half internal.
        dst = (
            external_nets[rng.integers(0, len(external_nets))]
            if rng.random() < 0.6
            else internal_nets[rng.integers(0, len(internal_nets))]
        )
        rules.append(
            {
                "source_ip": src,
                "destination_ip": dst,
                "port": str(port),
                "protocol": proto,
                "action": action,
            }
        )

    # 2) Add broader rules with same action (creates shadowing/redundancy)
    broad_count = max(20, int(0.25 * n_rules))
    for _ in range(broad_count):
        proto, port, action = services[rng.integers(0, len(services))]
        # broaden either src or dst or port
        src = "0.0.0.0/0" if rng.random() < 0.4 else internal_nets[rng.integers(0, len(internal_nets))]
        dst = "0.0.0.0/0" if rng.random() < 0.4 else external_nets[rng.integers(0, len(external_nets))]
        if rng.random() < 0.35:
            port_field = "ANY"
        else:
            # small port range around the service
            start = max(1, port - int(rng.integers(0, 10)))
            end = min(65535, port + int(rng.integers(0, 10)))
            port_field = f"{start}-{end}" if start != end else str(port)

        proto_field = proto if rng.random() < 0.75 else "ANY"
        rules.append(
            {
                "source_ip": src,
                "destination_ip": dst,
                "port": port_field,
                "protocol": proto_field,
                "action": action,
            }
        )

    # 3) Sprinkle some opposite-action overlaps (small amount)
    # Keep opposite-action overlaps relatively small so the optimizer can preserve
    # policy behavior (coverage) while improving performance.
    overlap_count = max(6, int(0.04 * n_rules))
    for _ in range(overlap_count):
        proto, port, action = services[rng.integers(0, len(services))]
        flipped = "DENY" if action == "ALLOW" else "ALLOW"
        src = internal_nets[rng.integers(0, len(internal_nets))]
        dst = external_nets[rng.integers(0, len(external_nets))]
        rules.append(
            {
                "source_ip": src,
                "destination_ip": dst,
                "port": str(port),
                "protocol": proto,
                "action": flipped,
            }
        )

    # 4) Fill remainder with random-ish rules
    while len(rules) < n_rules:
        src = internal_nets[rng.integers(0, len(internal_nets))] if rng.random() < 0.8 else "0.0.0.0/0"
        dst = external_nets[rng.integers(0, len(external_nets))] if rng.random() < 0.7 else "0.0.0.0/0"
        proto = rng.choice(["TCP", "UDP", "ICMP", "ANY"], p=[0.45, 0.35, 0.05, 0.15])
        if proto == "ICMP":
            port = "ANY"
        else:
            if rng.random() < 0.25:
                port = "ANY"
            else:
                a = int(rng.integers(1, 65535))
                b = int(rng.integers(1, 65535))
                start, end = (a, b) if a <= b else (b, a)
                port = str(start) if start == end else f"{start}-{end}"

        action = rng.choice(["ALLOW", "DENY"], p=[0.55, 0.45])
        rules.append(
            {
                "source_ip": src,
                "destination_ip": dst,
                "port": port,
                "protocol": proto,
                "action": action,
            }
        )

    # Shuffle to simulate a messy real-world ruleset.
    rng.shuffle(rules)
    return rules


def save_rules_json(rules: list[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rules, indent=2), encoding="utf-8")
