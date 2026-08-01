from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from ..preprocessing.normalize import normalize_rule_dict


@dataclass(frozen=True)
class Rule:
    """A normalized firewall rule.

    All fields are normalized into numeric ranges for fast evaluation.
    """

    rule_id: int
    src_start: int
    src_end: int
    dst_start: int
    dst_end: int
    port_start: int
    port_end: int
    proto: int  # 0=ANY, 1=TCP, 2=UDP, 3=ICMP
    action: int  # 1=ALLOW, 0=DENY
    raw: dict[str, Any]


def load_rules(path: str | Path) -> list[Rule]:
    """Load firewall rules from JSON or CSV.

    Expected columns/keys:
      - source_ip, destination_ip, port, protocol, action
    Where "port" can be:
      - single int "443"
      - range "1000-2000"
      - "ANY"
    And IPs can be:
      - CIDR (e.g., 10.0.0.0/8)
      - single IP (e.g., 192.168.1.10)
      - "ANY"
    """

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Rules file not found: {path}")

    ext = path.suffix.lower()
    if ext == ".json":
        items = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(items, dict) and "rules" in items:
            items = items["rules"]
        if not isinstance(items, list):
            raise ValueError("JSON rules file must contain a list of rule objects")
    elif ext == ".csv":
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            items = list(reader)
    else:
        raise ValueError("Unsupported rules file type. Use .json or .csv")

    rules: list[Rule] = []
    for idx, raw in enumerate(items):
        if not isinstance(raw, dict):
            raise ValueError("Each rule must be an object/dict")
        norm = normalize_rule_dict(raw)
        rules.append(
            Rule(
                rule_id=idx,
                src_start=norm["src_start"],
                src_end=norm["src_end"],
                dst_start=norm["dst_start"],
                dst_end=norm["dst_end"],
                port_start=norm["port_start"],
                port_end=norm["port_end"],
                proto=norm["proto"],
                action=norm["action"],
                raw=norm["raw"],
            )
        )
    return rules


def export_rules_json(rules: Iterable[Rule], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = []
    for r in rules:
        payload.append(
            {
                "rule_id": r.rule_id,
                "source_ip": r.raw.get("source_ip"),
                "destination_ip": r.raw.get("destination_ip"),
                "port": r.raw.get("port"),
                "protocol": r.raw.get("protocol"),
                "action": "ALLOW" if r.action == 1 else "DENY",
            }
        )
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
