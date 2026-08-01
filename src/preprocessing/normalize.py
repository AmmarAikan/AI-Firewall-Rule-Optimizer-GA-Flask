from __future__ import annotations

import ipaddress
from typing import Any


PROTO_MAP = {
    "ANY": 0,
    "TCP": 1,
    "UDP": 2,
    "ICMP": 3,
}


def _parse_ip_range(val: Any) -> tuple[int, int, str]:
    """Return (start_int, end_int, normalized_str)."""
    if val is None:
        val = "ANY"
    s = str(val).strip().upper()
    if s in {"ANY", "*", "0.0.0.0/0"}:
        return 0, (2**32) - 1, "0.0.0.0/0"

    # CIDR
    if "/" in s:
        net = ipaddress.ip_network(s, strict=False)
        start = int(net.network_address)
        end = int(net.broadcast_address)
        return start, end, str(net)

    # Single IP
    ip = ipaddress.ip_address(s)
    v = int(ip)
    return v, v, str(ip)


def _parse_port_range(val: Any) -> tuple[int, int, str]:
    if val is None:
        val = "ANY"
    s = str(val).strip().upper()
    if s in {"ANY", "*"}:
        return 0, 65535, "ANY"
    if "-" in s:
        a, b = s.split("-", 1)
        start, end = int(a), int(b)
    else:
        start = end = int(s)
    if not (0 <= start <= end <= 65535):
        raise ValueError(f"Invalid port range: {val}")
    if start == end:
        return start, end, str(start)
    return start, end, f"{start}-{end}"


def _parse_proto(val: Any) -> tuple[int, str]:
    if val is None:
        val = "ANY"
    s = str(val).strip().upper()
    if s not in PROTO_MAP:
        raise ValueError(f"Invalid protocol: {val}. Use TCP/UDP/ICMP/ANY")
    return PROTO_MAP[s], s


def _parse_action(val: Any) -> tuple[int, str]:
    if val is None:
        raise ValueError("Missing action")
    s = str(val).strip().upper()
    if s in {"ALLOW", "ACCEPT", "PERMIT", "1", "TRUE"}:
        return 1, "ALLOW"
    if s in {"DENY", "DROP", "REJECT", "0", "FALSE"}:
        return 0, "DENY"
    raise ValueError(f"Invalid action: {val}. Use ALLOW or DENY")


def normalize_rule_dict(raw: dict[str, Any]) -> dict[str, Any]:
    """Normalize a rule dictionary and return numeric ranges.

    Required keys (case-insensitive accepted):
      - source_ip
      - destination_ip
      - port
      - protocol
      - action
    """

    # allow a few aliases
    def g(*keys: str) -> Any:
        for k in keys:
            if k in raw:
                return raw[k]
            # case-insensitive
            for kk in raw.keys():
                if kk.lower() == k.lower():
                    return raw[kk]
        return None

    src_val = g("source_ip", "src", "src_ip")
    dst_val = g("destination_ip", "dst", "dst_ip")
    port_val = g("port", "dst_port")
    proto_val = g("protocol", "proto")
    action_val = g("action")

    src_start, src_end, src_str = _parse_ip_range(src_val)
    dst_start, dst_end, dst_str = _parse_ip_range(dst_val)
    port_start, port_end, port_str = _parse_port_range(port_val)
    proto, proto_str = _parse_proto(proto_val)
    action, action_str = _parse_action(action_val)

    normalized_raw = {
        "source_ip": src_str,
        "destination_ip": dst_str,
        "port": port_str,
        "protocol": proto_str,
        "action": action_str,
    }

    return {
        "src_start": src_start,
        "src_end": src_end,
        "dst_start": dst_start,
        "dst_end": dst_end,
        "port_start": port_start,
        "port_end": port_end,
        "proto": proto,
        "action": action,
        "raw": normalized_raw,
    }
