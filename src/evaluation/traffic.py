from __future__ import annotations

import ipaddress
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TrafficSamples:
    """Vectorized traffic samples."""

    src_ip: np.ndarray  # uint32
    dst_ip: np.ndarray  # uint32
    dst_port: np.ndarray  # uint16
    proto: np.ndarray  # uint8 (1=TCP,2=UDP,3=ICMP)


def ip_to_int(ip: str) -> int:
    return int(ipaddress.ip_address(ip))


def generate_traffic(n_packets: int, seed: int = 0) -> TrafficSamples:
    """Generate synthetic traffic.

    The distribution is intentionally mixed:
      - internal <-> internal
      - internal <-> external
      - random ports with heavier weight on common services
    """

    rng = np.random.default_rng(seed)
    # Internal segments
    internal_prefixes = [
        (ip_to_int("10.0.0.0"), 8),
        (ip_to_int("192.168.0.0"), 16),
        (ip_to_int("172.16.0.0"), 12),
    ]

    def sample_from_prefix(base_int: int, prefix_len: int, size: int) -> np.ndarray:
        host_bits = 32 - prefix_len
        max_hosts = 2 ** host_bits
        offsets = rng.integers(0, max_hosts, size=size, dtype=np.uint32)
        return (np.uint32(base_int) + offsets).astype(np.uint32)

    internal_choice = rng.integers(0, len(internal_prefixes), size=n_packets)
    src_is_internal = rng.random(n_packets) < 0.75
    dst_is_internal = rng.random(n_packets) < 0.55

    src = np.empty(n_packets, dtype=np.uint32)
    dst = np.empty(n_packets, dtype=np.uint32)
    for i in range(len(internal_prefixes)):
        base, pre = internal_prefixes[i]
        mask = internal_choice == i
        # src
        src_mask = mask & src_is_internal
        src[src_mask] = sample_from_prefix(base, pre, int(src_mask.sum()))
        # dst
        dst_mask = mask & dst_is_internal
        dst[dst_mask] = sample_from_prefix(base, pre, int(dst_mask.sum()))

    # external IPs (avoid private ranges; synthetic simplification)
    ext_src_mask = ~src_is_internal
    ext_dst_mask = ~dst_is_internal
    src[ext_src_mask] = rng.integers(1, 2**32 - 2, size=int(ext_src_mask.sum()), dtype=np.uint32)
    dst[ext_dst_mask] = rng.integers(1, 2**32 - 2, size=int(ext_dst_mask.sum()), dtype=np.uint32)

    # Protocol distribution
    proto = rng.choice(np.array([1, 2, 3], dtype=np.uint8), size=n_packets, p=[0.55, 0.40, 0.05])

    # Ports: bias toward common ones
    common_ports = np.array([22, 53, 80, 443, 123, 445, 3389], dtype=np.uint16)
    use_common = rng.random(n_packets) < 0.55
    ports = rng.integers(1, 65535, size=n_packets, dtype=np.uint16)
    ports[use_common] = rng.choice(common_ports, size=int(use_common.sum()))

    return TrafficSamples(src_ip=src, dst_ip=dst, dst_port=ports, proto=proto)
