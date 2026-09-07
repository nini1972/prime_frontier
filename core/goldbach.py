#!/usr/bin/env python3
"""
core/goldbach.py - Goldbach Partition Counting Engine

For every even n >= 4, counts g(n) = #{(p, q) : p <= q primes, p + q = n},
the number of "Goldbach partitions" of n. Goldbach's conjecture (1742, still
unproven) asserts g(n) >= 1 for all even n >= 4. Empirically g(n) grows
roughly like n / (2 ln^2 n) on average (Hardy-Littlewood heuristic), with rare
"hard" numbers that set new record lows.
"""

import math
import numpy as np
from typing import Dict, List

from core.sieve import generate_primes


def goldbach_partition_counts(limit: int) -> Dict[int, int]:
    """Compute g(n) for every even n in [4, limit]."""
    if limit < 4:
        return {}

    primes = generate_primes(limit)
    is_prime = np.zeros(limit + 1, dtype=bool)
    is_prime[primes] = True

    counts: Dict[int, int] = {}
    for n in range(4, limit + 1, 2):
        half = n // 2
        # primes is sorted; only need those <= n/2 (avoids double-counting p<=q)
        idx = np.searchsorted(primes, half, side="right")
        candidate_primes = primes[:idx]
        complements = n - candidate_primes
        counts[n] = int(np.count_nonzero(is_prime[complements]))

    return counts


def goldbach_stats(counts: Dict[int, int]) -> Dict:
    """Summarize Goldbach partition growth, record-low counts, and HL comparison."""
    if not counts:
        return {}

    ns = np.array(list(counts.keys()))
    gs = np.array(list(counts.values()))

    # Record lows: n whose g(n) is a new minimum among all n' <= n (n=4 -> g=1 seeds it)
    record_lows: List[Dict] = []
    current_min = math.inf
    for n, g in counts.items():
        if g < current_min:
            current_min = g
            record_lows.append({"n": int(n), "g": int(g)})

    # Rough Hardy-Littlewood-style heuristic tail comparison (unconstant-corrected form)
    tail_ns = [int(n) for n in ns[-5:]]
    hl_estimate_tail = {n: round(n / (2.0 * math.log(n) ** 2), 2) for n in tail_ns}

    return {
        "max_n": int(ns[-1]),
        "count_checked": len(ns),
        "min_g": int(gs.min()),
        "max_g": int(gs.max()),
        "mean_g": round(float(gs.mean()), 2),
        "record_lows": record_lows[-6:],
        "hl_estimate_tail": hl_estimate_tail,
        "verified_no_counterexample": bool(np.all(gs >= 1)),
    }


if __name__ == "__main__":
    print("Testing Goldbach Partition Engine up to 20,000...")
    counts = goldbach_partition_counts(20_000)
    stats = goldbach_stats(counts)
    print(f"Checked {stats['count_checked']} even numbers. Goldbach holds (no counterexample): "
          f"{stats['verified_no_counterexample']}")
    print(f"Mean g(n): {stats['mean_g']}, Min g(n): {stats['min_g']}, Max g(n): {stats['max_g']}")
    print(f"Record-low partition counts: {stats['record_lows']}")
    print(f"Hardy-Littlewood tail estimate: {stats['hl_estimate_tail']}")
    assert stats["verified_no_counterexample"], "Goldbach counterexample found?! Check the sieve."
    print("Goldbach Engine verified!")
