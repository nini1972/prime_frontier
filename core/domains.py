#!/usr/bin/env python3
"""
core/domains.py - Research Domain Registry for the Prime Frontier Agent World

Each `Domain` describes one distinct area of prime-number phenomena that the
autonomous agents can investigate. A domain packages:
  - `harvest()`   : a function that computes fresh empirical data using the
                    core engines (sieve, modular_orbits, zeta_harmonics, goldbach)
  - `prompt_template`: a research-brief template that turns that data into a
                    prompt for the reasoning model

Adding a new area of mathematics for the agents to explore is as simple as
writing a harvest function and registering a new `Domain` in `DOMAINS` below --
no changes needed anywhere else in the agent pipeline.
"""

import math
from dataclasses import dataclass
from typing import Any, Callable, Dict

from core.sieve import (
    generate_primes,
    compute_prime_gaps,
    compute_gap_statistics,
    find_maximal_gap_records,
)
from core.modular_orbits import compute_modular_transitions, compute_residue_autocorrelation
from core.goldbach import goldbach_partition_counts, goldbach_stats


@dataclass(frozen=True)
class Domain:
    id: str
    name: str
    description: str
    harvest: Callable[[], Dict[str, Any]]
    prompt_template: str


# Shared output-format contract every domain prompt ends with, so the parser
# in models/math_researcher.py can rely on a single stable schema regardless
# of which domain generated the prompt.
OUTPUT_FORMAT_INSTRUCTIONS = """
Your task:
Formulate ONE novel, mathematically precise conjecture or predictive heuristic grounded in the data above.
Then, write a self-contained Python script to test and either support or falsify your conjecture empirically.
IMPORTANT: Execution must complete in under 5 seconds. Do not use slow trial division.
You can directly import and use the fast project utilities:
    from core.sieve import generate_primes, compute_prime_gaps
Do not import third-party packages other than numpy and mpmath.

Format your output strictly as:
## CONJECTURE_NAME: <concise title>
## MATHEMATICAL_FORMULATION: <LaTeX formula and rigorous explanation>
## HYPOTHESIS_TARGET: <what exact quantitative threshold must hold>
## PYTHON_VERIFICATION:
```python
# Self-contained python script that prints 'VERDICT: SUPPORTED' or 'VERDICT: FALSIFIED' along with quantitative evidence
```
"""


# ---------------------------------------------------------------------------
# Domain: Prime gap dynamics
# ---------------------------------------------------------------------------
def _harvest_gap_dynamics() -> Dict[str, Any]:
    primes = generate_primes(1_300_000)[:100_000]
    gaps = compute_prime_gaps(primes)
    stats = compute_gap_statistics(primes, gaps)
    records = find_maximal_gap_records(primes, gaps)
    top = records[-1] if records else {}
    return {
        "mean_gap": stats.get("mean_gap"),
        "max_gap": stats.get("max_gap"),
        "twin_primes": f"{stats.get('twin_primes_count', 0):,}",
        "cousin_primes": f"{stats.get('cousin_primes_count', 0):,}",
        "sexy_primes": f"{stats.get('sexy_primes_count', 0):,}",
        "max_prime": f"{stats.get('max_prime', 0):,}",
        "latest_record_gap": top.get("gap"),
        "latest_record_prime": top.get("prime"),
        "latest_cramer_ratio": top.get("cramer_ratio"),
    }


GAP_DYNAMICS_PROMPT = """You are an elite research mathematician at the Prime Frontier laboratory.
Below is real empirical data computed on the first 100,000 prime numbers:

Gap Statistics (up to prime {max_prime}):
- Mean Gap: {mean_gap}
- Max Gap Observed: {max_gap}
- Twin Primes (g=2): {twin_primes}
- Cousin Primes (g=4): {cousin_primes}
- Sexy Primes (g=6): {sexy_primes} (most frequent nonzero gap)
- Latest maximal-gap record: gap {latest_record_gap} after prime {latest_record_prime}
  (Cramer ratio g / ln^2(p) = {latest_cramer_ratio})
""" + OUTPUT_FORMAT_INSTRUCTIONS


# ---------------------------------------------------------------------------
# Domain: Modular transition bias, modulo 10 (Lemke Oliver-Soundararajan)
# ---------------------------------------------------------------------------
def _harvest_modular_mod10() -> Dict[str, Any]:
    primes = generate_primes(1_300_000)[:100_000]
    mod10 = compute_modular_transitions(primes, mod=10)
    autocorr = compute_residue_autocorrelation(primes, mod=10, max_lag=5)
    m = mod10["transition_matrix"]
    return {
        "p11": round(m[0][0] * 100, 1),
        "p13": round(m[0][1] * 100, 1),
        "p17": round(m[0][2] * 100, 1),
        "p19": round(m[0][3] * 100, 1),
        "expected_uniform": round(mod10["expected_uniform"] * 100, 1),
        "autocorr_lag1": autocorr.get(1),
        "autocorr_lag2": autocorr.get(2),
    }


MODULAR_MOD10_PROMPT = """You are an elite research mathematician at the Prime Frontier laboratory.
Below is real empirical data on consecutive prime modular transitions modulo 10 (first 100,000 primes):

- P(1 -> 1): {p11}% (expected uniform: {expected_uniform}%, extreme repulsion)
- P(1 -> 3): {p13}% (attraction)
- P(1 -> 7): {p17}% (attraction)
- P(1 -> 9): {p19}%
- Residue autocorrelation excess at lag 1: {autocorr_lag1}
- Residue autocorrelation excess at lag 2: {autocorr_lag2}

This is the Lemke Oliver-Soundararajan (2016) phenomenon: consecutive primes exhibit short-range
repulsion/attraction modulo q that violates naive independence assumptions.
""" + OUTPUT_FORMAT_INSTRUCTIONS


# ---------------------------------------------------------------------------
# Domain: Modular transition bias, modulo 30 (finer residue structure)
# ---------------------------------------------------------------------------
def _harvest_modular_mod30() -> Dict[str, Any]:
    primes = generate_primes(1_300_000)[:100_000]
    mod30 = compute_modular_transitions(primes, mod=30)
    residues = mod30.get("residues", [])
    m = mod30.get("transition_matrix", [])
    expected = mod30.get("expected_uniform", 0.0)
    strongest_repulsion = None
    strongest_attraction = None
    if m and residues:
        best_low, best_high = 1.0, 0.0
        for i, r_i in enumerate(residues):
            for j, r_j in enumerate(residues):
                p = m[i][j]
                if p < best_low:
                    best_low, strongest_repulsion = p, (r_i, r_j)
                if p > best_high:
                    best_high, strongest_attraction = p, (r_i, r_j)
    return {
        "num_residues": len(residues),
        "expected_uniform": round(expected * 100, 2),
        "strongest_repulsion": strongest_repulsion,
        "strongest_repulsion_pct": round(best_low * 100, 2) if strongest_repulsion else None,
        "strongest_attraction": strongest_attraction,
        "strongest_attraction_pct": round(best_high * 100, 2) if strongest_attraction else None,
    }


MODULAR_MOD30_PROMPT = """You are an elite research mathematician at the Prime Frontier laboratory.
Below is real empirical data on consecutive prime modular transitions modulo 30 (first 100,000 primes),
i.e. the {num_residues} residue classes coprime to 30 ({{1,7,11,13,17,19,23,29}}):

- Expected uniform transition probability: {expected_uniform}%
- Strongest repulsion observed: residue {strongest_repulsion} at {strongest_repulsion_pct}%
- Strongest attraction observed: residue {strongest_attraction} at {strongest_attraction_pct}%

Modulo 30 refines the modulo-10 Lemke Oliver-Soundararajan bias since 30 = 2*3*5 removes more
small-prime obstructions, exposing finer-grained modular memory effects between consecutive primes.
""" + OUTPUT_FORMAT_INSTRUCTIONS


# ---------------------------------------------------------------------------
# Domain: Goldbach partitions
# ---------------------------------------------------------------------------
def _harvest_goldbach() -> Dict[str, Any]:
    counts = goldbach_partition_counts(20_000)
    stats = goldbach_stats(counts)
    return {
        "max_n": stats.get("max_n"),
        "min_g": stats.get("min_g"),
        "max_g": stats.get("max_g"),
        "mean_g": stats.get("mean_g"),
        "record_lows": stats.get("record_lows"),
        "hl_estimate_tail": stats.get("hl_estimate_tail"),
    }


GOLDBACH_PROMPT = """You are an elite research mathematician at the Prime Frontier laboratory.
Below is real empirical data on Goldbach partitions g(n) = #{{(p,q) : p<=q prime, p+q=n}}
for every even n up to {max_n}:

- Minimum g(n) observed: {min_g}
- Maximum g(n) observed: {max_g}
- Mean g(n): {mean_g}
- Record-low partition counts (new minima as n grows): {record_lows}
- Hardy-Littlewood-style heuristic tail estimate n/(2 ln^2 n): {hl_estimate_tail}

Goldbach's conjecture (1742, unproven in general) asserts g(n) >= 1 for all even n >= 4.
""" + OUTPUT_FORMAT_INSTRUCTIONS


# ---------------------------------------------------------------------------
# Domain: Riemann zeta harmonics
# ---------------------------------------------------------------------------
def _harvest_zeta() -> Dict[str, Any]:
    from core.zeta_harmonics import get_zeta_zeros

    gammas = get_zeta_zeros(15)
    gaps = [round(gammas[i + 1] - gammas[i], 4) for i in range(len(gammas) - 1)]
    return {
        "first_5_zeros": [round(g, 4) for g in gammas[:5]],
        "mean_zero_gap": round(sum(gaps) / len(gaps), 4) if gaps else None,
        "num_zeros": len(gammas),
    }


ZETA_HARMONICS_PROMPT = """You are an elite research mathematician at the Prime Frontier laboratory.
Below is real empirical data on the first {num_zeros} non-trivial Riemann zeta zeros
rho_k = 1/2 + i*gamma_k:

- First 5 imaginary parts gamma_k: {first_5_zeros}
- Mean spacing between consecutive gamma_k: {mean_zero_gap}

By Riemann's explicit formula, these zeros are the harmonic frequencies whose superposition
reconstructs the exact prime-counting staircase psi_0(x). The Montgomery pair correlation
conjecture predicts these spacings, when unfolded, follow GUE random matrix statistics.
""" + OUTPUT_FORMAT_INSTRUCTIONS


# ---------------------------------------------------------------------------
# Domain: Base Prime Powers & Sieve Mechanics
# ---------------------------------------------------------------------------
def _harvest_prime_powers() -> Dict[str, Any]:
    N = 1000
    sqrt_n = int(math.isqrt(N))
    primes = generate_primes(N)
    base_primes = [int(p) for p in primes if p <= sqrt_n and p > 2]

    # Highest powers p^k <= N with k >= 2
    powers_info = {}
    for p in base_primes:
        k = 2
        powers = []
        while p**k <= N:
            powers.append(p**k)
            k += 1
        if powers:
            powers_info[p] = powers

    # Elimination counts: odd multiples p * m <= N where m >= p is odd
    odds = list(range(3, N + 1, 2))
    eliminated = set()
    elim_counts = {}
    for p in base_primes:
        multiples = [p * m for m in range(p, N // p + 1, 2) if (p * m) not in eliminated]
        elim_counts[p] = len(multiples)
        eliminated.update(multiples)

    total_odd_composites = len(eliminated)
    total_odd_primes = len(odds) - total_odd_composites

    return {
        "N": N,
        "sqrt_N": sqrt_n,
        "base_primes": base_primes,
        "powers_info": powers_info,
        "sample_elim": {p: elim_counts[p] for p in base_primes[:6]},
        "total_odd_composites": total_odd_composites,
        "total_odd_primes": total_odd_primes,
        "pi_N": len(primes),
    }


PRIME_POWERS_PROMPT = """You are an elite research mathematician at the Prime Frontier laboratory.
Below is real empirical data on Base Prime Powers and Sieve Mechanics up to N = {N} (sqrt(N) = {sqrt_N}):

- Base odd primes p <= sqrt(N): {base_primes}
- Prime powers p^k <= {N} (k >= 2): {powers_info}
- In sieve theory, any composite number must have a prime factor <= sqrt(N).
- New odd composites for prime p first appear at p^2 (all smaller multiples p*m with m < p are crossed out by smaller primes).
- Number of new odd composites eliminated by each base prime starting at p^2: {sample_elim}
- Total odd composites in [1, {N}]: {total_odd_composites}
- Total odd primes in [1, {N}]: {total_odd_primes}
- Total primes pi({N}): {pi_N}
""" + OUTPUT_FORMAT_INSTRUCTIONS


DOMAINS: Dict[str, Domain] = {
    "gap_dynamics": Domain(
        id="gap_dynamics",
        name="Prime Gap Dynamics",
        description="Gap distribution, twin/cousin/sexy prime families, Cramer conjecture records.",
        harvest=_harvest_gap_dynamics,
        prompt_template=GAP_DYNAMICS_PROMPT,
    ),
    "modular_bias_mod10": Domain(
        id="modular_bias_mod10",
        name="Modular Transition Bias (mod 10)",
        description="Lemke Oliver-Soundararajan consecutive-prime residue bias modulo 10.",
        harvest=_harvest_modular_mod10,
        prompt_template=MODULAR_MOD10_PROMPT,
    ),
    "modular_bias_mod30": Domain(
        id="modular_bias_mod30",
        name="Modular Transition Bias (mod 30)",
        description="Finer-grained consecutive-prime residue bias modulo 30.",
        harvest=_harvest_modular_mod30,
        prompt_template=MODULAR_MOD30_PROMPT,
    ),
    "goldbach_partitions": Domain(
        id="goldbach_partitions",
        name="Goldbach Partition Growth",
        description="Growth and record-low behavior of Goldbach partition counts g(n).",
        harvest=_harvest_goldbach,
        prompt_template=GOLDBACH_PROMPT,
    ),
    "zeta_harmonics": Domain(
        id="zeta_harmonics",
        name="Riemann Zeta Harmonics",
        description="Non-trivial zeta zero spacing statistics and the explicit formula.",
        harvest=_harvest_zeta,
        prompt_template=ZETA_HARMONICS_PROMPT,
    ),
    "prime_powers": Domain(
        id="prime_powers",
        name="Base Prime Powers & Sieve Mechanics",
        description="Higher prime powers (p^k <= N), p^2 composite onset, and bottom-up sieve counting.",
        harvest=_harvest_prime_powers,
        prompt_template=PRIME_POWERS_PROMPT,
    ),
}


def list_domain_ids() -> list:
    return list(DOMAINS.keys())


def get_domain(domain_id: str) -> Domain:
    if domain_id not in DOMAINS:
        raise KeyError(f"Unknown research domain: {domain_id}. Known: {list_domain_ids()}")
    return DOMAINS[domain_id]


if __name__ == "__main__":
    print("Testing Domain Registry...")
    for did, domain in DOMAINS.items():
        print(f"\n--- {domain.name} ({did}) ---")
        data = domain.harvest()
        print(f"Harvested fields: {list(data.keys())}")
        prompt = domain.prompt_template.format(**data)
        assert "PYTHON_VERIFICATION" in prompt
        print(f"Prompt length: {len(prompt)} chars (formatted OK)")
    print("\nAll domains harvested and formatted successfully!")
