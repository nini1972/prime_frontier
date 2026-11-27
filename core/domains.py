#!/usr/bin/env python3
"""
core/domains.py - Research Domain Registry for the Prime Frontier Agent World

Each `Domain` describes one distinct area of prime-number phenomena that the
autonomous agents can investigate. A domain packages:
  - `harvest()`      : computes fresh empirical data across dynamically varied
                       scale windows using the core engines
  - `prompt_template`: turns that data into a structured research brief for the
                       mathematical reasoning model
"""

import math
import random
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
from core.arithmetic_functions import (
    mobius_sieve,
    mertens_function,
    liouville_sieve,
    sum_liouville,
    largest_prime_factor,
    smooth_number_counts,
    chebyshev_prime_race,
    prime_constellation_counts,
)


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
If a 'Prior Domain Memory' section is provided below, do NOT repeat already-confirmed laws or previously falsified titles; instead, formulate an orthogonal hypothesis or calibrate a deeper multi-scale law using lessons from past falsifications.
Then, write a self-contained Python script to test and either support or falsify your conjecture empirically.
IMPORTANT:
- Execution must complete in under 60 seconds (hard timeout: 120s). Use vectorized NumPy operations instead of slow Python loops over millions of elements.
- Test your conjecture across multiple scales (e.g., comparing N_1 < N_2 < N_3 including out-of-sample prime ranges larger than the prompt snapshot).
- NULL-BASELINE / ANTI-TRIVIALITY GATE (MANDATORY):
  Do NOT propose loose, wide inequality bands that would trivially hold for random numbers!
  Your verification script MUST include a quantitative Null / Baseline Contrast:
  (a) Compare your proposed predictor/scaling law against a standard first-order baseline (e.g., naive PNT, uniform residue distribution, first-order Hardy-Littlewood, or u^-u), OR
  (b) Evaluate the exact same observable on a synthetic Cramér random-prime control (`cramer_random_primes(limit, seed=...)`) or shuffled prime gaps to prove the effect is genuinely arithmetic rather than generic 1/ln(x) density noise.
  If your conjecture passes its raw bound ONLY because the bound is wide and fails to beat the baseline or separate from the Cramér/shuffled null, print `VERDICT: TRIVIAL_NULL`.
- VERDICT PRINTING RULE:
  For intermediate per-scale checks, print `PASS` or `FAIL` (do NOT print `VERDICT:` on intermediate lines).
  Print `VERDICT: SUPPORTED`, `VERDICT: FALSIFIED`, or `VERDICT: TRIVIAL_NULL` ONLY ONCE on the final line of stdout!
- You can directly import and use any of the fast project utilities (all pre-imported in the execution harness — NEVER redefine them with your own `def generate_primes(...)`, `def goldbach_partition_counts(...)`, `def get_zeta_zeros(...)`, `def mobius_sieve(...)`, `def mertens_function(...)`, `def liouville_sieve(...)`, or `def largest_prime_factor(...)`, as shadowing pre-imported C/NumPy engines raises a `PerformanceError`!):
    from core.sieve import generate_primes, compute_prime_gaps, compute_gap_statistics, find_maximal_gap_records
        # generate_primes(limit: int) -> np.ndarray of primes <= limit
        # compute_prime_gaps(primes) -> np.ndarray of consecutive gaps
    from core.modular_orbits import compute_modular_transitions, compute_residue_autocorrelation
        # compute_modular_transitions(primes, mod=10) -> dict with keys:
        #   "residues" (list of K coprime residues, e.g. [1,3,7,9] for mod=10 or 8 residues for mod=30),
        #   "transition_matrix" (K x K 2D list indexed 0..K-1 matching "residues"),
        #   "transition_counts" (K x K 2D list), "expected_uniform" (float)
        #   NOTE: If iterating over primes manually for modulo `mod`, ALWAYS filter `primes = primes[primes > mod]` first!
        # compute_residue_autocorrelation(primes, mod=10, max_lag=15) -> dict {{lag_int: excess_float}}
    from core.goldbach import goldbach_partition_counts, goldbach_stats
        # goldbach_partition_counts(limit: int) -> dict {{even_n: g_n_count}} (FFT-accelerated up to limit=500,000)
    from core.zeta_harmonics import get_zeta_zeros, exact_psi, approximate_psi_riemann
        # get_zeta_zeros(count: int = 50) -> list of float imaginary parts gamma_k (pre-cached and instantaneous up to count=3000; NEVER call mpmath.zetazero directly!)
    from core.arithmetic_functions import (
        mobius_sieve, mertens_function, liouville_sieve, sum_liouville,
        largest_prime_factor, smooth_number_counts, chebyshev_prime_race,
        prime_constellation_counts, cramer_random_primes,
    )
        # mobius_sieve(limit: int) -> np.ndarray of mu(0..limit) in {{-1, 0, 1}}
        # mertens_function(limit: int) -> np.ndarray of M(0..limit)
        # liouville_sieve(limit: int) -> np.ndarray of lambda(0..limit) in {{-1, 1}}
        # sum_liouville(limit: int) -> np.ndarray of L(0..limit)
        # largest_prime_factor(limit: int) -> np.ndarray of P^+(0..limit)
        # smooth_number_counts(limit: int, y_bound: int) -> dict with "psi_count", "u_param", "density", "max_smooth_gap"
        # chebyshev_prime_race(limit: int, mod=4, r_lead=3, r_trail=1) -> dict with "lead_fraction", "sign_changes", "mean_rs_bias"
        # prime_constellation_counts(limit: int) -> dict with exact keys:
        #   "twins_0_2", "cousins_0_4", "sexy_0_6", "triplets_0_2_6", "triplets_0_4_6", "quadruplets_0_2_6_8",
        #   "ratio_twin_to_hl", "ratio_sexy_to_hl", "ratio_triplet026_to_hl", "ratio_quad_to_hl"
        # cramer_random_primes(limit: int, seed=42) -> np.ndarray synthetic Cramér random-prime control sequence
- Do not import third-party packages other than numpy, scipy, and mpmath.

Format your output strictly as:
## CONJECTURE_NAME: <concise, distinctive title — avoid generic names like 'Harmonic Gap Resonance Conjecture'>
## MATHEMATICAL_FORMULATION: <LaTeX formula and rigorous explanation, including null/baseline contrast>
## HYPOTHESIS_TARGET: <what exact quantitative threshold AND null/baseline contrast must hold across scales>
## PYTHON_VERIFICATION:
```python
# Self-contained python script testing across multiple scales + null/baseline control,
# ending with a single final line: 'VERDICT: SUPPORTED', 'VERDICT: FALSIFIED', or 'VERDICT: TRIVIAL_NULL'
```
"""


# ---------------------------------------------------------------------------
# Domain 1: Prime gap dynamics
# ---------------------------------------------------------------------------
def _harvest_gap_dynamics() -> Dict[str, Any]:
    n_primes = random.choice([60_000, 100_000, 150_000, 200_000])
    upper = int(n_primes * (math.log(n_primes) + math.log(math.log(n_primes)) + 1.5))
    primes = generate_primes(upper)[:n_primes]
    gaps = compute_prime_gaps(primes)
    stats = compute_gap_statistics(primes, gaps)
    records = find_maximal_gap_records(primes, gaps)
    top = records[-1] if records else {}
    return {
        "n_primes": f"{len(primes):,}",
        "mean_gap": stats.get("mean_gap"),
        "std_gap": stats.get("std_gap"),
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
Below is real empirical data computed on the first {n_primes} prime numbers:

Gap Statistics (up to prime {max_prime}):
- Mean Gap: {mean_gap} (Std Gap: {std_gap})
- Max Gap Observed: {max_gap}
- Twin Primes (g=2): {twin_primes}
- Cousin Primes (g=4): {cousin_primes}
- Sexy Primes (g=6): {sexy_primes} (most frequent nonzero gap)
- Latest maximal-gap record: gap {latest_record_gap} after prime {latest_record_prime}
  (Cramer ratio g / ln^2(p) = {latest_cramer_ratio})
""" + OUTPUT_FORMAT_INSTRUCTIONS


# ---------------------------------------------------------------------------
# Domain 2: Modular transition bias, modulo 10 (Lemke Oliver-Soundararajan)
# ---------------------------------------------------------------------------
def _harvest_modular_mod10() -> Dict[str, Any]:
    n_primes = random.choice([75_000, 100_000, 150_000, 220_000])
    upper = int(n_primes * (math.log(n_primes) + math.log(math.log(n_primes)) + 1.5))
    primes = generate_primes(upper)[:n_primes]
    mod10 = compute_modular_transitions(primes, mod=10)
    autocorr = compute_residue_autocorrelation(primes, mod=10, max_lag=6)
    m = mod10["transition_matrix"]
    return {
        "n_primes": f"{len(primes):,}",
        "p11": round(m[0][0] * 100, 2),
        "p13": round(m[0][1] * 100, 2),
        "p17": round(m[0][2] * 100, 2),
        "p19": round(m[0][3] * 100, 2),
        "expected_uniform": round(mod10["expected_uniform"] * 100, 1),
        "autocorr_lag1": autocorr.get(1),
        "autocorr_lag2": autocorr.get(2),
        "autocorr_lag3": autocorr.get(3),
    }


MODULAR_MOD10_PROMPT = """You are an elite research mathematician at the Prime Frontier laboratory.
Below is real empirical data on consecutive prime modular transitions modulo 10 (first {n_primes} primes):

- P(1 -> 1): {p11}% (expected uniform: {expected_uniform}%, diagonal self-repulsion)
- P(1 -> 3): {p13}% (attraction)
- P(1 -> 7): {p17}% (attraction)
- P(1 -> 9): {p19}%
- Residue autocorrelation excess at lag 1: {autocorr_lag1}, lag 2: {autocorr_lag2}, lag 3: {autocorr_lag3}

This is the Lemke Oliver-Soundararajan (2016) phenomenon: consecutive primes exhibit short-range
repulsion/attraction modulo q driven by singular-series bias and logarithmic decay ~ 1/ln(x).
""" + OUTPUT_FORMAT_INSTRUCTIONS


# ---------------------------------------------------------------------------
# Domain 3: Modular transition bias, modulo 30 (finer residue structure)
# ---------------------------------------------------------------------------
def _harvest_modular_mod30() -> Dict[str, Any]:
    n_primes = random.choice([80_000, 120_000, 180_000, 240_000])
    upper = int(n_primes * (math.log(n_primes) + math.log(math.log(n_primes)) + 1.5))
    primes = generate_primes(upper)[:n_primes]
    mod30 = compute_modular_transitions(primes, mod=30)
    residues = mod30.get("residues", [])
    m = mod30.get("transition_matrix", [])
    expected = mod30.get("expected_uniform", 0.0)
    strongest_repulsion = None
    strongest_attraction = None
    best_low, best_high = 1.0, 0.0
    if m and residues:
        for i, r_i in enumerate(residues):
            for j, r_j in enumerate(residues):
                p = m[i][j]
                if p < best_low:
                    best_low, strongest_repulsion = p, (r_i, r_j)
                if p > best_high:
                    best_high, strongest_attraction = p, (r_i, r_j)
    return {
        "n_primes": f"{len(primes):,}",
        "num_residues": len(residues),
        "expected_uniform": round(expected * 100, 2),
        "strongest_repulsion": strongest_repulsion,
        "strongest_repulsion_pct": round(best_low * 100, 2) if strongest_repulsion else None,
        "strongest_attraction": strongest_attraction,
        "strongest_attraction_pct": round(best_high * 100, 2) if strongest_attraction else None,
    }


MODULAR_MOD30_PROMPT = """You are an elite research mathematician at the Prime Frontier laboratory.
Below is real empirical data on consecutive prime modular transitions modulo 30 (first {n_primes} primes),
i.e. the {num_residues} residue classes coprime to 30 ({{1,7,11,13,17,19,23,29}}):

- Expected uniform transition probability: {expected_uniform}%
- Strongest repulsion observed: residue {strongest_repulsion} at {strongest_repulsion_pct}%
- Strongest attraction observed: residue {strongest_attraction} at {strongest_attraction_pct}%

Modulo 30 = 2*3*5 removes small-prime divisibility obstructions, exposing second-order singular-series
asymmetries and spectral Markov eigenvalues of the transition operator.
""" + OUTPUT_FORMAT_INSTRUCTIONS


# ---------------------------------------------------------------------------
# Domain 4: Goldbach partitions
# ---------------------------------------------------------------------------
def _harvest_goldbach() -> Dict[str, Any]:
    limit = random.choice([20_000, 40_000, 60_000, 80_000])
    counts = goldbach_partition_counts(limit)
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
- Uncorrected tail baseline n/(2 ln^2 n): {hl_estimate_tail}

Recall Sylvester/Hardy-Littlewood's singular-series factor ("Goldbach's comet"): even integers divisible
by small odd primes (e.g., 6 | n or 30 | n) carry systematic multiplicative boosts prod_{{p|n, p>2}} (p-1)/(p-2).
""" + OUTPUT_FORMAT_INSTRUCTIONS


# ---------------------------------------------------------------------------
# Domain 5: Riemann zeta harmonics
# ---------------------------------------------------------------------------
def _harvest_zeta() -> Dict[str, Any]:
    from core.zeta_harmonics import get_zeta_zeros

    window_start, window_len = random.choice([(0, 25), (20, 40), (100, 50), (250, 60)])
    all_gammas = get_zeta_zeros(window_start + window_len)
    gammas = all_gammas[window_start : window_start + window_len]
    gaps = [round(gammas[i + 1] - gammas[i], 4) for i in range(len(gammas) - 1)]
    mean_t = sum(gammas) / len(gammas) if gammas else 14.13
    expected_gap = (2.0 * math.pi) / math.log(mean_t / (2.0 * math.pi)) if mean_t > 2.0 * math.pi else 0.0
    return {
        "idx_start": window_start + 1,
        "idx_end": window_start + len(gammas),
        "sample_zeros": [round(g, 4) for g in gammas[:5]],
        "mean_zero_gap": round(sum(gaps) / len(gaps), 4) if gaps else None,
        "rv_expected_gap": round(expected_gap, 4),
        "num_zeros": len(gammas),
    }


ZETA_HARMONICS_PROMPT = """You are an elite research mathematician at the Prime Frontier laboratory.
Below is real empirical data on non-trivial Riemann zeta zeros rho_k = 1/2 + i*gamma_k
in index window k in [{idx_start}, {idx_end}] ({num_zeros} consecutive zeros; up to 3,000 pre-cached in `get_zeta_zeros(3000)`):

- First 5 imaginary parts in this window: {sample_zeros}
- Empirical mean spacing in window: {mean_zero_gap} (Riemann-von Mangoldt asymptotic 2*pi / ln(T / 2*pi): {rv_expected_gap})

By Riemann's explicit formula, psi_0(x) = x - sum_rho x^rho / rho - ln(2*pi) - 1/2 ln(1 - x^-2),
and unfolded spacings s_k = (gamma_{{k+1}} - gamma_k) * ln(gamma_k / (2*pi)) / (2*pi) exhibit GUE level repulsion.

Frontier Context (Nov 2026 RH Formalization & Active Confirmed Law `PRIME-CONJ-1795263018`):
With Mathlib's `RiemannHypothesis` (Re(rho) = 1/2) formally verified in Lean 4 v4.33.1 (eliminating off-line
x^beta drift, locking |psi(x) - x| <= (1/(8*pi)) * sqrt(x) * ln^2(x) for x >= 73.2), explore finite-T
sub-leading structures beyond raw Re(rho) = 1/2 — such as Berry prime-orbit resummations of our confirmed
Two-Term Richardson Variance Deficit Law for unfolded zeta spacings or explicit-formula truncation residuals
inside Schoenfeld's sqrt(x)*ln^2(x)/(8*pi) envelope (verified on `get_zeta_zeros(3000)` + null control).
""" + OUTPUT_FORMAT_INSTRUCTIONS


# ---------------------------------------------------------------------------
# Domain 6: Base Prime Powers & Sieve Mechanics
# ---------------------------------------------------------------------------
def _harvest_prime_powers() -> Dict[str, Any]:
    N = random.choice([1_000, 2_500, 5_000, 10_000])
    sqrt_n = int(math.isqrt(N))
    primes = generate_primes(N)
    base_primes = [int(p) for p in primes if p <= sqrt_n and p > 2]

    powers_info = {}
    for p in base_primes[:8]:
        k = 2
        powers = []
        while p**k <= N:
            powers.append(p**k)
            k += 1
        if powers:
            powers_info[p] = powers

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
        "base_primes_preview": base_primes[:12],
        "num_base_primes": len(base_primes),
        "powers_info": powers_info,
        "sample_elim": {p: elim_counts[p] for p in base_primes[:6]},
        "total_odd_composites": total_odd_composites,
        "total_odd_primes": total_odd_primes,
        "pi_N": len(primes),
    }


PRIME_POWERS_PROMPT = """You are an elite research mathematician at the Prime Frontier laboratory.
Below is real empirical data on Base Prime Powers and Eratosthenes Sieve Mechanics up to N = {N} (sqrt(N) = {sqrt_N}):

- Count of base odd primes p <= sqrt(N): {num_base_primes} (first few: {base_primes_preview})
- Sample prime powers p^k <= {N} (k >= 2): {powers_info}
- Newly eliminated odd composites C_p({N}) starting at p^2 (smallest prime factor = p): {sample_elim}
- Total odd composites in [3, {N}]: {total_odd_composites}; Total odd primes: {total_odd_primes}; pi({N}) = {pi_N}

Notice C_p(N) counts odd integers in [3, N] whose least prime factor lpf(n) = p, connecting sieve attrition
directly to Mertens' third theorem prod_{{p<=z}} (1 - 1/p) ~ e^{{-gamma}} / ln(z) and Buchstab's function omega(u).
""" + OUTPUT_FORMAT_INSTRUCTIONS


# ---------------------------------------------------------------------------
# Domain 7: Chebyshev Prime Races & Quadratic Residue Bias (NEW)
# ---------------------------------------------------------------------------
def _harvest_chebyshev_race() -> Dict[str, Any]:
    mod, r_lead, r_trail = random.choice([(4, 3, 1), (8, 3, 1), (8, 5, 1), (8, 7, 1), (12, 5, 1), (12, 11, 1)])
    limit = random.choice([200_000, 400_000, 600_000])
    race = chebyshev_prime_race(limit=limit, mod=mod, r_lead=r_lead, r_trail=r_trail)
    return race


CHEBYSHEV_RACE_PROMPT = """You are an elite research mathematician at the Prime Frontier laboratory.
Below is real empirical data on Chebyshev's Prime Race Delta(p) = pi(p; {mod}, {r_lead}) - pi(p; {mod}, {r_trail})
evaluated across all {primes_checked} primes p <= {limit} coprime to {mod}:

- Leading residue candidate {r_lead} (quadratic residue mod {mod}? {r_lead_is_qr}) vs {r_trail} (quadratic residue? {r_trail_is_qr})
- Fraction of primes where pi(p; {mod}, {r_lead}) > pi(p; {mod}, {r_trail}): {lead_fraction} (ties: {tie_fraction})
- Final race lead Delta({limit}): {final_delta}
- Strict sign reversals observed up to {limit}: {sign_changes}
- Normalized Rubinstein-Sarnak bias Delta(p)*ln(p)/sqrt(p): mean = {mean_rs_bias}, std = {std_rs_bias}

Chebyshev (1853) noticed non-quadratic residues systematically lead quadratic residues even though Dirichlet's
theorem guarantees asymptotic 50/50 density; Rubinstein & Sarnak (1994) linked logarithmic race densities to
prime-square offsets and zeros of Dirichlet L-functions L(s, chi).

Frontier Context (Nov 2026 Dirichlet GRH Formalization & Active Confirmed Laws `SMOL-8`, `HWCL-8`, `LEVF-8`):
Note the sharp dividing line after the Lean 4 verification of `DirichletLZerosOnLine` (Dirichlet GRH): while
GRH anchors every oscillatory term at sqrt(x), full Rubinstein-Sarnak logarithmic race densities STILL depend
on Linear Independence (LI) of zero ordinates gamma_chi > 0 over Q. Connect our confirmed mod-8 laws
(`PRIME-CONJ-1795269576` SMOL-8, `PRIME-CONJ-1795262117` HWCL-8, `PRIME-CONJ-1795228384` LEVF-8) to finite-scale
cross-modulus LI signatures, rational-relation resonance diagnostics, or prime-power correction terms beyond
naive sqrt(x) envelopes.
""" + OUTPUT_FORMAT_INSTRUCTIONS


# ---------------------------------------------------------------------------
# Domain 8: Möbius, Mertens & Liouville Multiplicative Walks (NEW)
# ---------------------------------------------------------------------------
def _harvest_multiplicative_walk() -> Dict[str, Any]:
    import numpy as _np

    limit = random.choice([100_000, 200_000, 300_000])
    mu = mobius_sieve(limit)
    m_arr = mertens_function(limit)
    l_arr = sum_liouville(limit)

    xs = [1_000, 10_000, limit]
    m_norm = {x: round(int(m_arr[x]) / math.sqrt(x), 5) for x in xs}
    l_norm = {x: round(int(l_arr[x]) / math.sqrt(x), 5) for x in xs}
    sqfree_ratio = round(float((mu[1:] != 0).mean()), 6)
    six_over_pi2 = round(6.0 / (math.pi**2), 6)

    grid = _np.sqrt(_np.arange(100, limit + 1, dtype=_np.float64))
    max_abs_m_sqrt = round(float(_np.max(_np.abs(m_arr[100:]) / grid)), 5)
    min_l_sqrt = round(float(_np.min(l_arr[100:] / grid)), 5)
    max_l_sqrt = round(float(_np.max(l_arr[100:] / grid)), 5)

    return {
        "limit": limit,
        "m_norm": m_norm,
        "l_norm": l_norm,
        "sqfree_ratio": sqfree_ratio,
        "six_over_pi2": six_over_pi2,
        "max_abs_m_sqrt": max_abs_m_sqrt,
        "min_l_sqrt": min_l_sqrt,
        "max_l_sqrt": max_l_sqrt,
    }


MULTIPLICATIVE_WALK_PROMPT = """You are an elite research mathematician at the Prime Frontier laboratory.
Below is real empirical data on the Möbius function mu(n), Mertens function M(x) = sum_{{n<=x}} mu(n),
and Liouville/Pólya walk L(x) = sum_{{n<=x}} lambda(n) up to x = {limit}:

- Normalized Mertens values M(x)/sqrt(x): {m_norm}
- Normalized Liouville values L(x)/sqrt(x): {l_norm}
- Extrema on [100, {limit}]: max |M(x)|/sqrt(x) = {max_abs_m_sqrt}; L(x)/sqrt(x) in [{min_l_sqrt}, {max_l_sqrt}]
- Empirical squarefree density Q({limit})/{limit} = {sqfree_ratio} (theoretical 6/pi^2 = {six_over_pi2})

Recall that M(x) = O(x^{{1/2 + eps}}) is equivalent to the Riemann Hypothesis, while Chowla's conjecture
predicts higher-order sign correlations sum_{{n<=x}} mu(n) mu(n+h) = o(x), yet short-scale multiplicative
constraints create structured deviations from an i.i.d. Rademacher random walk.

Frontier Context (Nov 2026 RH Formalization & Active Confirmed Law `PRIME-CONJ-1795314014`):
With critical-line spectral rigidity confining M(x) = O(x^{{1/2 + eps}}), polynomial deviations above x^{{1/2}} are
ruled out, shifting the research frontier to fine-scale sub-polynomial / iterated-logarithmic extremal envelopes
of |M(x)|/sqrt(x) and L(x)/sqrt(x) and their connection to our confirmed Arcsine-Broken Liouville Occupation Law
(`PRIME-CONJ-1795314014`), contrasting directly with i.i.d. Rademacher and Cramer random-walk null controls.
""" + OUTPUT_FORMAT_INSTRUCTIONS


# ---------------------------------------------------------------------------
# Domain 9: Admissible Prime k-Tuples & Hardy-Littlewood Singular Series (NEW)
# ---------------------------------------------------------------------------
def _harvest_prime_constellations() -> Dict[str, Any]:
    limit = random.choice([200_000, 350_000, 500_000])
    return prime_constellation_counts(limit)


PRIME_CONSTELLATIONS_PROMPT = """You are an elite research mathematician at the Prime Frontier laboratory.
Below is real empirical data on admissible prime k-tuple constellations up to N = {limit}:

- 2-tuples: Twins (0,2) = {twins_0_2}; Cousins (0,4) = {cousins_0_4}; Sexy (0,6) = {sexy_0_6}
- 3-tuples: Triplets (0,2,6) = {triplets_0_2_6}; Triplets (0,4,6) = {triplets_0_4_6}
- 4-tuples: Quadruplets (0,2,6,8) = {quadruplets_0_2_6_8}
- Observed / First-Order Hardy-Littlewood Integral Ratios:
  * Twin / HL_2: {ratio_twin_to_hl}
  * Sexy / HL_2: {ratio_sexy_to_hl}
  * Triplet(0,2,6) / HL_3: {ratio_triplet026_to_hl}
  * Quadruplet(0,2,6,8) / HL_4: {ratio_quad_to_hl}

Explore quantitative laws governing k-tuple clustering, second-order asymptotic biases between symmetric
constellations (e.g., (0,2,6) vs (0,4,6) or twin vs cousin balances), inter-constellation waiting times,
or higher-order singular-series correction terms.
""" + OUTPUT_FORMAT_INSTRUCTIONS


# ---------------------------------------------------------------------------
# Domain 10: Friable (y-Smooth) Numbers & Largest Prime Factor Scaling (NEW)
# ---------------------------------------------------------------------------
def _harvest_smooth_numbers() -> Dict[str, Any]:
    limit, y_bound = random.choice([(120_000, 47), (180_000, 79), (240_000, 113), (300_000, 151)])
    return smooth_number_counts(limit=limit, y_bound=y_bound)


SMOOTH_NUMBERS_PROMPT = """You are an elite research mathematician at the Prime Frontier laboratory.
Below is real empirical data on y-smooth (friable) integers Psi(x, y) = #{{1 <= n <= x : P^+(n) <= y}}
for x = {limit}, smoothness bound y = {y_bound}:

- Dickman parameter u = ln(x) / ln(y) = {u_param}
- Exact smooth count Psi({limit}, {y_bound}) = {psi_count} (empirical density Psi/x = {density})
- Crude u^{{-u}} reference = {u_neg_u_heuristic}
- Smooth gap statistics in [1, {limit}]: mean gap = {mean_smooth_gap}, max gap = {max_smooth_gap}

De Bruijn-Dickman theory states Psi(x, x^{{1/u}}) ~ x * rho(u) where u*rho'(u) = -rho(u - 1), while Hildebrand-Tenenbaum
saddle-point corrections govern finite-scale deviations, largest-prime-factor quantiles ln P^+(n) / ln n,
and clustering of adjacent smooth integers.
""" + OUTPUT_FORMAT_INSTRUCTIONS


DOMAINS: Dict[str, Domain] = {
    "gap_dynamics": Domain(
        id="gap_dynamics",
        name="Prime Gap Dynamics",
        description="Gap distribution, twin/cousin/sexy prime families, Cramér conjecture records.",
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
        description="Growth, singular-series comet modulation, and record lows of Goldbach g(n).",
        harvest=_harvest_goldbach,
        prompt_template=GOLDBACH_PROMPT,
    ),
    "zeta_harmonics": Domain(
        id="zeta_harmonics",
        name="Riemann Zeta Harmonics",
        description="Non-trivial zeta zero spacing statistics, GUE repulsion, and the explicit formula.",
        harvest=_harvest_zeta,
        prompt_template=ZETA_HARMONICS_PROMPT,
    ),
    "prime_powers": Domain(
        id="prime_powers",
        name="Base Prime Powers & Sieve Mechanics",
        description="Higher prime powers (p^k <= N), least-prime-factor sieve attrition, and Buchstab scaling.",
        harvest=_harvest_prime_powers,
        prompt_template=PRIME_POWERS_PROMPT,
    ),
    "chebyshev_race": Domain(
        id="chebyshev_race",
        name="Chebyshev Prime Races & Quadratic Bias",
        description="Rubinstein-Sarnak prime race bias pi(x; q, a) - pi(x; q, b) and quadratic-residue repulsion.",
        harvest=_harvest_chebyshev_race,
        prompt_template=CHEBYSHEV_RACE_PROMPT,
    ),
    "multiplicative_walk": Domain(
        id="multiplicative_walk",
        name="Möbius, Mertens & Liouville Walks",
        description="Squarefree sieve fluctuations, Mertens M(x)/sqrt(x) bounds, and Chowla sign correlations.",
        harvest=_harvest_multiplicative_walk,
        prompt_template=MULTIPLICATIVE_WALK_PROMPT,
    ),
    "prime_constellations": Domain(
        id="prime_constellations",
        name="Prime k-Tuples & Singular Series",
        description="Twin, cousin, sexy, triplet, and quadruplet constellations vs Hardy-Littlewood integrals.",
        harvest=_harvest_prime_constellations,
        prompt_template=PRIME_CONSTELLATIONS_PROMPT,
    ),
    "smooth_numbers": Domain(
        id="smooth_numbers",
        name="Friable (y-Smooth) Numbers & P^+(n)",
        description="Dickman-de Bruijn smooth density Psi(x, y), largest prime factor laws, and smooth gaps.",
        harvest=_harvest_smooth_numbers,
        prompt_template=SMOOTH_NUMBERS_PROMPT,
    ),
}


def list_domain_ids() -> list:
    return list(DOMAINS.keys())


def get_domain(domain_id: str) -> Domain:
    if domain_id not in DOMAINS:
        raise KeyError(f"Unknown research domain: {domain_id}. Known: {list_domain_ids()}")
    return DOMAINS[domain_id]


if __name__ == "__main__":
    print("Testing Domain Registry (10 Domains)...")
    for did, domain in DOMAINS.items():
        print(f"\n--- {domain.name} ({did}) ---")
        data = domain.harvest()
        print(f"Harvested fields: {list(data.keys())}")
        prompt = domain.prompt_template.format(**data)
        assert "PYTHON_VERIFICATION" in prompt
        assert "TRIVIAL_NULL" in prompt
        print(f"Prompt length: {len(prompt)} chars (formatted OK)")
    print("\nAll 10 domains harvested and formatted successfully!")
