#!/usr/bin/env python3
"""
core/arithmetic_functions.py - Multiplicative Functions, Chebyshev Races, Constellations,
                               Smooth Numbers & Cramér Null Baseline Engine

High-performance vectorized NumPy engines supporting expanded Prime Frontier domains
and null-hypothesis anti-triviality testing:
  - mobius_sieve / mertens_function: Möbius mu(n) and Mertens M(x) = sum_{n<=x} mu(n)
  - liouville_sieve / sum_liouville: Liouville lambda(n) = (-1)^Omega(n) and L(x)
  - chebyshev_prime_race: Rubinstein-Sarnak prime race pi(x; q, a) - pi(x; q, b)
  - prime_constellation_counts: Twin, cousin, sexy, triplet, and quadruplet k-tuples vs Hardy-Littlewood
  - largest_prime_factor / smooth_number_counts: P^+(n) and y-smooth friable counts Psi(x, y)
  - cramer_random_primes: Synthetic Cramér random-prime sequence matching Li(x) density for null-model gating
"""

import math
import os
import sys
from typing import Dict, List

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.sieve import generate_primes


def mobius_sieve(limit: int) -> np.ndarray:
    """
    Compute Möbius mu(n) in {-1, 0, 1} for 0 <= n <= limit in O(limit log log sqrt(limit)).
    mu(0) = 0, mu(1) = 1.
    """
    if limit < 1:
        return np.zeros(max(0, limit + 1), dtype=np.int8)

    mu = np.ones(limit + 1, dtype=np.int8)
    mu[0] = 0
    rem = np.arange(limit + 1, dtype=np.int64)

    sqrt_limit = int(math.isqrt(limit))
    small_primes = generate_primes(sqrt_limit)
    for p in small_primes:
        p_int = int(p)
        mu[p_int::p_int] *= -1
        rem[p_int::p_int] //= p_int
        p2 = p_int * p_int
        mu[p2::p2] = 0

    # Any composite with a squared prime factor already has mu == 0.
    # Remaining rem[n] > 1 for squarefree n corresponds to a single prime factor > sqrt(limit).
    mu[rem > 1] *= -1
    return mu


def mertens_function(limit: int) -> np.ndarray:
    """Compute Mertens function M(x) = sum_{1 <= n <= x} mu(n) for 0 <= x <= limit."""
    return np.cumsum(mobius_sieve(limit), dtype=np.int64)


def liouville_sieve(limit: int) -> np.ndarray:
    """
    Compute Liouville function lambda(n) = (-1)^Omega(n) for 0 <= n <= limit,
    where Omega(n) counts prime factors with multiplicity. lambda(0) = 0, lambda(1) = 1.
    """
    if limit < 1:
        return np.zeros(max(0, limit + 1), dtype=np.int8)

    omega = np.zeros(limit + 1, dtype=np.int16)
    rem = np.arange(limit + 1, dtype=np.int64)

    sqrt_limit = int(math.isqrt(limit))
    small_primes = generate_primes(sqrt_limit)
    for p in small_primes:
        p_int = int(p)
        pk = p_int
        while pk <= limit:
            omega[pk::pk] += 1
            rem[pk::pk] //= p_int
            pk *= p_int

    mask = rem > 1
    mask[:2] = False
    omega[mask] += 1

    lam = np.where((omega & 1) == 0, 1, -1).astype(np.int8)
    lam[0] = 0
    return lam


def sum_liouville(limit: int) -> np.ndarray:
    """Compute Pólya summatory Liouville function L(x) = sum_{1 <= n <= x} lambda(n)."""
    return np.cumsum(liouville_sieve(limit), dtype=np.int64)


def largest_prime_factor(limit: int) -> np.ndarray:
    """
    Compute largest prime factor P^+(n) for 0 <= n <= limit (with P^+(0)=0, P^+(1)=1).
    Uses O(sqrt(limit)) prime division plus vectorized remainder assignment.
    """
    if limit < 1:
        return np.zeros(max(0, limit + 1), dtype=np.int64)

    lpf = np.ones(limit + 1, dtype=np.int64)
    lpf[0] = 0
    rem = np.arange(limit + 1, dtype=np.int64)

    sqrt_limit = int(math.isqrt(limit))
    small_primes = generate_primes(sqrt_limit)
    for p in small_primes:
        p_int = int(p)
        lpf[p_int::p_int] = p_int
        pk = p_int
        while pk <= limit:
            rem[pk::pk] //= p_int
            pk *= p_int

    gt1 = rem > 1
    lpf[gt1] = rem[gt1]
    return lpf


def smooth_number_counts(limit: int, y_bound: int) -> Dict:
    """
    Count y-smooth (friable) integers Psi(limit, y_bound) = #{1 <= n <= limit : P^+(n) <= y_bound}
    and compare against the Dickman-de Bruijn parameter u = ln(limit) / ln(y_bound).
    """
    if limit < 1 or y_bound < 2:
        return {}

    lpf = largest_prime_factor(limit)
    is_smooth = lpf[1:] <= y_bound
    psi_xy = int(np.count_nonzero(is_smooth))
    u = math.log(limit) / math.log(y_bound) if limit > 1 else 0.0
    smooth_indices = np.nonzero(is_smooth)[0] + 1
    smooth_gaps = np.diff(smooth_indices) if len(smooth_indices) >= 2 else np.array([0])

    return {
        "limit": int(limit),
        "y_bound": int(y_bound),
        "u_param": round(u, 4),
        "psi_count": psi_xy,
        "density": round(psi_xy / limit, 6),
        "u_neg_u_heuristic": round(u ** (-u), 6) if u > 0 else 1.0,
        "max_smooth_gap": int(np.max(smooth_gaps)),
        "mean_smooth_gap": round(float(np.mean(smooth_gaps)), 4),
    }


def chebyshev_prime_race(limit: int, mod: int = 4, r_lead: int = 3, r_trail: int = 1) -> Dict:
    """
    Analyze Chebyshev's prime race Delta(p) = pi(p; mod, r_lead) - pi(p; mod, r_trail)
    across primes p <= limit coprime to `mod` (Rubinstein-Sarnak phenomenon).
    """
    primes = generate_primes(limit)
    valid = primes[primes > mod]
    if len(valid) == 0:
        return {}

    res = valid % mod
    step = np.where(res == r_lead, 1, np.where(res == r_trail, -1, 0)).astype(np.int64)
    delta = np.cumsum(step)

    lead_frac = float(np.mean(delta > 0))
    tie_frac = float(np.mean(delta == 0))
    # Count strict sign reversals ignoring zeros
    nonzero = delta[delta != 0]
    sign_changes = int(np.count_nonzero(nonzero[:-1] * nonzero[1:] < 0)) if len(nonzero) >= 2 else 0

    # Rubinstein-Sarnak logarithmic/normalized bias: Delta(p) * ln(p) / sqrt(p)
    norm_bias = delta * np.log(valid) / np.sqrt(valid)

    quad_residues = {(x * x) % mod for x in range(1, mod) if math.gcd(x, mod) == 1}
    return {
        "limit": int(limit),
        "mod": int(mod),
        "r_lead": int(r_lead),
        "r_trail": int(r_trail),
        "r_lead_is_qr": r_lead in quad_residues,
        "r_trail_is_qr": r_trail in quad_residues,
        "primes_checked": int(len(valid)),
        "final_delta": int(delta[-1]),
        "lead_fraction": round(lead_frac, 5),
        "tie_fraction": round(tie_frac, 5),
        "sign_changes": sign_changes,
        "mean_rs_bias": round(float(np.mean(norm_bias)), 5),
        "std_rs_bias": round(float(np.std(norm_bias)), 5),
    }


def prime_constellation_counts(limit: int) -> Dict:
    """
    Count admissible prime k-tuples up to `limit` and compare against first-order
    Hardy-Littlewood singular-series integrals.
    """
    if limit < 15:
        return {}

    primes = generate_primes(limit)
    is_p = np.zeros(limit + 1, dtype=bool)
    is_p[primes] = True

    twins = int(np.count_nonzero(is_p[: limit - 1] & is_p[2:]))
    cousins = int(np.count_nonzero(is_p[: limit - 3] & is_p[4:]))
    sexy = int(np.count_nonzero(is_p[: limit - 5] & is_p[6:]))
    triplet_026 = int(np.count_nonzero(is_p[: limit - 5] & is_p[2 : limit - 3] & is_p[6:]))
    triplet_046 = int(np.count_nonzero(is_p[: limit - 5] & is_p[4 : limit - 1] & is_p[6:]))
    quad_0268 = int(
        np.count_nonzero(
            is_p[: limit - 7] & is_p[2 : limit - 5] & is_p[6 : limit - 1] & is_p[8:]
        )
    )

    # Numerical logarithmic integrals on grid dt = 1
    ts = np.arange(3.0, float(limit) + 1.0, 1.0)
    inv_ln = 1.0 / np.log(ts)
    li2 = float(np.sum(inv_ln**2))
    li3 = float(np.sum(inv_ln**3))
    li4 = float(np.sum(inv_ln**4))

    c2 = 0.6601618158468696  # Twin prime constant C_2
    c3 = 2.8582485957192204  # Singular series S(0,2,6) = S(0,4,6)
    c4 = 4.1511808632374158  # Singular series S(0,2,6,8)

    hl_twin = 2.0 * c2 * li2
    hl_sexy = 4.0 * c2 * li2
    hl_trip = c3 * li3
    hl_quad = c4 * li4

    return {
        "limit": int(limit),
        "twins_0_2": twins,
        "cousins_0_4": cousins,
        "sexy_0_6": sexy,
        "triplets_0_2_6": triplet_026,
        "triplets_0_4_6": triplet_046,
        "quadruplets_0_2_6_8": quad_0268,
        "ratio_twin_to_hl": round(twins / hl_twin, 5) if hl_twin > 0 else 0.0,
        "ratio_sexy_to_hl": round(sexy / hl_sexy, 5) if hl_sexy > 0 else 0.0,
        "ratio_triplet026_to_hl": round(triplet_026 / hl_trip, 5) if hl_trip > 0 else 0.0,
        "ratio_quad_to_hl": round(quad_0268 / hl_quad, 5) if hl_quad > 0 else 0.0,
    }


def cramer_random_primes(limit: int, seed: int = 42) -> np.ndarray:
    """
    Generate a synthetic Cramér random-prime sequence up to `limit` for null-hypothesis
    control tests: includes 2, and samples each odd integer n >= 3 independently with
    probability min(1, 2 / ln(n)) so expected density matches Li(x) without arithmetic correlations.
    """
    if limit < 2:
        return np.array([], dtype=np.int64)
    if limit == 2:
        return np.array([2], dtype=np.int64)

    rng = np.random.default_rng(seed)
    odds = np.arange(3, limit + 1, 2, dtype=np.int64)
    probs = np.minimum(1.0, 2.0 / np.log(odds.astype(np.float64)))
    chosen = odds[rng.random(len(odds)) < probs]
    return np.concatenate(([np.int64(2)], chosen))


def cramer_mobius_liouville(limit: int, seed: int = 42):
    """
    Fast vectorized Wintner/Rademacher random-multiplicative null control (mu_null, lam_null)
    for 0 <= n <= limit:
      - Assigns independent random signs eps_p in {-1, +1} to every prime p <= limit.
      - mu_null(n) = mu(n)^2 * prod_{p | n} eps_p in {-1, 0, 1} (random Möbius on squarefree n).
      - lam_null(n) = prod_{p^a || n} eps_p^a in {-1, 1} (completely multiplicative random Liouville).
    Runs in O(limit log log sqrt(limit)) (~0.08s at limit=10^7) without Python integer loops.
    """
    if limit < 1:
        z = np.zeros(max(0, limit + 1), dtype=np.int8)
        return z, z.copy()

    rng = np.random.default_rng(seed)
    mu_null = np.ones(limit + 1, dtype=np.int8)
    lam_null = np.ones(limit + 1, dtype=np.int8)
    mu_null[0] = 0
    lam_null[0] = 0
    rem = np.arange(limit + 1, dtype=np.int64)

    sqrt_limit = int(math.isqrt(limit))
    small_primes = generate_primes(sqrt_limit)
    small_signs = rng.choice(np.array([-1, 1], dtype=np.int8), size=len(small_primes))

    for p, s in zip(small_primes, small_signs):
        p_int = int(p)
        s_int = int(s)
        if s_int == -1:
            mu_null[p_int::p_int] *= -1
        p2 = p_int * p_int
        mu_null[p2::p2] = 0

        pk = p_int
        while pk <= limit:
            rem[pk::pk] //= p_int
            if s_int == -1:
                lam_null[pk::pk] *= -1
            pk *= p_int

    # Any remaining rem[n] > 1 is a single prime factor q > sqrt_limit (multiplicity 1).
    large_signs = rng.choice(np.array([-1, 1], dtype=np.int8), size=limit + 1)
    mask = rem > 1
    s_large = large_signs[rem[mask]]
    mu_null[mask] *= s_large
    lam_null[mask] *= s_large
    return mu_null, lam_null


if __name__ == "__main__":
    print("Testing Multiplicative, Race, Constellation, Smooth & Cramér Null Engines...")
    mu = mobius_sieve(100_000)
    # Known values: M(10) = -1, M(100) = 1, M(1000) = 2, M(10000) = -23, M(100000) = -48
    m_vals = np.cumsum(mu)
    assert int(m_vals[10]) == -1, f"Expected M(10)=-1, got {m_vals[10]}"
    assert int(m_vals[100]) == 1, f"Expected M(100)=1, got {m_vals[100]}"
    assert int(m_vals[1000]) == 2, f"Expected M(1000)=2, got {m_vals[1000]}"
    assert int(m_vals[10000]) == -23, f"Expected M(10000)=-23, got {m_vals[10000]}"
    assert int(m_vals[100000]) == -48, f"Expected M(100000)=-48, got {m_vals[100000]}"
    sqfree_density = float(np.mean(mu[1:] != 0))
    assert abs(sqfree_density - 6.0 / (math.pi**2)) < 5e-4

    lam = liouville_sieve(10_000)
    l_vals = np.cumsum(lam)
    # L(x) <= 0 for 2 <= x <= 10000 (first Pólya counterexample is around 9.06e8)
    assert np.all(l_vals[2:] <= 0), "Expected L(x) <= 0 up to 10,000"

    race = chebyshev_prime_race(200_000, mod=4, r_lead=3, r_trail=1)
    assert race["lead_fraction"] > 0.99, f"Expected mod-4 Chebyshev bias > 0.99, got {race['lead_fraction']}"

    const = prime_constellation_counts(200_000)
    assert 0.95 < const["ratio_twin_to_hl"] < 1.05
    assert const["quadruplets_0_2_6_8"] > 0

    sm = smooth_number_counts(100_000, 100)
    assert sm["psi_count"] > 1000

    c_primes = cramer_random_primes(100_000, seed=7)
    assert 8500 < len(c_primes) < 10700  # pi(10^5) = 9592

    mu_null, lam_null = cramer_mobius_liouville(200_000, seed=11)
    assert len(mu_null) == 200_001 and len(lam_null) == 200_001
    assert int(mu_null[0]) == 0 and int(mu_null[1]) == 1 and int(lam_null[1]) == 1
    # Verify exact support on squarefree integers and complete multiplicativity of lam_null
    mu_true_200k = mobius_sieve(200_000)
    assert np.array_equal(mu_null != 0, mu_true_200k != 0), "mu_null support must match squarefree integers"
    assert np.all(np.isin(lam_null[1:], [-1, 1])), "lam_null must take values in {-1, +1}"
    for a, b in ((6, 35), (12, 25), (14, 15), (97, 103), (199, 1001)):
        assert int(lam_null[a * b]) == int(lam_null[a]) * int(lam_null[b])
        if math.gcd(a, b) == 1:
            assert int(mu_null[a * b]) == int(mu_null[a]) * int(mu_null[b])

    print("All Arithmetic Function & Null Baseline engines verified!")
