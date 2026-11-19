#!/usr/bin/env python3
"""
tests/test_research_pipeline.py - Regression tests for Prime Frontier verdict parsing,
                                  head+tail output preservation, AST anti-tautology gate,
                                  domain-scoped deduplication, and 10-domain coverage.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.domains import DOMAINS, list_domain_ids
from models import knowledge_base as kb
from models.math_researcher import (
    execute_verification_script,
    format_head_tail_output,
    parse_verification_verdict,
    validate_script_falsifiability,
)


def test_verdict_shadowing_fixed():
    # Regression for PRIME-CONJ-1794449900: early per-scale SUPPORTED must NOT shadow final FALSIFIED
    multi_scale_out = (
        "N: 15, VERDICT: SUPPORTED\n"
        "N: 30, VERDICT: SUPPORTED\n"
        "N: 500, VERDICT: FALSIFIED\n"
        "N: 1000, VERDICT: FALSIFIED\n"
        "VERDICT: FALSIFIED\n"
    )
    assert parse_verification_verdict(multi_scale_out, "", 0) == "EMPIRICALLY FALSIFIED"

    # Non-zero exit code must never pass as SUPPORTED
    assert parse_verification_verdict("VERDICT: SUPPORTED\n", "Traceback...", 1) == "EXECUTION ERROR (code 1)"

    # Trivial null rejection
    assert (
        parse_verification_verdict("Null baseline indistinguishable\nVERDICT: TRIVIAL_NULL\n", "", 0)
        == "REJECTED (TRIVIAL / INDISTINGUISHABLE FROM NULL)"
    )

    # Clean SUPPORTED with NaN must fail
    assert "NaN" in parse_verification_verdict("ratio = nan\nVERDICT: SUPPORTED\n", "", 0)

    # Clean SUPPORTED without NaN passes
    assert parse_verification_verdict("All scales + null contrast passed\nVERDICT: SUPPORTED\n", "", 0) == "EMPIRICALLY SUPPORTED"


def test_head_tail_output_preservation():
    head = "HEADER_START\n" + ("row_early\n" * 250)
    tail = ("row_late\n" * 250) + "OUT_OF_SAMPLE_N=1000000: resid=-0.042\nVERDICT: FALSIFIED"
    combined = head + tail
    assert len(combined) > 3000

    summarized = format_head_tail_output(combined, head_chars=400, tail_chars=600)
    assert "HEADER_START" in summarized
    assert "...[middle output truncated]..." in summarized
    assert "OUT_OF_SAMPLE_N=1000000: resid=-0.042" in summarized
    assert summarized.endswith("VERDICT: FALSIFIED")


def test_ast_anti_tautology_gate():
    # Unconditional SUPPORTED without FALSIFIED branch must be rejected pre-execution
    bad_unconditional = 'print("VERDICT: SUPPORTED")\n'
    assert validate_script_falsifiability(bad_unconditional) is not None

    # SyntaxError caught cleanly
    bad_syntax = 'if True\n    print("VERDICT: SUPPORTED")\n'
    err = validate_script_falsifiability(bad_syntax)
    assert err is not None and "SyntaxError" in err

    # Valid falsifiable script using pre-imported cramer_random_primes executes end-to-end
    valid_code = """
primes = generate_primes(200_000)
null_p = cramer_random_primes(200_000, seed=11)
g_real = compute_prime_gaps(primes)
g_null = compute_prime_gaps(null_p)
real_ratio = float(np.sum(g_real == 6)) / max(1, int(np.sum(g_real == 2)))
null_ratio = float(np.sum(g_null == 6)) / max(1, int(np.sum(g_null == 2)))
print(f"real_sexy_to_twin={real_ratio:.4f}, cramer_null_sexy_to_twin={null_ratio:.4f}")
if real_ratio > 1.5 and null_ratio < 1.1:
    print("VERDICT: SUPPORTED")
else:
    print("VERDICT: FALSIFIED")
"""
    res = execute_verification_script(valid_code, timeout_sec=30)
    assert res["verdict"] == "EMPIRICALLY SUPPORTED", f"Unexpected verdict: {res}"


def test_dedup_domain_scope_and_boilerplate():
    state = {"domains": {}, "models": {}, "confirmed_laws": [], "seen_signatures": []}
    kb.remember_signature(
        state,
        "Prime Power Sieve Eliminated Composite Decay Law",
        "Let C_p(N) be odd composites eliminated by p^2 with decay rate 1/(p ln p).",
        domain_id="prime_powers",
    )

    # Different theorem in same domain sharing only boilerplate words ('Prime Power Sieve ... Law')
    # should NOT collide (fixes PRIME-CONJ-1795055115 false-positive duplicate)
    m = kb.find_duplicate_match(
        state,
        "Prime Power Sieve Spectral Momentum Conservation Law",
        "Define normalized spectral momentum M_hat(N) = sum (C_p(N)/p^2) log(p) / pi(sqrt(N)) with loglog transfer tau_0 = 0.12.",
        domain_id="prime_powers",
    )
    assert m is None, f"Should not false-positive on boilerplate words, got {m}"

    # Cross-domain independence
    m_cross = kb.find_duplicate_match(
        state,
        "Prime Power Sieve Eliminated Composite Decay Law",
        "Let C_p(N) be odd composites eliminated by p^2 with decay rate 1/(p ln p).",
        domain_id="smooth_numbers",
    )
    assert m_cross is None, "Signatures in different domains should not collide"

    # True near-duplicate in the SAME domain MUST still be caught
    m_dup = kb.find_duplicate_match(
        state,
        "Prime Power Sieve Eliminated Composite Decay Principle",
        "Let C_p(N) denote odd composites eliminated starting at p^2 obeying decay rate 1/(p ln p).",
        domain_id="prime_powers",
    )
    assert m_dup is not None, "Expected true duplicate in same domain to be caught"


def test_ten_domains_registered():
    ids = list_domain_ids()
    assert len(ids) == 10, f"Expected 10 domains, got {len(ids)}: {ids}"
    for expected in ("chebyshev_race", "multiplicative_walk", "prime_constellations", "smooth_numbers"):
        assert expected in DOMAINS


if __name__ == "__main__":
    print("Running Prime Frontier Research Pipeline regression suite...")
    test_verdict_shadowing_fixed()
    test_head_tail_output_preservation()
    test_ast_anti_tautology_gate()
    test_dedup_domain_scope_and_boilerplate()
    test_ten_domains_registered()
    print("All 5 pipeline regression test groups passed!")
