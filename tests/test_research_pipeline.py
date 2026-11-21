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
    _build_sandboxed_env,
    execute_verification_script,
    format_head_tail_output,
    parse_verification_verdict,
    validate_script_falsifiability,
)
from visualizer.compile_dashboard import generate_html


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


def test_dashboard_xss_escaping_and_katex_preservation():
    sample_data = {
        "generated_at": "Nov 21, 2026",
        "stats": {
            "total_primes": 10,
            "max_prime": 29,
            "max_gap": 6,
            "mean_gap": 2.8,
            "sexy_primes_count": 3,
            "twin_primes_count": 2,
            "cousin_primes_count": 2,
        },
        "gap_records": [],
        "mod10": {},
        "autocorr": {1: -0.05},
        "zeta_zeros": [14.1347],
        "images": {"zeta_staircase": None, "modular_bias": None, "ulam_spiral": None, "sacks_spiral": None},
        "agent_world": {
            "totals": {"attempts": 1, "supported": 1, "falsified": 0, "errors": 0, "timeout": 0, "duplicates": 0},
            "domain_coverage": [],
            "confirmed_laws_count": 1,
        },
        "conjectures": [
            {
                "id": "PRIME-CONJ-XSS",
                "timestamp": "2026-11-21 12:00:00",
                "domain": "gap_dynamics",
                "name": "Test <script>alert(1)</script>",
                "model": "sonnet",
                "verdict": "EMPIRICALLY SUPPORTED",
                "formulation": "For $p_n < x$, <script>alert('xss_form')</script> holds.",
                "target": "<img src=x onerror=alert('xss_target')> with $\\Delta < 0.05$",
                "code": "if 1 < 2:\n    print('VERDICT: SUPPORTED')\nelse:\n    print('VERDICT: FALSIFIED')",
                "output": "VERDICT: SUPPORTED",
            }
        ],
    }
    rendered = generate_html(sample_data)
    assert "<script>alert('xss_form')</script>" not in rendered
    assert "<img src=x onerror=" not in rendered
    assert "&lt;script&gt;alert(&#x27;xss_form&#x27;)&lt;/script&gt;" in rendered
    assert "$p_n &lt; x$" in rendered


def test_sandbox_blocks_secrets_network_and_slow_zetazero():
    # 1. Secret stripping in child environment
    os.environ["OPENROUTER_API_KEY"] = "sk-or-test-secret-do-not-leak"
    env = _build_sandboxed_env("/tmp/pf_test")
    assert "OPENROUTER_API_KEY" not in env
    del os.environ["OPENROUTER_API_KEY"]

    # 2. AST gate blocks dangerous imports, builtins, env access, and raw mpmath.zetazero
    for forbidden in (
        "import subprocess\nif True:\n    print('VERDICT: SUPPORTED')\nelse:\n    print('VERDICT: FALSIFIED')",
        "import urllib.request\nif True:\n    print('VERDICT: SUPPORTED')\nelse:\n    print('VERDICT: FALSIFIED')",
        "import os\nx = os.environ\nif x:\n    print('VERDICT: SUPPORTED')\nelse:\n    print('VERDICT: FALSIFIED')",
        "f = open('pwn.txt', 'w')\nif f:\n    print('VERDICT: SUPPORTED')\nelse:\n    print('VERDICT: FALSIFIED')",
        "import mpmath as mp\nz = mp.zetazero(1)\nif z:\n    print('VERDICT: SUPPORTED')\nelse:\n    print('VERDICT: FALSIFIED')",
    ):
        err = validate_script_falsifiability(forbidden)
        assert err is not None, f"Expected AST gate to reject: {forbidden}"

    # 3. Legitimate scipy + numpy verification script runs cleanly inside the 4-layer sandbox
    legit_scipy = """
from scipy.stats import spearmanr
xs = np.arange(1, 20, dtype=float)
ys = 2.0 * xs + 1.0
rho, _ = spearmanr(xs, ys)
print(f"rho={rho:.4f}")
if rho > 0.99:
    print("VERDICT: SUPPORTED")
else:
    print("VERDICT: FALSIFIED")
"""
    res = execute_verification_script(legit_scipy, timeout_sec=30)
    assert res["verdict"] == "EMPIRICALLY SUPPORTED", f"Sandbox broke legitimate scipy script: {res}"


if __name__ == "__main__":
    print("Running Prime Frontier Research Pipeline regression suite...")
    test_verdict_shadowing_fixed()
    test_head_tail_output_preservation()
    test_ast_anti_tautology_gate()
    test_dedup_domain_scope_and_boilerplate()
    test_ten_domains_registered()
    test_dashboard_xss_escaping_and_katex_preservation()
    test_sandbox_blocks_secrets_network_and_slow_zetazero()
    print("All 7 pipeline & security regression test groups passed!")
