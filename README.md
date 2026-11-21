# Prime Frontier

An empirical number-theory laboratory combining fast deterministic computational engines with an autonomous, multi-model AI **"Agent World"** that formulates, stress-tests against null baselines, refines, and archives mathematical conjectures across **10 prime-number domains**—compiling everything into an interactive dark-mode laboratory dashboard.

## Architecture

```
core/                       Deterministic math engines (vectorized NumPy / FFT / pre-cached tables)
  sieve.py                    Segmented prime sieve, gap statistics, Cramér maximal-gap records
  modular_orbits.py           Lemke Oliver–Soundararajan consecutive-prime modular bias (mod 10 / mod 30)
  zeta_harmonics.py           Riemann zeta zero harmonics (3,000 pre-cached zeros) & explicit-formula staircase
  goldbach.py                 FFT-accelerated Goldbach partition counts g(n) and record-low tracking
  arithmetic_functions.py     Möbius/Mertens/Liouville sieves, Chebyshev prime races, k-tuple constellations,
                              y-smooth friable numbers, and Cramér random-prime null controls
  domains.py                  Registry packaging all 10 research domains + Null-Baseline prompt contracts

models/                     Autonomous research & verification agents
  llm_client.py               Zero-dependency OpenRouter client with CoT token-budget protection
  math_researcher.py          Single inquiry cycle: harvest + Domain Memory -> conjecture -> 4-layer sandboxed
                              verification + AST/Null gates -> self-repair -> 1-step Scientific Refinement
  knowledge_base.py           Persistent memory: domain coverage, boilerplate-stripped dedup, Confirmed Laws
  agent_world.py              Multi-cycle orchestrator rotating across 10 domains and frontier reasoning models

visualizer/                 Presentation layer
  compile_dashboard.py        Compiles empirical metrics and escaped KaTeX cards into dashboard.html
  ulam_sacks.py               Generates Ulam square and Sacks Archimedean prime spiral plots

experiments/                Standalone exploratory scripts (e.g., analyze_user_formula.py)
tests/                      Automated pipeline, verdict-gate, dedup, XSS, and sandbox regression tests
outputs/                    Persistent research state (conjectures.json, knowledge_base.json, *.png)
AUDIT.md                    Security, performance, and scientific pipeline audit log
```

## 10 Research Domains

1. `gap_dynamics` — Prime gap distribution, twin/cousin/sexy ratios, and Cramér records
2. `modular_bias_mod10` — Consecutive-prime transition memory modulo 10
3. `modular_bias_mod30` — 8-residue singular-series & Markov spectral structure modulo 30
4. `goldbach_partitions` — Goldbach comet modulation, dyadic scaling, and record lows of $g(n)$
5. `zeta_harmonics` — Riemann zeta zero spacing statistics (3,000 cached zeros) & GUE repulsion
6. `prime_powers` — Least-prime-factor Eratosthenes attrition starting at $p^2$ & Buchstab scaling
7. `chebyshev_race` — Rubinstein–Sarnak prime races $\pi(x; q, a) - \pi(x; q, b)$
8. `multiplicative_walk` — Möbius $\mu(n)$, Mertens $M(x)/\sqrt{x}$, and Liouville/Pólya walks
9. `prime_constellations` — Admissible prime $k$-tuples vs. Hardy–Littlewood singular series
10. `smooth_numbers` — Friable ($y$-smooth) counts $\Psi(x, y)$, Dickman $\rho(u)$, and $P^+(n)$

## Setup & Quick Start

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Run all deterministic engine & security/pipeline regression tests
python -m tests.test_research_pipeline

# Compile the interactive dashboard (no API key needed)
python -m visualizer.compile_dashboard
```

### Running Autonomous Research Cycles

Provide an OpenRouter API key via `$env:OPENROUTER_API_KEY = "sk-or-..."` or `config/.env` (gitignored):

```powershell
python -m models.agent_world --cycles 9 --sleep 9 --temperature 0.75 --models sonnet,glm,deepseek
python -m visualizer.compile_dashboard
```

## Verification Sandbox & Null-Baseline Gate

Every LLM-generated verification script in `models/math_researcher.py` runs inside a **4-layer sandbox**:
- Static AST validation blocking network/process/filesystem modules, dynamic execution (`eval`/`exec`/`open`), and uncached `mpmath.zetazero` loops;
- Secret-stripped allowlist subprocess environment (`OPENROUTER_API_KEY` is never passed to child scripts);
- Isolated ephemeral working directory (`TemporaryDirectory`) + POSIX memory/file limits;
- CPython `sys.addaudithook` runtime enforcement blocking sockets, subprocesses, and file writes.

## License

Released under the [MIT License](LICENSE).
