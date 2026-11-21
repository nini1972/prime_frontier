# Prime Frontier — Security, Engineering & Scientific Pipeline Audit

**Date:** 2026-11-21  
**Scope:** Deterministic number-theory engines (`core/`), autonomous LLM research & verification pipeline (`models/`), interactive dashboard compiler (`visualizer/`), regression suite (`tests/`), CI/CD workflows (`.github/workflows/`), and full 108-commit Git history.

---

## 🔴 High-Priority Security Findings (✅ Resolved)

1. **Stored XSS & Unescaped Inequalities in `dashboard.html` (`visualizer/compile_dashboard.py`) — ✅ RESOLVED**
   - **Finding:** `compile_dashboard.py` escaped `id`, `timestamp`, `name`, `model`, `code`, and `output`, but injected `formulation` and `target` unescaped into HTML. Adversarial or malformed model output containing `<script>` or `<img onerror=...>` could execute in the browser, and valid LaTeX inequalities (`$p_n < x$`) risked DOM misparsing.
   - **Fix Implemented:** Wrapped `formulation` and `target` in `html.escape()` (preserving KaTeX auto-rendering via decoded DOM text nodes) and added regression test `test_dashboard_xss_escaping_and_katex_preservation` in `tests/test_research_pipeline.py`.

2. **Unsandboxed Execution of LLM-Generated Verification Code (`models/math_researcher.py`) — ✅ RESOLVED**
   - **Finding:** `execute_verification_script` previously ran model-authored Python inside `outputs/` using `env = os.environ.copy()`, exposing `OPENROUTER_API_KEY` and repository write access to generated scripts.
   - **Fix Implemented:** Enforced a 4-layer verification sandbox in `models/math_researcher.py`:
     1. **Static AST Security Gate:** Blocks dangerous modules (`subprocess`, `socket`, `urllib`, `http`, `requests`, `shutil`, `ctypes`, `pickle`, `importlib`, `pathlib`, etc.), dynamic execution builtins (`eval`, `exec`, `compile`, `__import__`, `open`), and OS/environment mutation attributes (`os.environ`, `os.getenv`, `os.system`, `os.remove`, etc.).
     2. **Secret-Free Allowlist Environment (`_build_sandboxed_env`):** Strips `OPENROUTER_API_KEY`, `GITHUB_TOKEN`, and all non-essential host variables from child subprocesses; also removed the legacy sibling-directory `.env` lookup from `models/llm_client.py`.
     3. **Isolated Ephemeral Working Directory & POSIX Resource Caps:** Executes inside `tempfile.TemporaryDirectory()` (`cwd`) with `-B` and applies `RLIMIT_AS` (3 GB), `RLIMIT_FSIZE` (4 MB), and `RLIMIT_CORE` (0) on Linux CI runners.
     4. **Unremovable CPython `sys.addaudithook`:** Blocks outbound socket/HTTP events, process spawning, and file write modes (`w`, `a`, `x`, `+`) at the CPython runtime layer.

---

## ⚡ Performance & Scientific Pipeline Findings (✅ Resolved)

3. **`zeta_harmonics` Subprocess Timeouts Above `N = 1,000` (`core/zeta_harmonics.py`) — ✅ RESOLVED**
   - **Finding:** `get_zeta_zeros(count)` previously shipped 1,000 precomputed zeros; scripts requesting `count = 2,500` (e.g., `PRIME-CONJ-1795263018`) fell back to `mpmath.zetazero(k)` (~0.3s/zero) in a fresh subprocess and hit the 120s timeout before finishing. Separately, `PRIME-CONJ-1795228698` bypassed `get_zeta_zeros` and called `mpmath.zetazero` directly at `dps=30`.
   - **Fix Implemented:**
     - Expanded the read-only precomputed zero dataset (`core/zeta_zeros_3000.json`) to **3,000 non-trivial Riemann zeta zeros** ($\gamma_1 = 14.1347$ through $\gamma_{3000}$), making `get_zeta_zeros(count <= 3000)` instantaneous ($<1\text{ ms}$) without requiring untrusted subprocess write access to `outputs/`.
     - Added an AST check rejecting direct `.zetazero()` calls so models automatically self-repair to use `get_zeta_zeros(count)`.

4. **Undocumented `prime_constellation_counts` Dictionary Keys (`core/domains.py`) — ✅ RESOLVED**
   - **Finding:** `PRIME-CONJ-1795261966` failed with `KeyError: 'twin_count'` because `OUTPUT_FORMAT_INSTRUCTIONS` omitted the exact dictionary keys returned by `prime_constellation_counts(limit)`.
   - **Fix Implemented:** Documented all exact return keys (`"twins_0_2"`, `"cousins_0_4"`, `"sexy_0_6"`, `"triplets_0_2_6"`, `"triplets_0_4_6"`, `"quadruplets_0_2_6_8"`, and HL ratios) in both `core/domains.py` and `REPAIR_PROMPT_TEMPLATE`.

---

## 🟡 Medium & 🟢 Hygiene Items

5. **License & Dependency Pinning — ✅ RESOLVED**
   - Added MIT [`LICENSE`](LICENSE).
   - Added compatible upper-bound version constraints in [`requirements.txt`](requirements.txt) and documented why `scipy` is installed for autonomous verification scripts.
   - Untracked `.github/instructions/codacy.instructions.md` from the Git index (already listed in `.gitignore`) and updated [`README.md`](README.md) to document all 10 domains, `core/arithmetic_functions.py`, `experiments/`, and the verification sandbox.

6. **Full Git History Credential Scan — ✅ VERIFIED CLEAN**
   - Scanned diffs across all **108 commits and all branches** for OpenRouter (`sk-or-v1-...`), Anthropic, OpenAI, and GitHub tokens: **0 leaked secrets found** (only literal `sk-or-...` documentation placeholders in `README.md`).

7. **Future Architectural Consideration (Artifact Growth)**
   - `outputs/conjectures.json` (~2.0 MB) provides serverless persistence for nightly GitHub Actions runs; as the archive passes 1,000+ cycles, consider deploying `dashboard.html` via GitHub Pages Actions artifacts instead of committing compiled HTML on each nightly run.
