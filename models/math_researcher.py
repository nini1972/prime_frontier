#!/usr/bin/env python3
"""
models/math_researcher.py - Autonomous Mathematical Conjecture & Verification Engine

Harnesses frontier mathematical reasoning models to autonomously formulate
hypotheses across 10 number-theory domains (core/domains.py), enforce a
Null-Baseline / Anti-Triviality Gate, execute vectorized Python tests,
self-repair broken scripts, pivot away from near-duplicates via in-cycle
novelty retries, and refine falsified bounds using tail-preserved empirical
counter-evidence.
"""

import ast
import sys
import os
import json
import re
import tempfile
import time
import subprocess
from datetime import datetime
from typing import Dict, Optional

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.llm_client import query_math_model
from models import knowledge_base as kb
from core.domains import get_domain, list_domain_ids

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
CONJECTURES_FILE = os.path.join(OUTPUTS_DIR, "conjectures.json")

REPAIR_PROMPT_TEMPLATE = """The following self-contained Python script was meant to empirically test a
mathematical conjecture about prime numbers, but it failed to run or lacked a valid falsifiable verdict.

--- ORIGINAL SCRIPT ---
```python
{code}
```

--- ERROR / OUTPUT ---
{error}

Fix the bug and return the corrected, complete verification script. Keep the same conjecture and
intent, just make it run cleanly and print quantitative multi-scale + null-baseline diagnostics.
IMPORTANT PERFORMANCE RULE: All core C/NumPy/FFT project utilities listed below are ALREADY pre-imported in the execution harness — call them directly (or `from core... import ...`) and NEVER redefine them with your own `def generate_primes(...)`, `def compute_modular_transitions(...)`, `def goldbach_partition_counts(...)`, `def get_zeta_zeros(...)`, `def mobius_sieve(...)`, `def mertens_function(...)`, `def liouville_sieve(...)`, or `def largest_prime_factor(...)` (redefining protected engines raises `PerformanceError`)!
Core utility return signatures if used:
- generate_primes(limit: int) -> np.ndarray
- compute_modular_transitions(primes, mod=10) -> dict with keys "residues", "transition_matrix" (2D list), "transition_counts" (2D list), "expected_uniform"
- goldbach_partition_counts(limit: int) -> dict {{even_n: count}}
- get_zeta_zeros(count: int) -> list of floats (up to 3000 cached; NEVER call mpmath.zetazero directly)
- mobius_sieve(limit), mertens_function(limit), liouville_sieve(limit), sum_liouville(limit), largest_prime_factor(limit) -> np.ndarray
- smooth_number_counts(limit, y_bound), chebyshev_prime_race(limit, mod, r_lead, r_trail) -> dict
- prime_constellation_counts(limit) -> dict with keys "twins_0_2", "cousins_0_4", "sexy_0_6", "triplets_0_2_6", "triplets_0_4_6", "quadruplets_0_2_6_8"
- cramer_random_primes(limit, seed=42) -> np.ndarray
- cramer_mobius_liouville(limit, seed=42) -> (mu_null, lam_null) vectorized random-multiplicative Möbius/Liouville arrays
- njit (from numba import njit): use `@njit` (WITHOUT `cache=True`) to compile any remaining tight numeric loops to C speed
IMPORTANT: Do NOT print 'VERDICT:' on intermediate per-scale lines (use PASS/FAIL instead).
Print 'VERDICT: SUPPORTED', 'VERDICT: FALSIFIED', or 'VERDICT: TRIVIAL_NULL' ONLY ONCE on the very last line, with genuine branches for both SUPPORTED and FALSIFIED!
Return ONLY the corrected script as a single Python code block, with no other commentary.
"""

REFINEMENT_PROMPT_TEMPLATE = """Your initial mathematical conjecture in domain '{domain_id}' was tested against empirical data and returned '{prior_verdict}':

--- INITIAL CONJECTURE ---
Name: {name}
Formulation: {formulation}
Target: {target}

--- VERIFICATION SCRIPT ---
```python
{code}
```

--- EMPIRICAL OUTPUT (COUNTER-EVIDENCE) ---
{output}

Scientific Refinement Task:
Use the exact numerical observations in the empirical output above (especially the large-N out-of-sample rows and null/baseline comparison at the bottom) to diagnose why your initial bound, scaling exponent, or null contrast failed.
Formulate a REFINED, mathematically non-trivial conjecture that captures the true empirical behavior AND beats or separates from a trivial null baseline (e.g., Cramér random primes, shuffled gaps, or first-order PNT/HL heuristic).
Test your refined law across MULTIPLE scales (including a larger out-of-sample scale than the initial script, to prevent overfitting).
IMPORTANT PERFORMANCE RULE: All fast C/NumPy/FFT project utilities (`generate_primes`, `compute_prime_gaps`, `compute_modular_transitions`, `goldbach_partition_counts`, `get_zeta_zeros`, `mobius_sieve`, `mertens_function`, `liouville_sieve`, `sum_liouville`, `largest_prime_factor`, `smooth_number_counts`, `chebyshev_prime_race`, `prime_constellation_counts`, `cramer_random_primes`, `cramer_mobius_liouville`, and `@njit`) are ALREADY pre-imported in the execution harness — use them directly and NEVER redefine slow Python-loop versions (e.g. do NOT write your own `def goldbach_partition_counts`, `def generate_primes`, or `mpmath.zetazero` loop)!
Do NOT widen tolerances into a trivial tautology! Print PASS/FAIL on per-scale lines and print 'VERDICT: SUPPORTED', 'VERDICT: FALSIFIED', or 'VERDICT: TRIVIAL_NULL' ONLY ONCE on the final line.

Format your output strictly as:
## CONJECTURE_NAME: <distinctive title for the calibrated law>
## MATHEMATICAL_FORMULATION: <LaTeX formula and rigorous explanation incorporating the empirical calibration & null contrast>
## HYPOTHESIS_TARGET: <exact quantitative threshold + null/baseline separation across multiple scales>
## PYTHON_VERIFICATION:
```python
# Self-contained python script testing across multiple scales + null control, ending with ONE final VERDICT line
```
"""

NOVELTY_RETRY_PROMPT_TEMPLATE = """Your proposed conjecture in domain '{domain_id}' is too similar to prior research already recorded in the Prime Frontier Knowledge Base:

- Your proposed title: {proposed_name}
- Matched prior research: {matched_title} (title overlap={title_overlap}, body overlap={body_overlap})

{original_prompt}

CRITICAL NOVELTY OVERRIDE:
Formulate a COMPLETELY ORTHOGONAL conjecture in domain '{domain_id}' using a DIFFERENT mathematical observable, statistic, transform, or higher-order interaction than '{matched_title}' and all titles listed in Prior Domain Memory!
Follow the exact `## CONJECTURE_NAME:`, `## MATHEMATICAL_FORMULATION:`, `## HYPOTHESIS_TARGET:`, and `## PYTHON_VERIFICATION:` format above.
"""


def extract_code_block(text: str) -> str:
    """Extract the primary Python verification script (preferring the last ```python block)."""
    py_blocks = re.findall(r"```python\s*([\s\S]*?)```", text or "", re.IGNORECASE)
    if py_blocks:
        return py_blocks[-1].strip()
    any_blocks = re.findall(r"```\w*\s*([\s\S]*?)```", text or "")
    if any_blocks:
        return any_blocks[-1].strip()
    return ""


def parse_conjecture_sections(content: str) -> Dict[str, str]:
    """Parse structured sections (name, formulation, target, code) resilient to markdown variations."""
    text = content or ""
    name_m = re.search(
        r"(?:#{1,4}\s*|\*\*)?CONJECTURE_NAME\s*(?:\*\*)?\s*:\s*([^\n\r*]+)",
        text,
        re.IGNORECASE,
    )
    name = name_m.group(1).strip().strip("*` ") if name_m else "Empirical Prime Invariant"

    form_m = re.search(
        r"(?:#{1,4}\s*|\*\*)?MATHEMATICAL_FORMULATION\s*(?:\*\*)?\s*:\s*([\s\S]*?)(?=(?:#{1,4}\s*|\*\*)?(?:HYPOTHESIS_TARGET|PYTHON_VERIFICATION)\b|\Z)",
        text,
        re.IGNORECASE,
    )
    formulation = form_m.group(1).strip() if form_m else text[:400]

    target_m = re.search(
        r"(?:#{1,4}\s*|\*\*)?HYPOTHESIS_TARGET\s*(?:\*\*)?\s*:\s*([\s\S]*?)(?=(?:#{1,4}\s*|\*\*)?PYTHON_VERIFICATION\b|```|\Z)",
        text,
        re.IGNORECASE,
    )
    target = target_m.group(1).strip() if target_m else "Quantitative multi-scale & null-baseline validation"

    code = extract_code_block(text)
    return {
        "name": name,
        "formulation": formulation,
        "target": target,
        "code": code,
    }


def format_head_tail_output(text: str, head_chars: int = 1000, tail_chars: int = 2200) -> str:
    """
    Preserve BOTH the beginning (setup/small-scale table header) and the end
    (large-N out-of-sample rows, null comparison, Python tracebacks, and final VERDICT)
    instead of slicing off everything past character 1000.
    """
    cleaned = (text or "").strip()
    limit = head_chars + tail_chars + 40
    if len(cleaned) <= limit:
        return cleaned
    return f"{cleaned[:head_chars]}\n...[middle output truncated]...\n{cleaned[-tail_chars:]}"


_BLOCKED_IMPORT_ROOTS = {
    "subprocess", "socket", "ssl", "urllib", "http", "requests", "httpx", "aiohttp",
    "ftplib", "smtplib", "poplib", "imaplib", "telnetlib", "xmlrpc", "shutil", "ctypes",
    "multiprocessing", "webbrowser", "pty", "pickle", "marshal", "shelve", "importlib",
    "builtins", "tempfile", "pathlib", "glob",
}

_BLOCKED_CALL_NAMES = {"eval", "exec", "compile", "__import__", "open", "input", "breakpoint"}

_BLOCKED_ATTR_NAMES = {
    "system", "popen", "spawnl", "spawnle", "spawnlp", "spawnlpe", "spawnv", "spawnve",
    "spawnvp", "spawnvpe", "execl", "execle", "execlp", "execlpe", "execv", "execve",
    "execvp", "execvpe", "fork", "kill", "remove", "unlink", "rmdir", "removedirs",
    "rename", "replace", "makedirs", "mkdir", "chmod", "chown", "chdir", "listdir",
    "scandir", "walk", "environ", "getenv", "putenv",
}


_PROTECTED_ENGINE_FUNCS = {
    "generate_primes", "goldbach_partition_counts", "get_zeta_zeros",
    "mobius_sieve", "mertens_function", "liouville_sieve", "largest_prime_factor",
    "cramer_mobius_liouville",
}


def validate_script_falsifiability(code: str) -> Optional[str]:
    """
    Static AST security, performance, & anti-tautology pre-check before subprocess execution:
      1. Catch Python SyntaxErrors immediately.
      2. Block dangerous imports, dynamic execution builtins, filesystem/process/env access,
         and uncached `mpmath.zetazero` loops.
      3. Ensure the script has at least one conditional (`If` / `IfExp`) and contains
         both 'SUPPORTED' and 'FALSIFIED' tokens so a script cannot unconditionally pass.
    Returns an error message string if invalid, or None if valid.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return f"SyntaxError at line {exc.lineno}: {exc.msg}"

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root in _BLOCKED_IMPORT_ROOTS:
                    return f"SecurityError: importing '{root}' is forbidden in verification scripts."
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if root in _BLOCKED_IMPORT_ROOTS:
                return f"SecurityError: importing from '{root}' is forbidden in verification scripts."
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in _BLOCKED_CALL_NAMES:
                return f"SecurityError: calling '{node.func.id}()' is forbidden in verification scripts."
        elif isinstance(node, ast.FunctionDef) and node.name in _PROTECTED_ENGINE_FUNCS:
            return (
                f"PerformanceError: do not redefine pre-imported fast engine '{node.name}()' "
                "with a slow Python loop; call the built-in vectorized/FFT helper directly."
            )
        elif isinstance(node, ast.Attribute):
            if node.attr == "zetazero":
                return (
                    "PerformanceError: direct mpmath.zetazero() calls time out; "
                    "use the pre-cached get_zeta_zeros(count) helper (instantaneous up to count=3000)."
                )
            if node.attr in _BLOCKED_ATTR_NAMES:
                return f"SecurityError: accessing '.{node.attr}' is forbidden in verification scripts."

    upper = code.upper()
    if "SUPPORTED" not in upper or "FALSIFIED" not in upper:
        return (
            "Verification script must contain explicit conditional branches capable of "
            "printing both 'VERDICT: SUPPORTED' and 'VERDICT: FALSIFIED'."
        )

    has_branch = any(isinstance(node, (ast.If, ast.IfExp)) for node in ast.walk(tree))
    if not has_branch:
        return "Verification script has no conditional `if` logic; verdict cannot be unconditional."

    return None


def parse_verification_verdict(stdout: str, stderr: str, returncode: int) -> str:
    """
    Determine the authoritative empirical verdict from subprocess output:
      - Non-zero exit codes NEVER qualify as SUPPORTED.
      - Uses the LAST 'VERDICT: <TOKEN>' emitted in stdout so per-scale intermediate
        prints at small N can never shadow a final FALSIFIED verdict at large N.
      - Guards against NaN / empty-slice numerical corruption.
    """
    combined = (stdout or "") + "\n" + (stderr or "")
    has_nan = bool(
        re.search(r"(?<![a-zA-Z0-9_])nan(?![a-zA-Z0-9_])", stdout or "", re.IGNORECASE)
        or "Mean of empty slice" in combined
        or "invalid value encountered" in combined
    )

    if returncode != 0:
        return f"EXECUTION ERROR (code {returncode})"

    matches = re.findall(r"VERDICT:\s*(SUPPORTED|FALSIFIED|TRIVIAL_NULL)\b", stdout or "", re.IGNORECASE)
    if not matches:
        return "INCONCLUSIVE"

    final_token = matches[-1].upper()
    if final_token == "TRIVIAL_NULL":
        return "REJECTED (TRIVIAL / INDISTINGUISHABLE FROM NULL)"
    if final_token == "FALSIFIED":
        return "EMPIRICALLY FALSIFIED"
    if final_token == "SUPPORTED":
        if has_nan:
            return "EXECUTION ERROR (NaN detected in numerical output; cannot confirm law with NaN values)"
        return "EMPIRICALLY SUPPORTED"

    return "INCONCLUSIVE"


def _build_sandboxed_env(tmp_dir: str) -> Dict[str, str]:
    """
    Construct a minimal, secret-free environment allowlist for verification subprocesses
    so API keys (OPENROUTER_API_KEY, GITHUB_TOKEN, etc.) are never exposed to LLM code.
    """
    env: Dict[str, str] = {}
    safe_os_keys = (
        "PATH", "SYSTEMROOT", "WINDIR", "COMSPEC", "LD_LIBRARY_PATH",
        "LANG", "LC_ALL", "APPDATA", "LOCALAPPDATA", "USERPROFILE",
        "HOMEDRIVE", "HOMEPATH", "VIRTUAL_ENV",
    )
    for safe_key in safe_os_keys:
        val = os.environ.get(safe_key)
        if val:
            env[safe_key] = val
    env.setdefault("HOME", os.environ.get("HOME") or tmp_dir)
    env["TMPDIR"] = tmp_dir
    env["TEMP"] = tmp_dir
    env["TMP"] = tmp_dir
    env["MPLCONFIGDIR"] = tmp_dir
    env["MPLBACKEND"] = "Agg"
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["NUMBA_CACHE_DIR"] = tmp_dir
    env["PYTHONPATH"] = BASE_DIR
    return env


def execute_verification_script(code: str, timeout_sec: int = 120) -> Dict:
    """
    Execute generated verification code inside a 4-layer sandbox:
      1. Static AST security & falsifiability gate (`validate_script_falsifiability`)
      2. Stripped secret-free allowlist environment (`_build_sandboxed_env`)
      3. Isolated ephemeral `TemporaryDirectory` working directory + POSIX `resource` caps
      4. Unremovable CPython `sys.addaudithook` blocking outbound sockets, subprocesses, and file writes
    """
    static_err = validate_script_falsifiability(code)
    if static_err:
        return {
            "verdict": f"EXECUTION ERROR ({static_err})",
            "output": static_err,
            "elapsed_s": 0.0,
            "success": False,
        }

    preamble = (
        "import sys, math\n"
        "import numpy as np\n"
        f"sys.path.insert(0, {repr(BASE_DIR)})\n"
        "from core.sieve import generate_primes, compute_prime_gaps, compute_gap_statistics, find_maximal_gap_records\n"
        "from core.modular_orbits import compute_modular_transitions, compute_residue_autocorrelation\n"
        "from core.goldbach import goldbach_partition_counts, goldbach_stats\n"
        "from core.zeta_harmonics import get_zeta_zeros, exact_psi, approximate_psi_riemann\n"
        "from core.arithmetic_functions import (\n"
        "    mobius_sieve, mertens_function, liouville_sieve, sum_liouville,\n"
        "    largest_prime_factor, smooth_number_counts, chebyshev_prime_race,\n"
        "    prime_constellation_counts, cramer_random_primes, cramer_mobius_liouville,\n"
        ")\n"
        "try:\n"
        "    from numba import njit as _raw_njit\n"
        "    import numba as _numba_mod\n"
        "    def njit(*args, **kwargs):\n"
        "        kwargs.pop('cache', None)\n"
        "        return _raw_njit(*args, **kwargs)\n"
        "    _numba_mod.njit = njit\n"
        "except Exception:\n"
        "    def njit(*args, **kwargs):\n"
        "        if len(args) == 1 and callable(args[0]) and not kwargs:\n"
        "            return args[0]\n"
        "        return lambda fn: fn\n"
        "try:\n"
        "    import resource as _res\n"
        "    _res.setrlimit(_res.RLIMIT_CORE, (0, 0))\n"
        "    _res.setrlimit(_res.RLIMIT_FSIZE, (4 * 1024 * 1024, 4 * 1024 * 1024))\n"
        "    _res.setrlimit(_res.RLIMIT_AS, (3 * 1024 * 1024 * 1024, 3 * 1024 * 1024 * 1024))\n"
        "except Exception:\n"
        "    pass\n"
        "def _pf_audit_guard(event, args):\n"
        "    if event.startswith(('socket.', 'urllib.', 'http.', 'ftplib.', 'smtplib.', 'telnetlib.')):\n"
        "        raise PermissionError(f'Network access blocked by sandbox: {event}')\n"
        "    if event in ('subprocess.Popen', 'os.system', 'os.exec', 'os.posix_spawn', 'os.spawn', 'os.fork', 'os.remove', 'os.unlink', 'os.rmdir', 'os.rename', 'os.replace', 'os.chmod', 'os.chown'):\n"
        "        raise PermissionError(f'System/filesystem mutation blocked by sandbox: {event}')\n"
        "    if event == 'open' and len(args) >= 2 and isinstance(args[1], str) and any(c in args[1] for c in 'wax+'):\n"
        "        raise PermissionError('File writes are blocked by sandbox')\n"
        "sys.addaudithook(_pf_audit_guard)\n"
        "del _pf_audit_guard\n"
    )
    full_code = preamble + "\n" + code

    with tempfile.TemporaryDirectory(prefix="pf_verify_") as tmp_dir:
        test_script_path = os.path.join(tmp_dir, "_verify.py")
        with open(test_script_path, "w", encoding="utf-8") as f:
            f.write(full_code)

        env = _build_sandboxed_env(tmp_dir)
        t0 = time.time()
        try:
            proc = subprocess.run(
                [sys.executable, "-B", test_script_path],
                cwd=tmp_dir,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=env,
                timeout=timeout_sec,
            )
            elapsed = time.time() - t0
            verdict = parse_verification_verdict(proc.stdout, proc.stderr, proc.returncode)
            combined_output = (proc.stdout or "") + (("\n" + proc.stderr) if proc.stderr else "")
            stored_output = format_head_tail_output(combined_output, head_chars=1000, tail_chars=2200)

            return {
                "verdict": verdict,
                "output": stored_output,
                "elapsed_s": round(elapsed, 2),
                "success": proc.returncode == 0 and not _needs_repair(verdict),
            }
        except subprocess.TimeoutExpired:
            return {
                "verdict": "TIMEOUT EXCEEDED",
                "output": (
                    f"Computation exceeded {timeout_sec} seconds due to slow Python loops or excessive N. "
                    "Vectorize loops with NumPy, decorate custom tight numeric loops with `@njit` "
                    "(without cache=True), use pre-imported vectorized helpers such as "
                    "`cramer_mobius_liouville(limit, seed)` and `generate_primes(limit)`, "
                    "or slightly reduce the largest scale N so total runtime stays under 45s."
                ),
                "elapsed_s": timeout_sec,
                "success": False,
            }


def _needs_repair(verdict: str) -> bool:
    v = (verdict or "").upper()
    return ("ERROR" in v) or (v == "INCONCLUSIVE") or (v == "TIMEOUT EXCEEDED")


def _eligible_for_refinement(verdict: str) -> bool:
    v = (verdict or "").upper()
    return v == "EMPIRICALLY FALSIFIED" or v.startswith("REJECTED (TRIVIAL")


def attempt_repair(code: str, error_output: str, model_alias: str) -> str:
    """Ask the reasoning model to fix its own broken verification script using tail-preserved traceback."""
    prompt = REPAIR_PROMPT_TEMPLATE.format(
        code=code,
        error=format_head_tail_output(error_output, head_chars=500, tail_chars=1800),
    )
    res = query_math_model(
        prompt=prompt,
        system_prompt="You are a meticulous Python debugger fixing a numerical number-theory verification script.",
        model_alias=model_alias,
        temperature=0.1,
        max_tokens=4500,
    )
    if not res["success"]:
        return code
    fixed = extract_code_block(res["content"])
    return fixed if fixed else code


def attempt_refinement(
    domain_id: str,
    name: str,
    formulation: str,
    target: str,
    code: str,
    empirical_output: str,
    prior_verdict: str,
    model_alias: str,
    temperature: float = 0.5,
) -> Optional[Dict[str, str]]:
    """
    1-Step Scientific Refinement Loop:
    Feed tail-preserved empirical counter-evidence back to the reasoning model so it
    can calibrate its scaling law across multiple scales + null baseline.
    """
    prompt = REFINEMENT_PROMPT_TEMPLATE.format(
        domain_id=domain_id,
        prior_verdict=prior_verdict,
        name=name,
        formulation=formulation[:1400],
        target=target[:500],
        code=code[:2400],
        output=format_head_tail_output(empirical_output, head_chars=700, tail_chars=1800),
    )
    res = query_math_model(
        prompt=prompt,
        system_prompt=(
            "You are a senior experimental number theorist calibrating a falsified hypothesis "
            "using empirical counter-evidence. Formulate a sharp, non-trivial multi-scale mathematical law."
        ),
        model_alias=model_alias,
        temperature=max(0.2, min(temperature, 0.6)),
        max_tokens=5000,
    )
    if not res["success"]:
        return None
    content = res.get("content") or res.get("reasoning") or ""
    parsed = parse_conjecture_sections(content)
    if not parsed["code"]:
        return None
    return parsed


def run_hypothesis_cycle(
    model_alias: str = "haiku",
    domain_id: Optional[str] = None,
    max_repairs: int = 2,
    max_novelty_retries: int = 2,
    temperature: float = 0.7,
    enable_refinement: bool = True,
) -> Dict:
    """
    Run one full scientific inquiry cycle:
    1. Pick a research domain
    2. Gather fresh dynamically-windowed empirical data + Prior Domain Memory
    3. Prompt reasoning model for a novel conjecture, retrying in-cycle if near-duplicate
    4. Validate & execute Python verification script (with Null-Baseline Gate & self-repair)
    5. On falsification or trivial-null verdict, run 1-step Scientific Refinement
    6. Persist conjecture, verdict, and updated knowledge-base state
    """
    state = kb.load_state()
    if domain_id is None:
        domain_id = kb.select_next_domain(state, list_domain_ids())
    domain = get_domain(domain_id)

    def log(msg: str) -> None:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

    log(f"Domain selected: {domain.name} ({domain_id})")
    log("Gathering baseline empirical data & prior domain memory...")
    harvested = domain.harvest()
    base_prompt = domain.prompt_template.format(**harvested)
    memory_block = kb.build_domain_memory_prompt(domain_id, CONJECTURES_FILE)
    if memory_block and "Your task:" in base_prompt:
        prompt = base_prompt.replace("Your task:", f"{memory_block}\nYour task:", 1)
    else:
        prompt = base_prompt + memory_block

    system_prompt = (
        "You are a visionary mathematical researcher at the Prime Frontier laboratory. "
        "Formulate bold, non-trivial, quantitative conjectures in number theory that beat or separate "
        "from naive random baselines. Seek deep empirical invariants, singular-series corrections, "
        "or spectral bounds rather than loose tautologies."
    )

    log(f"Querying mathematical reasoning model ({model_alias}, T={temperature})...")
    res = query_math_model(
        prompt=prompt,
        system_prompt=system_prompt,
        model_alias=model_alias,
        temperature=temperature,
        max_tokens=5000,
    )

    if not res["success"]:
        print(f"Error querying model: {res['error']}")
        return {"success": False, "error": res["error"], "domain": domain_id}

    content = res.get("content") or ""
    reasoning = res.get("reasoning") or ""
    parsed = parse_conjecture_sections(content)
    name = parsed["name"]
    formulation = parsed["formulation"]
    target = parsed["target"]
    code = parsed["code"]
    if not code or name == "Empirical Prime Invariant":
        log("Incomplete structured sections or missing code block; requesting concise structured completion...")
        fmt_res = query_math_model(
            prompt=(
                prompt
                + "\n\nIMPORTANT: Output ALL FOUR sections (`## CONJECTURE_NAME:`, `## MATHEMATICAL_FORMULATION:`, "
                "`## HYPOTHESIS_TARGET:`, and `## PYTHON_VERIFICATION:` with a complete ```python script) concisely."
            ),
            system_prompt=system_prompt,
            model_alias=model_alias,
            temperature=max(0.3, temperature - 0.15),
            max_tokens=16384,
        )
        if fmt_res["success"]:
            fmt_content = fmt_res.get("content") or ""
            fmt_parsed = parse_conjecture_sections(fmt_content)
            if fmt_parsed["code"]:
                res = fmt_res
                reasoning = fmt_res.get("reasoning") or reasoning
                name = fmt_parsed["name"]
                formulation = fmt_parsed["formulation"]
                target = fmt_parsed["target"]
                code = fmt_parsed["code"]
    log(f"Conjectured: {name}")

    # In-cycle novelty retry loop: avoid wasting nightly slots on near-duplicates
    novelty_retries = 0
    dup_match = kb.find_duplicate_match(state, name, formulation, domain_id=domain_id)
    while dup_match is not None and novelty_retries < max_novelty_retries:
        novelty_retries += 1
        log(
            f"Near-duplicate of '{dup_match['matched_title']}' "
            f"(title_ov={dup_match['title_overlap']}, body_ov={dup_match['body_overlap']}). "
            f"Requesting orthogonal conjecture retry {novelty_retries}/{max_novelty_retries}..."
        )
        retry_prompt = NOVELTY_RETRY_PROMPT_TEMPLATE.format(
            domain_id=domain_id,
            proposed_name=name,
            matched_title=dup_match["matched_title"],
            title_overlap=dup_match["title_overlap"],
            body_overlap=dup_match["body_overlap"],
            original_prompt=prompt,
        )
        retry_res = query_math_model(
            prompt=retry_prompt,
            system_prompt=system_prompt,
            model_alias=model_alias,
            temperature=min(0.95, temperature + 0.1 * novelty_retries),
            max_tokens=5000,
        )
        if not retry_res["success"]:
            break
        retry_content = retry_res.get("content") or ""
        retry_parsed = parse_conjecture_sections(retry_content)
        if retry_parsed["code"]:
            res = retry_res
            reasoning = retry_res.get("reasoning") or reasoning
            name = retry_parsed["name"]
            formulation = retry_parsed["formulation"]
            target = retry_parsed["target"]
            code = retry_parsed["code"]
            log(f"Re-conjectured ({novelty_retries}): {name}")
            dup_match = kb.find_duplicate_match(state, name, formulation, domain_id=domain_id)
        else:
            break

    repairs_attempted = 0
    refined_from = None
    if not code:
        log("Model failed to generate a Python verification script after retry; skipping persistence of empty stub.")
        return {
            "success": False,
            "error": "NO CODE GENERATED after structured retry",
            "domain": domain_id,
        }
    if dup_match is not None:
        log("Rejected after novelty retries: still too similar to prior research. Skipping execution.")
        verif = {
            "verdict": "SKIPPED (DUPLICATE OF PRIOR RESEARCH)",
            "output": f"Matched prior signature: {dup_match['matched_title']}",
            "elapsed_s": 0.0,
            "success": False,
        }
    else:
        verif = {"verdict": "NO CODE GENERATED", "output": "", "elapsed_s": 0.0, "success": False}
        if code:
            log("Executing verification script...")
            verif = execute_verification_script(code)
            timeout_repairs = 0
            while _needs_repair(verif["verdict"]) and repairs_attempted < max_repairs:
                if verif["verdict"] == "TIMEOUT EXCEEDED":
                    if timeout_repairs >= 1:
                        break
                    timeout_repairs += 1
                repairs_attempted += 1
                log(
                    f"Verification errored ({verif['verdict']}). "
                    f"Requesting self-repair attempt {repairs_attempted}/{max_repairs}..."
                )
                repaired_code = attempt_repair(code, verif["output"], model_alias)
                if repaired_code.strip() == code.strip():
                    log("Model returned no usable fix. Giving up on repair.")
                    break
                code = repaired_code
                verif = execute_verification_script(code)

            if enable_refinement and _eligible_for_refinement(verif["verdict"]) and verif["output"]:
                log(
                    f"Initial hypothesis returned '{verif['verdict']}'. "
                    "Running 1-step scientific refinement using tail-preserved empirical evidence..."
                )
                refined = attempt_refinement(
                    domain_id=domain_id,
                    name=name,
                    formulation=formulation,
                    target=target,
                    code=code,
                    empirical_output=verif["output"],
                    prior_verdict=verif["verdict"],
                    model_alias=model_alias,
                    temperature=temperature,
                )
                if refined and not kb.is_duplicate(
                    state, refined["name"], refined["formulation"], domain_id=domain_id
                ):
                    refined_verif = execute_verification_script(refined["code"])
                    if _needs_repair(refined_verif["verdict"]) and repairs_attempted < max_repairs:
                        repairs_attempted += 1
                        fixed_ref = attempt_repair(refined["code"], refined_verif["output"], model_alias)
                        if fixed_ref.strip() != refined["code"].strip():
                            refined["code"] = fixed_ref
                            refined_verif = execute_verification_script(refined["code"])
                    if not _needs_repair(refined_verif["verdict"]):
                        kb.remember_signature(state, name, formulation, domain_id=domain_id)
                        refined_from = name
                        name = refined["name"]
                        formulation = refined["formulation"]
                        target = refined["target"]
                        code = refined["code"]
                        verif = refined_verif
                        log(f"Refined Conjecture: {name} -> {verif['verdict']}")

        log(f"Verification Result: {verif['verdict']}")
        kb.remember_signature(state, name, formulation, domain_id=domain_id)

    conjecture_record = {
        "id": f"PRIME-CONJ-{int(time.time())}",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model": res["model"],
        "domain": domain_id,
        "name": name,
        "formulation": formulation,
        "target": target,
        "reasoning": reasoning[:1000] if reasoning else "Direct derivation",
        "code": code,
        "verdict": verif["verdict"],
        "output": verif["output"],
        "elapsed_s": verif["elapsed_s"],
        "repairs_attempted": repairs_attempted,
        "novelty_retries": novelty_retries,
        "refined_from": refined_from,
    }

    kb.record_result(state, domain_id, model_alias, verif["verdict"])
    kb.promote_if_confirmed(state, conjecture_record)
    kb.save_state(state)

    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    conjectures = []
    if os.path.exists(CONJECTURES_FILE):
        try:
            with open(CONJECTURES_FILE, "r", encoding="utf-8") as f:
                conjectures = json.load(f)
        except Exception:
            conjectures = []

    conjectures.insert(0, conjecture_record)
    with open(CONJECTURES_FILE, "w", encoding="utf-8") as f:
        json.dump(conjectures, f, indent=2)

    return conjecture_record


if __name__ == "__main__":
    print("Testing Autonomous Math Researcher Cycle...")
    print(f"Available domains ({len(list_domain_ids())}): {', '.join(list_domain_ids())}")
    rec = run_hypothesis_cycle(model_alias="haiku")
    if rec.get("success", True) is not False:
        print(f"\nRecorded Conjecture: {rec['name']}  [domain={rec.get('domain')}]")
        print(f"Verdict: {rec['verdict']}")
        print(f"Execution Output:\n{rec['output']}")
