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

Fix the bug and return the corrected, complete, self-contained script. Keep the same conjecture and
intent, just make it run cleanly and print quantitative multi-scale + null-baseline diagnostics.
Note core utility return signatures if used:
- generate_primes(limit: int) -> np.ndarray
- compute_modular_transitions(primes, mod=10) -> dict with keys "residues", "transition_matrix" (2D list), "transition_counts" (2D list), "expected_uniform"
- goldbach_partition_counts(limit: int) -> dict {{even_n: count}}
- get_zeta_zeros(count: int) -> list of floats (up to 1000 cached)
- mobius_sieve(limit), mertens_function(limit), liouville_sieve(limit), sum_liouville(limit), largest_prime_factor(limit) -> np.ndarray
- smooth_number_counts(limit, y_bound), chebyshev_prime_race(limit, mod, r_lead, r_trail), prime_constellation_counts(limit) -> dict
- cramer_random_primes(limit, seed=42) -> np.ndarray
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
    """Extract Python code inside triple backticks."""
    match = re.search(r"```(?:python)?\s*([\s\S]*?)```", text or "")
    if match:
        return match.group(1).strip()
    return ""


def parse_conjecture_sections(content: str) -> Dict[str, str]:
    """Parse structured sections (name, formulation, target, code) without single-line truncation."""
    name_m = re.search(r"##\s*CONJECTURE_NAME:\s*([^\n\r]+)", content)
    name = name_m.group(1).strip() if name_m else "Empirical Prime Invariant"

    form_m = re.search(r"##\s*MATHEMATICAL_FORMULATION:\s*([\s\S]*?)(?:##|\Z)", content)
    formulation = form_m.group(1).strip() if form_m else content[:400]

    target_m = re.search(r"##\s*HYPOTHESIS_TARGET:\s*([\s\S]*?)(?:##|```|\Z)", content)
    target = target_m.group(1).strip() if target_m else "Quantitative validation"

    code = extract_code_block(content)
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


def validate_script_falsifiability(code: str) -> Optional[str]:
    """
    Static AST & anti-tautology pre-check before subprocess execution:
      1. Catch Python SyntaxErrors immediately.
      2. Ensure the script has at least one conditional (`If` / `IfExp`) and contains
         both 'SUPPORTED' and 'FALSIFIED' tokens so a script cannot unconditionally pass.
    Returns an error message string if invalid, or None if valid.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return f"SyntaxError at line {exc.lineno}: {exc.msg}"

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


def execute_verification_script(code: str, timeout_sec: int = 120) -> Dict:
    """Safely execute generated verification code with pre-imported fast domain helpers and capture stdout/verdict."""
    static_err = validate_script_falsifiability(code)
    if static_err:
        return {
            "verdict": f"EXECUTION ERROR ({static_err})",
            "output": static_err,
            "elapsed_s": 0.0,
            "success": False,
        }

    test_script_path = os.path.join(OUTPUTS_DIR, f"_temp_verify_{os.getpid()}.py")
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    preamble = (
        "import sys, os, math\n"
        "import numpy as np\n"
        f"sys.path.insert(0, {repr(BASE_DIR)})\n"
        "from core.sieve import generate_primes, compute_prime_gaps, compute_gap_statistics, find_maximal_gap_records\n"
        "from core.modular_orbits import compute_modular_transitions, compute_residue_autocorrelation\n"
        "from core.goldbach import goldbach_partition_counts, goldbach_stats\n"
        "from core.zeta_harmonics import get_zeta_zeros, exact_psi, approximate_psi_riemann\n"
        "from core.arithmetic_functions import (\n"
        "    mobius_sieve, mertens_function, liouville_sieve, sum_liouville,\n"
        "    largest_prime_factor, smooth_number_counts, chebyshev_prime_race,\n"
        "    prime_constellation_counts, cramer_random_primes,\n"
        ")\n"
    )
    full_code = preamble + "\n" + code

    with open(test_script_path, "w", encoding="utf-8") as f:
        f.write(full_code)

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["MPLBACKEND"] = "Agg"
    env["PYTHONPATH"] = BASE_DIR + os.pathsep + env.get("PYTHONPATH", "")

    t0 = time.time()
    try:
        proc = subprocess.run(
            [sys.executable, test_script_path],
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
            "output": f"Computation exceeded {timeout_sec} seconds.",
            "elapsed_s": timeout_sec,
            "success": False,
        }
    finally:
        if os.path.exists(test_script_path):
            try:
                os.remove(test_script_path)
            except Exception:
                pass


def _needs_repair(verdict: str) -> bool:
    v = (verdict or "").upper()
    return ("ERROR" in v) or (v == "INCONCLUSIVE")


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

    content = res.get("content") or res.get("reasoning") or ""
    reasoning = res.get("reasoning") or ""
    parsed = parse_conjecture_sections(content)
    name = parsed["name"]
    formulation = parsed["formulation"]
    target = parsed["target"]
    code = parsed["code"]
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
        retry_content = retry_res.get("content") or retry_res.get("reasoning") or ""
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
            while _needs_repair(verif["verdict"]) and repairs_attempted < max_repairs:
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
