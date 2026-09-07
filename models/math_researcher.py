#!/usr/bin/env python3
"""
models/math_researcher.py - Autonomous Mathematical Conjecture & Verification Engine

Harnesses DeepSeek R1, Qwen 2.5 Math, and Claude Sonnet to autonomously formulate
mathematical hypotheses across multiple prime-number research domains (see
core/domains.py), write executable Python tests, run them, self-repair broken
verification code, and log the scientific findings -- persisting coverage and
duplicate-detection state in models/knowledge_base.py so the world accumulates
knowledge across runs instead of restarting from zero every time.
"""

import sys
import os
import json
import re
import time
import subprocess
from datetime import datetime
from typing import Dict, List, Optional

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.llm_client import query_math_model
from models import knowledge_base as kb
from core.domains import get_domain, list_domain_ids

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
CONJECTURES_FILE = os.path.join(OUTPUTS_DIR, "conjectures.json")

REPAIR_PROMPT_TEMPLATE = """The following self-contained Python script was meant to empirically test a
mathematical conjecture about prime numbers, but it failed to run correctly.

--- ORIGINAL SCRIPT ---
```python
{code}
```

--- ERROR / OUTPUT ---
{error}

Fix the bug and return the corrected, complete, self-contained script. Keep the same conjecture and
intent, just make it run correctly. It must still print 'VERDICT: SUPPORTED' or 'VERDICT: FALSIFIED'.
Return ONLY the corrected script as a single Python code block, with no other commentary.
"""

def extract_code_block(text: str) -> str:
    """Extract Python code inside triple backticks."""
    match = re.search(r'```(?:python)?\s*([\s\S]*?)```', text)
    if match:
        return match.group(1).strip()
    return ""

def execute_verification_script(code: str, timeout_sec: int = 45) -> Dict:
    """Safely execute generated verification code with pre-imported fast sieve helpers and capture stdout/verdict."""
    test_script_path = os.path.join(OUTPUTS_DIR, "_temp_verify.py")
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    
    # Prepend project root to sys.path and provide core sieve utilities
    preamble = (
        "import sys, os\n"
        f"sys.path.insert(0, {repr(BASE_DIR)})\n"
        "from core.sieve import generate_primes, compute_prime_gaps, compute_gap_statistics\n"
    )
    full_code = preamble + code

    with open(test_script_path, "w", encoding="utf-8") as f:
        f.write(full_code)
        
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
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
            timeout=timeout_sec
        )
        elapsed = time.time() - t0
        output = proc.stdout + proc.stderr
        
        verdict = "INCONCLUSIVE"
        if "VERDICT: SUPPORTED" in output:
            verdict = "EMPIRICALLY SUPPORTED"
        elif "VERDICT: FALSIFIED" in output:
            verdict = "EMPIRICALLY FALSIFIED"
        elif proc.returncode != 0:
            verdict = f"EXECUTION ERROR (code {proc.returncode})"
            
        return {
            "verdict": verdict,
            "output": output.strip()[:1000],
            "elapsed_s": round(elapsed, 2),
            "success": proc.returncode == 0
        }
    except subprocess.TimeoutExpired:
        return {
            "verdict": "TIMEOUT EXCEEDED",
            "output": f"Computation exceeded {timeout_sec} seconds.",
            "elapsed_s": timeout_sec,
            "success": False
        }
    finally:
        if os.path.exists(test_script_path):
            try:
                os.remove(test_script_path)
            except Exception:
                pass

def _needs_repair(verdict: str) -> bool:
    """
    Execution errors and inconclusive runs are worth an automatic self-repair attempt.
    Timeouts and genuine SUPPORTED/FALSIFIED verdicts are not: retrying slow code just
    times out again, and a real verdict is a real result, not a bug.
    """
    v = (verdict or "").upper()
    return ("ERROR" in v) or (v == "INCONCLUSIVE")


def attempt_repair(code: str, error_output: str, model_alias: str) -> str:
    """Ask the reasoning model to fix its own broken verification script."""
    prompt = REPAIR_PROMPT_TEMPLATE.format(code=code, error=error_output[:1500])
    res = query_math_model(
        prompt=prompt,
        system_prompt="You are a meticulous Python debugger fixing a broken numerical verification script.",
        model_alias=model_alias,
        temperature=0.1,
        max_tokens=2000,
    )
    if not res["success"]:
        return code
    fixed = extract_code_block(res["content"])
    return fixed if fixed else code


def run_hypothesis_cycle(model_alias: str = "haiku", domain_id: Optional[str] = None, max_repairs: int = 2) -> Dict:
    """
    Run one full scientific inquiry cycle:
    1. Pick a research domain (explicit, or chosen by the knowledge base's coverage-driven explorer)
    2. Gather fresh empirical data for that domain (core/domains.py)
    3. Prompt reasoning model (Claude / Qwen / DeepSeek) for a novel conjecture
    4. Extract & execute its Python verification script, self-repairing on execution errors
    5. Reject near-duplicates of prior conjectures using the knowledge base
    6. Save the conjecture, empirical verdict, and updated knowledge-base coverage stats
    """
    state = kb.load_state()
    if domain_id is None:
        domain_id = kb.select_next_domain(state, list_domain_ids())
    domain = get_domain(domain_id)

    def log(msg: str) -> None:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

    log(f"Domain selected: {domain.name} ({domain_id})")
    log("Gathering baseline empirical data...")
    harvested = domain.harvest()
    prompt = domain.prompt_template.format(**harvested)

    log(f"Querying mathematical reasoning model ({model_alias})...")
    res = query_math_model(
        prompt=prompt,
        system_prompt="You are an elite research mathematician formulating testable conjectures in number theory.",
        model_alias=model_alias,
        temperature=0.2,
        max_tokens=3500 if "r1" in model_alias else 2000,
    )

    if not res["success"]:
        print(f"Error querying model: {res['error']}")
        return {"success": False, "error": res["error"], "domain": domain_id}

    content = res.get("content") or res.get("reasoning") or ""
    reasoning = res.get("reasoning") or ""

    # Parse sections
    name_m = re.search(r'##\s*CONJECTURE_NAME:\s*([^\n\r]+)', content)
    name = name_m.group(1).strip() if name_m else "Empirical Prime Invariant"

    form_m = re.search(r'##\s*MATHEMATICAL_FORMULATION:\s*([\s\S]*?)(?:##|\Z)', content)
    formulation = form_m.group(1).strip() if form_m else content[:400]

    target_m = re.search(r'##\s*HYPOTHESIS_TARGET:\s*([^\n\r]+)', content)
    target = target_m.group(1).strip() if target_m else "Quantitative validation"

    code = extract_code_block(content)
    log(f"Conjectured: {name}")

    repairs_attempted = 0
    if kb.is_duplicate(state, name, formulation):
        log("Rejected: too similar to a previously explored conjecture. Skipping execution.")
        verif = {
            "verdict": "SKIPPED (DUPLICATE OF PRIOR RESEARCH)",
            "output": "",
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
                log(f"Verification errored ({verif['verdict']}). "
                    f"Requesting self-repair attempt {repairs_attempted}/{max_repairs}...")
                repaired_code = attempt_repair(code, verif["output"], model_alias)
                if repaired_code.strip() == code.strip():
                    log("Model returned no usable fix. Giving up on repair.")
                    break
                code = repaired_code
                verif = execute_verification_script(code)
        log(f"Verification Result: {verif['verdict']}")
        kb.remember_signature(state, name, formulation)

    conjecture_record = {
        "id": f"PRIME-CONJ-{int(time.time())}",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model": res["model"],
        "domain": domain_id,
        "name": name,
        "formulation": formulation,
        "target": target,
        "reasoning": reasoning[:800] if reasoning else "Direct derivation",
        "code": code,
        "verdict": verif["verdict"],
        "output": verif["output"],
        "elapsed_s": verif["elapsed_s"],
        "repairs_attempted": repairs_attempted,
    }

    kb.record_result(state, domain_id, model_alias, verif["verdict"])
    kb.promote_if_confirmed(state, conjecture_record)
    kb.save_state(state)

    # Save to conjectures log
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
    print(f"Available domains: {', '.join(list_domain_ids())}")
    rec = run_hypothesis_cycle(model_alias="haiku")
    if rec.get("success", True) is not False:
        print(f"\nRecorded Conjecture: {rec['name']}  [domain={rec.get('domain')}]")
        print(f"Verdict: {rec['verdict']}")
        print(f"Execution Output:\n{rec['output']}")
