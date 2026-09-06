#!/usr/bin/env python3
"""
models/math_researcher.py - Autonomous Mathematical Conjecture & Verification Engine

Harnesses DeepSeek R1, Qwen 2.5 Math, and Claude Sonnet to autonomously formulate
mathematical hypotheses on prime numbers, write executable Python tests, run them,
and log the scientific findings.
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
from core.sieve import generate_primes, compute_prime_gaps, compute_gap_statistics
from core.modular_orbits import compute_modular_transitions

OUTPUTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")
CONJECTURES_FILE = os.path.join(OUTPUTS_DIR, "conjectures.json")

RESEARCH_PROMPT_TEMPLATE = """You are an elite research mathematician at the Prime Frontier laboratory.
Below is real empirical data computed on the first 100,000 prime numbers:

1. Gap Statistics:
- Mean Gap: {mean_gap}
- Max Gap: {max_gap}
- Twin Primes (g=2): {twin_primes}
- Sexy Primes (g=6): {sexy_primes} (most frequent gap)

2. Consecutive Modular Transition Bias (Mod 10):
- P(1 -> 1): {p11}% (extreme repulsion)
- P(1 -> 3): {p13}% (attraction)
- P(1 -> 7): {p17}% (attraction)
- P(1 -> 9): {p19}%

Your task:
Formulate ONE novel, mathematically precise conjecture or predictive heuristic regarding prime numbers (e.g. next-prime gap upper bounds, modular autocorrelation patterns, or divisibility of gap moments).
Then, write a self-contained Python script to test and either support or falsify your conjecture empirically up to prime 200,000.
IMPORTANT: In your script, use a fast boolean array sieve (Sieve of Eratosthenes) for generating primes so execution completes in under 2 seconds. Do not use slow trial division.

Format your output strictly as:
## CONJECTURE_NAME: <concise title>
## MATHEMATICAL_FORMULATION: <LaTeX formula and rigorous explanation>
## HYPOTHESIS_TARGET: <what exact quantitative threshold must hold>
## PYTHON_VERIFICATION:
```python
# Self-contained python script that prints 'VERDICT: SUPPORTED' or 'VERDICT: FALSIFIED' along with quantitative evidence
```
"""

def extract_code_block(text: str) -> str:
    """Extract Python code inside triple backticks."""
    match = re.search(r'```(?:python)?\s*([\s\S]*?)```', text)
    if match:
        return match.group(1).strip()
    return ""

def execute_verification_script(code: str, timeout_sec: int = 45) -> Dict:
    """Safely execute generated verification code and capture stdout/verdict."""
    test_script_path = os.path.join(OUTPUTS_DIR, "_temp_verify.py")
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    
    with open(test_script_path, "w", encoding="utf-8") as f:
        f.write(code)
        
    t0 = time.time()
    try:
        proc = subprocess.run(
            [sys.executable, test_script_path],
            capture_output=True,
            text=True,
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
            "output": "Computation exceeded 30 seconds.",
            "elapsed_s": timeout_sec,
            "success": False
        }
    finally:
        if os.path.exists(test_script_path):
            try:
                os.remove(test_script_path)
            except Exception:
                pass

def run_hypothesis_cycle(model_alias: str = "haiku") -> Dict:
    """
    Run one full scientific inquiry cycle:
    1. Gather empirical prime data
    2. Prompt reasoning model (Claude / Qwen / DeepSeek)
    3. Extract & execute Python test
    4. Save conjecture and empirical verdict
    """
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Gathering baseline prime statistics...", flush=True)
    primes = generate_primes(1_300_000)[:100_000]
    gaps = compute_prime_gaps(primes)
    stats = compute_gap_statistics(primes, gaps)
    mod10 = compute_modular_transitions(primes, mod=10)
    
    # Fill prompt template
    prompt = RESEARCH_PROMPT_TEMPLATE.format(
        mean_gap=stats["mean_gap"],
        max_gap=stats["max_gap"],
        twin_primes=f"{stats['twin_primes_count']:,}",
        sexy_primes=f"{stats['sexy_primes_count']:,}",
        p11=round(mod10["transition_matrix"][0][0] * 100, 1),
        p13=round(mod10["transition_matrix"][0][1] * 100, 1),
        p17=round(mod10["transition_matrix"][0][2] * 100, 1),
        p19=round(mod10["transition_matrix"][0][3] * 100, 1)
    )
    
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Querying mathematical reasoning model ({model_alias})...", flush=True)
    res = query_math_model(
        prompt=prompt,
        system_prompt="You are an elite research mathematician formulating testable conjectures in number theory.",
        model_alias=model_alias,
        temperature=0.2,
        max_tokens=2000
    )
    
    if not res["success"]:
        print(f"Error querying model: {res['error']}")
        return {"success": False, "error": res["error"]}
        
    content = res["content"]
    reasoning = res.get("reasoning", "")
    
    # Parse sections
    name_m = re.search(r'##\s*CONJECTURE_NAME:\s*([^\n\r]+)', content)
    name = name_m.group(1).strip() if name_m else "Empirical Prime Invariant"
    
    form_m = re.search(r'##\s*MATHEMATICAL_FORMULATION:\s*([\s\S]*?)(?:##|\Z)', content)
    formulation = form_m.group(1).strip() if form_m else content[:400]
    
    target_m = re.search(r'##\s*HYPOTHESIS_TARGET:\s*([^\n\r]+)', content)
    target = target_m.group(1).strip() if target_m else "Quantitative validation"
    
    code = extract_code_block(content)
    
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Conjectured: {name}", flush=True)
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Executing verification script...", flush=True)
    
    verif = {"verdict": "NO CODE GENERATED", "output": "", "elapsed_s": 0.0, "success": False}
    if code:
        verif = execute_verification_script(code)
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Verification Result: {verif['verdict']}", flush=True)
    
    conjecture_record = {
        "id": f"PRIME-CONJ-{int(time.time())}",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model": res["model"],
        "name": name,
        "formulation": formulation,
        "target": target,
        "reasoning": reasoning[:800] if reasoning else "Direct derivation",
        "code": code,
        "verdict": verif["verdict"],
        "output": verif["output"],
        "elapsed_s": verif["elapsed_s"]
    }
    
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
    rec = run_hypothesis_cycle(model_alias="haiku")
    if rec.get("success", True):
        print(f"\nRecorded Conjecture: {rec['name']}")
        print(f"Verdict: {rec['verdict']}")
        print(f"Execution Output:\n{rec['output']}")
