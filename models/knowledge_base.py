#!/usr/bin/env python3
"""
models/knowledge_base.py - Persistent Memory for the Prime Frontier Agent World

Gives the autonomous agents long-term memory across runs, stored in
outputs/knowledge_base.json:

  - per-domain attempt/verdict tallies, used to steer exploration toward
    under-covered territory (`select_next_domain`)
  - a coarse token-overlap fingerprint index used to reject near-duplicate
    conjectures (`is_duplicate`) so the world doesn't keep "re-discovering"
    (and re-breaking on) the same idea
  - a "Hall of Fame" of empirically confirmed laws (`promote_if_confirmed`)
  - per-model reliability stats (attempts/success/errors)

Without this, each run of the researcher is a stateless one-shot; with it,
the repo behaves like a persistent research world that remembers what it has
already tried.
"""

import os
import re
import json
import random
import datetime
from typing import Dict, List, Optional

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KB_FILE = os.path.join(BASE_DIR, "outputs", "knowledge_base.json")

_DEFAULT_STATE = {
    "domains": {},         # domain_id -> {attempts, supported, falsified, errors, timeout, duplicates, last_run}
    "models": {},          # model_alias -> {attempts, success, errors}
    "confirmed_laws": [],  # promoted subset of conjectures with a SUPPORTED verdict
    "seen_signatures": [], # token-set fingerprints of past conjecture name+formulation
}


def load_state() -> Dict:
    """Load persistent agent-world state, seeding sane defaults for any missing keys."""
    state = json.loads(json.dumps(_DEFAULT_STATE))  # deep copy of defaults
    if os.path.exists(KB_FILE):
        try:
            with open(KB_FILE, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            state.update(loaded)
        except Exception:
            pass
    return state


def save_state(state: Dict) -> None:
    os.makedirs(os.path.dirname(KB_FILE), exist_ok=True)
    with open(KB_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


def _fingerprint(text: str) -> str:
    """Normalize a conjecture name/formulation into a coarse token-set fingerprint."""
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    stopwords = {
        "the", "a", "an", "of", "and", "or", "to", "is", "are", "for", "that",
        "this", "with", "let", "be", "we", "conjecture", "prime", "primes",
        "empirically", "following", "define", "over", "all", "up", "first",
    }
    tokens = sorted(set(t for t in tokens if t not in stopwords and len(t) > 2))
    return "|".join(tokens)


def is_duplicate(state: Dict, name: str, formulation: str, overlap_threshold: float = 0.6) -> bool:
    """Check whether a proposed conjecture is too similar to a previously seen one (Jaccard token overlap)."""
    candidate = set(_fingerprint(f"{name} {formulation}").split("|"))
    candidate.discard("")
    if not candidate:
        return False
    for sig in state.get("seen_signatures", []):
        existing = set(sig.split("|"))
        existing.discard("")
        if not existing:
            continue
        overlap = len(candidate & existing) / max(1, len(candidate | existing))
        if overlap >= overlap_threshold:
            return True
    return False


def remember_signature(state: Dict, name: str, formulation: str) -> None:
    state.setdefault("seen_signatures", []).append(_fingerprint(f"{name} {formulation}"))
    state["seen_signatures"] = state["seen_signatures"][-500:]  # keep bounded


def record_result(state: Dict, domain_id: str, model_alias: str, verdict: str) -> None:
    d = state.setdefault("domains", {}).setdefault(domain_id, {
        "attempts": 0, "supported": 0, "falsified": 0, "errors": 0, "timeout": 0,
        "duplicates": 0, "last_run": None,
    })
    m = state.setdefault("models", {}).setdefault(model_alias, {
        "attempts": 0, "success": 0, "errors": 0,
    })
    d["attempts"] += 1
    m["attempts"] += 1
    verdict_upper = (verdict or "").upper()

    if "DUPLICATE" in verdict_upper:
        d["duplicates"] += 1
    elif "SUPPORTED" in verdict_upper:
        d["supported"] += 1
        m["success"] += 1
    elif "FALSIFIED" in verdict_upper:
        d["falsified"] += 1
        m["success"] += 1
    elif "TIMEOUT" in verdict_upper:
        d["timeout"] += 1
        m["errors"] += 1
    else:
        d["errors"] += 1
        m["errors"] += 1

    d["last_run"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def promote_if_confirmed(state: Dict, conjecture_record: Dict) -> None:
    if "SUPPORTED" in (conjecture_record.get("verdict") or "").upper():
        laws = state.setdefault("confirmed_laws", [])
        laws.insert(0, {
            "id": conjecture_record.get("id"),
            "name": conjecture_record.get("name"),
            "domain": conjecture_record.get("domain"),
            "timestamp": conjecture_record.get("timestamp"),
        })
        state["confirmed_laws"] = laws[:200]


def select_next_domain(state: Dict, domain_ids: List[str], exclude: Optional[List[str]] = None) -> str:
    """
    Pick the next research domain to explore. Biased toward domains attempted
    least so far (coverage-driven exploration), with enough randomness that
    the world doesn't collapse into a deterministic round-robin.
    """
    exclude = exclude or []
    candidates = [d for d in domain_ids if d not in exclude] or list(domain_ids)
    domains_state = state.get("domains", {})
    weights = [1.0 / (1.0 + domains_state.get(d, {}).get("attempts", 0)) for d in candidates]
    return random.choices(candidates, weights=weights, k=1)[0]


def summarize(state: Dict) -> Dict:
    """Produce a compact summary dict, handy for CLI printing and dashboard rendering."""
    domains = state.get("domains", {})
    totals = {"attempts": 0, "supported": 0, "falsified": 0, "errors": 0, "timeout": 0, "duplicates": 0}
    for d in domains.values():
        for k in totals:
            totals[k] += d.get(k, 0)
    return {
        "totals": totals,
        "domains": domains,
        "models": state.get("models", {}),
        "confirmed_laws_count": len(state.get("confirmed_laws", [])),
        "confirmed_laws": state.get("confirmed_laws", [])[:20],
    }


if __name__ == "__main__":
    print("Testing Knowledge Base...")
    state = load_state()
    print(f"Loaded state with {len(state['domains'])} known domains, "
          f"{len(state['seen_signatures'])} remembered signatures.")

    # Simulate a couple of research cycles
    record_result(state, "gap_dynamics", "haiku", "EMPIRICALLY SUPPORTED")
    remember_signature(state, "Sexy Prime Density Bound", "gap 6 dominates all other gaps up to N")
    record_result(state, "modular_bias_mod10", "haiku", "EMPIRICALLY FALSIFIED")

    dup = is_duplicate(state, "Sexy Prime Density Bound II", "gap 6 dominates every other gap up to N")
    print(f"Near-duplicate detection working: {dup}")
    assert dup, "Expected the near-identical conjecture to be flagged as duplicate"

    next_domain = select_next_domain(state, ["gap_dynamics", "modular_bias_mod10", "zeta_harmonics"])
    print(f"Suggested next domain (should favor unexplored 'zeta_harmonics'): {next_domain}")

    print(f"Summary: {summarize(state)}")
    save_state(state)
    print(f"Saved to {KB_FILE}")
    print("Knowledge Base self-test passed!")
