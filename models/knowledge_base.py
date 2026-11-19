#!/usr/bin/env python3
"""
models/knowledge_base.py - Persistent Memory for the Prime Frontier Agent World

Gives the autonomous agents long-term memory across runs, stored in
outputs/knowledge_base.json:
  - per-domain attempt/verdict tallies, used to steer exploration toward
    under-covered territory (`select_next_domain`)
  - domain-scoped, boilerplate-stripped token fingerprint index used to detect
    near-duplicate conjectures (`find_duplicate_match` / `is_duplicate`) without
    false-positive collisions on generic domain nouns
  - a "Hall of Fame" of empirically confirmed laws (`promote_if_confirmed`)
  - per-model reliability stats (attempts/success/errors)
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
    "seen_signatures": [], # structured fingerprints of past conjecture name+formulation
}

_COMMON_STOPWORDS = {
    "the", "a", "an", "of", "and", "or", "to", "is", "are", "for", "that",
    "this", "with", "let", "be", "we", "conjecture", "prime", "primes",
    "empirically", "following", "define", "over", "all", "up", "first",
    "from", "into", "under", "across", "between", "where", "when", "such",
}

# Domain-boilerplate words stripped ONLY when computing title similarity so two
# distinct theorems in the same domain don't collide solely on generic nouns.
_TITLE_BOILERPLATE = _COMMON_STOPWORDS | {
    "law", "laws", "bound", "bounds", "scaling", "asymptotic", "empirical",
    "refined", "generalized", "adaptive", "principle", "theorem", "hypothesis",
    "zeta", "riemann", "zero", "zeros", "spacing", "spacings",
    "goldbach", "partition", "partitions", "even",
    "modular", "modulo", "mod", "residue", "residues", "transition", "transitions",
    "gap", "gaps", "sieve", "base", "power", "powers", "composite", "composites",
    "harmonic", "harmonics", "resonance", "chebyshev", "race", "races",
    "mobius", "mertens", "liouville", "walk", "walks", "multiplicative",
    "smooth", "friable", "constellation", "constellations", "tuple", "tuples",
}


def load_state() -> Dict:
    """Load persistent agent-world state, seeding sane defaults for any missing keys."""
    state = json.loads(json.dumps(_DEFAULT_STATE))
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


def _fingerprint(text: str, title_mode: bool = False) -> str:
    """Normalize a conjecture name or formulation into a sorted token-set fingerprint."""
    tokens = re.findall(r"[a-z0-9_]+", (text or "").lower())
    stops = _TITLE_BOILERPLATE if title_mode else _COMMON_STOPWORDS
    filtered = sorted(set(t for t in tokens if t not in stops and len(t) > 2))
    # Fallback if a short title consisted purely of boilerplate words
    if not filtered and title_mode:
        filtered = sorted(set(t for t in tokens if t not in _COMMON_STOPWORDS and len(t) > 2))
    return "|".join(filtered)


def _token_set(text: str, title_mode: bool = False) -> set:
    fp = _fingerprint(text, title_mode=title_mode)
    s = set(fp.split("|")) if fp else set()
    s.discard("")
    return s


def _parse_stored_signature(sig: str) -> Dict:
    """
    Parse both modern 4-part ('domain::title_fp::body_fp::raw_title'),
    2-part ('title_fp::body_fp'), and legacy 1-part ('flat_fp') signatures.
    """
    parts = sig.split("::")
    if len(parts) >= 4:
        dom, t_str, b_str, raw_title = parts[0], parts[1], parts[2], "::".join(parts[3:])
        ex_title = (set(t_str.split("|")) - {""}) - _TITLE_BOILERPLATE
        if not ex_title:
            ex_title = set(t_str.split("|")) - {""}
        ex_body = set(b_str.split("|")) - {""}
        return {
            "domain": dom,
            "title_tokens": ex_title,
            "body_tokens": ex_body,
            "all_tokens": ex_title | ex_body,
            "raw_title": raw_title,
        }
    if len(parts) == 2:
        t_str, b_str = parts
        raw_t = set(t_str.split("|")) - {""}
        ex_title = raw_t - _TITLE_BOILERPLATE
        if not ex_title:
            ex_title = raw_t
        ex_body = set(b_str.split("|")) - {""}
        return {
            "domain": None,
            "title_tokens": ex_title,
            "body_tokens": ex_body,
            "all_tokens": raw_t | ex_body,
            "raw_title": " ".join(sorted(raw_t))[:80],
        }
    ex_all = set(sig.split("|")) - {""}
    return {
        "domain": None,
        "title_tokens": set(),
        "body_tokens": ex_all,
        "all_tokens": ex_all,
        "raw_title": " ".join(sorted(list(ex_all)[:8])),
    }


def find_duplicate_match(
    state: Dict,
    name: str,
    formulation: str,
    domain_id: Optional[str] = None,
    overlap_threshold: float = 0.56,
) -> Optional[Dict]:
    """
    Domain-aware, boilerplate-stripped near-duplicate detector:
      1. Informative title token Jaccard >= 0.75 (with >= 2 informative tokens)
         paired with body/all token Jaccard >= 0.45, OR
      2. Overall mathematical formulation token Jaccard >= overlap_threshold (default 0.56).
    Returns metadata about the matched signature, or None if novel.
    """
    cand_title = _token_set(name, title_mode=True)
    cand_body = _token_set(formulation, title_mode=False)
    cand_all = cand_title | cand_body
    if not cand_all:
        return None

    for sig in state.get("seen_signatures", []):
        parsed = _parse_stored_signature(sig)
        # If both candidate and stored signature have explicit domains and they differ, skip
        if domain_id and parsed["domain"] and parsed["domain"] != "any" and parsed["domain"] != domain_id:
            continue

        ex_title = parsed["title_tokens"]
        ex_all = parsed["all_tokens"]
        if not ex_all:
            continue

        body_ov = len(cand_all & ex_all) / max(1, len(cand_all | ex_all))
        title_ov = 0.0
        if cand_title and ex_title:
            title_ov = len(cand_title & ex_title) / max(1, len(cand_title | ex_title))
            min_toks = min(len(cand_title), len(ex_title))
            req_body = 0.45 if min_toks >= 2 else 0.52
            if title_ov >= 0.75 and body_ov >= req_body:
                return {
                    "matched_title": parsed["raw_title"] or "Prior domain conjecture",
                    "title_overlap": round(title_ov, 3),
                    "body_overlap": round(body_ov, 3),
                }

        if body_ov >= overlap_threshold:
            return {
                "matched_title": parsed["raw_title"] or "Prior domain conjecture",
                "title_overlap": round(title_ov, 3),
                "body_overlap": round(body_ov, 3),
            }

    return None


def is_duplicate(
    state: Dict,
    name: str,
    formulation: str,
    overlap_threshold: float = 0.56,
    domain_id: Optional[str] = None,
) -> bool:
    """Boolean wrapper around `find_duplicate_match`."""
    return (
        find_duplicate_match(
            state=state,
            name=name,
            formulation=formulation,
            domain_id=domain_id,
            overlap_threshold=overlap_threshold,
        )
        is not None
    )


def remember_signature(
    state: Dict,
    name: str,
    formulation: str,
    domain_id: Optional[str] = None,
) -> None:
    clean_title = re.sub(r"\s+", " ", (name or "").replace("::", " ")).strip()[:120]
    dom = domain_id or "any"
    sig = f"{dom}::{_fingerprint(name, title_mode=True)}::{_fingerprint(formulation, title_mode=False)}::{clean_title}"
    state.setdefault("seen_signatures", []).append(sig)
    state["seen_signatures"] = state["seen_signatures"][-600:]


def build_domain_memory_prompt(
    domain_id: str,
    conjectures_file: Optional[str] = None,
    max_confirmed: int = 6,
    max_falsified: int = 6,
) -> str:
    """
    Build a dynamic 'Prior Domain Memory' prompt block from past runs in `domain_id`
    so the reasoning model sees already-confirmed laws and learns from recent falsifications
    (using tail-preserved empirical counter-evidence) instead of operating blind.
    """
    if conjectures_file is None:
        conjectures_file = os.path.join(BASE_DIR, "outputs", "conjectures.json")
    if not os.path.exists(conjectures_file):
        return ""
    try:
        with open(conjectures_file, "r", encoding="utf-8") as f:
            all_conjs = json.load(f)
    except Exception:
        return ""

    domain_conjs = [c for c in all_conjs if c.get("domain") == domain_id]
    if not domain_conjs:
        return ""

    confirmed = []
    seen_conf_names = set()
    for c in domain_conjs:
        v_up = (c.get("verdict") or "").upper()
        if "SUPPORTED" in v_up and "TRIVIAL" not in v_up:
            nm = (c.get("name") or "").strip().lower()
            if nm not in seen_conf_names:
                seen_conf_names.add(nm)
                confirmed.append(c)
                if len(confirmed) >= max_confirmed:
                    break

    falsified = []
    seen_fals_names = set()
    for c in domain_conjs:
        v_up = (c.get("verdict") or "").upper()
        if "FALSIFIED" in v_up or "TRIVIAL" in v_up:
            nm = (c.get("name") or "").strip().lower()
            if nm not in seen_fals_names:
                seen_fals_names.add(nm)
                falsified.append(c)
                if len(falsified) >= max_falsified:
                    break

    seen_names = []
    for c in domain_conjs:
        n = (c.get("name") or "").strip()
        if n and n not in seen_names:
            seen_names.append(n)

    lines = ["\n--- PRIOR DOMAIN MEMORY (ACCUMULATED LABORATORY KNOWLEDGE) ---"]
    if confirmed:
        lines.append("Already Confirmed Laws in this Domain (DO NOT re-propose these; build deeper orthogonal invariants):")
        for c in confirmed:
            tgt = (c.get("target") or "").replace("\n", " ").strip()[:160]
            lines.append(f"  * [CONFIRMED] {c.get('name')}: {tgt}")

    if falsified:
        lines.append("\nRecently Falsified / Trivial Hypotheses in this Domain (Learn from their counter-evidence; avoid repeating):")
        for c in falsified:
            tgt = (c.get("target") or "").replace("\n", " ").strip()[:140]
            out_lines = [
                ln.strip()
                for ln in (c.get("output") or "").splitlines()
                if ln.strip() and "VERDICT:" not in ln and ln.strip() != "...[middle output truncated]..."
            ]
            empirical_hint = " | ".join(out_lines[-3:])[:200] if out_lines else "Threshold violated"
            tag = "TRIVIAL_NULL" if "TRIVIAL" in (c.get("verdict") or "").upper() else "FALSIFIED"
            lines.append(f"  * [{tag}] {c.get('name')} (Target: {tgt}) -> Output tail: {empirical_hint}")

    if seen_names:
        lines.append(f"\nPreviously explored titles in this domain (formulate a distinct mathematical observable & title): {', '.join(seen_names[:14])}")
    lines.append("--- END PRIOR DOMAIN MEMORY ---\n")
    return "\n".join(lines)


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
    elif "SUPPORTED" in verdict_upper and "TRIVIAL" not in verdict_upper:
        d["supported"] += 1
        m["success"] += 1
    elif "FALSIFIED" in verdict_upper or "TRIVIAL" in verdict_upper:
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
    v_up = (conjecture_record.get("verdict") or "").upper()
    if "SUPPORTED" in v_up and "TRIVIAL" not in v_up:
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
    least so far (coverage-driven exploration), while avoiding domains in `exclude`
    (such as domains already visited during the current nightly run).
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
    print("Testing Knowledge Base (in-memory non-mutating self-test)...")
    test_state = json.loads(json.dumps(_DEFAULT_STATE))

    record_result(test_state, "gap_dynamics", "haiku", "EMPIRICALLY SUPPORTED")
    remember_signature(
        test_state,
        "Sexy Prime Density Dominance",
        "gap 6 dominates every other twin cousin gap frequency up to N",
        domain_id="gap_dynamics",
    )
    record_result(test_state, "modular_bias_mod10", "haiku", "EMPIRICALLY FALSIFIED")

    # Near-identical conjecture in same domain MUST be caught
    dup = is_duplicate(
        test_state,
        "Sexy Prime Density Dominance II",
        "gap 6 dominates every other twin cousin gap frequency up to N",
        domain_id="gap_dynamics",
    )
    assert dup, "Expected near-identical conjecture in same domain to be flagged as duplicate"

    # Distinct mathematical concept sharing only domain boilerplate words MUST NOT be falsely flagged
    non_dup = is_duplicate(
        test_state,
        "Prime Gap Cramér Record Spacing Variance Law",
        "maximal gap record ratios follow Gumbel extreme value distribution with variance pi^2/6",
        domain_id="gap_dynamics",
    )
    assert not non_dup, "Expected orthogonal conjecture sharing only boilerplate nouns NOT to be flagged"

    next_domain = select_next_domain(test_state, ["gap_dynamics", "modular_bias_mod10", "chebyshev_race"])
    print(f"Suggested next domain (favors unexplored 'chebyshev_race'): {next_domain}")
    print("Knowledge Base self-test passed without polluting outputs/knowledge_base.json!")
