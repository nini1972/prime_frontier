#!/usr/bin/env python3
"""
models/agent_world.py - Prime Frontier Agent World Orchestrator

Turns math_researcher.py's single-shot "one conjecture per run" script into a
persistent, self-directed, multi-domain research world:

  Explorer    -> picks the least-explored research domain
                 (knowledge_base.select_next_domain)
  Conjecturer -> queries a reasoning model for a novel, testable conjecture
                 grounded in that domain's freshly harvested data
  Skeptic     -> executes the generated verification script; on execution
                 errors, asks the model to repair its own code (bounded
                 retries) instead of just logging a failure
  Archivist   -> saves the verdict to outputs/conjectures.json and updates
                 outputs/knowledge_base.json so future cycles avoid repeats
                 and steer toward under-explored territory

Usage:
    python -m models.agent_world --cycles 10 --sleep 5 --models haiku,sonnet
    python -m models.agent_world --cycles 1 --domain modular_bias_mod10
    python -m models.agent_world --cycles 20 --sleep 0 --models r1,qwen_math,sonnet,haiku
"""

import argparse
import itertools
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import knowledge_base as kb
from core.domains import list_domain_ids
from models.math_researcher import run_hypothesis_cycle
from models.llm_client import load_api_key


def run_world(cycles: int, sleep_s: float, model_aliases, domain_id=None, max_repairs: int = 2, temperature: float = 0.7) -> None:
    if not load_api_key():
        print("=" * 72)
        print("PRIME FRONTIER AGENT WORLD -- DRY RUN (no OPENROUTER_API_KEY set)")
        print("Domain rotation and knowledge-base bookkeeping will still run each")
        print("cycle, but no real conjectures can be generated without an API key.")
        print("Set OPENROUTER_API_KEY (env var or config/.env) to go live.")
        print("=" * 72)

    model_cycle = itertools.cycle(model_aliases)
    state = kb.load_state()
    domains = list_domain_ids()
    prior_attempts = sum(d.get("attempts", 0) for d in state["domains"].values())
    print(f"[world] Known domains ({len(domains)}): {', '.join(domains)}")
    print(f"[world] Prior attempts recorded in knowledge base: {prior_attempts}")
    print(f"[world] Speculation Temperature set to: {temperature}")

    completed = 0
    try:
        for i in range(1, cycles + 1):
            model_alias = next(model_cycle)
            chosen_domain = domain_id or kb.select_next_domain(state, domains)
            print(f"\n=== Cycle {i}/{cycles} | domain={chosen_domain} | model={model_alias} | T={temperature} ===")
            try:
                rec = run_hypothesis_cycle(
                    model_alias=model_alias,
                    domain_id=chosen_domain,
                    max_repairs=max_repairs,
                    temperature=temperature,
                )
            except Exception as e:
                print(f"[world] Cycle {i} crashed unexpectedly: {e}")
                rec = None

            state = kb.load_state()  # reload: the cycle persisted its own updates
            completed += 1
            if rec and rec.get("success", True) is not False:
                print(f"[world] -> {rec.get('name')}: {rec.get('verdict')}")
            elif rec:
                print(f"[world] -> cycle did not produce a conjecture: {rec.get('error')}")

            if i < cycles and sleep_s > 0:
                time.sleep(sleep_s)
    except KeyboardInterrupt:
        print(f"\n[world] Interrupted by user after {completed}/{cycles} cycles. State is saved incrementally, nothing lost.")

    print("\n[world] Run complete. Knowledge base summary:")
    summary = kb.summarize(state)
    for d, s in summary["domains"].items():
        print(f"  {d:24s} attempts={s['attempts']:3d} supported={s['supported']:2d} "
              f"falsified={s['falsified']:2d} errors={s['errors']:2d} timeout={s['timeout']:2d} "
              f"duplicates={s['duplicates']:2d}")
    print(f"  {'TOTAL':24s} {summary['totals']}")
    print(f"[world] Confirmed laws so far: {summary['confirmed_laws_count']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prime Frontier autonomous agent world orchestrator")
    parser.add_argument("--cycles", type=int, default=3, help="Number of research cycles to run")
    parser.add_argument("--sleep", type=float, default=5.0, help="Seconds to sleep between cycles")
    parser.add_argument("--models", type=str, default="haiku",
                        help="Comma-separated model aliases to rotate through (haiku,sonnet,r1,qwen_math,r1_full)")
    parser.add_argument("--domain", type=str, default=None,
                        help="Pin every cycle to one domain id instead of letting the explorer pick")
    parser.add_argument("--max-repairs", type=int, default=2, help="Max self-repair attempts per broken script")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature for radical/speculative conjectures (default: 0.7)")
    args = parser.parse_args()

    run_world(
        cycles=args.cycles,
        sleep_s=args.sleep,
        model_aliases=[m.strip() for m in args.models.split(",") if m.strip()],
        domain_id=args.domain,
        max_repairs=args.max_repairs,
        temperature=args.temperature,
    )


if __name__ == "__main__":
    main()
