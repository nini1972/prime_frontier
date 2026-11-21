#!/usr/bin/env python3
"""
models/llm_client.py - Unified OpenRouter Client for Mathematical Reasoning Models

Supports frontier mathematical & reasoning lineages via OpenRouter:
- Claude Sonnet 4.5 (sonnet) & Haiku 4.5 (haiku)
- DeepSeek R1 (r1 / r1_full) & DeepSeek V4.1 Flash (deepseek)
- Z.AI GLM 5.2 (glm)
- Qwen 2.5 72B / Qwen 3.8 27B (qwen_math / qwen3)
- Xiaomi MiMo v2.6 Flash (mimo)

Zero external dependencies (uses standard urllib.request).
"""

import os
import json
import time
import urllib.request
import urllib.error
from typing import Dict

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

DEFAULT_MODELS = {
    "r1": "deepseek/deepseek-r1-0528",
    "r1_full": "deepseek/deepseek-r1-0528",
    "deepseek": "deepseek/deepseek-v4.1-flash",
    "glm": "z-ai/glm-5.3-flash",
    "glm_5_2": "z-ai/glm-5.2",
    "qwen_math": "qwen/qwen-2.5-72b-instruct",
    "qwen3": "qwen/qwen3.8-27b",
    "mimo": "xiaomi/mimo-v2.6-flash",
    "sonnet": "anthropic/claude-sonnet-4.5",
    "haiku": "anthropic/claude-haiku-4.5",
}

# Models that emit internal chain-of-thought (`reasoning`) tokens sharing max_tokens.
# Using max_tokens=16384 + reasoning={"effort": "medium"} ensures ample room for both
# mathematical derivation and the complete Python verification script.
REASONING_MODEL_PREFIXES = (
    "deepseek/",
    "z-ai/glm",
    "qwen/qwen3",
    "xiaomi/mimo",
)

# OpenRouter provider slugs that forwarded MiniMax requests to MiniMax's non-ZDR moderation endpoints
MINIMAX_BLOCKED_OPENROUTER_PROVIDERS = (
    "minimax",
    "novita",
    "streamlake",
    "atlas-cloud",
)


def load_api_key() -> str:
    """Load OPENROUTER_API_KEY from environment or configuration files."""
    if os.environ.get("OPENROUTER_API_KEY"):
        return os.environ["OPENROUTER_API_KEY"].strip()

    candidates = [
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config", ".env"),
        os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "evolution_sandbox",
            "config",
            ".env",
        ),
    ]

    for path in candidates:
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.startswith("OPENROUTER_API_KEY="):
                            key = line.split("=", 1)[1].strip().strip("\"'")
                            if key:
                                return key
            except Exception:
                pass

    return ""


def query_math_model(
    prompt: str,
    system_prompt: str = "You are an elite research mathematician specializing in analytic and combinatorial number theory.",
    model_alias: str = "r1",
    temperature: float = 0.2,
    max_tokens: int = 8192,
    timeout: int = 180,
) -> Dict:
    """
    Query an OpenRouter reasoning or mathematical model with automatic CoT budget protection.
    """
    api_key = load_api_key()
    if not api_key:
        return {
            "content": "",
            "reasoning": None,
            "model": model_alias,
            "latency_s": 0.0,
            "success": False,
            "error": "OPENROUTER_API_KEY not found in environment or config files.",
        }

    model_id = DEFAULT_MODELS.get(model_alias, model_alias)
    is_reasoning = any(model_id.startswith(pref) for pref in REASONING_MODEL_PREFIXES)
    if model_id.startswith("z-ai/glm-5.2") or model_id.startswith("deepseek/deepseek-r1"):
        max_tokens = max(max_tokens, 32768)
        timeout = max(timeout, 300)
    elif is_reasoning:
        max_tokens = max(max_tokens, 16384)
        timeout = max(timeout, 240)
    else:
        max_tokens = max(max_tokens, 8192)

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": "https://github.com/nini1972/prime_frontier",
        "X-Title": "Prime Frontier Mathematics Engine",
    }

    t0 = time.time()
    max_retries = 3
    last_err = None
    effort_level = "medium"

    for attempt in range(max_retries):
        sys_msg = system_prompt
        if attempt > 0 and is_reasoning and effort_level == "low":
            sys_msg = (
                system_prompt
                + "\nCRITICAL: Keep internal thinking concise (<1,200 words) so your final output "
                "contains all four required sections including the complete ```python verification block."
            )

        payload = {
            "model": model_id,
            "messages": [
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if is_reasoning:
            payload["reasoning"] = {"effort": effort_level}
        if "minimax" in model_id.lower():
            payload["provider"] = {"ignore": list(MINIMAX_BLOCKED_OPENROUTER_PROVIDERS)}

        req = urllib.request.Request(
            OPENROUTER_URL,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                res_data = json.loads(response.read().decode("utf-8"))
                elapsed = time.time() - t0

                choice = res_data.get("choices", [{}])[0]
                finish_reason = choice.get("finish_reason") or ""
                msg = choice.get("message", {})
                content = (msg.get("content") or "").strip()
                reasoning = (msg.get("reasoning") or "").strip()

                # DeepSeek R1 may emit inline <think>...</think> tags
                if not reasoning and "<think>" in content and "</think>" in content:
                    parts = content.split("</think>", 1)
                    reasoning = parts[0].replace("<think>", "").strip()
                    content = parts[1].strip()

                # Guard against reasoning token exhaustion leaving `content` empty:
                # retry immediately with lower reasoning effort rather than dumping raw CoT into content!
                if not content and reasoning:
                    if ("CONJECTURE_NAME" in reasoning and "```" in reasoning) or attempt == max_retries - 1:
                        content = reasoning
                    else:
                        effort_level = "low"
                        last_err = f"Reasoning exhausted max_tokens (finish_reason={finish_reason}) with empty content"
                        continue

                return {
                    "content": content,
                    "reasoning": reasoning,
                    "model": model_id,
                    "latency_s": round(elapsed, 2),
                    "success": True,
                    "error": None,
                }
        except urllib.error.HTTPError as e:
            err_msg = e.read().decode("utf-8", errors="ignore")
            last_err = f"HTTP {e.code}: {err_msg}"
            if e.code in (429, 500, 502, 503, 504) and attempt < max_retries - 1:
                time.sleep(4 * (attempt + 1))
                continue
            return {
                "content": "",
                "reasoning": None,
                "model": model_id,
                "latency_s": round(time.time() - t0, 2),
                "success": False,
                "error": last_err,
            }
        except Exception as e:
            last_err = str(e)
            if attempt < max_retries - 1:
                time.sleep(3 * (attempt + 1))
                continue
            return {
                "content": "",
                "reasoning": None,
                "model": model_id,
                "latency_s": round(time.time() - t0, 2),
                "success": False,
                "error": last_err,
            }
    return {
        "content": "",
        "reasoning": None,
        "model": model_id,
        "latency_s": round(time.time() - t0, 2),
        "success": False,
        "error": last_err or "Exhausted retries",
    }


if __name__ == "__main__":
    print("Testing LLM Client connectivity...")
    key = load_api_key()
    if key:
        print(f"Loaded API Key: {key[:8]}...{key[-4:]}")
        print("Sending test query to Haiku (fast check)...")
        res = query_math_model(
            "What is the asymptotic density of primes by the Prime Number Theorem? Answer in 1 sentence.",
            model_alias="haiku",
        )
        print(f"Success: {res['success']}, Latency: {res['latency_s']}s")
        print("Response:", res["content"])
    else:
        print("No API key available for testing.")
