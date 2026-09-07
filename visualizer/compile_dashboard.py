#!/usr/bin/env python3
"""
visualizer/compile_dashboard.py - Automated Compiler for Prime Frontier Laboratory Dashboard

Compiles empirical number theory data, Riemann Zeta harmonic wave synthesis,
modular memory transition matrices, geometric spirals, and live AI mathematical
conjectures into a self-contained, interactive dark-mode dashboard.html.
"""

import sys
import os
import json
import math
from datetime import datetime

# Ensure project root is in path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from core.sieve import generate_primes, compute_prime_gaps, compute_gap_statistics, find_maximal_gap_records
from core.modular_orbits import compute_modular_transitions, compute_residue_autocorrelation
from core.zeta_harmonics import get_zeta_zeros
from core.domains import DOMAINS
from models import knowledge_base as kb

def harvest_dashboard_data() -> dict:
    """Harvest all mathematical metrics, plots, and conjecture records."""
    print("[1/4] Computing prime sequence & gap statistics (N = 1,000,000)...")
    primes = generate_primes(1_000_000)
    gaps = compute_prime_gaps(primes)
    stats = compute_gap_statistics(primes, gaps)
    records = find_maximal_gap_records(primes, gaps)
    
    print("[2/4] Computing modular transition matrix & autocorrelation (Mod 10)...")
    mod10 = compute_modular_transitions(primes, mod=10)
    autocorr = compute_residue_autocorrelation(primes, mod=10, max_lag=8)
    
    print("[3/4] Harvesting Zeta frequencies & conjecture logs...")
    zeta_zeros = get_zeta_zeros(10)
    
    conjectures = []
    conjectures_file = os.path.join(BASE_DIR, "outputs", "conjectures.json")
    if os.path.exists(conjectures_file):
        try:
            with open(conjectures_file, "r", encoding="utf-8") as f:
                conjectures = json.load(f)
        except Exception:
            conjectures = []
            
    # Check generated image artifacts
    images = {
        "zeta_staircase": "outputs/zeta_harmonics_staircase.png" if os.path.exists(os.path.join(BASE_DIR, "outputs", "zeta_harmonics_staircase.png")) else None,
        "modular_bias": "outputs/modular_bias_mod10.png" if os.path.exists(os.path.join(BASE_DIR, "outputs", "modular_bias_mod10.png")) else None,
        "ulam_spiral": "outputs/ulam_spiral.png" if os.path.exists(os.path.join(BASE_DIR, "outputs", "ulam_spiral.png")) else None,
        "sacks_spiral": "outputs/sacks_spiral.png" if os.path.exists(os.path.join(BASE_DIR, "outputs", "sacks_spiral.png")) else None,
    }

    # Merge the full domain registry with knowledge-base coverage stats, so every
    # research domain shows up in the dashboard even before it has been attempted.
    agent_world = kb.summarize(kb.load_state())
    domain_coverage = []
    for did, domain in DOMAINS.items():
        s = agent_world["domains"].get(did, {})
        domain_coverage.append({
            "id": did,
            "name": domain.name,
            "description": domain.description,
            "attempts": s.get("attempts", 0),
            "supported": s.get("supported", 0),
            "falsified": s.get("falsified", 0),
            "errors": s.get("errors", 0),
            "timeout": s.get("timeout", 0),
            "duplicates": s.get("duplicates", 0),
            "last_run": s.get("last_run"),
        })
    agent_world["domain_coverage"] = domain_coverage

    return {
        "generated_at": datetime.now().strftime("%B %d, %Y, %H:%M UTC"),
        "stats": stats,
        "gap_records": records,
        "mod10": mod10,
        "autocorr": autocorr,
        "zeta_zeros": zeta_zeros,
        "conjectures": conjectures,
        "images": images,
        "agent_world": agent_world,
    }

def generate_html(data: dict) -> str:
    """Generate modern, responsive, dark-mode Prime Frontier dashboard."""
    json_blob = json.dumps(data)
    
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Prime Frontier — Mathematical Reverse-Engineering & Prediction Lab</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;600;700&family=Outfit:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
    <!-- KaTeX for formula rendering -->
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.css">
    <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.js"></script>
    <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/contrib/auto-render.min.js"></script>
    <style>
        :root {{
            --bg-base: #060913;
            --bg-card: rgba(13, 21, 38, 0.72);
            --border-glass: rgba(255, 255, 255, 0.08);
            --border-highlight: rgba(0, 242, 254, 0.35);
            --accent-cyan: #00f2fe;
            --accent-purple: #9d4edd;
            --accent-emerald: #10b981;
            --accent-amber: #f59e0b;
            --accent-rose: #f43f5e;
            --text-primary: #f8fafc;
            --text-secondary: #94a3b8;
            --text-muted: #64748b;
            --font-main: 'Outfit', sans-serif;
            --font-mono: 'JetBrains Mono', monospace;
            --shadow-glass: 0 8px 32px 0 rgba(0, 0, 0, 0.45);
            --glow-cyan: 0 0 24px rgba(0, 242, 254, 0.18);
        }}

        * {{ margin: 0; padding: 0; box-sizing: border-box; }}

        body {{
            background-color: var(--bg-base);
            background-image: 
                radial-gradient(at 0% 0%, rgba(0, 242, 254, 0.08) 0px, transparent 50%),
                radial-gradient(at 100% 10%, rgba(157, 78, 221, 0.08) 0px, transparent 50%),
                radial-gradient(at 50% 100%, rgba(16, 185, 129, 0.05) 0px, transparent 60%);
            background-attachment: fixed;
            color: var(--text-primary);
            font-family: var(--font-main);
            line-height: 1.6;
            min-height: 100vh;
            padding-bottom: 5rem;
        }}

        .container {{
            max-width: 1520px;
            margin: 0 auto;
            padding: 2rem 1.5rem;
        }}

        header {{
            margin-bottom: 2.5rem;
        }}

        .top-status-bar {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            flex-wrap: wrap;
            gap: 1rem;
            margin-bottom: 1.5rem;
            padding-bottom: 1rem;
            border-bottom: 1px solid var(--border-glass);
        }}

        .status-pill {{
            font-family: var(--font-mono);
            font-size: 0.75rem;
            padding: 0.35rem 0.85rem;
            border-radius: 9999px;
            background: rgba(255, 255, 255, 0.04);
            border: 1px solid var(--border-glass);
            color: var(--text-secondary);
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
        }}

        .pulse-dot {{
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: var(--accent-cyan);
            box-shadow: 0 0 10px var(--accent-cyan);
            animation: pulse-glow 2s infinite;
        }}

        @keyframes pulse-glow {{
            0%, 100% {{ transform: scale(1); opacity: 1; }}
            50% {{ transform: scale(1.4); opacity: 0.6; }}
        }}

        .header-title-wrap h1 {{
            font-size: 3.2rem;
            font-weight: 800;
            letter-spacing: -1.5px;
            background: linear-gradient(135deg, #ffffff 10%, var(--accent-cyan) 60%, var(--accent-purple) 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
        }}

        .header-subtitle {{
            color: var(--text-secondary);
            font-size: 1.15rem;
            font-weight: 300;
            max-width: 950px;
        }}

        /* Hero Telemetry */
        .telemetry-strip {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 1.25rem;
            margin-bottom: 2.5rem;
        }}

        .kpi-card {{
            background: var(--bg-card);
            border: 1px solid var(--border-glass);
            border-radius: 18px;
            padding: 1.5rem;
            backdrop-filter: blur(16px);
            box-shadow: var(--shadow-glass);
            transition: all 0.3s ease;
            position: relative;
            overflow: hidden;
        }}

        .kpi-card:hover {{
            border-color: var(--border-highlight);
            transform: translateY(-3px);
            box-shadow: var(--shadow-glass), var(--glow-cyan);
        }}

        .kpi-card::before {{
            content: '';
            position: absolute;
            top: 0; left: 0; width: 100%; height: 3px;
            background: linear-gradient(90deg, var(--accent-cyan), var(--accent-purple));
        }}

        .kpi-label {{
            font-size: 0.8rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 1.2px;
            color: var(--text-muted);
            margin-bottom: 0.5rem;
            display: flex;
            justify-content: space-between;
        }}

        .kpi-value {{
            font-family: var(--font-mono);
            font-size: 2.1rem;
            font-weight: 700;
            color: var(--text-primary);
            line-height: 1.1;
            margin-bottom: 0.25rem;
        }}

        .kpi-subtext {{
            font-size: 0.8rem;
            color: var(--text-secondary);
        }}

        /* Tab Navigation */
        .tab-nav-container {{
            position: sticky;
            top: 1rem;
            z-index: 100;
            margin-bottom: 2.5rem;
        }}

        .tab-menu {{
            display: flex;
            gap: 0.75rem;
            background: rgba(10, 16, 30, 0.85);
            padding: 0.6rem;
            border-radius: 9999px;
            border: 1px solid var(--border-glass);
            backdrop-filter: blur(20px);
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
            overflow-x: auto;
            scrollbar-width: none;
            -ms-overflow-style: none;
        }}

        .tab-menu::-webkit-scrollbar {{
            display: none;
        }}

        .tab-btn {{
            background: transparent;
            border: none;
            color: var(--text-secondary);
            font-family: var(--font-main);
            font-size: 0.95rem;
            font-weight: 600;
            padding: 0.65rem 1.6rem;
            border-radius: 9999px;
            cursor: pointer;
            transition: all 0.25s ease;
            white-space: nowrap;
        }}

        .tab-btn:hover {{
            color: var(--text-primary);
            background: rgba(255, 255, 255, 0.05);
        }}

        .tab-btn.active {{
            background: linear-gradient(135deg, var(--accent-cyan), #00b4d8);
            color: #030712;
            font-weight: 700;
            box-shadow: 0 0 20px rgba(0, 242, 254, 0.4);
        }}

        .tab-content {{
            display: none;
            animation: fadeInTab 0.35s ease forwards;
        }}

        .tab-content.active {{
            display: block;
        }}

        @keyframes fadeInTab {{
            from {{ opacity: 0; transform: translateY(12px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}

        /* Grid & Cards */
        .grid-2col {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 2rem;
        }}

        @media (max-width: 1024px) {{
            .grid-2col {{ grid-template-columns: 1fr; }}
        }}

        .glass-panel {{
            background: var(--bg-card);
            border: 1px solid var(--border-glass);
            border-radius: 20px;
            padding: 2rem;
            box-shadow: var(--shadow-glass);
            backdrop-filter: blur(16px);
        }}

        .panel-title {{
            font-size: 1.4rem;
            font-weight: 800;
            margin-bottom: 0.75rem;
            display: flex;
            align-items: center;
            gap: 0.6rem;
        }}

        .panel-desc {{
            color: var(--text-secondary);
            font-size: 0.95rem;
            margin-bottom: 1.5rem;
        }}

        .img-display-wrap {{
            background: #03050a;
            border: 1px solid var(--border-glass);
            border-radius: 14px;
            overflow: hidden;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 1rem;
            min-height: 380px;
        }}

        .img-display-wrap img {{
            max-width: 100%;
            height: auto;
            border-radius: 8px;
            box-shadow: 0 5px 25px rgba(0,0,0,0.5);
        }}

        /* Tables */
        .data-table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.85rem;
            margin-top: 1rem;
        }}

        .data-table th, .data-table td {{
            padding: 0.65rem 0.85rem;
            text-align: left;
            border-bottom: 1px solid var(--border-glass);
        }}

        .data-table th {{
            color: var(--text-muted);
            text-transform: uppercase;
            font-size: 0.75rem;
            letter-spacing: 0.5px;
        }}

        .data-table tr:hover td {{
            background: rgba(255, 255, 255, 0.02);
        }}

        /* Conjectures Feed */
        .conjecture-card {{
            background: var(--bg-card);
            border: 1px solid var(--border-glass);
            border-radius: 18px;
            padding: 1.75rem;
            margin-bottom: 1.5rem;
            box-shadow: var(--shadow-glass);
        }}

        .conj-header {{
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            margin-bottom: 0.75rem;
            flex-wrap: wrap;
            gap: 0.5rem;
        }}

        .conj-title {{
            font-size: 1.2rem;
            font-weight: 700;
            color: var(--text-primary);
        }}

        .verdict-badge {{
            font-family: var(--font-mono);
            font-size: 0.75rem;
            font-weight: 700;
            padding: 0.25rem 0.65rem;
            border-radius: 6px;
            text-transform: uppercase;
        }}

        .verdict-supported {{
            background: rgba(16, 185, 129, 0.15);
            color: var(--accent-emerald);
            border: 1px solid rgba(16, 185, 129, 0.3);
        }}

        .verdict-falsified {{
            background: rgba(244, 63, 94, 0.15);
            color: var(--accent-rose);
            border: 1px solid rgba(244, 63, 94, 0.3);
        }}

        .code-box {{
            background: rgba(0, 0, 0, 0.45);
            border: 1px solid var(--border-glass);
            border-radius: 10px;
            padding: 1rem;
            font-family: var(--font-mono);
            font-size: 0.8rem;
            color: #93c5fd;
            white-space: pre-wrap;
            margin-top: 0.75rem;
            max-height: 250px;
            overflow-y: auto;
        }}

        /* Footer */
        footer {{
            margin-top: 5rem;
            text-align: center;
            border-top: 1px solid var(--border-glass);
            padding-top: 2rem;
            color: var(--text-muted);
            font-size: 0.85rem;
        }}
    </style>
</head>
<body>

<div class="container">
    <header>
        <div class="top-status-bar">
            <div style="display: flex; gap: 0.75rem;">
                <span class="status-pill"><span class="pulse-dot"></span> Prime Frontier Lab: Operational</span>
                <span class="status-pill">⚡ Engine: SymPy • mpmath • NumPy</span>
                <span class="status-pill">🧠 AI: DeepSeek R1 • Qwen 2.5 Math • Claude Sonnet</span>
            </div>
            <div class="status-pill">{data['generated_at']}</div>
        </div>

        <div class="header-title-wrap">
            <h1>Prime Frontier</h1>
            <p class="header-subtitle">
                An advanced empirical laboratory dedicated to reverse-engineering prime number distributions,
                predicting consecutive prime gaps, and synthesizing the discrete prime staircase from Riemann Zeta harmonics.
            </p>
        </div>
    </header>

    <!-- Telemetry -->
    <div class="telemetry-strip">
        <div class="kpi-card">
            <div class="kpi-label">Primes Computed <span>🔢</span></div>
            <div class="kpi-value">{data['stats']['total_primes']:,}</div>
            <div class="kpi-subtext">Max prime: {data['stats']['max_prime']:,}</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Max Gap Record <span>📏</span></div>
            <div class="kpi-value">{data['stats']['max_gap']}</div>
            <div class="kpi-subtext">Mean gap: {data['stats']['mean_gap']}</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Sexy Primes (g=6) <span>✨</span></div>
            <div class="kpi-value">{data['stats']['sexy_primes_count']:,}</div>
            <div class="kpi-subtext">Dominant gap ({data['stats']['sexy_primes_count'] / data['stats']['twin_primes_count']:.1f}x twin primes)</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Zeta Harmonics <span>🌊</span></div>
            <div class="kpi-value">50 Zeros</div>
            <div class="kpi-subtext">First frequency: $\\gamma_1 = 14.1347$</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Conjectures Tested <span>🧠</span></div>
            <div class="kpi-value">{len(data['conjectures'])}</div>
            <div class="kpi-subtext">Formulated by DeepSeek R1 / Qwen Math</div>
        </div>
    </div>

    <!-- Navigation -->
    <div class="tab-nav-container">
        <nav class="tab-menu">
            <button class="tab-btn active" onclick="switchTab('gaps')">📏 Gap Spectrum & Cramér Records</button>
            <button class="tab-btn" onclick="switchTab('zeta')">🌊 Riemann Zeta Wave Synthesizer</button>
            <button class="tab-btn" onclick="switchTab('modular')">🔄 Modular Memory & Chebyshev Bias</button>
            <button class="tab-btn" onclick="switchTab('geometry')">🌀 Geometric Spirals (Ulam & Sacks)</button>
            <button class="tab-btn" onclick="switchTab('ai_lab')">🧠 Autonomous Conjectures Lab ({len(data['conjectures'])})</button>
        </nav>
    </div>

    <!-- TAB 1: GAPS & CRAMER -->
    <div id="tab-gaps" class="tab-content active">
        <div class="grid-2col">
            <div class="glass-panel">
                <div class="panel-title"><span>📏</span> Prime Gap Distribution & Special Pairs</div>
                <div class="panel-desc">
                    The gap between consecutive primes $g_n = p_{{n+1}} - p_n$ reveals striking empirical structure.
                    While primes appear locally stochastic, gap frequencies obey strict divisibility preferences.
                </div>
                <ul style="list-style: none; display: flex; flex-direction: column; gap: 0.75rem; font-size: 0.9rem;">
                    <li style="display: flex; justify-content: space-between; border-bottom: 1px solid var(--border-glass); padding-bottom: 0.4rem;">
                        <span>Twin Primes ($g=2$):</span>
                        <b style="font-family: var(--font-mono); color: var(--accent-cyan);">{data['stats']['twin_primes_count']:,} pairs</b>
                    </li>
                    <li style="display: flex; justify-content: space-between; border-bottom: 1px solid var(--border-glass); padding-bottom: 0.4rem;">
                        <span>Cousin Primes ($g=4$):</span>
                        <b style="font-family: var(--font-mono); color: var(--accent-cyan);">{data['stats']['cousin_primes_count']:,} pairs</b>
                    </li>
                    <li style="display: flex; justify-content: space-between; border-bottom: 1px solid var(--border-glass); padding-bottom: 0.4rem;">
                        <span>Sexy Primes ($g=6$):</span>
                        <b style="font-family: var(--font-mono); color: var(--accent-emerald);">{data['stats']['sexy_primes_count']:,} pairs (Highest Frequency)</b>
                    </li>
                </ul>
                <div style="margin-top: 1.5rem; font-size: 0.85rem; color: var(--text-secondary);">
                    <b>Why is Gap 6 dominant?</b> By the Hardy-Littlewood prime k-tuple conjecture, differences that are multiples of $2 \\times 3 = 6$ avoid modular obstructions modulo 2 and 3 simultaneously, giving them a theoretical density multiplier of 2 relative to twin primes!
                </div>
            </div>

            <div class="glass-panel">
                <div class="panel-title"><span>🏆</span> Maximal Gap Records & Cramér Conjecture</div>
                <div class="panel-desc">
                    Harold Cramér (1936) conjectured that the maximal gap satisfies $g_n = O(\\ln^2 p_n)$.
                    Below are the record-setting gaps discovered in our empirical sweep:
                </div>
                <div style="max-height: 280px; overflow-y: auto;">
                    <table class="data-table">
                        <thead>
                            <tr>
                                <th>Prime $p_n$</th>
                                <th>Gap $g_n$</th>
                                <th>$\\ln^2(p_n)$</th>
                                <th>Cramér Ratio $\\frac{{g_n}}{{\\ln^2 p_n}}$</th>
                            </tr>
                        </thead>
                        <tbody>
                            {"".join(f"<tr><td style='font-family: var(--font-mono); color: #fff;'>{r['prime']:,}</td><td style='font-family: var(--font-mono); color: var(--accent-cyan); font-weight: bold;'>{r['gap']}</td><td style='font-family: var(--font-mono); color: var(--text-muted);'>{r['log2_p']}</td><td style='font-family: var(--font-mono); color: var(--accent-emerald);'>{r['cramer_ratio']}</td></tr>" for r in data['gap_records'])}
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
    </div>

    <!-- TAB 2: ZETA HARMONICS -->
    <div id="tab-zeta" class="tab-content">
        <div class="grid-2col">
            <div class="glass-panel">
                <div class="panel-title"><span>🌊</span> Riemann Zeta Spectral Synthesis</div>
                <div class="panel-desc">
                    Bernhard Riemann's 1859 Explicit Formula establishes that primes are the Fourier-like superposition of the non-trivial zeros $\\rho = \\frac{{1}}{{2}} + i\\gamma$:
                    $$\\psi_0(x) = x - \\sum_{{\\rho}} \\frac{{x^\\rho}}{{\\rho}} - \\ln(2\\pi) - \\frac{{1}}{{2}}\\ln(1 - x^{{-2}})$$
                </div>
                <p style="font-size: 0.9rem; color: var(--text-secondary); margin-bottom: 1rem;">
                    Each zero $\\rho_k = \\frac{{1}}{{2}} + i\\gamma_k$ acts as a harmonic frequency $\\cos(\\gamma_k \\ln x)$ oscillating in logarithmic space. As more zero pairs are summed, continuous smooth waves abruptly sharpen into vertical cliffs at each prime power ($2, 3, 4, 5, 7, 8, 9, 11, 13\\dots$).
                </p>
                <h4 style="font-size: 0.95rem; margin-bottom: 0.5rem; color: #fff;">First 10 Non-Trivial Zero Frequencies $\\gamma_k$:</h4>
                <div style="display: flex; flex-wrap: wrap; gap: 0.5rem; font-family: var(--font-mono); font-size: 0.8rem;">
                    {"".join(f"<span style='background: rgba(0,242,254,0.1); border: 1px solid var(--border-highlight); color: var(--accent-cyan); padding: 0.2rem 0.5rem; border-radius: 6px;'>$\\gamma_{{{i+1}}} = {z:.4f}$</span>" for i, z in enumerate(data['zeta_zeros']))}
                </div>
            </div>

            <div class="glass-panel">
                <div class="panel-title"><span>📊</span> The Prime Staircase Emergence</div>
                <div class="img-display-wrap">
                    <img src="{data['images']['zeta_staircase'] or ''}" alt="Riemann Zeta Prime Staircase">
                </div>
            </div>
        </div>
    </div>

    <!-- TAB 3: MODULAR MEMORY -->
    <div id="tab-modular" class="tab-content">
        <div class="grid-2col">
            <div class="glass-panel">
                <div class="panel-title"><span>🔄</span> Lemke Oliver-Soundararajan Phenomenon</div>
                <div class="panel-desc">
                    In 2016, Soundararajan and Lemke Oliver proved that consecutive primes modulo $q$ exhibit strong short-term memory and repulsion.
                </div>
                <p style="font-size: 0.9rem; color: var(--text-secondary); margin-bottom: 1rem;">
                    For consecutive primes ending in 1, 3, 7, 9 (mod 10), a prime ending in 1 is followed by another 1 only <b>16.9%</b> of the time (strong repulsion), while being followed by 3 or 7 over <b>31%</b> of the time!
                </p>
                <div class="img-display-wrap">
                    <img src="{data['images']['modular_bias'] or ''}" alt="Modular Bias Heatmap">
                </div>
            </div>

            <div class="glass-panel">
                <div class="panel-title"><span>📉</span> Residue Autocorrelation Memory (Mod 10)</div>
                <div class="panel-desc">
                    How many consecutive primes does this short-term memory persist?
                    Below is the empirical autocorrelation deviation from uniform random across lags $k = 1\\dots 8$:
                </div>
                <table class="data-table">
                    <thead>
                        <tr>
                            <th>Lag $k$ ($p_n \\to p_{{n+k}}$)</th>
                            <th>Autocorrelation Excess</th>
                            <th>Status</th>
                        </tr>
                    </thead>
                    <tbody>
                        {"".join(f"<tr><td style='font-family: var(--font-mono); color: #fff;'>Lag {k}</td><td style='font-family: var(--font-mono); color: {'var(--accent-rose)' if v < 0 else 'var(--accent-cyan)'}; font-weight: bold;'>{v:+.5f}</td><td style='color: var(--text-muted);'>{'Strong Repulsion' if abs(v) > 0.05 else ('Moderate Repulsion' if abs(v) > 0.01 else 'Decaying Memory')}</td></tr>" for k, v in data['autocorr'].items())}
                    </tbody>
                </table>
            </div>
        </div>
    </div>

    <!-- TAB 4: GEOMETRY -->
    <div id="tab-geometry" class="tab-content">
        <div class="grid-2col">
            <div class="glass-panel">
                <div class="panel-title"><span>🌀</span> Ulam Square Spiral</div>
                <div class="panel-desc">
                    Discovered by Stanislaw Ulam in 1963. Integers spiraling outward on a square lattice reveal dramatic diagonal tracks corresponding to Euler polynomials $n^2 + n + 41$ and quadratic forms.
                </div>
                <div class="img-display-wrap">
                    <img src="{data['images']['ulam_spiral'] or ''}" alt="Ulam Spiral">
                </div>
            </div>

            <div class="glass-panel">
                <div class="panel-title"><span>🍥</span> Sacks Archimedean Spiral</div>
                <div class="panel-desc">
                    Discovered by Robert Sacks in 1994. Maps numbers to polar coordinates $r = \\sqrt{{n}}, \\theta = 2\\pi\\sqrt{{n}}$, causing quadratic polynomial curves to align into continuous smooth Archimedean trajectories.
                </div>
                <div class="img-display-wrap">
                    <img src="{data['images']['sacks_spiral'] or ''}" alt="Sacks Spiral">
                </div>
            </div>
        </div>
    </div>

    <!-- TAB 5: AI CONJECTURES LAB -->
    <div id="tab-ai_lab" class="tab-content">
        <div style="margin-bottom: 2rem;">
            <h2 style="font-size: 1.8rem; font-weight: 800; margin-bottom: 0.5rem;">Autonomous Mathematical Conjectures Lab</h2>
            <p style="color: var(--text-secondary); max-width: 850px;">
                Hypotheses formulated by <b>DeepSeek R1</b>, <b>Qwen 2.5 Math</b>, and <b>Claude Sonnet</b> based on empirical prime datasets, paired with self-contained Python validation tests.
            </p>
        </div>

        <div class="glass-panel" style="margin-bottom: 2rem;">
            <div class="panel-title"><span>🤖</span> Agent World Control Room</div>
            <div class="panel-desc">
                {len(data['agent_world']['domain_coverage'])} research domains, explored autonomously by
                <code>models/agent_world.py</code>. Coverage-weighted domain selection favors under-explored
                territory; a self-repair loop asks the model to fix its own broken verification code before
                giving up; a duplicate-detection index rejects conjectures too similar to ones already tried.
            </div>
            <div style="display:flex; gap:1.5rem; flex-wrap:wrap; margin: 0.5rem 0 1.25rem;">
                <div class="status-pill">🔁 Total Cycles: {data['agent_world']['totals']['attempts']}</div>
                <div class="status-pill">✅ Supported: {data['agent_world']['totals']['supported']}</div>
                <div class="status-pill">❌ Falsified: {data['agent_world']['totals']['falsified']}</div>
                <div class="status-pill">⚠️ Errors/Timeouts: {data['agent_world']['totals']['errors'] + data['agent_world']['totals']['timeout']}</div>
                <div class="status-pill">🪞 Duplicates Rejected: {data['agent_world']['totals']['duplicates']}</div>
                <div class="status-pill">🏆 Confirmed Laws: {data['agent_world']['confirmed_laws_count']}</div>
            </div>
            <div style="max-height: 320px; overflow-y: auto;">
                <table class="data-table">
                    <thead>
                        <tr>
                            <th>Research Domain</th>
                            <th>Attempts</th>
                            <th>Supported</th>
                            <th>Falsified</th>
                            <th>Errors/Timeout</th>
                            <th>Duplicates</th>
                            <th>Last Run</th>
                        </tr>
                    </thead>
                    <tbody>
                        {"".join(f'''<tr>
                            <td style="color:#fff;">{d['name']}<div style="font-size:0.75rem;color:var(--text-muted);">{d['description']}</div></td>
                            <td style="font-family: var(--font-mono); color: var(--text-secondary);">{d['attempts']}</td>
                            <td style="font-family: var(--font-mono); color: var(--accent-emerald);">{d['supported']}</td>
                            <td style="font-family: var(--font-mono); color: var(--accent-rose);">{d['falsified']}</td>
                            <td style="font-family: var(--font-mono); color: var(--accent-amber);">{d['errors'] + d['timeout']}</td>
                            <td style="font-family: var(--font-mono); color: var(--text-muted);">{d['duplicates']}</td>
                            <td style="font-family: var(--font-mono); color: var(--text-muted); font-size:0.8rem;">{d['last_run'] or '—'}</td>
                        </tr>''' for d in data['agent_world']['domain_coverage'])}
                    </tbody>
                </table>
            </div>
        </div>

        <div id="conjecturesFeed">
            {"".join(f'''
            <div class="conjecture-card">
                <div class="conj-header">
                    <div>
                        <span style="font-size: 0.75rem; font-family: var(--font-mono); color: var(--accent-cyan);">{c.get('id', 'CONJ')} • {c.get('timestamp', '')} • <span style="color: var(--accent-purple);">{DOMAINS[c['domain']].name if c.get('domain') in DOMAINS else 'legacy domain'}</span></span>
                        <div class="conj-title">{c.get('name', 'Conjecture')}</div>
                    </div>
                    <span class="verdict-badge {'verdict-supported' if 'SUPPORTED' in c.get('verdict', '') else 'verdict-falsified'}">{c.get('verdict', 'PENDING')}</span>
                </div>
                <div style="margin-bottom: 0.75rem; font-size: 0.9rem; color: #cbd5e1;">
                    <b>Formulation:</b> {c.get('formulation', '')}
                </div>
                <div style="font-size: 0.85rem; color: var(--text-muted); margin-bottom: 0.5rem;">
                    <b>Target:</b> {c.get('target', '')} • Model: <code>{c.get('model', '')}</code>
                </div>
                <div class="code-box">{c.get('code', '')}</div>
                <div style="margin-top: 0.5rem; font-family: var(--font-mono); font-size: 0.8rem; color: var(--accent-emerald);">
                    <b>Execution Output:</b> {c.get('output', '')}
                </div>
            </div>
            ''' for c in data['conjectures']) if data['conjectures'] else '<div class="glass-panel" style="text-align: center; padding: 3rem;"><p style="color: var(--text-muted);">No conjectures generated yet. Run models/math_researcher.py to begin autonomous inquiry cycles.</p></div>'}
        </div>
    </div>

    <footer>
        <p>Prime Frontier — Mathematical Reverse-Engineering & Prediction Laboratory</p>
        <p style="margin-top: 0.5rem; font-family: var(--font-mono); font-size: 0.75rem;">Compiled with SymPy, mpmath & NumPy • Powered by DeepSeek R1 & Qwen 2.5 Math</p>
    </footer>
</div>

<script>
    const PRIME_DATA = {json_blob};

    function switchTab(tabId) {{
        document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
        document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));

        const targetBtn = Array.from(document.querySelectorAll('.tab-btn')).find(b => b.getAttribute('onclick').includes(tabId));
        if (targetBtn) targetBtn.classList.add('active');

        const targetContent = document.getElementById(`tab-${{tabId}}`);
        if (targetContent) targetContent.classList.add('active');

        window.scrollTo({{ top: 0, behavior: 'smooth' }});
    }}

    document.addEventListener("DOMContentLoaded", function() {{
        if (typeof renderMathInElement !== 'undefined') {{
            renderMathInElement(document.body, {{
                delimiters: [
                    {{left: '$$', right: '$$', display: true}},
                    {{left: '$', right: '$', display: false}}
                ],
                throwOnError : false
            }});
        }}
    }});
</script>

</body>
</html>
"""
    return html_content

def main():
    print("==================================================")
    print("  PRIME FRONTIER — DASHBOARD COMPILER")
    print("==================================================")
    data = harvest_dashboard_data()
    print("[4/4] Compiling dashboard.html...")
    html_output = generate_html(data)
    out_path = os.path.join(BASE_DIR, "dashboard.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html_output)
    print(f"Successfully generated: {out_path} ({round(os.path.getsize(out_path)/1024, 1)} KB)")
    print("==================================================")

if __name__ == "__main__":
    main()
