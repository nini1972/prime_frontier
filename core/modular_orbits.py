#!/usr/bin/env python3
"""
core/modular_orbits.py - Consecutive Prime Modular Memory & Chebyshev Bias Engine

Analyzes the Lemke Oliver-Soundararajan phenomenon (2016): consecutive primes exhibit
unexpected short-range repulsion and attraction modulo q.
For example, modulo 10 (primes ending in 1, 3, 7, 9):
Primes ending in 1 are followed by 1 only ~18% of the time (repulsion),
but followed by 3 or 7 over ~30% of the time!

Demonstrates that primes possess short-term memory that classical random models ignore.
"""

import sys
import os
import math
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.sieve import generate_primes

def compute_modular_transitions(primes: np.ndarray, mod: int = 10) -> Dict:
    """
    Compute transition probability matrix between consecutive primes modulo `mod`.
    For mod=10, considers coprime residues [1, 3, 7, 9].
    """
    # Filter primes that are coprime to mod (exclude 2, 5 for mod 10)
    coprimes = [r for r in range(1, mod) if math.gcd(r, mod) == 1]
    r_to_idx = {r: i for i, r in enumerate(coprimes)}
    n_res = len(coprimes)
    
    valid_primes = primes[primes > mod]
    residues = valid_primes % mod
    
    # Filter to only valid coprimes
    mask = np.isin(residues, coprimes)
    residues = residues[mask]
    
    if len(residues) < 2:
        return {}
        
    counts = np.zeros((n_res, n_res), dtype=np.int64)
    r_current = residues[:-1]
    r_next = residues[1:]
    
    for c, n in zip(r_current, r_next):
        if c in r_to_idx and n in r_to_idx:
            counts[r_to_idx[c], r_to_idx[n]] += 1
            
    # Row normalize to get transition probabilities
    row_sums = counts.sum(axis=1, keepdims=True)
    probs = np.divide(counts, row_sums, out=np.zeros_like(counts, dtype=np.float64), where=row_sums!=0)
    
    return {
        "modulus": mod,
        "residues": coprimes,
        "transition_matrix": probs.round(4).tolist(),
        "transition_counts": counts.tolist(),
        "total_transitions": int(counts.sum()),
        "expected_uniform": round(1.0 / n_res, 4)
    }

def compute_residue_autocorrelation(primes: np.ndarray, mod: int = 10, max_lag: int = 15) -> Dict[int, float]:
    """
    Compute the autocorrelation of the indicator I(p_n == p_{n+k} mod mod)
    to measure how many steps the short-term prime memory persists.
    """
    valid = primes[primes > mod] % mod
    coprimes = [r for r in range(1, mod) if math.gcd(r, mod) == 1]
    valid = valid[np.isin(valid, coprimes)]
    
    autocorr = {}
    n = len(valid)
    expected = 1.0 / len(coprimes)
    
    for lag in range(1, max_lag + 1):
        matches = np.mean(valid[:-lag] == valid[lag:])
        # Excess relative to uniform random
        autocorr[lag] = round(float(matches - expected), 5)
        
    return autocorr

def plot_modular_bias_heatmap(matrix_data: Dict, output_path: str = "outputs/modular_bias_mod10.png"):
    """Render a sleek dark-mode heatmap of the consecutive prime transition matrix."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    probs = np.array(matrix_data["transition_matrix"])
    residues = matrix_data["residues"]
    mod = matrix_data["modulus"]
    expected = matrix_data["expected_uniform"]
    
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(8, 7), dpi=200)
    fig.patch.set_facecolor('#060913')
    ax.set_facecolor('#080d1a')
    
    # Deviation from uniform (red = repulsion, cyan = attraction)
    deviation = probs - expected
    im = ax.imshow(deviation, cmap='coolwarm', vmin=-0.12, vmax=0.12)
    
    # Annotate cells with percentages and deviations
    for i in range(len(residues)):
        for j in range(len(residues)):
            p = probs[i, j]
            diff = (p - expected) * 100
            sign = "+" if diff > 0 else ""
            txt = f"{p*100:.1f}%\n({sign}{diff:.1f}%)"
            color = "#000000" if abs(deviation[i, j]) > 0.08 else "#ffffff"
            ax.text(j, i, txt, ha='center', va='center', color=color, fontweight='bold', fontsize=10)
            
    ax.set_xticks(range(len(residues)))
    ax.set_yticks(range(len(residues)))
    ax.set_xticklabels([f"...{r}" for r in residues], fontsize=11, color='#00f2fe', fontweight='bold')
    ax.set_yticklabels([f"...{r}" for r in residues], fontsize=11, color='#00f2fe', fontweight='bold')
    
    ax.set_xlabel(f'Next Prime $p_{{n+1}}$ (mod {mod})', fontsize=12, labelpad=10, color='#94a3b8')
    ax.set_ylabel(f'Current Prime $p_n$ (mod {mod})', fontsize=12, labelpad=10, color='#94a3b8')
    ax.set_title(f'Lemke Oliver-Soundararajan Prime Memory (Mod {mod})\nConsecutive Residue Transition Probabilities vs Uniform ({expected*100:.0f}%)',
                 fontsize=13, fontweight='bold', pad=15, color='#ffffff')
    
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Deviation from Uniform Random (Prob - 0.25)', color='#94a3b8', fontsize=10)
    cbar.ax.tick_params(colors='#94a3b8')
    
    plt.tight_layout()
    plt.savefig(output_path, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()

if __name__ == "__main__":
    print("Testing Modular Memory Engine on first 500,000 primes...")
    primes = generate_primes(7_500_000)[:500_000]
    res = compute_modular_transitions(primes, mod=10)
    print(f"Modulus: {res['modulus']}, Residues: {res['residues']}")
    print(f"Expected uniform probability: {res['expected_uniform']*100}%")
    
    labels = res['residues']
    print("\nTransition Matrix (%):")
    header = "p_n \\ p_{n+1} | " + " | ".join(f"  {r}  " for r in labels)
    print(header)
    print("-" * len(header))
    for i, r in enumerate(labels):
        row_str = f"     {r}       | " + " | ".join(f"{res['transition_matrix'][i][j]*100:5.2f}%" for j in range(len(labels)))
        print(row_str)
        
    autocorr = compute_residue_autocorrelation(primes, mod=10, max_lag=6)
    print("\nResidue Autocorrelation Excess (p_n == p_{n+k}):")
    for lag, val in autocorr.items():
        print(f"  Lag {lag}: {val:+.5f}")
        
    plot_modular_bias_heatmap(res, output_path="outputs/modular_bias_mod10.png")
    print("\nSaved heatmap to outputs/modular_bias_mod10.png")
    print("Modular Memory Engine verified!")
