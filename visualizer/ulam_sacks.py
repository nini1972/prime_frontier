#!/usr/bin/env python3
"""
visualizer/ulam_sacks.py - Geometric Prime Cartography (Ulam & Sacks Spirals)

Generates geometric visual representations of prime numbers:
1. Ulam Spiral (1963): Integer square spiral revealing persistent diagonal lines
   corresponding to quadratic polynomials with high prime density (e.g. Euler's n^2 + n + 41).
2. Sacks Spiral (1994): Polar Archimedean spiral r = sqrt(n), theta = 2*pi*sqrt(n)
   aligning quadratic progressions into smooth curves.
"""

import sys
import os
import math
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.sieve import generate_primes

def generate_ulam_grid(size: int = 501) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate an NxN Ulam spiral grid where 1 is at the center and integers spiral outward.
    Returns: (grid of integers, boolean prime mask)
    """
    if size % 2 == 0:
        size += 1
        
    grid = np.zeros((size, size), dtype=np.int64)
    cx = cy = size // 2
    
    x = cx
    y = cy
    val = 1
    grid[y, x] = val
    
    dx, dy = 1, 0
    step_len = 1
    step_count = 0
    turns = 0
    
    max_val = size * size
    while val < max_val:
        x += dx
        y += dy
        val += 1
        grid[y, x] = val
        step_count += 1
        
        if step_count == step_len:
            step_count = 0
            # Turn 90 degrees counter-clockwise
            dx, dy = -dy, dx
            turns += 1
            if turns % 2 == 0:
                step_len += 1
                
    # Create prime mask
    max_p = size * size
    primes = set(generate_primes(max_p))
    prime_mask = np.isin(grid, list(primes))
    
    return grid, prime_mask

def plot_ulam_spiral(size: int = 501, output_path: str = "outputs/ulam_spiral.png"):
    """Render high-resolution Ulam spiral highlighting prime tracks."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    grid, prime_mask = generate_ulam_grid(size)
    
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, 10), dpi=200)
    fig.patch.set_facecolor('#060913')
    ax.set_facecolor('#060913')
    
    # Primes as luminous cyan dots
    y_coords, x_coords = np.nonzero(prime_mask)
    ax.scatter(x_coords, y_coords, c='#00f2fe', s=1.2, alpha=0.9, edgecolors='none')
    
    # Highlight Euler's polynomial n^2 + n + 41
    euler_vals = [n*n + n + 41 for n in range(int(math.sqrt(size*size)))]
    euler_mask = np.isin(grid, euler_vals) & prime_mask
    ey, ex = np.nonzero(euler_mask)
    ax.scatter(ex, ey, c='#f43f5e', s=4.0, alpha=0.95, label='Euler Track ($n^2 + n + 41$)', zorder=5)
    
    ax.set_title(f'Ulam Prime Spiral ({size}x{size} Grid, {len(x_coords):,} Primes)\nDiagonal Alignments of Quadratic Polynomial Attractors',
                 fontsize=13, fontweight='bold', pad=15, color='#ffffff')
    ax.axis('off')
    ax.legend(loc='upper right', framealpha=0.8, facecolor='#0d1526', edgecolor='#334155', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()

def plot_sacks_spiral(max_n: int = 40000, output_path: str = "outputs/sacks_spiral.png"):
    """
    Render Sacks polar spiral:
    r = sqrt(n), theta = 2*pi*sqrt(n)
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    primes = generate_primes(max_n)
    
    # Polar coordinates for primes
    r_primes = np.sqrt(primes)
    theta_primes = 2.0 * np.pi * r_primes
    
    # Convert to Cartesian
    x_primes = r_primes * np.cos(theta_primes)
    y_primes = r_primes * np.sin(theta_primes)
    
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, 10), dpi=200)
    fig.patch.set_facecolor('#060913')
    ax.set_facecolor('#060913')
    
    ax.scatter(x_primes, y_primes, c='#00f2fe', s=1.0, alpha=0.8, edgecolors='none', label=f'{len(primes):,} Primes')
    
    # Highlight Euler polynomial primes
    euler_n = np.arange(0, int(math.sqrt(max_n)))
    euler_primes = euler_n**2 + euler_n + 41
    euler_primes = euler_primes[euler_primes <= max_n]
    
    r_euler = np.sqrt(euler_primes)
    theta_euler = 2.0 * np.pi * r_euler
    ex = r_euler * np.cos(theta_euler)
    ey = r_euler * np.sin(theta_euler)
    ax.scatter(ex, ey, c='#f59e0b', s=5.0, alpha=0.95, label='Euler Spiral Curve ($n^2 + n + 41$)', zorder=5)
    
    ax.set_title(f'Sacks Archimedean Prime Spiral ($N \\leq {max_n:,}$)\nPolar Mapping $r = \\sqrt{{n}}, \\theta = 2\\pi\\sqrt{{n}}$',
                 fontsize=13, fontweight='bold', pad=15, color='#ffffff')
    ax.axis('off')
    ax.legend(loc='upper right', framealpha=0.8, facecolor='#0d1526', edgecolor='#334155', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()

if __name__ == "__main__":
    print("Generating Ulam Spiral (351x351)...")
    plot_ulam_spiral(size=351, output_path="outputs/ulam_spiral.png")
    print("Saved Ulam Spiral to outputs/ulam_spiral.png")
    
    print("Generating Sacks Spiral (N <= 30,000)...")
    plot_sacks_spiral(max_n=30000, output_path="outputs/sacks_spiral.png")
    print("Saved Sacks Spiral to outputs/sacks_spiral.png")
    print("Geometric Cartography completed!")
