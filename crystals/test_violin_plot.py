#!/usr/bin/env python
"""
Test script for plot_crystal_violin_errors function.

This script demonstrates how to use the new plot_crystal_violin_errors function
which creates violin plots showing error distributions for each crystal.
"""

from crystal_main import plot_crystal_violin_errors

# Example 1: Plot violin errors for N=0 (no reference points, pure ML)
print("Generating violin plots for N=0...")
crystal_errors_apprx, crystal_errors_bm = plot_crystal_violin_errors(N=0, sft=False, tl_N=100)
print("Done!")
print("\nCrystal errors (approximate):")
for crystal, errors in sorted(crystal_errors_apprx.items()):
    print(f"  {crystal}: {len(errors)} methods")
    for method, error in errors.items():
        print(f"    {method}: {error:.4f} kJ/mol")

print("\nCrystal errors (benchmark):")
for crystal, errors in sorted(crystal_errors_bm.items()):
    print(f"  {crystal}: {len(errors)} methods")
    for method, error in errors.items():
        print(f"    {method}: {error:.4f} kJ/mol")

# Example 2: Plot violin errors for N=1 (1 reference point + ML for rest)
print("\n\nGenerating violin plots for N=1...")
crystal_errors_apprx_N1, crystal_errors_bm_N1 = plot_crystal_violin_errors(N=1, sft=False, tl_N=100)
print("Done!")

# Example 3: Plot violin errors for N=10 with different tl_N
print("\n\nGenerating violin plots for N=10, tl_N=1000...")
crystal_errors_apprx_N10, crystal_errors_bm_N10 = plot_crystal_violin_errors(N=10, sft=False, tl_N=1000)
print("Done!")
