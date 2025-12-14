#!/usr/bin/env python3
"""
Test script to verify that FF energies change for different dimers.
"""

import pandas as pd
from crystal_ff import run_ff_dimer

if __name__ == "__main__":
    print("Testing FF energy calculations for different dimers...")
    print("=" * 70)
    run_ff_dimer()
    print("=" * 70)
    print("\nTest complete! Check output above for:")
    print("1. Different geometry hashes for each dimer")
    print("2. Unique energy values (no duplicates)")
    print("3. Summary statistics showing energy variation")
