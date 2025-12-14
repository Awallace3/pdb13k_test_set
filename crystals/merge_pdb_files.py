#!/usr/bin/env python3
"""
Merge atom types from mol_A.openmm.pdb with connectivity from mol_A-debug.pdb
for all crystals in ff_params directory.
"""

import os
from crystal_ff import merge_pdb_files, crystal_names_all


def process_all_crystals():
    """Process all crystals and create merged PDB files."""
    for c in crystal_names_all:
        c_path = f"ff_params/{c}"

        if not os.path.exists(c_path):
            print(f"Skipping {c}: directory not found")
            continue

        openmm_pdb = f"{c_path}/mol_A.openmm.pdb"
        debug_pdb = f"{c_path}/mol_A-debug.pdb"
        merged_pdb = f"{c_path}/mol_A.openmm_merged.pdb"

        if not os.path.exists(openmm_pdb):
            print(f"Skipping {c}: {openmm_pdb} not found")
            continue

        if not os.path.exists(debug_pdb):
            print(f"Skipping {c}: {debug_pdb} not found")
            continue

        try:
            merge_pdb_files(openmm_pdb, debug_pdb, merged_pdb)
            print(f"Created: {merged_pdb}")
        except Exception as e:
            print(f"Error processing {c}: {e}")


if __name__ == "__main__":
    print("Merging PDB files for all crystals...")
    print("=" * 70)
    process_all_crystals()
    print("=" * 70)
    print("Done!")
