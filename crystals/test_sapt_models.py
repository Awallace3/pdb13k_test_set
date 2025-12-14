#!/usr/bin/env python3
"""
Test script for SAPT model inference and visualization.

Usage:
    python test_sapt_models.py --generate  # Generate predictions
    python test_sapt_models.py --plot      # Create visualizations
    python test_sapt_models.py --both      # Do both
"""

import argparse
from crystal_main import ap2_ap3_df_energies_sapt_models, plot_crystal_lattice_energies_sapt_models


def main():
    parser = argparse.ArgumentParser(description='Test SAPT models for crystal predictions')
    parser.add_argument('--generate', action='store_true', 
                        help='Generate predictions from SAPT models')
    parser.add_argument('--plot', action='store_true',
                        help='Create visualization plots')
    parser.add_argument('--both', action='store_true',
                        help='Generate predictions and create plots')
    parser.add_argument('--version', '-v', default='bm', choices=['bm', 'apprx'],
                        help='Version to use: bm (benchmark) or apprx (approximate)')
    parser.add_argument('--N-values', nargs='+', type=int, default=[0, 1, 5, 10, 20],
                        help='N-body truncation values for plotting')
    
    args = parser.parse_args()
    
    # If no arguments provided, show help
    if not (args.generate or args.plot or args.both):
        parser.print_help()
        return
    
    version = args.version
    n_values = args.N_values
    
    # Generate predictions
    if args.generate or args.both:
        print("\n" + "="*80)
        print(f"GENERATING PREDICTIONS FOR VERSION: {version}")
        print("="*80)
        
        df = ap2_ap3_df_energies_sapt_models(generate=True, v=version)
        
        print(f"\nGenerated dataframe shape: {df.shape}")
        sapt_cols = [col for col in df.columns if 'SAPT' in col or 'sapt' in col]
        print(f"Number of SAPT columns added: {len(sapt_cols)}")
        print(f"Sample columns: {sapt_cols[:10]}")  # Show first 10 columns
    
    # Create plots
    if args.plot or args.both:
        print("\n" + "="*80)
        print(f"CREATING VISUALIZATIONS FOR VERSION: {version}")
        print(f"N-body truncation values: {n_values}")
        print("="*80)
        
        plot_crystal_lattice_energies_sapt_models(v=version, N_values=n_values)
    
    print("\n" + "="*80)
    print("DONE!")
    print("="*80)


if __name__ == "__main__":
    main()
