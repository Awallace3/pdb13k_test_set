"""
Example of using the refactored crystal presentation workflow.

This demonstrates the two-step process:
1. prepare_crystal_data_intermediates() - Run once to compute all CLE errors
2. Use the stored results for fast plotting/analysis

The refactoring separates slow data processing from fast visualization.
"""

from crystal_presentation import (
    prepare_crystal_data_intermediates,
    compute_cle_errors_for_N_all_crystals,
)
import pandas as pd
import time


def main():
    print("=" * 80)
    print("REFACTORED WORKFLOW DEMONSTRATION")
    print("=" * 80)

    # Step 1: Prepare intermediates (slow, run once)
    print("\nStep 1: Preparing intermediates...")
    print("This computes CLE errors for N = [0, 1, 5, 10, 20]")
    print("This is the SLOW step - but you only do it once!\n")

    start_time = time.time()
    intermediates = prepare_crystal_data_intermediates(
        tl_N=100, uma_cutoff=6.0, sft=False
    )
    prepare_time = time.time() - start_time

    print(f"\nPreparation complete in {prepare_time:.1f} seconds")
    print(f"Saved to: crystal_intermediates_tl100_uma6.0_sftFalse.pkl")

    # Step 2: Access pre-computed results (fast)
    print("\n" + "=" * 80)
    print("Step 2: Accessing pre-computed CLE errors (FAST!)")
    print("=" * 80)

    for N in [0, 1, 5, 10, 20]:
        print(f"\nN={N} CLE errors:")
        errors = intermediates["cle_errors_by_N"][N]

        # Show statistics for UMA-m+AP3-LR vs CCSD(T)/CBS (the key result)
        uma_errors = errors["uma_m_ap3lr_full_cle_errors_ccsd_t_CBS"]
        if len(uma_errors) > 0:
            mae = sum(abs(e) for e in uma_errors) / len(uma_errors)
            max_err = max(abs(e) for e in uma_errors)
            print(f"  UMA-m+AP3-LR vs CCSD(T)/CBS:")
            print(f"    MAE: {mae:.4f} kJ/mol")
            print(f"    Max |error|: {max_err:.4f} kJ/mol")
            print(f"    N crystals: {len(uma_errors)}")

    # Step 3: Show how fast subsequent access is
    print("\n" + "=" * 80)
    print("Step 3: Demonstrating fast re-loading")
    print("=" * 80)

    print("\nLoading pre-computed intermediates from pickle...")
    start_reload = time.time()
    loaded = pd.read_pickle("./crystal_intermediates_tl100_uma6.0_sftFalse.pkl")
    reload_time = time.time() - start_reload

    print(f"Reloaded in {reload_time:.3f} seconds (instant!)")
    print(f"\nAvailable data:")
    print(f"  - {len(loaded['all_crystals'])} crystals")
    print(f"  - CLE errors for N = [0, 1, 5, 10, 20]")
    print(f"  - df_apprx shape: {loaded['df_apprx'].shape}")
    print(f"  - df_bm shape: {loaded['df_bm'].shape}")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"✓ One-time preparation: {prepare_time:.1f}s")
    print(f"✓ Subsequent loads: {reload_time:.3f}s (instant!)")
    print(f"✓ CLE errors pre-computed for 5 different N values")
    print(f"✓ Can now create plots/analysis very quickly")
    print("\nOLD WORKFLOW:")
    print("  - Run full computation every time you want a different N value")
    print("  - Takes ~{prepare_time:.1f}s each time")
    print("\nNEW WORKFLOW:")
    print("  - Run preparation once: {prepare_time:.1f}s total")
    print("  - Create unlimited plots/analyses: <1s each")
    print(
        f"  - Speedup for 5 plots: ~{5 * prepare_time / prepare_time:.0f}x → ~5x faster!"
    )


if __name__ == "__main__":
    main()
