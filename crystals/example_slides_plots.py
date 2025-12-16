"""
Example usage of plot_from_intermediates_slides() for creating presentation-ready figures.

This function creates clean, compact plots suitable for slides by:
- Showing only selected crystals (not all 20+)
- Using a 2x2 or Nx2 grid layout
- Focusing on either 'apprx' or 'bm' data
- Optimized font sizes and spacing for presentations
"""

from crystal_presentation import (
    prepare_crystal_data_intermediates,
    plot_from_intermediates_slides,
)


def main():
    print("=" * 80)
    print("CREATING PRESENTATION-READY CRYSTAL PLOTS")
    print("=" * 80)

    # Step 1: Prepare intermediates (if not already done)
    print("\nStep 1: Checking for pre-computed intermediates...")
    import os

    if not os.path.exists("./crystal_intermediates_tl100_uma6.0_sftFalse.pkl"):
        print("No intermediates found. Preparing now...")
        prepare_crystal_data_intermediates(tl_N=100, uma_cutoff=6.0, sft=False)
    else:
        print("Intermediates already exist. Skipping preparation.")

    # Step 2: Create slide plots for benchmark (CCSD(T)/CBS) data
    print("\n" + "=" * 80)
    print("Step 2: Creating benchmark (CCSD(T)/CBS) plots")
    print("=" * 80)

    # Select interesting crystals for presentation
    selected_crystals = ["ice", "urea", "benzene", "formamide"]

    print(f"\nSelected crystals: {selected_crystals}")
    print("Creating 2x2 grid plots for N = 0, 1, 5, 10...")

    for N in [0, 1, 5, 10]:
        print(f"\n  Creating plot for N={N}...")
        output_path = plot_from_intermediates_slides(
            N=N,
            v="bm",  # benchmark data (vs CCSD(T)/CBS)
            crystals=selected_crystals,
            sft=False,
        )
        print(f"  Saved to: {output_path}")

    # Step 3: Create slide plots for approximate (SAPT0/aDZ) data
    print("\n" + "=" * 80)
    print("Step 3: Creating approximate (SAPT0/aDZ) plots")
    print("=" * 80)

    # Can use different crystals for approximate data
    apprx_crystals = ["ice", "ammonia", "benzene", "pyrazole"]

    print(f"\nSelected crystals: {apprx_crystals}")
    print("Creating 2x2 grid plots for N = 0, 1...")

    for N in [0, 1]:
        print(f"\n  Creating plot for N={N}...")
        output_path = plot_from_intermediates_slides(
            N=N,
            v="apprx",  # approximate data (vs SAPT0/aDZ)
            crystals=apprx_crystals,
            sft=False,
        )
        print(f"  Saved to: {output_path}")

    # Step 4: Create a larger grid (3x2 = 6 crystals)
    print("\n" + "=" * 80)
    print("Step 4: Creating larger 3x2 grid with 6 crystals")
    print("=" * 80)

    many_crystals = ["ice", "urea", "benzene", "formamide", "pyrazole", "imidazole"]

    print(f"\nSelected crystals: {many_crystals}")
    output_path = plot_from_intermediates_slides(
        N=1,
        v="bm",
        crystals=many_crystals,
        sft=False,
        output_path="./x23_plots/CLE_bm_N1_6crystals_slides.png",
    )
    print(f"Saved to: {output_path}")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("\nGenerated plots suitable for presentations:")
    print("  ✓ Compact 2x2 or Nx2 grid layouts")
    print("  ✓ Optimized font sizes and spacing")
    print("  ✓ Clean legends (only on first subplot)")
    print("  ✓ Focus on key methods (AP3+D4, UMA-m+AP3-LR, OPLS)")
    print("  ✓ Customizable crystal selection")
    print("  ✓ Separate plots for 'apprx' vs 'bm' datasets")
    print("\nAll plots saved to ./x23_plots/")
    print("\nComparison with standard plots:")
    print("  - Standard: All 20+ crystals in Nx2 grid → very tall, hard to read")
    print("  - Slides: Selected 4-6 crystals in 2x2/3x2 grid → fits on one slide!")


if __name__ == "__main__":
    main()
