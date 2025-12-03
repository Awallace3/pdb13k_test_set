import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import AutoMinorLocator
from matplotlib.colors import BoundaryNorm, ListedColormap
import qcelemental as qcel
import os
from crystalatte.plugins import force_fields
from mpl_toolkits.axes_grid1 import make_axes_locatable


def setup_plot_style(use_latex=True):
    """Apply consistent plot styling."""

    # LaTeX font configuration
    if use_latex:
        try:
            mpl.rcParams['text.usetex'] = True
            mpl.rcParams['font.family'] = 'serif'
            mpl.rcParams['font.serif'] = ['Computer Modern Roman']
            mpl.rcParams['mathtext.fontset'] = 'cm'
        except:
            print("Warning: LaTeX not available, falling back to standard fonts")
            use_latex = False

    if not use_latex:
        mpl.rcParams['font.family'] = 'DejaVu Sans'
        mpl.rcParams['mathtext.fontset'] = 'dejavusans'

    # Override with specific settings
    mpl.rcParams['figure.figsize'] = (10, 6)
    mpl.rcParams['font.size'] = 12
    mpl.rcParams['axes.labelsize'] = 14
    mpl.rcParams['xtick.labelsize'] = 12
    mpl.rcParams['ytick.labelsize'] = 12
    mpl.rcParams['legend.fontsize'] = 12
    mpl.rcParams['lines.linewidth'] = 2.5
    mpl.rcParams['lines.markersize'] = 8
    mpl.rcParams['savefig.dpi'] = 300
    mpl.rcParams['savefig.bbox'] = 'tight'

    # Enhanced legend defaults
    mpl.rcParams['legend.frameon'] = True
    mpl.rcParams['legend.facecolor'] = 'white'
    mpl.rcParams['legend.edgecolor'] = 'gray'
    mpl.rcParams['legend.framealpha'] = 0.9
    mpl.rcParams['legend.borderpad'] = 0.4


def style_axes(ax, border_width=2.0):
    """Apply consistent axis styling."""
    # Set spine widths
    for spine in ax.spines.values():
        spine.set_linewidth(border_width)

    # Tick styling
    ax.tick_params(which='major', direction='in',
                  length=8, width=border_width,
                  top=True, right=True, pad=10)
    ax.tick_params(which='minor', direction='in',
                  length=4, width=border_width,
                  top=True, right=True)

    # Minor ticks
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())

    # No grid
    ax.grid(False)


def cle_contributions(non_additive_mb_energy, num_replicas, num_monomers):
    """Calculate CLE contributions."""
    return non_additive_mb_energy * num_replicas / int(num_monomers)


def compute_jax_induction_energy(geom_dir, df, xml_opt=""):
    """Compute JAX induction energy for molecules."""
    def compute_energy(row):
        mol = row["qcel_mol"]
        nmer = {
            "monomers": [0 for i in range(int(row["N-mer Name"][0]))],
            "nambe": 0,
        }

        nambe = force_fields.polarization_energy_function(
            qcel_mol=mol,
            cif_output=None,
            nmers=nmer,
            nmer=nmer,
            keynmer=row["N-mer Name"],
            rminseps=None,
            rcomseps=None,
            cle_run_type=["custom"],
            pdb_file=f"extra_files/{geom_dir}.pdb",
            residue_file=f"extra_files/{geom_dir}_residue.xml",
            xml_file=f"extra_files/{geom_dir}{xml_opt}.xml",
            atom_types_map=f"extra_files/{geom_dir}_map.csv",
        )

        if nambe:
            nambe *= qcel.constants.hartree2kJmol
            return nambe
        return np.nan

    df[f"jax_induction_energy{xml_opt}"] = df.apply(compute_energy, axis=1)
    return df


def CLE_drude_mp2_figure(
    geom_dirs=None,
    names=None,
    xml_opts=None,
    mp2_switch_mms_distance=None,
    figure_params=None,
    use_latex=True,
    show_annotations=True,
    use_hybrid=True
):
    """
    Create CLE drude MP2 figure with cleaner implementation.

    Parameters:
    -----------
    geom_dirs : list
        List of geometry directories to process
    names : list
        List of subplot labels for each geometry
    xml_opts : list
        List of XML options for each geometry
    mp2_switch_mms_distance : list
        List of MMS switch distances for each geometry
    figure_params : dict
        Additional figure parameters (colormap bounds, etc.)
    use_latex : bool
        Whether to use LaTeX for rendering text
    show_annotations : bool
        Whether to show value annotations and reference lines (default: True)
    use_hybrid : bool
        Whether to use hybrid MP2/Drude method (MP2 below cutoff, Drude above) instead of pure Drude (default: True)
    """

    # Default values
    if geom_dirs is None:
        geom_dirs = ["./imidazole", "./pyrazole", "./acetic_acid", "./formamide"]

    if names is None:
        names = [
            r"\textbf{(A) Imidazole}",
            r"\textbf{(B) Pyrazole}",
            r"\textbf{(C) Acetic Acid}",
            r"\textbf{(D) Formamide}",
        ]

    if xml_opts is None:
        xml_opts = ["", "", "_elst", ""]

    if mp2_switch_mms_distance is None:
        mp2_switch_mms_distance = [4.2, 4.2, 4.2, 4.2]

    if figure_params is None:
        figure_params = {
            "colormap_min": -1.0,
            "colormap_max": 1.0
        }

    # Validate inputs
    N = len(geom_dirs)
    assert N == len(xml_opts), "geom_dirs and xml_opts must have the same length"
    assert N == len(names), "geom_dirs and names must have the same length"
    assert N == len(mp2_switch_mms_distance), "geom_dirs and mp2_switch_mms_distance must have the same length"

    # Set up plot styling
    setup_plot_style(use_latex=use_latex)

    # Create figure and axes
    fig, axes = plt.subplots(N, 1, figsize=(6, N + 3))
    if N == 1:
        axes = [axes]

    # Define separation distance range
    increment = 0.25
    sep_distances = np.arange(1.0 - 0.05, 20.0 + 0.05, increment)

    # Process each geometry
    for n, geom_dir in enumerate(geom_dirs):
        xml_opt = xml_opts[n]
        ax1 = axes[n]

        # Share x-axis with first subplot
        if n > 0:
            ax1.sharex(axes[0])

        # Load data
        df_path = geom_dir + "_mols.pkl"
        if not os.path.exists(df_path):
            print(f"Warning: {df_path} not found, skipping {geom_dir}")
            continue

        df = pd.read_pickle(df_path)

        # Define reference columns
        ref = "MP2/aTZ"
        mp2_cle = ref + " CLE"

        # Check if JAX energy needs to be computed
        jax_col = f"jax_induction_energy{xml_opt}"
        if jax_col not in df.columns:
            print(f"Computing JAX induction energy for {geom_dir}{xml_opt}...")
            df["qcel_mol"] = df.get("mol", df.get("qcel_mol"))
            df = compute_jax_induction_energy(geom_dir, df, xml_opt)
            df.to_pickle(df_path)

        # Calculate CLE contributions
        df["JAX IND CLE"] = df.apply(
            lambda r: cle_contributions(
                r[jax_col],
                r["Num. Rep. (#)"],
                int(r["N-mer Name"][0]),
            ),
            axis=1,
        )

        df[mp2_cle] = df.apply(
            lambda r: cle_contributions(
                r[ref],
                r["Num. Rep. (#)"],
                int(r["N-mer Name"][0])
            ),
            axis=1,
        )

        # Print total CLE
        total_cle = df[mp2_cle].sum()
        print(f"{n}: {geom_dir}{xml_opt} - Total MP2 CLE: {total_cle:.2f} kJ/mol")

        # Calculate values for each separation distance
        jax_ind_cle_below = []
        jax_ind_cle_above = []
        mp2_cle_below = []
        mp2_cle_above = []
        hybrid_cle_below = []  # Hybrid: MP2 below cutoff, Drude above cutoff
        hybrid_cle_above = []

        for d in sep_distances:
            # Below threshold
            df_below = df[df["Minimum Monomer Separations (A)"] <= d]
            jax_below = df_below["JAX IND CLE"].sum()
            mp2_below = df_below[mp2_cle].sum()

            # Above threshold
            df_above = df[df["Minimum Monomer Separations (A)"] > d]
            jax_above = df_above["JAX IND CLE"].sum()
            mp2_above = df_above[mp2_cle].sum()

            # Hybrid calculation using current x-axis value as switchover
            if use_hybrid:
                # The hybrid uses the current x-axis value 'd' as the switchover:
                # - MP2 for trimers with MMS <= d
                # - Drude for trimers with MMS > d
                # This gives us the TOTAL CLE using hybrid approach at switchover distance d

                # Calculate hybrid total:
                # Use MP2 for all trimers with separation <= d
                mp2_part_total = df[df["Minimum Monomer Separations (A)"] <= d][mp2_cle].sum()
                # Use Drude for all trimers with separation > d
                drude_part_total = df[df["Minimum Monomer Separations (A)"] > d]["JAX IND CLE"].sum()

                # Total hybrid CLE = MP2 (for close) + Drude (for far)
                hybrid_total = mp2_part_total + drude_part_total

                # For the colormap, we need the "above" value
                # This is just the Drude part (trimers above d)
                hybrid_above = drude_part_total

                hybrid_cle_below.append(hybrid_total)
                hybrid_cle_above.append(hybrid_above)

            jax_ind_cle_below.append(jax_below)
            jax_ind_cle_above.append(jax_above)
            mp2_cle_below.append(mp2_below)
            mp2_cle_above.append(mp2_above)

        # Use continuous seismic colormap with standard normalization
        cmap = plt.cm.seismic
        norm = plt.Normalize(
            figure_params["colormap_min"],
            figure_params["colormap_max"]
        )

        # Scatter plot for Drude or Hybrid
        if use_hybrid:
            # Plot hybrid (MP2 below cutoff, Drude above cutoff)
            sc_drude = ax1.scatter(
                sep_distances,
                hybrid_cle_below,
                s=28,  # Marker size
                marker="o",
                label="$E^{(3)}_{\\rm below}$(MP2/aTZ) + $E^{(3)}_{\\rm above}$(JAX SAPT-FF)" if n == 1 else "",
                c=hybrid_cle_above,
                cmap=cmap,
                norm=norm,
                zorder=10,  # High z-order to be on top
                edgecolors='black',  # Black border for open circles
                facecolors='none',  # No fill color (open circles)
                linewidths=1.0
            )
        else:
            # Plot pure Drude
            sc_drude = ax1.scatter(
                sep_distances,
                jax_ind_cle_below,
                s=28,  # Marker size
                marker="o",
                label="Drude" if n == 1 else "",  # Only label on second subplot
                c=jax_ind_cle_above,
                cmap=cmap,
                norm=norm,
                zorder=10,  # High z-order to be on top
                edgecolors=(0, 0, 0, 0.1),  # Very transparent black border
                linewidths=0.5
            )

        # Scatter plot for MP2
        sc_mp2 = ax1.scatter(
            sep_distances,
            mp2_cle_below,
            c=mp2_cle_above,
            cmap=cmap,
            norm=norm,
            s=28,  # Marker size
            marker="x",
            label="$E^{(3)}_{\\rm below}$(MP2/aTZ)" if n == 1 else "",  # Only label on second subplot
            zorder=10,  # High z-order to be on top
            edgecolors=(0, 0, 0, 0.1),  # Very transparent black border
            linewidths=0.5
        )

        # Apply axis styling
        style_axes(ax1)

        # Set y-axis label for each subplot
        ax1.set_ylabel(f"{names[n]}\n$E^{{(3)}}_{{\\rm below}}$ (kJ/mol)", fontsize=12)

        # Set axis limits based on whether annotations are shown
        x_min_data = min(sep_distances)
        x_max_data = max(sep_distances)
        x_offset = 0.2  # Small offset to start slightly before data minimum

        if show_annotations:
            ax1.set_xlim(x_min_data - x_offset, 15.0)
        else:
            # Use natural x-limits based on data with small padding
            ax1.set_xlim(x_min_data - x_offset, x_max_data)

        # Conditionally add annotations and reference lines
        if show_annotations:
            # Add single horizontal reference line at MP2 asymptote
            ax1.hlines(
                y=mp2_cle_below[-1],
                xmin=0,
                xmax=15.0,
                linestyles="--",
                colors='black',
                alpha=1.0,
                linewidth=1.2,
                zorder=15  # High z-order to be in front of data
            )

            # Set reasonable y-limits based on data
            # Get all data values to determine range
            all_values = []
            all_values.extend(mp2_cle_below)
            if use_hybrid:
                all_values.extend(hybrid_cle_below)
            else:
                all_values.extend(jax_ind_cle_below)

            data_min = min(all_values)
            data_max = max(all_values)
            data_range = data_max - data_min

            # Add 10% padding on both sides for clarity
            y_min = data_min - 0.1 * data_range
            y_max = data_max + 0.1 * data_range

            ax1.set_ylim(y_min, y_max)
            y_range = y_max - y_min

            # Annotate MP2 asymptote value
            mp2_final = mp2_cle_below[-1]
            mp2_text_y = mp2_final + y_range * 0.03  # Slightly above the line

            ax1.annotate(
                f"$\\mathbf{{CLE = {mp2_final:.2f}}}$",
                xy=(13.5, mp2_final),  # Position so annotation ends ~0.5 Å from right edge
                xytext=(13.5, mp2_text_y),
                fontsize=10,
                color='black',
                ha='center',
                va='bottom',
                weight='bold'
            )

        # Remove x-axis tick labels for all but the bottom subplot
        if n < N - 1:
            ax1.tick_params(axis='x', labelbottom=False)

        # Store the last scatter plot for colorbar reference
        if n == N - 1:
            last_scatter = sc_drude

    # Set x-axis label on bottom plot
    axes[-1].set_xlabel("Minimum Monomer Separation Switchover Distance, $R^*$ (Å)", fontsize=14)

    # Create a single colorbar for all subplots with much more space and width
    fig.subplots_adjust(right=0.70)  # Reduce subplot area to make more room
    cbar_ax = fig.add_axes([1.02, 0.15, 0.03, 0.7])  # [left, bottom, width, height] - positioned just beyond normal bounds
    cbar = plt.colorbar(last_scatter, cax=cbar_ax)
    cbar.set_label("$E^{(3)}_{\\rm above}$ (kJ/mol)", fontsize=14)

    # Use default colorbar ticks and labels with bigger font
    cbar.ax.tick_params(labelsize=12)

    # Add single legend to the second subplot with enhanced styling
    legend = axes[1].legend(
        loc="lower right",
        markerscale=1.4,  # Slightly smaller legend markers
        frameon=True,
        fancybox=False,  # Sharp edges instead of rounded
        shadow=True,  # Add subtle shadow
        framealpha=1.0,  # Fully opaque background
        edgecolor='black',  # Sharp black edge
        facecolor='white',
        borderpad=0.4,  # Reduced padding
        columnspacing=0.9,  # Reduced spacing
        handletextpad=0.4,  # Reduced spacing
        borderaxespad=0.4,  # Reduced spacing
        fontsize=11  # Slightly smaller font
    )

    # Make the legend border more prominent
    legend.get_frame().set_linewidth(1.5)

    # Finalize layout
    plt.tight_layout()

    # Save figure in both PNG and SVG formats
    os.makedirs("Figures_SP", exist_ok=True)

    # Save as PNG
    output_path_png = os.path.join("Figures_SP", "CLE_MP2_Drude_Etotal.png")
    plt.savefig(output_path_png, dpi=600)
    print(f"Figure saved to: {output_path_png}")

    # Save as SVG
    output_path_svg = os.path.join("Figures_SP", "CLE_MP2_Drude_Etotal.svg")
    plt.savefig(output_path_svg, format='svg')
    print(f"Figure saved to: {output_path_svg}")

    return fig, axes

def main():
    """Main function to generate CLE figure."""

    # Generate the CLE figure with default parameters
    fig, axes = CLE_drude_mp2_figure()
    # Optional: Generate with custom parameters
    # custom_params = {
    #     "colormap_min": -1.0,
    #     "colormap_max": 1.0
    # }
    # fig, axes = CLE_drude_mp2_figure(figure_params=custom_params)


if __name__ == "__main__":
    main()
