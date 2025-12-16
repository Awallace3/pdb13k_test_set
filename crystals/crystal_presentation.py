import pandas as pd
import qcelemental as qcel
import numpy as np
import os
from cdsg_plot import error_statistics
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import AutoMinorLocator

qcml_model_dir = os.path.expanduser("~/gits/qcmlforge/models")
kcalmol_to_kjmol = qcel.constants.conversion_factor("kcal/mol", "kJ/mol")
ha_to_kjmol = qcel.constants.conversion_factor("hartree", "kJ/mol")

crystal_name_mapping = {
    "14-cyclohexanedione": "14-Cyclohexanedione",
    "acetic_acid": "Acetic Acid",
    # "adamantane": '',
    "ammonia": "Ammonia",
    # "anthracene": '', # missing becnhmark-cc
    "benzene": "Benzene",
    "cyanamide": "Cyanamide",
    "cytosine": "Cytosine",
    "ethyl_carbamate": "Ethyl Carbamate",
    "formamide": "Formamide",
    "hexamine": "Hexamine",
    "ice": "Ice",
    "imidazole": "Imidazole",
    # "naphthalene": '',
    "oxalic_acid_alpha": "Oxalic Acid (α)",
    "oxalic_acid_beta": "Oxalic Acid (β)",
    "pyrazine": "Pyrazine",
    "pyrazole": "Pyrazole",
    "succinic_acid": "Succinic Acid",
    "triazine": "Triazine",
    "trioxane": "Trioxane",
    "uracil": "Uracil",
    "urea": "Urea",
    "CO2": "Carbon Dioxide",
}


def prepare_crystal_data_intermediates(tl_N=100, uma_cutoff=6.0, sft=False):
    """
    Prepare and save intermediate dataframes (df_c_apprx and df_c_bm) for all crystals.
    This separates data preparation from plotting for faster iteration.

    Args:
        tl_N: Transfer learning N value for DES models
        uma_cutoff: Cutoff distance for UMA+AP3-LR switching
        sft: Whether to use SFT models

    Returns:
        Dictionary containing intermediate data for all crystals
    """
    # Load dataframes
    if sft:
        df_apprx = pd.read_pickle("./sft_crystals_ap2_ap3_results_mol_apprx.pkl")
        df_bm = pd.read_pickle("./sft_crystals_ap2_ap3_results_mol_bm.pkl")
    else:
        df_apprx = pd.read_pickle("./crystals_ap2_ap3_des_results_mol_apprx.pkl")
        df_bm = pd.read_pickle("./crystals_ap2_ap3_des_results_mol_bm.pkl")

    # Load SAPT model results if available
    try:
        df_sapt_bm = pd.read_pickle("./crystals_ap2_ap3_sapt_results_mol_bm.pkl")
        df_bm["AP3-SAPT2p3CCDDMP2_atz TOTAL"] = df_sapt_bm[
            "AP3-SAPT2p3CCDDMP2_atz TOTAL"
        ]
        df_bm["AP3-SAPT2p3CCDDMP2_atz DISP"] = df_sapt_bm["AP3-SAPT2p3CCDDMP2_atz DISP"]
    except FileNotFoundError:
        print("Warning: SAPT model results not found.")
        df_bm["AP3-SAPT2p3CCDDMP2_atz TOTAL"] = None
        df_bm["AP3-SAPT2p3CCDDMP2_atz DISP"] = None

    # Add derived columns
    df_bm["ap3+d4"] = df_bm["AP3 TOTAL"]
    df_apprx["ap3+d4"] = df_apprx["AP3 TOTAL"]

    df_bm["d"] = df_bm["Minimum Monomer Separations (A) CCSD(T)/CBS"]
    df_bm["ref"] = df_bm["Non-Additive MB Energy (kJ/mol) CCSD(T)/CBS"]
    df_apprx["d"] = df_apprx["Minimum Monomer Separations (A) sapt0-dz-aug"]
    df_apprx["ref"] = df_apprx["Non-Additive MB Energy (kJ/mol) sapt0-dz-aug"]

    # Process UMA models
    for i in ["uma-s-1p1", "uma-m-1p1"]:
        df_uma_bm = pd.read_pickle(f"./crystals_ap2_ap3_results_{i}_mol_bm.pkl")
        df_bm[f"{i} IE (kJ/mol)"] = df_uma_bm[f"{i} IE (kJ/mol)"]
        df_uma_apprx = pd.read_pickle(f"./crystals_ap2_ap3_results_{i}_mol_apprx.pkl")
        df_apprx[f"{i} IE (kJ/mol)"] = df_uma_apprx[f"{i} IE (kJ/mol)"]

        # UMA+AP3-LR switching logic
        uma_ap3_lr = []
        df_bm.sort_values(
            by="Minimum Monomer Separations (A) CCSD(T)/CBS", inplace=True
        )
        for n, r in df_bm.iterrows():
            if r["Minimum Monomer Separations (A) CCSD(T)/CBS"] > uma_cutoff:
                val = (
                    r["ap3_d_elst"]
                    + r["ap3_classical_ind_energy"]
                    + r["d4_s IE (kJ/mol)"]
                )
                uma_ap3_lr.append(val)
            else:
                uma_ap3_lr.append(r[f"{i} IE (kJ/mol)"])
        df_bm[f"{i}+ap3_lr IE (kJ/mol)"] = uma_ap3_lr

        uma_ap3_lr = []
        for n, r in df_apprx.iterrows():
            if r["Minimum Monomer Separations (A) sapt0-dz-aug"] > uma_cutoff:
                val = (
                    r["ap3_d_elst"]
                    + r["ap3_classical_ind_energy"]
                    + r["d4_s IE (kJ/mol)"]
                )
                uma_ap3_lr.append(val)
            else:
                uma_ap3_lr.append(r[f"{i} IE (kJ/mol)"])
        df_apprx[f"{i}+ap3_lr IE (kJ/mol)"] = uma_ap3_lr

    pd.set_option("display.max_rows", None)
    pd.set_option("display.float_format", "{:.4f}".format)

    df_bm["ap3+d4"] = df_bm["AP3 TOTAL"] + df_bm["d4_s IE (kJ/mol)"] - df_bm["AP3 DISP"]
    df_apprx["ap3+d4"] = (
        df_apprx["AP3 TOTAL"] + df_apprx["d4_s IE (kJ/mol)"] - df_apprx["AP3 DISP"]
    )

    # Create ap3-des+d4
    df_bm[f"ap3-des+d4"] = (
        df_bm[f"AP3-des-tl{tl_N} TOTAL"]
        + df_bm["d4_i IE (kJ/mol)"]
        - df_bm[f"AP3-des-tl{tl_N} DISP"]
    )
    df_apprx[f"ap3-des+d4"] = (
        df_apprx[f"AP3-des-tl{tl_N} TOTAL"]
        + df_apprx["d4_i IE (kJ/mol)"]
        - df_apprx[f"AP3-des-tl{tl_N} DISP"]
    )

    # Create AP3+D4(SAPT2+3)
    if (
        "AP3-SAPT2p3CCDDMP2_atz TOTAL" in df_bm.columns
        and df_bm["AP3-SAPT2p3CCDDMP2_atz TOTAL"].notna().any()
    ):
        df_bm["ap3_sapt2p3+d4"] = (
            df_bm["AP3-SAPT2p3CCDDMP2_atz TOTAL"]
            + df_bm["d4_i IE (kJ/mol)"]
            - df_bm["AP3-SAPT2p3CCDDMP2_atz DISP"]
        )
        print("Created AP3+D4(SAPT2+3) column for df_bm")
    else:
        df_bm["ap3_sapt2p3+d4"] = None
        print("Warning: AP3-SAPT2p3CCDDMP2_atz TOTAL not available")

    df_bm.sort_values(by="Minimum Monomer Separations (A) CCSD(T)/CBS", inplace=True)
    df_apprx.sort_values(
        by="Minimum Monomer Separations (A) sapt0-dz-aug", inplace=True
    )

    # Get unique crystals
    crystals_apprx = sorted(df_apprx["crystal apprx"].dropna().unique())
    crystals_bm = sorted(df_bm["crystal bm"].dropna().unique())
    all_crystals = sorted(list(set(crystals_apprx) | set(crystals_bm)))
    skip = ["triazine"]
    all_crystals = [c for c in all_crystals if c not in skip]

    # Compute CLE errors for multiple N values
    print("Computing CLE errors for N = 0, 1, 5, 10, 20...")
    cle_errors_by_N = {}
    for N in [0, 1, 5, 10, 20]:
        print(f"  N={N}...")
        cle_errors_by_N[N] = compute_cle_errors_for_N_all_crystals(
            N, df_apprx, df_bm, all_crystals, crystals_apprx, crystals_bm, tl_N
        )

    # Save processed dataframes and CLE errors
    intermediates = {
        "df_apprx": df_apprx,
        "df_bm": df_bm,
        "all_crystals": all_crystals,
        "crystals_apprx": crystals_apprx,
        "crystals_bm": crystals_bm,
        "tl_N": tl_N,
        "uma_cutoff": uma_cutoff,
        "cle_errors_by_N": cle_errors_by_N,
    }

    # Save to pickle
    output_file = f"./crystal_intermediates_tl{tl_N}_uma{uma_cutoff}_sft{sft}.pkl"
    pd.to_pickle(intermediates, output_file)
    print(f"Saved intermediates with CLE errors to {output_file}")

    return intermediates


def compute_cle_errors_for_N_all_crystals(
    N, df_apprx, df_bm, all_crystals, crystals_apprx, crystals_bm, tl_N
):
    """
    Compute CLE errors for a specific N value across all crystals.
    Helper function for prepare_crystal_data_intermediates().

    Returns dict with all error lists for this N value.
    """
    # Define separation distance range
    increment = 0.25
    sep_distances_full = np.arange(1.0, 20.0 + 0.05, increment)

    # Initialize error lists
    errors = {
        "ap2_sapt0": [],
        "ap3_sapt0": [],
        "ap2_des_sapt0": [],
        "ap3_des_sapt0": [],
        "uma_s_sapt0": [],
        "uma_s_ap3lr_sapt0": [],
        "uma_m_ap3lr_sapt0": [],
        "uma_m_sapt0": [],
        "ap3_d4_sapt0": [],
        "ap3_des_d4_sapt0": [],
        "opls_sapt0": [],
        "ap2_ccsd": [],
        "ap3_ccsd": [],
        "ap2_des_ccsd": [],
        "ap3_des_ccsd": [],
        "uma_s_ccsd": [],
        "uma_s_ap3lr_ccsd": [],
        "uma_m_ap3lr_ccsd": [],
        "uma_m_ccsd": [],
        "ap3_d4_ccsd": [],
        "ap3_des_d4_ccsd": [],
        "opls_ccsd": [],
        "ap3_sapt2p3_ccsd": [],
        "ap3_sapt2p3_d4_ccsd": [],
    }

    # Process each crystal
    for crystal in all_crystals:
        # Process apprx (SAPT0/aDZ reference)
        if crystal in crystals_apprx:
            df_c = df_apprx[df_apprx["crystal apprx"] == crystal].copy()
            ref_col = "Non-Additive MB Energy (kJ/mol) sapt0-dz-aug"
            num_rep_col = "Num. Rep. (#) sapt0-dz-aug"
            nmer_col = "N-mer Name apprx"
            mms_col = "Minimum Monomer Separations (A) sapt0-dz-aug"

            if ref_col in df_c.columns and "AP2 TOTAL" in df_c.columns:
                # Calculate CLE contributions for all methods
                for method_key, method_col in [
                    ("ref", ref_col),
                    ("ap2", "AP2 TOTAL"),
                    ("ap3", "AP3 TOTAL"),
                    ("ap2_des", f"AP2-des-tl{tl_N} TOTAL"),
                    ("ap3_des", f"AP3-des-tl{tl_N} TOTAL"),
                    ("uma_s", "uma-s-1p1 IE (kJ/mol)"),
                    ("uma_m", "uma-m-1p1 IE (kJ/mol)"),
                    ("uma_s_ap3lr", "uma-s-1p1+ap3_lr IE (kJ/mol)"),
                    ("uma_m_ap3lr", "uma-m-1p1+ap3_lr IE (kJ/mol)"),
                    ("ap3_d4", "ap3+d4"),
                    ("ap3_des_d4", f"ap3-des+d4"),
                    ("opls", "OPLS Interaction Energy (kJ/mol)"),
                ]:
                    df_c[f"{method_key}_cle"] = df_c.apply(
                        lambda r: r[method_col] * r[num_rep_col] / int(r[nmer_col][0])
                        if pd.notnull(r[nmer_col])
                        and method_col in r
                        and pd.notnull(r[method_col])
                        else 0,
                        axis=1,
                    )

                # Determine relevant sep_distances
                sep_distances = sep_distances_full
                for d in sep_distances_full:
                    if df_c[df_c[mms_col] < d]["ref_cle"].sum() != 0.0:
                        sep_distances = np.arange(d, sep_distances_full[-1], increment)
                        break

                # Sort and split
                df_c = df_c.sort_values(by=mms_col, ascending=True)
                df_c_N = df_c.iloc[:N]
                df_c_above = df_c.iloc[N:]

                # Compute final energies
                methods = [
                    "ap2",
                    "ap3",
                    "ap3_d4",
                    "ap3_des_d4",
                    "uma_s",
                    "uma_s_ap3lr",
                    "uma_m_ap3lr",
                    "uma_m",
                    "ap2_des",
                    "ap3_des",
                    "opls",
                ]
                energies = {m: [] for m in methods}
                ref_energies = []

                for d in sep_distances:
                    ref_N_sum = df_c_N[df_c_N[mms_col] < d]["ref_cle"].sum()
                    ref_total = df_c[df_c[mms_col] < d]["ref_cle"].sum()

                    if len(df_c_above[df_c_above[mms_col] < d]) > 0:
                        for m in methods:
                            above_sum = df_c_above[df_c_above[mms_col] < d][
                                f"{m}_cle"
                            ].sum()
                            energies[m].append(above_sum + ref_N_sum)
                    ref_energies.append(ref_total)

                if len(energies["ap2"]) > 0 and ref_energies[-1] != 0.0:
                    for m in methods:
                        errors[f"{m}_sapt0"].append(energies[m][-1] - ref_energies[-1])

        # Process bm (CCSD(T)/CBS reference)
        if crystal in crystals_bm:
            df_c = df_bm[df_bm["crystal bm"] == crystal].copy()
            ref_col = "Non-Additive MB Energy (kJ/mol) CCSD(T)/CBS"
            num_rep_col = "Num. Rep. (#) CCSD(T)/CBS"
            nmer_col = "N-mer Name bm"
            mms_col = "Minimum Monomer Separations (A) CCSD(T)/CBS"

            if ref_col in df_c.columns and "AP2 TOTAL" in df_c.columns:
                # Calculate CLE contributions for all methods
                for method_key, method_col in [
                    ("ref", ref_col),
                    ("ap2", "AP2 TOTAL"),
                    ("ap3", "AP3 TOTAL"),
                    ("ap2_des", f"AP2-des-tl{tl_N} TOTAL"),
                    ("ap3_des", f"AP3-des-tl{tl_N} TOTAL"),
                    ("uma_s", "uma-s-1p1 IE (kJ/mol)"),
                    ("uma_m", "uma-m-1p1 IE (kJ/mol)"),
                    ("uma_s_ap3lr", "uma-s-1p1+ap3_lr IE (kJ/mol)"),
                    ("uma_m_ap3lr", "uma-m-1p1+ap3_lr IE (kJ/mol)"),
                    ("ap3_d4", "ap3+d4"),
                    ("ap3_des_d4", f"ap3-des+d4"),
                    ("ap3_sapt2p3", "AP3-SAPT2p3CCDDMP2_atz TOTAL"),
                    ("ap3_sapt2p3_d4", "ap3_sapt2p3+d4"),
                    ("opls", "OPLS Interaction Energy (kJ/mol)"),
                ]:
                    df_c[f"{method_key}_cle"] = df_c.apply(
                        lambda r: r[method_col] * r[num_rep_col] / int(r[nmer_col][0])
                        if pd.notnull(r[nmer_col])
                        and method_col in r
                        and pd.notnull(r[method_col])
                        else 0,
                        axis=1,
                    )

                # Determine relevant sep_distances
                sep_distances = sep_distances_full
                for d in sep_distances_full:
                    if df_c[df_c[mms_col] < d]["ref_cle"].sum() != 0.0:
                        sep_distances = np.arange(d, sep_distances_full[-1], increment)
                        break

                # Sort and split
                df_c = df_c.sort_values(by=mms_col, ascending=True)
                df_c_N = df_c.iloc[:N]
                df_c_above = df_c.iloc[N:]

                # Compute final energies
                methods = [
                    "ap2",
                    "ap3",
                    "ap3_d4",
                    "ap3_des_d4",
                    "uma_s",
                    "uma_s_ap3lr",
                    "uma_m_ap3lr",
                    "uma_m",
                    "ap2_des",
                    "ap3_des",
                    "opls",
                    "ap3_sapt2p3",
                    "ap3_sapt2p3_d4",
                ]
                energies = {m: [] for m in methods}
                ref_energies = []

                for d in sep_distances:
                    ref_N_sum = df_c_N[df_c_N[mms_col] < d]["ref_cle"].sum()
                    ref_total = df_c[df_c[mms_col] < d]["ref_cle"].sum()

                    if len(df_c_above[df_c_above[mms_col] < d]) > 0:
                        for m in methods:
                            above_sum = df_c_above[df_c_above[mms_col] < d][
                                f"{m}_cle"
                            ].sum()
                            energies[m].append(above_sum + ref_N_sum)
                    ref_energies.append(ref_total)

                if len(energies["ap2"]) > 0 and not abs(energies["ap2"][-1]) < 8e-8:
                    for m in [
                        "ap2",
                        "ap3",
                        "ap3_d4",
                        "ap3_des_d4",
                        "uma_s",
                        "uma_s_ap3lr",
                        "uma_m_ap3lr",
                        "uma_m",
                        "ap2_des",
                        "ap3_des",
                        "opls",
                    ]:
                        errors[f"{m}_ccsd"].append(energies[m][-1] - ref_energies[-1])
                    errors["ap3_sapt2p3_ccsd"].append(
                        energies["ap3_sapt2p3"][-1] - ref_energies[-1]
                    )
                    errors["ap3_sapt2p3_d4_ccsd"].append(
                        energies["ap3_sapt2p3_d4"][-1] - ref_energies[-1]
                    )

    # Return in expected format
    return {
        "ap2_full_cle_errors_sapt0_aDZ": errors["ap2_sapt0"],
        "ap3_full_cle_errors_sapt0_aDZ": errors["ap3_sapt0"],
        "ap2_des_full_cle_errors_sapt0_aDZ": errors["ap2_des_sapt0"],
        "ap3_des_full_cle_errors_sapt0_aDZ": errors["ap3_des_sapt0"],
        "uma_s_full_cle_errors_sapt0_aDZ": errors["uma_s_sapt0"],
        "uma_s_ap3lr_full_cle_errors_sapt0_aDZ": errors["uma_s_ap3lr_sapt0"],
        "uma_m_ap3lr_full_cle_errors_sapt0_aDZ": errors["uma_m_ap3lr_sapt0"],
        "uma_m_full_cle_errors_sapt0_aDZ": errors["uma_m_sapt0"],
        "ap2_full_cle_errors_ccsd_t_CBS": errors["ap2_ccsd"],
        "ap3_full_cle_errors_ccsd_t_CBS": errors["ap3_ccsd"],
        "ap2_des_full_cle_errors_ccsd_t_CBS": errors["ap2_des_ccsd"],
        "ap3_des_full_cle_errors_ccsd_t_CBS": errors["ap3_des_ccsd"],
        "uma_s_full_cle_errors_ccsd_t_CBS": errors["uma_s_ccsd"],
        "uma_s_ap3lr_full_cle_errors_ccsd_t_CBS": errors["uma_s_ap3lr_ccsd"],
        "uma_m_ap3lr_full_cle_errors_ccsd_t_CBS": errors["uma_m_ap3lr_ccsd"],
        "uma_m_full_cle_errors_ccsd_t_CBS": errors["uma_m_ccsd"],
        "ap3_d4_full_cle_errors_ccsd_t_CBS": errors["ap3_d4_ccsd"],
        "ap3_d4_full_cle_errors_sapt0_aDZ": errors["ap3_d4_sapt0"],
        "ap3_des_d4_full_cle_errors_ccsd_t_CBS": errors["ap3_des_d4_ccsd"],
        "ap3_des_d4_full_cle_errors_sapt0_aDZ": errors["ap3_des_d4_sapt0"],
        "ap3_sapt2p3_full_cle_errors_ccsd_t_CBS": errors["ap3_sapt2p3_ccsd"],
        "ap3_sapt2p3_full_cle_errors_sapt0_aDZ": [],  # Not computed for sapt0
        "ap3_sapt2p3_d4_full_cle_errors_ccsd_t_CBS": errors["ap3_sapt2p3_d4_ccsd"],
        "ap3_sapt2p3_d4_full_cle_errors_sapt0_aDZ": [],  # Not computed for sapt0
        "opls_full_cle_errors_sapt0_aDZ": errors["opls_sapt0"],
        "opls_full_cle_errors_ccsd_t_CBS": errors["opls_ccsd"],
    }


def plot_crystal_lattice_energies_with_N(N=1, sft=False, tl_N=100, uma_cutoff=6.0):
    """
    For each crystal, plot rolling sum of CLE energy errors from all points
    above X. Creates subplots showing AP2/AP3
    """
    # Set up plot styling
    mpl.rcParams["figure.figsize"] = (10, 6)
    mpl.rcParams["font.size"] = 10
    mpl.rcParams["axes.labelsize"] = 12
    mpl.rcParams["xtick.labelsize"] = 10
    mpl.rcParams["ytick.labelsize"] = 10
    mpl.rcParams["legend.fontsize"] = 10
    mpl.rcParams["lines.linewidth"] = 2.0
    mpl.rcParams["lines.markersize"] = 6

    # Load dataframes
    if sft:
        df_apprx = pd.read_pickle("./sft_crystals_ap2_ap3_results_mol_apprx.pkl")
        df_bm = pd.read_pickle("./sft_crystals_ap2_ap3_results_mol_bm.pkl")
        # df_bm = pd.read_pickle("./sft_crystals_ap2_ap3_des_results_mol_bm.pkl")
        output_path = f"./x23_plots/CLE_all_crystals_N{N}_sft.png"
        output_violin_apprx = f"./x23_plots/N{N}_ap2_ap3_errors_vs_sapt0_sft.png"
        output_violin_bm = f"./x23_plots/N{N}_ap2_ap3_errors_vs_ccsdt_cbs_sft.png"
    else:
        # df_apprx = pd.read_pickle("./crystals_ap2_ap3_results_mol_apprx.pkl")
        # df_bm = pd.read_pickle("./crystals_ap2_ap3_results_mol_bm.pkl")
        df_apprx = pd.read_pickle("./crystals_ap2_ap3_des_results_mol_apprx.pkl")
        df_bm = pd.read_pickle("./crystals_ap2_ap3_des_results_mol_bm.pkl")
        output_path = f"./x23_plots/CLE_all_crystals_N{N}.png"
        output_violin_apprx = f"./x23_plots/N{N}_ap2_ap3_errors_vs_sapt0.png"
        output_violin_bm = f"./x23_plots/N{N}_ap2_ap3_errors_vs_ccsdt_cbs.png"

    # Load SAPT model results if available
    try:
        df_sapt_bm = pd.read_pickle("./crystals_ap2_ap3_sapt_results_mol_bm.pkl")
        # Add SAPT2+3 energies to df_bm
        df_bm["AP3-SAPT2p3CCDDMP2_atz TOTAL"] = df_sapt_bm[
            "AP3-SAPT2p3CCDDMP2_atz TOTAL"
        ]
        df_bm["AP3-SAPT2p3CCDDMP2_atz DISP"] = df_sapt_bm["AP3-SAPT2p3CCDDMP2_atz DISP"]
        # print("Loaded SAPT2+3(CCD)DMP2/aTZ model results")
    except FileNotFoundError:
        print(
            "Warning: SAPT model results not found. Run ap2_ap3_df_energies_sapt_models() first."
        )
        df_bm["AP3-SAPT2p3CCDDMP2_atz TOTAL"] = None
        df_bm["AP3-SAPT2p3CCDDMP2_atz DISP"] = None

    df_bm["ap3+d4"] = df_bm["AP3 TOTAL"]
    df_apprx["ap3+d4"] = df_apprx["AP3 TOTAL"]

    df_bm["d"] = df_bm["Minimum Monomer Separations (A) CCSD(T)/CBS"]
    df_bm["ref"] = df_bm["Non-Additive MB Energy (kJ/mol) CCSD(T)/CBS"]
    df_apprx["d"] = df_apprx["Minimum Monomer Separations (A) sapt0-dz-aug"]
    df_apprx["ref"] = df_apprx["Non-Additive MB Energy (kJ/mol) sapt0-dz-aug"]
    for i in ["uma-s-1p1", "uma-m-1p1"]:
        df_uma_bm = pd.read_pickle(f"./crystals_ap2_ap3_results_{i}_mol_bm.pkl")
        df_bm[f"{i} IE (kJ/mol)"] = df_uma_bm[f"{i} IE (kJ/mol)"]
        df_uma_apprx = pd.read_pickle(f"./crystals_ap2_ap3_results_{i}_mol_apprx.pkl")
        df_apprx[f"{i} IE (kJ/mol)"] = df_uma_apprx[f"{i} IE (kJ/mol)"]
        # where uma-m-1p1 IE (kJ/mol) 0, use ap3_d_elst+ap3_classical_ind_energy
        uma_ap3_lr = []
        df_bm.sort_values(
            by="Minimum Monomer Separations (A) CCSD(T)/CBS", inplace=True
        )
        for n, r in df_bm.iterrows():
            # if r['Minimum Monomer Separations (A) CCSD(T)/CBS'] > 6.0:
            if r["Minimum Monomer Separations (A) CCSD(T)/CBS"] > uma_cutoff:
                # if r[f"{i} IE (kJ/mol)"] == 0.0:
                # val = r["ap3_d_elst"] + r["ap3_classical_ind_energy"]
                val = (
                    r["ap3_d_elst"]
                    + r["ap3_classical_ind_energy"]
                    + r["d4_s IE (kJ/mol)"]
                    # + r['AP3 ELST']
                    # + r['AP3 INDU']
                    # + r['AP3 DISP']
                )
                uma_ap3_lr.append(val)
            else:
                uma_ap3_lr.append(r[f"{i} IE (kJ/mol)"])
        # print(df_bm[['crystal bm', 'd', 'ref', f"{i} IE (kJ/mol)", "ap3_d_elst", "d4_s IE (kJ/mol)"]])
        df_bm[f"{i}+ap3_lr IE (kJ/mol)"] = uma_ap3_lr
        uma_ap3_lr = []
        for n, r in df_apprx.iterrows():
            if r["Minimum Monomer Separations (A) sapt0-dz-aug"] > uma_cutoff:
                val = (
                    r["ap3_d_elst"]
                    + r["ap3_classical_ind_energy"]
                    + r["d4_s IE (kJ/mol)"]
                )
                uma_ap3_lr.append(val)
            else:
                uma_ap3_lr.append(r[f"{i} IE (kJ/mol)"])
        df_apprx[f"{i}+ap3_lr IE (kJ/mol)"] = uma_ap3_lr

    pd.set_option("display.max_rows", None)
    pd.set_option("display.float_format", "{:.4f}".format)
    df_bm["ap3+d4"] = df_bm["AP3 TOTAL"] + df_bm["d4_s IE (kJ/mol)"] - df_bm["AP3 DISP"]
    df_apprx["ap3+d4"] = (
        df_apprx["AP3 TOTAL"]
        # df_apprx["AP3 EXCH"]
        # + df_apprx["AP3 INDU"]
        # + r["ap3_d_elst"]
        # + r["ap3_classical_ind_energy"]
        + df_apprx["d4_s IE (kJ/mol)"]
        - df_apprx["AP3 DISP"]
    )
    # Create ap3-des+d4 using AP3-des-tl{tl_N} TOTAL and DISP
    df_bm[f"ap3-des+d4"] = (
        df_bm[f"AP3-des-tl{tl_N} TOTAL"]
        + df_bm["d4_i IE (kJ/mol)"]
        - df_bm[f"AP3-des-tl{tl_N} DISP"]
    )
    df_apprx[f"ap3-des+d4"] = (
        df_apprx[f"AP3-des-tl{tl_N} TOTAL"]
        + df_apprx["d4_i IE (kJ/mol)"]
        - df_apprx[f"AP3-des-tl{tl_N} DISP"]
    )
    # Create AP3+D4(SAPT2+3) using SAPT2+3(CCD)DMP2/aTZ energies
    if (
        "AP3-SAPT2p3CCDDMP2_atz TOTAL" in df_bm.columns
        and df_bm["AP3-SAPT2p3CCDDMP2_atz TOTAL"].notna().any()
    ):
        df_bm["ap3_sapt2p3+d4"] = (
            df_bm["AP3-SAPT2p3CCDDMP2_atz TOTAL"]
            + df_bm["d4_i IE (kJ/mol)"]
            - df_bm["AP3-SAPT2p3CCDDMP2_atz DISP"]
        )
        print("Created AP3+D4(SAPT2+3) column for df_bm")
    else:
        df_bm["ap3_sapt2p3+d4"] = None
        print("Warning: AP3-SAPT2p3CCDDMP2_atz TOTAL not available")
    df_bm.sort_values(by="Minimum Monomer Separations (A) CCSD(T)/CBS", inplace=True)
    df_apprx.sort_values(
        by="Minimum Monomer Separations (A) sapt0-dz-aug", inplace=True
    )

    # Get unique crystals
    crystals_apprx = sorted(df_apprx["crystal apprx"].dropna().unique())
    crystals_bm = sorted(df_bm["crystal bm"].dropna().unique())

    # Combine and get all unique crystals
    all_crystals = sorted(list(set(crystals_apprx) | set(crystals_bm)))
    # crystals to skip
    skip = ["triazine"]
    all_crystals = [c for c in all_crystals if c not in skip]
    N_crystals = len(all_crystals)

    # print(f"Processing {N_crystals} crystals for switchover error plots")

    # Create figure with subplots (2 columns: apprx and bm)
    fig, axes = plt.subplots(N_crystals, 2, figsize=(12, N_crystals * 2 + 2))
    if N_crystals == 1:
        axes = axes.reshape(1, -1)

    # Define separation distance range
    increment = 0.25
    sep_distances_full = np.arange(1.0, 20.0 + 0.05, increment)

    ap2_full_cle_errors_sapt0_aDZ = []
    ap3_full_cle_errors_sapt0_aDZ = []
    ap2_des_full_cle_errors_sapt0_aDZ = []
    ap3_des_full_cle_errors_sapt0_aDZ = []
    uma_s_full_cle_errors_sapt0_aDZ = []
    uma_s_ap3lr_full_cle_errors_sapt0_aDZ = []
    uma_m_ap3lr_full_cle_errors_sapt0_aDZ = []
    uma_m_full_cle_errors_sapt0_aDZ = []
    ap2_full_cle_errors_ccsd_t_CBS = []
    ap3_full_cle_errors_ccsd_t_CBS = []
    ap2_des_full_cle_errors_ccsd_t_CBS = []
    ap3_des_full_cle_errors_ccsd_t_CBS = []
    uma_s_full_cle_errors_ccsd_t_CBS = []
    uma_s_ap3lr_full_cle_errors_ccsd_t_CBS = []
    uma_m_ap3lr_full_cle_errors_ccsd_t_CBS = []
    uma_m_full_cle_errors_ccsd_t_CBS = []
    ap3_d4_full_cle_errors_ccsd_t_CBS = []
    ap3_d4_full_cle_errors_sapt0_aDZ = []
    ap3_des_d4_full_cle_errors_ccsd_t_CBS = []
    ap3_des_d4_full_cle_errors_sapt0_aDZ = []
    ap3_sapt2p3_full_cle_errors_ccsd_t_CBS = []
    ap3_sapt2p3_full_cle_errors_sapt0_aDZ = []
    ap3_sapt2p3_d4_full_cle_errors_ccsd_t_CBS = []
    ap3_sapt2p3_d4_full_cle_errors_sapt0_aDZ = []
    opls_full_cle_errors_sapt0_aDZ = []
    opls_full_cle_errors_ccsd_t_CBS = []

    # Process each crystal
    for idx, crystal in enumerate(all_crystals):
        # print(f"\nProcessing crystal {idx + 1}/{N}: {crystal}")

        # Left plot: apprx (vs SAPT0/aDZ)
        ax_apprx = axes[idx, 0]
        if crystal in crystals_apprx:
            df_c = df_apprx[df_apprx["crystal apprx"] == crystal].copy()

            # Reference method
            ref_col = "Non-Additive MB Energy (kJ/mol) sapt0-dz-aug"
            num_rep_col = "Num. Rep. (#) sapt0-dz-aug"
            nmer_col = "N-mer Name apprx"
            mms_col = "Minimum Monomer Separations (A) sapt0-dz-aug"

            if ref_col in df_c.columns and "AP2 TOTAL" in df_c.columns:
                # Calculate CLE contributions
                df_c["ref_cle"] = df_c.apply(
                    lambda r: r[ref_col] * r[num_rep_col] / int(r[nmer_col][0])
                    if pd.notnull(r[nmer_col])
                    else 0,
                    axis=1,
                )
                df_c["ap2_cle"] = df_c.apply(
                    lambda r: r["AP2 TOTAL"] * r[num_rep_col] / int(r[nmer_col][0])
                    if pd.notnull(r[nmer_col])
                    else 0,
                    axis=1,
                )
                df_c["ap3_cle"] = df_c.apply(
                    lambda r: r["AP3 TOTAL"] * r[num_rep_col] / int(r[nmer_col][0])
                    if pd.notnull(r[nmer_col])
                    else 0,
                    axis=1,
                )
                df_c["ap2_des_cle"] = df_c.apply(
                    lambda r: r[f"AP2-des-tl{tl_N} TOTAL"]
                    * r[num_rep_col]
                    / int(r[nmer_col][0])
                    if pd.notnull(r[nmer_col])
                    else 0,
                    axis=1,
                )
                df_c["ap3_des_cle"] = df_c.apply(
                    lambda r: r[f"AP3-des-tl{tl_N} TOTAL"]
                    * r[num_rep_col]
                    / int(r[nmer_col][0])
                    if pd.notnull(r[nmer_col])
                    else 0,
                    axis=1,
                )
                df_c["uma-s-1p1_cle"] = df_c.apply(
                    lambda r: r["uma-s-1p1 IE (kJ/mol)"]
                    * r[num_rep_col]
                    / int(r[nmer_col][0])
                    if pd.notnull(r[nmer_col]) and "uma-s-1p1 IE (kJ/mol)" in r
                    else 0,
                    axis=1,
                )
                df_c["uma-m-1p1_cle"] = df_c.apply(
                    lambda r: r["uma-m-1p1 IE (kJ/mol)"]
                    * r[num_rep_col]
                    / int(r[nmer_col][0])
                    if pd.notnull(r[nmer_col]) and "uma-m-1p1 IE (kJ/mol)" in r
                    else 0,
                    axis=1,
                )
                df_c["uma-s-1p1+ap3_lr_cle"] = df_c.apply(
                    lambda r: r["uma-s-1p1+ap3_lr IE (kJ/mol)"]
                    * r[num_rep_col]
                    / int(r[nmer_col][0])
                    if pd.notnull(r[nmer_col]) and "uma-s-1p1+ap3_lr IE (kJ/mol)" in r
                    else 0,
                    axis=1,
                )
                df_c["uma-m-1p1+ap3_lr_cle"] = df_c.apply(
                    lambda r: r["uma-m-1p1+ap3_lr IE (kJ/mol)"]
                    * r[num_rep_col]
                    / int(r[nmer_col][0])
                    if pd.notnull(r[nmer_col]) and "uma-m-1p1+ap3_lr IE (kJ/mol)" in r
                    else 0,
                    axis=1,
                )
                df_c["ap3_d4_cle"] = df_c.apply(
                    lambda r: r["ap3+d4"] * r[num_rep_col] / int(r[nmer_col][0])
                    if pd.notnull(r[nmer_col]) and "ap3+d4" in r
                    else 0,
                    axis=1,
                )
                df_c["ap3_des_d4_cle"] = df_c.apply(
                    lambda r: r["ap3-des+d4"] * r[num_rep_col] / int(r[nmer_col][0])
                    if pd.notnull(r[nmer_col]) and "ap3-des+d4" in r
                    else 0,
                    axis=1,
                )
                df_c["opls_cle"] = df_c.apply(
                    lambda r: r["OPLS Interaction Energy (kJ/mol)"]
                    * r[num_rep_col]
                    / int(r[nmer_col][0])
                    if pd.notnull(r[nmer_col])
                    and "OPLS Interaction Energy (kJ/mol)" in r
                    else 0,
                    axis=1,
                )
                # Determine relevant sep_distances per crystal starting point based on ref_cle non-zero
                for d in sep_distances_full:
                    ref_below = df_c[df_c[mms_col] < d]["ref_cle"].sum()
                    if ref_below != 0.0:
                        sep_distances = np.arange(d, sep_distances_full[-1], increment)
                        break

                ap2_2b_energies = []
                ap3_2b_energies = []
                ap3_d4_2b_energies = []
                ap3_des_d4_2b_energies = []
                uma_s_2b_energies = []
                uma_s_ap3lr_2b_energies = []
                uma_m_ap3lr_2b_energies = []
                uma_m_2b_energies = []
                ref_2b_energies = []
                ap2_des_2b_energies = []
                ap3_des_2b_energies = []
                opls_2b_energies = []
                df_c = df_c.sort_values(by=mms_col, ascending=True)
                df_c_N = df_c.iloc[:N]
                df_c_above = df_c.iloc[N:]
                ml_sep_distances = []
                for d in sep_distances:
                    ref_N = df_c_N[df_c_N[mms_col] < d]["ref_cle"].sum()
                    ap2_above = df_c_above[df_c_above[mms_col] < d]["ap2_cle"].sum()
                    ap3_above = df_c_above[df_c_above[mms_col] < d]["ap3_cle"].sum()
                    ap3_d4_above = df_c_above[df_c_above[mms_col] < d][
                        "ap3_d4_cle"
                    ].sum()
                    ap2_des_above = df_c_above[df_c_above[mms_col] < d][
                        "ap2_des_cle"
                    ].sum()
                    ap3_des_above = df_c_above[df_c_above[mms_col] < d][
                        "ap3_des_cle"
                    ].sum()
                    ap3_des_d4_above = df_c_above[df_c_above[mms_col] < d][
                        "ap3_des_d4_cle"
                    ].sum()
                    uma_s_above = df_c_above[df_c_above[mms_col] < d][
                        "uma-s-1p1_cle"
                    ].sum()
                    uma_s_ap3lr_above = df_c_above[df_c_above[mms_col] < d][
                        "uma-s-1p1+ap3_lr_cle"
                    ].sum()
                    uma_m_ap3lr_above = df_c_above[df_c_above[mms_col] < d][
                        "uma-m-1p1+ap3_lr_cle"
                    ].sum()
                    uma_m_above = df_c_above[df_c_above[mms_col] < d][
                        "uma-m-1p1_cle"
                    ].sum()
                    opls_above = df_c_above[df_c_above[mms_col] < d]["opls_cle"].sum()

                    ref_below = df_c[df_c[mms_col] < d]["ref_cle"].sum()

                    if len(df_c_above[df_c_above[mms_col] < d]) > 0:
                        ml_sep_distances.append(d)
                        ap2_2b_energies.append(ap2_above + ref_N)
                        ap3_2b_energies.append(ap3_above + ref_N)
                        uma_s_2b_energies.append(uma_s_above + ref_N)
                        uma_s_ap3lr_2b_energies.append(uma_s_ap3lr_above + ref_N)
                        uma_m_ap3lr_2b_energies.append(uma_m_ap3lr_above + ref_N)
                        uma_m_2b_energies.append(uma_m_above + ref_N)
                        ap2_des_2b_energies.append(ap2_des_above + ref_N)
                        ap3_des_2b_energies.append(ap3_des_above + ref_N)
                        ap3_d4_2b_energies.append(ap3_d4_above + ref_N)
                        ap3_des_d4_2b_energies.append(ap3_des_d4_above + ref_N)
                        opls_2b_energies.append(opls_above + ref_N)
                    ref_2b_energies.append(ref_below)

                # Plot
                if ref_2b_energies[-1] != 0.0:
                    # print(f"{crystal=}, {ml_sep_distances[0]}")
                    ax_apprx.plot(
                        ml_sep_distances,
                        ap2_2b_energies,
                        "o-",
                        label="AP2",
                        markersize=4,
                        linewidth=1.5,
                        alpha=0.8,
                    )
                    ax_apprx.plot(
                        ml_sep_distances,
                        ap3_2b_energies,
                        "s-",
                        label="AP3",
                        markersize=4,
                        linewidth=1.5,
                        alpha=0.8,
                    )
                    ax_apprx.plot(
                        ml_sep_distances,
                        ap3_d4_2b_energies,
                        "s-",
                        label="AP3+D4(S)",
                        markersize=4,
                        linewidth=1.5,
                        alpha=0.8,
                    )
                    ax_apprx.plot(
                        ml_sep_distances,
                        opls_2b_energies,
                        "d-",
                        c="grey",
                        label="OPLS",
                        markersize=4,
                        linewidth=1.5,
                        alpha=0.8,
                    )
                    # ax_apprx.plot(
                    #     ml_sep_distances,
                    #     uma_s_2b_energies,
                    #     "^-",
                    #     label="UMA-s",
                    #     markersize=4,
                    #     linewidth=1.5,
                    #     alpha=0.8,
                    # )
                    # ax_apprx.plot(
                    #     ml_sep_distances,
                    #     uma_m_2b_energies,
                    #     "^-",
                    #     label="UMA-m",
                    #     markersize=4,
                    #     linewidth=1.5,
                    #     alpha=0.8,
                    # )
                    # ax_apprx.plot(
                    #     ml_sep_distances,
                    #     uma_s_ap3lr_2b_energies,
                    #     "v-",
                    #     label="UMA-s+AP3-LR",
                    #     markersize=4,
                    #     linewidth=1.5,
                    #     alpha=0.8,
                    # )
                    # ax_apprx.plot(
                    #     ml_sep_distances,
                    #     uma_m_ap3lr_2b_energies,
                    #     "v-",
                    #     label="UMA-m+AP3-LR",
                    #     markersize=4,
                    #     linewidth=1.5,
                    #     alpha=0.8,
                    # )
                    # ax_apprx.plot(
                    #     ml_sep_distances,
                    #     ap2_des_2b_energies,
                    #     "o-",
                    #     label=f"AP2-DES{tl_N}",
                    #     markersize=4,
                    #     linewidth=1.5,
                    #     alpha=0.8,
                    # )
                    # ax_apprx.plot(
                    #     ml_sep_distances,
                    #     ap3_des_2b_energies,
                    #     "s-",
                    #     label=f"AP3-DES{tl_N}",
                    #     markersize=4,
                    #     linewidth=1.5,
                    #     alpha=0.8,
                    # )
                    ax_apprx.plot(
                        sep_distances,
                        ref_2b_energies,
                        "-",
                        color="red",
                        label="SAPT0/aDZ",
                        markersize=4,
                        linewidth=1.5,
                        alpha=0.8,
                    )
                # ax_apprx.axhline(
                #     y=0, color="black", linestyle="--", linewidth=1.0, alpha=0.5
                # )
                ax_apprx.axvline(
                    x=6.0, color="gray", linestyle="--", linewidth=1.0, alpha=0.5
                )
                ax_apprx.set_ylabel("CLE Error (kJ/mol)", fontsize=10)
                ax_apprx.set_title(f"{crystal}:{len(df_c)}\nvs SAPT0/aDZ", fontsize=10)

                if idx == 0:
                    ax_apprx.legend(loc="best", fontsize=8)

                if len(ap2_2b_energies) > 0:
                    ap2_full_cle_errors_sapt0_aDZ.append(
                        ap2_2b_energies[-1] - ref_2b_energies[-1]
                    )
                    ap3_full_cle_errors_sapt0_aDZ.append(
                        ap3_2b_energies[-1] - ref_2b_energies[-1]
                    )
                    uma_s_full_cle_errors_sapt0_aDZ.append(
                        uma_s_2b_energies[-1] - ref_2b_energies[-1]
                    )
                    uma_m_full_cle_errors_sapt0_aDZ.append(
                        uma_m_2b_energies[-1] - ref_2b_energies[-1]
                    )
                    uma_s_ap3lr_full_cle_errors_sapt0_aDZ.append(
                        uma_s_ap3lr_2b_energies[-1] - ref_2b_energies[-1]
                    )
                    uma_m_ap3lr_full_cle_errors_sapt0_aDZ.append(
                        uma_m_ap3lr_2b_energies[-1] - ref_2b_energies[-1]
                    )
                    ap2_des_full_cle_errors_sapt0_aDZ.append(
                        ap2_des_2b_energies[-1] - ref_2b_energies[-1]
                    )
                    ap3_des_full_cle_errors_sapt0_aDZ.append(
                        ap3_des_2b_energies[-1] - ref_2b_energies[-1]
                    )
                    ap3_d4_full_cle_errors_sapt0_aDZ.append(
                        ap3_d4_2b_energies[-1] - ref_2b_energies[-1]
                    )
                    ap3_des_d4_full_cle_errors_sapt0_aDZ.append(
                        ap3_des_d4_2b_energies[-1] - ref_2b_energies[-1]
                    )
                    opls_full_cle_errors_sapt0_aDZ.append(
                        opls_2b_energies[-1] - ref_2b_energies[-1]
                    )
                # ax_apprx.set_ylim(-5, 5)

        # Right plot: bm (vs CCSD(T)/CBS)
        ax_bm = axes[idx, 1]
        if crystal in crystals_bm:
            df_c = df_bm[df_bm["crystal bm"] == crystal].copy()

            # Reference method
            ref_col = "Non-Additive MB Energy (kJ/mol) CCSD(T)/CBS"
            num_rep_col = "Num. Rep. (#) CCSD(T)/CBS"
            nmer_col = "N-mer Name bm"
            mms_col = "Minimum Monomer Separations (A) CCSD(T)/CBS"
            df_c.sort_values(by=mms_col, inplace=True)

            if ref_col in df_c.columns and "AP2 TOTAL" in df_c.columns:
                # Calculate CLE contributions
                df_c["ref_cle"] = df_c.apply(
                    lambda r: r[ref_col] * r[num_rep_col] / int(r[nmer_col][0])
                    if pd.notnull(r[nmer_col])
                    else 0,
                    axis=1,
                )
                df_c["ap2_cle"] = df_c.apply(
                    lambda r: r["AP2 TOTAL"] * r[num_rep_col] / int(r[nmer_col][0])
                    if pd.notnull(r[nmer_col])
                    else 0,
                    axis=1,
                )
                df_c["ap3_cle"] = df_c.apply(
                    lambda r: r["AP3 TOTAL"] * r[num_rep_col] / int(r[nmer_col][0])
                    if pd.notnull(r[nmer_col])
                    else 0,
                    axis=1,
                )
                df_c["ap2_des_cle"] = df_c.apply(
                    lambda r: r[f"AP2-des-tl{tl_N} TOTAL"]
                    * r[num_rep_col]
                    / int(r[nmer_col][0])
                    if pd.notnull(r[nmer_col])
                    else 0,
                    axis=1,
                )
                df_c["ap3_des_cle"] = df_c.apply(
                    lambda r: r[f"AP3-des-tl{tl_N} TOTAL"]
                    * r[num_rep_col]
                    / int(r[nmer_col][0])
                    if pd.notnull(r[nmer_col])
                    else 0,
                    axis=1,
                )
                df_c["uma-s-1p1_cle"] = df_c.apply(
                    lambda r: r["uma-s-1p1 IE (kJ/mol)"]
                    * r[num_rep_col]
                    / int(r[nmer_col][0])
                    if pd.notnull(r[nmer_col]) and "uma-s-1p1 IE (kJ/mol)" in r
                    else 0,
                    axis=1,
                )
                df_c["uma-m-1p1_cle"] = df_c.apply(
                    lambda r: r["uma-m-1p1 IE (kJ/mol)"]
                    * r[num_rep_col]
                    / int(r[nmer_col][0])
                    if pd.notnull(r[nmer_col]) and "uma-m-1p1 IE (kJ/mol)" in r
                    else 0,
                    axis=1,
                )
                df_c["uma-s-1p1+ap3_lr_cle"] = df_c.apply(
                    lambda r: r["uma-s-1p1+ap3_lr IE (kJ/mol)"]
                    * r[num_rep_col]
                    / int(r[nmer_col][0])
                    if pd.notnull(r[nmer_col]) and "uma-s-1p1+ap3_lr IE (kJ/mol)" in r
                    else 0,
                    axis=1,
                )
                df_c["uma-m-1p1+ap3_lr_cle"] = df_c.apply(
                    lambda r: r["uma-m-1p1+ap3_lr IE (kJ/mol)"]
                    * r[num_rep_col]
                    / int(r[nmer_col][0])
                    if pd.notnull(r[nmer_col]) and "uma-m-1p1+ap3_lr IE (kJ/mol)" in r
                    else 0,
                    axis=1,
                )
                df_c["ap3_d4_cle"] = df_c.apply(
                    lambda r: r["ap3+d4"] * r[num_rep_col] / int(r[nmer_col][0])
                    if pd.notnull(r[nmer_col]) and "ap3+d4" in r
                    else 0,
                    axis=1,
                )
                df_c["ap3_des_d4_cle"] = df_c.apply(
                    lambda r: r["ap3-des+d4"] * r[num_rep_col] / int(r[nmer_col][0])
                    if pd.notnull(r[nmer_col]) and "ap3-des+d4" in r
                    else 0,
                    axis=1,
                )
                df_c["ap3_sapt2p3_cle"] = df_c.apply(
                    lambda r: r["AP3-SAPT2p3CCDDMP2_atz TOTAL"]
                    * r[num_rep_col]
                    / int(r[nmer_col][0])
                    if pd.notnull(r[nmer_col]) and "AP3-SAPT2p3CCDDMP2_atz TOTAL" in r
                    else 0,
                    axis=1,
                )
                df_c["ap3_sapt2p3_d4_cle"] = df_c.apply(
                    lambda r: r["ap3_sapt2p3+d4"] * r[num_rep_col] / int(r[nmer_col][0])
                    if pd.notnull(r[nmer_col]) and "ap3_sapt2p3+d4" in r
                    else 0,
                    axis=1,
                )
                df_c["opls_cle"] = df_c.apply(
                    lambda r: r["OPLS Interaction Energy (kJ/mol)"]
                    * r[num_rep_col]
                    / int(r[nmer_col][0])
                    if pd.notnull(r[nmer_col])
                    and "OPLS Interaction Energy (kJ/mol)" in r
                    else 0,
                    axis=1,
                )
                # Determine relevant sep_distances per crystal starting point based on ref_cle non-zero
                for d in sep_distances_full:
                    ref_below = df_c[df_c[mms_col] < d]["ref_cle"].sum()
                    if ref_below != 0.0:
                        sep_distances = np.arange(d, sep_distances_full[-1], increment)
                        break

                ap2_2b_energies = []
                ap3_2b_energies = []
                ap3_d4_2b_energies = []
                ap3_des_d4_2b_energies = []
                ap3_sapt2p3_2b_energies = []
                ap3_sapt2p3_d4_2b_energies = []
                uma_s_2b_energies = []
                uma_s_ap3lr_2b_energies = []
                uma_m_ap3lr_2b_energies = []
                uma_m_2b_energies = []
                ref_2b_energies = []
                ap2_des_2b_energies = []
                ap3_des_2b_energies = []
                opls_2b_energies = []
                df_c = df_c.sort_values(by=mms_col, ascending=True)
                df_c_N = df_c.iloc[:N]
                df_c_above = df_c.iloc[N:]
                ml_sep_distances = []
                for d in sep_distances:
                    ref_N = df_c_N[df_c_N[mms_col] < d]["ref_cle"].sum()
                    ap2_above = df_c_above[df_c_above[mms_col] < d]["ap2_cle"].sum()
                    ap3_above = df_c_above[df_c_above[mms_col] < d]["ap3_cle"].sum()
                    ap3_d4_above = df_c_above[df_c_above[mms_col] < d][
                        "ap3_d4_cle"
                    ].sum()
                    ap2_des_above = df_c_above[df_c_above[mms_col] < d][
                        "ap2_des_cle"
                    ].sum()
                    ap3_des_above = df_c_above[df_c_above[mms_col] < d][
                        "ap3_des_cle"
                    ].sum()
                    ap3_des_d4_above = df_c_above[df_c_above[mms_col] < d][
                        "ap3_des_d4_cle"
                    ].sum()
                    ap3_sapt2p3_above = df_c_above[df_c_above[mms_col] < d][
                        "ap3_sapt2p3_cle"
                    ].sum()
                    ap3_sapt2p3_d4_above = df_c_above[df_c_above[mms_col] < d][
                        "ap3_sapt2p3_d4_cle"
                    ].sum()
                    uma_s_above = df_c_above[df_c_above[mms_col] < d][
                        "uma-s-1p1_cle"
                    ].sum()
                    uma_s_ap3lr_above = df_c_above[df_c_above[mms_col] < d][
                        "uma-s-1p1+ap3_lr_cle"
                    ].sum()
                    uma_m_ap3lr_above = df_c_above[df_c_above[mms_col] < d][
                        "uma-m-1p1+ap3_lr_cle"
                    ].sum()
                    uma_m_above = df_c_above[df_c_above[mms_col] < d][
                        "uma-m-1p1_cle"
                    ].sum()
                    opls_above = df_c_above[df_c_above[mms_col] < d]["opls_cle"].sum()

                    ref_below = df_c[df_c[mms_col] < d]["ref_cle"].sum()

                    if len(df_c_above[df_c_above[mms_col] < d]) > 0:
                        ml_sep_distances.append(d)
                        ap2_2b_energies.append(ap2_above + ref_N)
                        ap3_2b_energies.append(ap3_above + ref_N)
                        uma_s_2b_energies.append(uma_s_above + ref_N)
                        uma_s_ap3lr_2b_energies.append(uma_s_ap3lr_above + ref_N)
                        uma_m_ap3lr_2b_energies.append(uma_m_ap3lr_above + ref_N)
                        uma_m_2b_energies.append(uma_m_above + ref_N)
                        ap2_des_2b_energies.append(ap2_des_above + ref_N)
                        ap3_des_2b_energies.append(ap3_des_above + ref_N)
                        ap3_d4_2b_energies.append(ap3_d4_above + ref_N)
                        ap3_des_d4_2b_energies.append(ap3_des_d4_above + ref_N)
                        ap3_sapt2p3_2b_energies.append(ap3_sapt2p3_above + ref_N)
                        ap3_sapt2p3_d4_2b_energies.append(ap3_sapt2p3_d4_above + ref_N)
                        opls_2b_energies.append(opls_above + ref_N)
                    ref_2b_energies.append(ref_below)

                # Plot
                if ref_2b_energies[-1] != 0.0:
                    # ax_bm.plot(
                    #     ml_sep_distances,
                    #     ap2_2b_energies,
                    #     "o-",
                    #     label="AP2",
                    #     markersize=4,
                    #     linewidth=1.5,
                    #     alpha=0.8,
                    # )
                    # ax_bm.plot(
                    #     ml_sep_distances,
                    #     ap3_2b_energies,
                    #     "s-",
                    #     label="AP3",
                    #     markersize=4,
                    #     linewidth=1.5,
                    #     alpha=0.8,
                    # )
                    # ax_bm.plot(
                    #     ml_sep_distances,
                    #     ap2_des_2b_energies,
                    #     "o-",
                    #     label=f"AP2-DES{tl_N}",
                    #     markersize=4,
                    #     linewidth=1.5,
                    #     alpha=0.8,
                    # )
                    # ax_bm.plot(
                    #     ml_sep_distances,
                    #     ap3_des_2b_energies,
                    #     "s-",
                    #     label=f"AP3-DES{tl_N}",
                    #     markersize=4,
                    #     linewidth=1.5,
                    #     alpha=0.8,
                    # )
                    ax_bm.plot(
                        ml_sep_distances,
                        ap3_d4_2b_energies,
                        "v-",
                        label=f"AP3+D4(S)",
                        markersize=4,
                        linewidth=1.5,
                        alpha=0.8,
                    )
                    # ax_bm.plot(
                    #     ml_sep_distances,
                    #     ap3_sapt2p3_2b_energies,
                    #     "<-",
                    #     label=f"AP3-SAPT2+3",
                    #     markersize=4,
                    #     linewidth=1.5,
                    #     alpha=0.8,
                    # )
                    ax_bm.plot(
                        ml_sep_distances,
                        ap3_sapt2p3_d4_2b_energies,
                        ">-",
                        label=f"AP3(SAPT2)+D4(I)",
                        markersize=4,
                        linewidth=1.5,
                        alpha=0.8,
                    )
                    ax_bm.plot(
                        ml_sep_distances,
                        opls_2b_energies,
                        "d-",
                        label="OPLS",
                        c="grey",
                        markersize=4,
                        linewidth=1.5,
                        alpha=0.8,
                    )
                    # ax_bm.plot(
                    #     ml_sep_distances,
                    #     uma_s_2b_energies,
                    #     "^-",
                    #     label="UMA-s",
                    #     markersize=4,
                    #     linewidth=1.5,
                    #     alpha=0.8,
                    # )
                    ax_bm.plot(
                        ml_sep_distances,
                        uma_m_2b_energies,
                        "^-",
                        label="UMA-m",
                        markersize=4,
                        linewidth=1.5,
                        alpha=0.8,
                    )
                    # ax_bm.plot(
                    #     ml_sep_distances,
                    #     uma_s_ap3lr_2b_energies,
                    #     "v-",
                    #     label="UMA-s+AP3-LR",
                    #     markersize=4,
                    #     linewidth=1.5,
                    #     alpha=0.8,
                    # )
                    ax_bm.plot(
                        ml_sep_distances,
                        uma_m_ap3lr_2b_energies,
                        "v-",
                        label="UMA-m+AP3-LR+D4(S)",
                        markersize=4,
                        linewidth=1.5,
                        alpha=0.8,
                    )
                    ax_bm.plot(
                        sep_distances,
                        ref_2b_energies,
                        "k-",
                        label="CCSD(T)/CBS",
                        markersize=4,
                        linewidth=1.5,
                        alpha=0.8,
                    )
                    ax_bm.set_xlim(sep_distances_full[0], sep_distances_full[-1])
                    # vertical line at 6.0 A
                    ax_bm.axvline(
                        x=6.0, color="gray", linestyle="--", linewidth=1.0, alpha=0.5
                    )
                    # ax_bm.axhline(
                    #     y=0, color="black", linestyle="--", linewidth=1.0, alpha=0.5
                    # )
                ax_bm.set_ylabel("CLE Error (kJ/mol)", fontsize=10)
                ax_bm.set_title(f"{crystal}:{len(df_c)}\nvs CCSD(T)/CBS", fontsize=10)

                if idx == 0:
                    ax_bm.legend(loc="best", fontsize=8)

                if not abs(ap2_2b_energies[-1]) < 8e-8:
                    ap2_full_cle_errors_ccsd_t_CBS.append(
                        ap2_2b_energies[-1] - ref_2b_energies[-1]
                    )
                    ap3_full_cle_errors_ccsd_t_CBS.append(
                        ap3_2b_energies[-1] - ref_2b_energies[-1]
                    )
                    uma_s_full_cle_errors_ccsd_t_CBS.append(
                        uma_s_2b_energies[-1] - ref_2b_energies[-1]
                    )
                    uma_m_full_cle_errors_ccsd_t_CBS.append(
                        uma_m_2b_energies[-1] - ref_2b_energies[-1]
                    )
                    uma_s_ap3lr_full_cle_errors_ccsd_t_CBS.append(
                        uma_s_ap3lr_2b_energies[-1] - ref_2b_energies[-1]
                    )
                    uma_m_ap3lr_full_cle_errors_ccsd_t_CBS.append(
                        uma_m_ap3lr_2b_energies[-1] - ref_2b_energies[-1]
                    )
                    ap2_des_full_cle_errors_ccsd_t_CBS.append(
                        ap2_des_2b_energies[-1] - ref_2b_energies[-1]
                    )
                    ap3_des_full_cle_errors_ccsd_t_CBS.append(
                        ap3_des_2b_energies[-1] - ref_2b_energies[-1]
                    )
                    ap3_d4_full_cle_errors_ccsd_t_CBS.append(
                        ap3_d4_2b_energies[-1] - ref_2b_energies[-1]
                    )
                    ap3_des_d4_full_cle_errors_ccsd_t_CBS.append(
                        ap3_des_d4_2b_energies[-1] - ref_2b_energies[-1]
                    )
                    opls_full_cle_errors_ccsd_t_CBS.append(
                        opls_2b_energies[-1] - ref_2b_energies[-1]
                    )

                    ap3_sapt2p3_full_cle_errors_ccsd_t_CBS.append(
                        ap3_sapt2p3_2b_energies[-1] - ref_2b_energies[-1]
                    )
                    ap3_sapt2p3_d4_full_cle_errors_ccsd_t_CBS.append(
                        ap3_sapt2p3_d4_2b_energies[-1] - ref_2b_energies[-1]
                    )
                    # difference between uma_s and uma_s_ap3lr
                    # print(
                    #     f"{crystal:19s} bm|UMA-s: {uma_s_2b_energies[-1]:.4f}, UMA-s+AP3-LR: {uma_s_ap3lr_2b_energies[-1]:.4f}, diff: {uma_s_2b_energies[-1] - uma_s_ap3lr_2b_energies[-1]:.8f}, ref: {ref_2b_energies[-1]:.4f} kJ/mol"
                    # )
                # ax_bm.set_ylim(-5, 5)

        # Style axes
        for ax in [ax_apprx, ax_bm]:
            ax.grid(True, alpha=0.3, linestyle=":")
            ax.tick_params(which="major", direction="in", top=True, right=True)

            # Only show x-label on bottom row
            if idx == N_crystals - 1:
                ax.set_xlabel("Min. Mon. Sep. (Å)", fontsize=11)
            else:
                ax.tick_params(axis="x", labelbottom=False)

    # Adjust layout
    plt.tight_layout()
    # print("Error CLE statistics")
    # mae_ap2_sapt = np.sum(np.abs(np.array(ap2_full_cle_errors_sapt0_aDZ))) / len(
    #     ap2_full_cle_errors_sapt0_aDZ
    # )
    # mae_ap3_sapt = np.sum(np.abs(np.array(ap3_full_cle_errors_sapt0_aDZ))) / len(
    #     ap3_full_cle_errors_sapt0_aDZ
    # )
    # mae_ap2_ccsd = np.sum(np.abs(np.array(ap2_full_cle_errors_ccsd_t_CBS))) / len(
    #     ap2_full_cle_errors_ccsd_t_CBS
    # )
    # mae_ap3_ccsd = np.sum(np.abs(np.array(ap3_full_cle_errors_ccsd_t_CBS))) / len(
    #     ap3_full_cle_errors_ccsd_t_CBS
    # )
    mae_uma_lr_ccsd = np.sum(
        np.abs(np.array(uma_m_ap3lr_full_cle_errors_ccsd_t_CBS))
    ) / len(uma_m_ap3lr_full_cle_errors_ccsd_t_CBS)
    print(f"{mae_uma_lr_ccsd=:.4f} kJ/mol")
    # print(f"{mae_ap2_sapt=:.4f} kJ/mol")
    # print(f"{mae_ap3_sapt=:.4f} kJ/mol")
    # print(f"{mae_ap2_ccsd=:.4f} kJ/mol")
    # print(f"{mae_ap3_ccsd=:.4f} kJ/mol")

    # Save figure
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\nFigure saved to: {output_path}")
    # error_df1 = pd.DataFrame(
    #     {
    #         "AP2 vs SAPT0 error": ap2_full_cle_errors_sapt0_aDZ,
    #         "AP3 vs SAPT0 error": ap3_full_cle_errors_sapt0_aDZ,
    #         "AP3-D4 vs SAPT0 error": ap3_d4_full_cle_errors_sapt0_aDZ,
    #         # f"AP2-DES-tl{tl_N} vs SAPT0 error": ap2_des_full_cle_errors_sapt0_aDZ,
    #         # f"AP3-DES-tl{tl_N} vs SAPT0 error": ap3_des_full_cle_errors_sapt0_aDZ,
    #         # f"AP3-DES-tl{tl_N}+D4 vs SAPT0 error": ap3_des_d4_full_cle_errors_sapt0_aDZ,
    #         # "UMA-s vs SAPT0 error": uma_s_full_cle_errors_sapt0_aDZ,
    #         "UMA-m vs SAPT0 error": uma_m_full_cle_errors_sapt0_aDZ,
    #         # "UMA-s+AP3-LR vs SAPT0 error": uma_s_ap3lr_full_cle_errors_sapt0_aDZ,
    #         "UMA-m+AP3-LR vs SAPT0 error": uma_m_ap3lr_full_cle_errors_sapt0_aDZ,
    #         "OPLS vs SAPT0 error": opls_full_cle_errors_sapt0_aDZ,
    #     }
    # )
    #
    # # Prepare df1 for violin plot
    # dfs1 = [
    #     {
    #         "df": error_df1,
    #         "basis": "",
    #         "label": "SAPT0/aDZ Reference",
    #         "ylim": [[-10, 10]],
    #     }
    # ]
    #
    # # Labels and columns for df1 (multiple methods)
    # df1_labels = {
    #     "AP2": "AP2 vs SAPT0 error",
    #     "AP3": "AP3 vs SAPT0 error",
    #     "AP3-D4": "AP3-D4 vs SAPT0 error",
    #     # f"AP2-DES-tl{tl_N}": f"AP2-DES-tl{tl_N} vs SAPT0 error",
    #     # f"AP3-DES-tl{tl_N}": f"AP3-DES-tl{tl_N} vs SAPT0 error",
    #     # f"AP3-DES-tl{tl_N}+D4": f"AP3-DES-tl{tl_N}+D4 vs SAPT0 error",
    #     # "UMA-s": "UMA-s vs SAPT0 error",
    #     "UMA-m": "UMA-m vs SAPT0 error",
    #     # "UMA-s+AP3-LR": "UMA-s+AP3-LR vs SAPT0 error",
    #     "UMA-m+AP3-LR": "UMA-m+AP3-LR vs SAPT0 error",
    #     "OPLS": "OPLS vs SAPT0 error",
    # }
    #
    # method_cols = [
    #     col
    #     for col in df_apprx.columns
    #     if "Non-Additive MB Energy (kJ/mol)" in col and "sapt0-dz-aug" not in col
    # ]
    # # Add other methods if available
    # for method_col in method_cols[:3]:
    #     method_name = method_col.replace("Non-Additive MB Energy (kJ/mol) ", "")
    #     if f"{method_name} error" in df_apprx.columns:
    #         df1_labels[method_name] = f"{method_name} error"

    error_df2 = pd.DataFrame(
        {
            "AP2 vs CCSD(T)/CBS error": ap2_full_cle_errors_ccsd_t_CBS,
            "AP3 vs CCSD(T)/CBS error": ap3_full_cle_errors_ccsd_t_CBS,
            "AP3+D4(S) vs CCSD(T)/CBS error": ap3_d4_full_cle_errors_ccsd_t_CBS,
            # f"AP2-DES-tl{tl_N} vs CCSD(T)/CBS error": ap2_des_full_cle_errors_ccsd_t_CBS,
            # f"AP3-DES-tl{tl_N} vs CCSD(T)/CBS error": ap3_des_full_cle_errors_ccsd_t_CBS,
            # f"AP3-DES-tl{tl_N}+D4 vs CCSD(T)/CBS error": ap3_des_d4_full_cle_errors_ccsd_t_CBS,
            # "UMA-s vs CCSD(T)/CBS error": uma_s_full_cle_errors_ccsd_t_CBS,
            "UMA-m vs CCSD(T)/CBS error": uma_m_full_cle_errors_ccsd_t_CBS,
            # "UMA-s+AP3-LR vs CCSD(T)/CBS error": uma_s_ap3lr_full_cle_errors_ccsd_t_CBS,
            "UMA-m+AP3-LR vs CCSD(T)/CBS error": uma_m_ap3lr_full_cle_errors_ccsd_t_CBS,
            "OPLS vs CCSD(T)/CBS error": opls_full_cle_errors_ccsd_t_CBS,
            # "AP3(SAPT2+3) vs CCSD(T)/CBS error": ap3_sapt2p3_full_cle_errors_ccsd_t_CBS,
            "AP3(SAPT2+3)+D4(I) vs CCSD(T)/CBS error": ap3_sapt2p3_d4_full_cle_errors_ccsd_t_CBS,
        }
    )

    # Prepare df2 for violin plot
    dfs2 = [
        {
            "df": error_df2,
            "basis": "",
            "label": "CCSD(T)/CBS Reference",
            "ylim": [[-10, 10]],
        }
    ]

    # Labels and columns for df2 (only AP2 and AP3 vs CCSD(T)/CBS)
    df2_labels = {
        "AP2": "AP2 vs CCSD(T)/CBS error",
        "AP3": "AP3 vs CCSD(T)/CBS error",
        # f"AP2-DES-tl{tl_N}": f"AP2-DES-tl{tl_N} vs CCSD(T)/CBS error",
        # f"AP3-DES-tl{tl_N}": f"AP3-DES-tl{tl_N} vs CCSD(T)/CBS error",
        # f"AP3-DES-tl{tl_N}+D4": f"AP3-DES-tl{tl_N}+D4 vs CCSD(T)/CBS error",
        "AP3+D4(S)": "AP3+D4(S) vs CCSD(T)/CBS error",
        # "UMA-s": "UMA-s vs CCSD(T)/CBS error",
        "UMA-m": "UMA-m vs CCSD(T)/CBS error",
        # "UMA-s+AP3-LR": "UMA-s+AP3-LR vs CCSD(T)/CBS error",
        "UMA-m\\\\+AP3-LR+D4(S)": "UMA-m+AP3-LR vs CCSD(T)/CBS error",
        "OPLS": "OPLS vs CCSD(T)/CBS error",
        # "AP3(SAPT2+3)": "AP3(SAPT2+3) vs CCSD(T)/CBS error",
        "AP3(SFT)+D4(I)": "AP3(SAPT2+3)+D4(I) vs CCSD(T)/CBS error",
    }

    # Create violin plot for df1
    # print("\nCreating violin plot for df1 (approximate methods)...")
    # error_statistics.violin_plot_table_multi_SAPT_components(
    #     dfs1,
    #     df_labels_and_columns_total=df1_labels,
    #     output_filename=output_violin_apprx,
    #     grid_heights=[0.3, 1.0],
    #     grid_widths=[1],
    #     legend_loc="upper left",
    #     annotations_texty=0.3,
    #     figure_size=(6, 2.5),
    #     add_title=False,
    #     name_violin=False,
    #     ylabel=r"Error (kJ$\cdot$mol$^{-1}$)",
    # )
    # Create violin plot for df2
    print("\nCreating violin plot for df2 (benchmark comparison)...")
    error_statistics.violin_plot_table_multi_SAPT_components(
        dfs2,
        df_labels_and_columns_total=df2_labels,
        output_filename=output_violin_bm,
        grid_heights=[0.3, 1.0],
        grid_widths=[1],
        legend_loc="upper left",
        annotations_texty=0.3,
        figure_size=(4, 2.5),
        add_title=False,
        name_violin=False,
        ylabel=r"Error (kJ$\cdot$mol$^{-1}$)",
    )
    return mae_uma_lr_ccsd


def plot_from_intermediates(N=1, intermediates_file=None, sft=False):
    """
    Create crystal lattice energy plots from pre-computed intermediates.
    This is faster than plot_crystal_lattice_energies_with_N since data is pre-processed.

    Args:
        N: Number of closest pairs to use reference method for
        intermediates_file: Path to pickled intermediates (if None, will try to find it)
        sft: Whether using SFT models
    """
    # Load intermediates
    if intermediates_file is None:
        # Try to find the file automatically
        import glob

        pattern = f"./crystal_intermediates_*_sft{sft}.pkl"
        files = glob.glob(pattern)
        if not files:
            raise FileNotFoundError(
                f"No intermediate files found matching {pattern}. "
                "Run prepare_crystal_data_intermediates() first."
            )
        intermediates_file = sorted(files)[-1]  # Use most recent
        print(f"Using intermediates file: {intermediates_file}")

    intermediates = pd.read_pickle(intermediates_file)
    df_apprx = intermediates["df_apprx"]
    df_bm = intermediates["df_bm"]
    all_crystals = intermediates["all_crystals"]
    crystals_apprx = intermediates["crystals_apprx"]
    crystals_bm = intermediates["crystals_bm"]
    tl_N = intermediates["tl_N"]
    uma_cutoff = intermediates["uma_cutoff"]

    # Set up plot styling
    mpl.rcParams["figure.figsize"] = (10, 6)
    mpl.rcParams["font.size"] = 10
    mpl.rcParams["axes.labelsize"] = 12
    mpl.rcParams["xtick.labelsize"] = 10
    mpl.rcParams["ytick.labelsize"] = 10
    mpl.rcParams["legend.fontsize"] = 10
    mpl.rcParams["lines.linewidth"] = 2.0
    mpl.rcParams["lines.markersize"] = 6

    # Set output paths
    if sft:
        output_path = f"./x23_plots/CLE_all_crystals_N{N}_sft.png"
        output_violin_bm = f"./x23_plots/N{N}_ap2_ap3_errors_vs_ccsdt_cbs_sft.png"
    else:
        output_path = f"./x23_plots/CLE_all_crystals_N{N}.png"
        output_violin_bm = f"./x23_plots/N{N}_ap2_ap3_errors_vs_ccsdt_cbs.png"

    N_crystals = len(all_crystals)

    # Create figure with subplots (2 columns: apprx and bm)
    fig, axes = plt.subplots(N_crystals, 2, figsize=(12, N_crystals * 2 + 2))
    if N_crystals == 1:
        axes = axes.reshape(1, -1)

    # Define separation distance range
    increment = 0.25
    sep_distances_full = np.arange(1.0, 20.0 + 0.05, increment)

    # Initialize error accumulation lists
    ap2_full_cle_errors_sapt0_aDZ = []
    ap3_full_cle_errors_sapt0_aDZ = []
    ap2_full_cle_errors_ccsd_t_CBS = []
    ap3_full_cle_errors_ccsd_t_CBS = []
    uma_m_ap3lr_full_cle_errors_ccsd_t_CBS = []
    uma_m_full_cle_errors_ccsd_t_CBS = []
    ap3_d4_full_cle_errors_ccsd_t_CBS = []
    ap3_d4_full_cle_errors_sapt0_aDZ = []
    ap3_sapt2p3_d4_full_cle_errors_ccsd_t_CBS = []
    opls_full_cle_errors_sapt0_aDZ = []
    opls_full_cle_errors_ccsd_t_CBS = []

    # Process each crystal (using same logic as original function but with pre-loaded data)
    for idx, crystal in enumerate(all_crystals):
        # Left plot: apprx (vs SAPT0/aDZ)
        ax_apprx = axes[idx, 0]
        if crystal in crystals_apprx:
            df_c = df_apprx[df_apprx["crystal apprx"] == crystal].copy()

            ref_col = "Non-Additive MB Energy (kJ/mol) sapt0-dz-aug"
            num_rep_col = "Num. Rep. (#) sapt0-dz-aug"
            nmer_col = "N-mer Name apprx"
            mms_col = "Minimum Monomer Separations (A) sapt0-dz-aug"

            if ref_col in df_c.columns and "AP2 TOTAL" in df_c.columns:
                # Calculate CLE contributions
                df_c["ref_cle"] = df_c.apply(
                    lambda r: r[ref_col] * r[num_rep_col] / int(r[nmer_col][0])
                    if pd.notnull(r[nmer_col])
                    else 0,
                    axis=1,
                )
                df_c["ap2_cle"] = df_c.apply(
                    lambda r: r["AP2 TOTAL"] * r[num_rep_col] / int(r[nmer_col][0])
                    if pd.notnull(r[nmer_col])
                    else 0,
                    axis=1,
                )
                df_c["ap3_cle"] = df_c.apply(
                    lambda r: r["AP3 TOTAL"] * r[num_rep_col] / int(r[nmer_col][0])
                    if pd.notnull(r[nmer_col])
                    else 0,
                    axis=1,
                )
                df_c["ap3_d4_cle"] = df_c.apply(
                    lambda r: r["ap3+d4"] * r[num_rep_col] / int(r[nmer_col][0])
                    if pd.notnull(r[nmer_col]) and "ap3+d4" in r
                    else 0,
                    axis=1,
                )
                df_c["opls_cle"] = df_c.apply(
                    lambda r: r["OPLS Interaction Energy (kJ/mol)"]
                    * r[num_rep_col]
                    / int(r[nmer_col][0])
                    if pd.notnull(r[nmer_col])
                    and "OPLS Interaction Energy (kJ/mol)" in r
                    else 0,
                    axis=1,
                )

                # Determine relevant sep_distances per crystal
                for d in sep_distances_full:
                    ref_below = df_c[df_c[mms_col] < d]["ref_cle"].sum()
                    if ref_below != 0.0:
                        sep_distances = np.arange(d, sep_distances_full[-1], increment)
                        break

                ap2_2b_energies = []
                ap3_2b_energies = []
                ap3_d4_2b_energies = []
                ref_2b_energies = []
                opls_2b_energies = []

                df_c = df_c.sort_values(by=mms_col, ascending=True)
                df_c_N = df_c.iloc[:N]
                df_c_above = df_c.iloc[N:]
                ml_sep_distances = []

                for d in sep_distances:
                    ref_N = df_c_N[df_c_N[mms_col] < d]["ref_cle"].sum()
                    ap2_above = df_c_above[df_c_above[mms_col] < d]["ap2_cle"].sum()
                    ap3_above = df_c_above[df_c_above[mms_col] < d]["ap3_cle"].sum()
                    ap3_d4_above = df_c_above[df_c_above[mms_col] < d][
                        "ap3_d4_cle"
                    ].sum()
                    opls_above = df_c_above[df_c_above[mms_col] < d]["opls_cle"].sum()
                    ref_below = df_c[df_c[mms_col] < d]["ref_cle"].sum()

                    if len(df_c_above[df_c_above[mms_col] < d]) > 0:
                        ml_sep_distances.append(d)
                        ap2_2b_energies.append(ap2_above + ref_N)
                        ap3_2b_energies.append(ap3_above + ref_N)
                        ap3_d4_2b_energies.append(ap3_d4_above + ref_N)
                        opls_2b_energies.append(opls_above + ref_N)
                    ref_2b_energies.append(ref_below)

                # Plot
                if ref_2b_energies[-1] != 0.0:
                    ax_apprx.plot(
                        ml_sep_distances,
                        ap2_2b_energies,
                        "o-",
                        label="AP2",
                        markersize=4,
                        linewidth=1.5,
                        alpha=0.8,
                    )
                    ax_apprx.plot(
                        ml_sep_distances,
                        ap3_2b_energies,
                        "s-",
                        label="AP3",
                        markersize=4,
                        linewidth=1.5,
                        alpha=0.8,
                    )
                    ax_apprx.plot(
                        ml_sep_distances,
                        ap3_d4_2b_energies,
                        "s-",
                        label="AP3+D4(S)",
                        markersize=4,
                        linewidth=1.5,
                        alpha=0.8,
                    )
                    ax_apprx.plot(
                        ml_sep_distances,
                        opls_2b_energies,
                        "d-",
                        c="grey",
                        label="OPLS",
                        markersize=4,
                        linewidth=1.5,
                        alpha=0.8,
                    )
                    ax_apprx.plot(
                        sep_distances,
                        ref_2b_energies,
                        "-",
                        color="red",
                        label="SAPT0/aDZ",
                        markersize=4,
                        linewidth=1.5,
                        alpha=0.8,
                    )

                ax_apprx.axvline(
                    x=6.0, color="gray", linestyle="--", linewidth=1.0, alpha=0.5
                )
                ax_apprx.set_ylabel("CLE Error (kJ/mol)", fontsize=10)
                ax_apprx.set_title(f"{crystal}:{len(df_c)}\nvs SAPT0/aDZ", fontsize=10)

                if idx == 0:
                    ax_apprx.legend(loc="best", fontsize=8)

                if len(ap2_2b_energies) > 0:
                    ap2_full_cle_errors_sapt0_aDZ.append(
                        ap2_2b_energies[-1] - ref_2b_energies[-1]
                    )
                    ap3_full_cle_errors_sapt0_aDZ.append(
                        ap3_2b_energies[-1] - ref_2b_energies[-1]
                    )
                    ap3_d4_full_cle_errors_sapt0_aDZ.append(
                        ap3_d4_2b_energies[-1] - ref_2b_energies[-1]
                    )
                    opls_full_cle_errors_sapt0_aDZ.append(
                        opls_2b_energies[-1] - ref_2b_energies[-1]
                    )

        # Right plot: bm (vs CCSD(T)/CBS)
        ax_bm = axes[idx, 1]
        if crystal in crystals_bm:
            df_c = df_bm[df_bm["crystal bm"] == crystal].copy()

            ref_col = "Non-Additive MB Energy (kJ/mol) CCSD(T)/CBS"
            num_rep_col = "Num. Rep. (#) CCSD(T)/CBS"
            nmer_col = "N-mer Name bm"
            mms_col = "Minimum Monomer Separations (A) CCSD(T)/CBS"
            df_c.sort_values(by=mms_col, inplace=True)

            if ref_col in df_c.columns and "AP2 TOTAL" in df_c.columns:
                # Calculate CLE contributions
                df_c["ref_cle"] = df_c.apply(
                    lambda r: r[ref_col] * r[num_rep_col] / int(r[nmer_col][0])
                    if pd.notnull(r[nmer_col])
                    else 0,
                    axis=1,
                )
                df_c["ap2_cle"] = df_c.apply(
                    lambda r: r["AP2 TOTAL"] * r[num_rep_col] / int(r[nmer_col][0])
                    if pd.notnull(r[nmer_col])
                    else 0,
                    axis=1,
                )
                df_c["ap3_cle"] = df_c.apply(
                    lambda r: r["AP3 TOTAL"] * r[num_rep_col] / int(r[nmer_col][0])
                    if pd.notnull(r[nmer_col])
                    else 0,
                    axis=1,
                )
                df_c["ap3_d4_cle"] = df_c.apply(
                    lambda r: r["ap3+d4"] * r[num_rep_col] / int(r[nmer_col][0])
                    if pd.notnull(r[nmer_col]) and "ap3+d4" in r
                    else 0,
                    axis=1,
                )
                df_c["uma-m-1p1_cle"] = df_c.apply(
                    lambda r: r["uma-m-1p1 IE (kJ/mol)"]
                    * r[num_rep_col]
                    / int(r[nmer_col][0])
                    if pd.notnull(r[nmer_col]) and "uma-m-1p1 IE (kJ/mol)" in r
                    else 0,
                    axis=1,
                )
                df_c["uma-m-1p1+ap3_lr_cle"] = df_c.apply(
                    lambda r: r["uma-m-1p1+ap3_lr IE (kJ/mol)"]
                    * r[num_rep_col]
                    / int(r[nmer_col][0])
                    if pd.notnull(r[nmer_col]) and "uma-m-1p1+ap3_lr IE (kJ/mol)" in r
                    else 0,
                    axis=1,
                )
                df_c["ap3_sapt2p3_d4_cle"] = df_c.apply(
                    lambda r: r["ap3_sapt2p3+d4"] * r[num_rep_col] / int(r[nmer_col][0])
                    if pd.notnull(r[nmer_col]) and "ap3_sapt2p3+d4" in r
                    else 0,
                    axis=1,
                )
                df_c["opls_cle"] = df_c.apply(
                    lambda r: r["OPLS Interaction Energy (kJ/mol)"]
                    * r[num_rep_col]
                    / int(r[nmer_col][0])
                    if pd.notnull(r[nmer_col])
                    and "OPLS Interaction Energy (kJ/mol)" in r
                    else 0,
                    axis=1,
                )

                # Determine relevant sep_distances per crystal
                for d in sep_distances_full:
                    ref_below = df_c[df_c[mms_col] < d]["ref_cle"].sum()
                    if ref_below != 0.0:
                        sep_distances = np.arange(d, sep_distances_full[-1], increment)
                        break

                ap2_2b_energies = []
                ap3_2b_energies = []
                ap3_d4_2b_energies = []
                ap3_sapt2p3_d4_2b_energies = []
                uma_m_2b_energies = []
                uma_m_ap3lr_2b_energies = []
                ref_2b_energies = []
                opls_2b_energies = []

                df_c = df_c.sort_values(by=mms_col, ascending=True)
                df_c_N = df_c.iloc[:N]
                df_c_above = df_c.iloc[N:]
                ml_sep_distances = []

                for d in sep_distances:
                    ref_N = df_c_N[df_c_N[mms_col] < d]["ref_cle"].sum()
                    ap2_above = df_c_above[df_c_above[mms_col] < d]["ap2_cle"].sum()
                    ap3_above = df_c_above[df_c_above[mms_col] < d]["ap3_cle"].sum()
                    ap3_d4_above = df_c_above[df_c_above[mms_col] < d][
                        "ap3_d4_cle"
                    ].sum()
                    ap3_sapt2p3_d4_above = df_c_above[df_c_above[mms_col] < d][
                        "ap3_sapt2p3_d4_cle"
                    ].sum()
                    uma_m_above = df_c_above[df_c_above[mms_col] < d][
                        "uma-m-1p1_cle"
                    ].sum()
                    uma_m_ap3lr_above = df_c_above[df_c_above[mms_col] < d][
                        "uma-m-1p1+ap3_lr_cle"
                    ].sum()
                    opls_above = df_c_above[df_c_above[mms_col] < d]["opls_cle"].sum()
                    ref_below = df_c[df_c[mms_col] < d]["ref_cle"].sum()

                    if len(df_c_above[df_c_above[mms_col] < d]) > 0:
                        ml_sep_distances.append(d)
                        ap2_2b_energies.append(ap2_above + ref_N)
                        ap3_2b_energies.append(ap3_above + ref_N)
                        uma_m_2b_energies.append(uma_m_above + ref_N)
                        uma_m_ap3lr_2b_energies.append(uma_m_ap3lr_above + ref_N)
                        ap3_d4_2b_energies.append(ap3_d4_above + ref_N)
                        ap3_sapt2p3_d4_2b_energies.append(ap3_sapt2p3_d4_above + ref_N)
                        opls_2b_energies.append(opls_above + ref_N)
                    ref_2b_energies.append(ref_below)

                # Plot
                if ref_2b_energies[-1] != 0.0:
                    ax_bm.plot(
                        ml_sep_distances,
                        ap3_d4_2b_energies,
                        "v-",
                        label="AP3+D4(S)",
                        markersize=4,
                        linewidth=1.5,
                        alpha=0.8,
                    )
                    ax_bm.plot(
                        ml_sep_distances,
                        ap3_sapt2p3_d4_2b_energies,
                        ">-",
                        label="AP3(SAPT2)+D4(I)",
                        markersize=4,
                        linewidth=1.5,
                        alpha=0.8,
                    )
                    ax_bm.plot(
                        ml_sep_distances,
                        opls_2b_energies,
                        "d-",
                        label="OPLS",
                        c="grey",
                        markersize=4,
                        linewidth=1.5,
                        alpha=0.8,
                    )
                    ax_bm.plot(
                        ml_sep_distances,
                        uma_m_2b_energies,
                        "^-",
                        label="UMA-m",
                        markersize=4,
                        linewidth=1.5,
                        alpha=0.8,
                    )
                    ax_bm.plot(
                        ml_sep_distances,
                        uma_m_ap3lr_2b_energies,
                        "v-",
                        label="UMA-m+AP3-LR+D4(S)",
                        markersize=4,
                        linewidth=1.5,
                        alpha=0.8,
                    )
                    ax_bm.plot(
                        sep_distances,
                        ref_2b_energies,
                        "k-",
                        label="CCSD(T)/CBS",
                        markersize=4,
                        linewidth=1.5,
                        alpha=0.8,
                    )
                    ax_bm.set_xlim(sep_distances_full[0], sep_distances_full[-1])
                    ax_bm.axvline(
                        x=6.0, color="gray", linestyle="--", linewidth=1.0, alpha=0.5
                    )

                ax_bm.set_ylabel("CLE Error (kJ/mol)", fontsize=10)
                ax_bm.set_title(f"{crystal}:{len(df_c)}\nvs CCSD(T)/CBS", fontsize=10)

                if idx == 0:
                    ax_bm.legend(loc="best", fontsize=8)

                if not abs(ap2_2b_energies[-1]) < 8e-8:
                    ap2_full_cle_errors_ccsd_t_CBS.append(
                        ap2_2b_energies[-1] - ref_2b_energies[-1]
                    )
                    ap3_full_cle_errors_ccsd_t_CBS.append(
                        ap3_2b_energies[-1] - ref_2b_energies[-1]
                    )
                    uma_m_full_cle_errors_ccsd_t_CBS.append(
                        uma_m_2b_energies[-1] - ref_2b_energies[-1]
                    )
                    uma_m_ap3lr_full_cle_errors_ccsd_t_CBS.append(
                        uma_m_ap3lr_2b_energies[-1] - ref_2b_energies[-1]
                    )
                    ap3_d4_full_cle_errors_ccsd_t_CBS.append(
                        ap3_d4_2b_energies[-1] - ref_2b_energies[-1]
                    )
                    opls_full_cle_errors_ccsd_t_CBS.append(
                        opls_2b_energies[-1] - ref_2b_energies[-1]
                    )
                    ap3_sapt2p3_d4_full_cle_errors_ccsd_t_CBS.append(
                        ap3_sapt2p3_d4_2b_energies[-1] - ref_2b_energies[-1]
                    )

        # Style axes
        for ax in [ax_apprx, ax_bm]:
            ax.grid(True, alpha=0.3, linestyle=":")
            ax.tick_params(which="major", direction="in", top=True, right=True)

            if idx == N_crystals - 1:
                ax.set_xlabel("Min. Mon. Sep. (Å)", fontsize=11)
            else:
                ax.tick_params(axis="x", labelbottom=False)

    # Adjust layout
    plt.tight_layout()

    mae_uma_lr_ccsd = np.sum(
        np.abs(np.array(uma_m_ap3lr_full_cle_errors_ccsd_t_CBS))
    ) / len(uma_m_ap3lr_full_cle_errors_ccsd_t_CBS)
    print(f"{mae_uma_lr_ccsd=:.4f} kJ/mol")

    # Save figure
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\nFigure saved to: {output_path}")

    # Create error dataframe for violin plot
    error_df2 = pd.DataFrame(
        {
            "AP2 vs CCSD(T)/CBS error": ap2_full_cle_errors_ccsd_t_CBS,
            "AP3 vs CCSD(T)/CBS error": ap3_full_cle_errors_ccsd_t_CBS,
            "AP3+D4(S) vs CCSD(T)/CBS error": ap3_d4_full_cle_errors_ccsd_t_CBS,
            "UMA-m vs CCSD(T)/CBS error": uma_m_full_cle_errors_ccsd_t_CBS,
            "UMA-m+AP3-LR vs CCSD(T)/CBS error": uma_m_ap3lr_full_cle_errors_ccsd_t_CBS,
            "OPLS vs CCSD(T)/CBS error": opls_full_cle_errors_ccsd_t_CBS,
            "AP3(SAPT2+3)+D4(I) vs CCSD(T)/CBS error": ap3_sapt2p3_d4_full_cle_errors_ccsd_t_CBS,
        }
    )

    if N==0:
        ylim = [-20, 20]
    else:
        ylim = [-10, 10]
    dfs2 = [
        {
            "df": error_df2,
            "basis": "",
            "label": f"QM={N},CCSD(T)/CBS Reference",
            "ylim": [ylim],
        }
    ]

    df2_labels = {
        "AP2": "AP2 vs CCSD(T)/CBS error",
        # "AP3": "AP3 vs CCSD(T)/CBS error",
        "AP3+D4(S)": "AP3+D4(S) vs CCSD(T)/CBS error",
        "UMA-m": "UMA-m vs CCSD(T)/CBS error",
        "UMA-m\\\\+AP3-LR+D4(S)": "UMA-m+AP3-LR vs CCSD(T)/CBS error",
        "OPLS": "OPLS vs CCSD(T)/CBS error",
        # "AP3(SFT)+D4(I)": "AP3(SAPT2+3)+D4(I) vs CCSD(T)/CBS error",
    }

    print("\nCreating violin plot for benchmark comparison...")
    error_statistics.violin_plot_table_multi_SAPT_components(
        dfs2,
        df_labels_and_columns_total=df2_labels,
        output_filename=output_violin_bm,
        grid_heights=[0.3, 1.0],
        grid_widths=[1],
        legend_loc="upper left",
        annotations_texty=0.3,
        figure_size=(4, 2.5),
        add_title=False,
        name_violin=False,
        ylabel=r"Error (kJ$\cdot$mol$^{-1}$)",
    )

    return mae_uma_lr_ccsd


def plot_from_intermediates_slides(
    N=1, intermediates_file=None, sft=False, v="bm", crystals=None, output_path=None
):
    """
    Create a compact N/2x2 grid plot for presentations (slides).

    This function generates cleaner plots than plot_from_intermediates by:
    - Showing only selected crystals (not all)
    - Using a 2x2 or Nx2 grid layout
    - Focusing on either 'apprx' or 'bm' data (not both)
    - Optimized sizing for presentation slides

    Args:
        N: Number of closest pairs to use reference method for (0, 1, 5, 10, 20)
        intermediates_file: Path to pickled intermediates (if None, auto-detects)
        sft: Whether using SFT models
        v: Which dataset to plot - 'apprx' (vs SAPT0/aDZ) or 'bm' (vs CCSD(T)/CBS)
        crystals: List of crystal names to plot (if None, uses first 4 crystals)
        output_path: Where to save the figure (if None, auto-generates)

    Returns:
        None (saves figure to disk)
    """
    # Load intermediates
    if intermediates_file is None:
        import glob

        pattern = f"./crystal_intermediates_*_sft{sft}.pkl"
        files = glob.glob(pattern)
        if not files:
            raise FileNotFoundError(
                f"No intermediate files found matching {pattern}. "
                "Run prepare_crystal_data_intermediates() first."
            )
        intermediates_file = sorted(files)[-1]
        print(f"Using intermediates file: {intermediates_file}")

    intermediates = pd.read_pickle(intermediates_file)

    # Check if CLE errors are pre-computed
    if "cle_errors_by_N" in intermediates and N in intermediates["cle_errors_by_N"]:
        print(f"Using pre-computed CLE errors for N={N}")
        use_precomputed = True
    else:
        print(f"Warning: No pre-computed errors for N={N}, will compute on-the-fly")
        use_precomputed = False

    df_apprx = intermediates["df_apprx"]
    df_bm = intermediates["df_bm"]
    all_crystals = intermediates["all_crystals"]
    crystals_apprx = intermediates["crystals_apprx"]
    crystals_bm = intermediates["crystals_bm"]
    tl_N = intermediates["tl_N"]

    # Validate version parameter
    if v not in ["apprx", "bm"]:
        raise ValueError(f"v must be 'apprx' or 'bm', got '{v}'")

    # Select crystals to plot
    if crystals is None:
        # Default: use first 4 crystals that have data for the selected version
        available = crystals_apprx if v == "apprx" else crystals_bm
        crystals = [c for c in all_crystals if c in available][:4]

    # Validate crystals
    available = crystals_apprx if v == "apprx" else crystals_bm
    invalid = [c for c in crystals if c not in available]
    if invalid:
        raise ValueError(f"Crystals {invalid} not available in '{v}' dataset")

    n_crystals = len(crystals)
    if n_crystals == 0:
        raise ValueError("No crystals to plot")

    # Calculate grid layout
    n_cols = 2
    n_rows = (n_crystals + 1) // 2  # Ceiling division

    # Set up plot styling
    mpl.rcParams["font.size"] = 9
    mpl.rcParams["axes.labelsize"] = 10
    mpl.rcParams["xtick.labelsize"] = 8
    mpl.rcParams["ytick.labelsize"] = 8
    mpl.rcParams["legend.fontsize"] = 7
    mpl.rcParams["lines.linewidth"] = 1.5
    mpl.rcParams["lines.markersize"] = 4

    # Create figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, n_rows * 2.5))
    if n_crystals == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)

    # Set output path
    if output_path is None:
        suffix = "_sft" if sft else ""
        output_path = f"./x23_plots/CLE_{v}_N{N}_slides{suffix}.png"

    # Define separation distance range
    increment = 0.3
    sep_distances_full = np.arange(1.0, 15.0 + 0.05, increment)

    # Select dataset and columns
    if v == "apprx":
        df = df_apprx
        crystal_col = "crystal apprx"
        ref_col = "Non-Additive MB Energy (kJ/mol) sapt0-dz-aug"
        num_rep_col = "Num. Rep. (#) sapt0-dz-aug"
        nmer_col = "N-mer Name apprx"
        mms_col = "Minimum Monomer Separations (A) sapt0-dz-aug"
        ref_label = "SAPT0/aDZ"
    else:  # v == 'bm'
        df = df_bm
        crystal_col = "crystal bm"
        ref_col = "Non-Additive MB Energy (kJ/mol) CCSD(T)/CBS"
        num_rep_col = "Num. Rep. (#) CCSD(T)/CBS"
        nmer_col = "N-mer Name bm"
        mms_col = "Minimum Monomer Separations (A) CCSD(T)/CBS"
        ref_label = "CCSD(T)/CBS"

    # Process each crystal
    for idx, crystal in enumerate(crystals):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]

        df_c = df[df[crystal_col] == crystal].copy()

        if ref_col not in df_c.columns or "AP2 TOTAL" not in df_c.columns:
            ax.text(
                0.5,
                0.5,
                f"No data for {crystal}",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title(f"{crystal}", fontsize=10)
            continue

        # Calculate CLE contributions
        df_c["ref_cle"] = df_c.apply(
            lambda r: r[ref_col] * r[num_rep_col] / int(r[nmer_col][0])
            if pd.notnull(r[nmer_col])
            else 0,
            axis=1,
        )

        methods = {
            "AP2": "AP2 TOTAL",
            "AP3": "AP3 TOTAL",
            "AP3+D4(S)": "ap3+d4",
            "UMA-m": "uma-m-1p1 IE (kJ/mol)",
            "UMA-m+AP3-LR+D4(S)": "uma-m-1p1+ap3_lr IE (kJ/mol)",
            "OPLS": "OPLS Interaction Energy (kJ/mol)",
        }

        # Add SAPT2+3 methods for bm dataset
        if v == "bm":
            methods["AP3(SFT)+D4(I)"] = "ap3_sapt2p3+d4"

        # Calculate CLE for each method
        for method_name, method_col in methods.items():
            if method_col in df_c.columns:
                df_c[f"{method_name}_cle"] = df_c.apply(
                    lambda r: r[method_col] * r[num_rep_col] / int(r[nmer_col][0])
                    if pd.notnull(r[nmer_col]) and pd.notnull(r[method_col])
                    else 0,
                    axis=1,
                )

        # Determine starting distance (where ref_cle becomes non-zero)
        sep_distances = sep_distances_full
        for d in sep_distances_full:
            if df_c[df_c[mms_col] < d]["ref_cle"].sum() != 0.0:
                sep_distances = np.arange(d, sep_distances_full[-1], increment)
                break

        # Sort by distance and split into N closest and rest
        df_c = df_c.sort_values(by=mms_col, ascending=True)
        df_c_N = df_c.iloc[:N]
        df_c_above = df_c.iloc[N:]

        # Compute cumulative lattice energies
        ref_energies = []
        method_energies = {
            name: [] for name in methods.keys() if f"{name}_cle" in df_c.columns
        }
        ml_sep_distances = []
        ref_final_cle = df_c["ref_cle"].sum()
        for d in sep_distances:
            ref_N_sum = df_c_N[df_c_N[mms_col] < d]["ref_cle"].sum()
            ref_total = df_c[df_c[mms_col] < d]["ref_cle"].sum()

            if len(df_c_above[df_c_above[mms_col] < d]) > 0:
                ml_sep_distances.append(d)
                for method_name in method_energies.keys():
                    above_sum = df_c_above[df_c_above[mms_col] < d][
                        f"{method_name}_cle"
                    ].sum()
                    method_energies[method_name].append(above_sum + ref_N_sum)
            ref_energies.append(ref_total)

        # Plot
        if len(ref_energies) > 0 and ref_energies[-1] != 0.0:
            # Define colors and markers for key methods
            plot_config = {
                "AP2": {"marker": "v", "color": None, "alpha": 0.7},
                "AP3": {"marker": "v", "color": None, "alpha": 1.0},
                "AP3+D4(S)": {"marker": "v", "color": None, "alpha": 1.0},
                "UMA-m": {"marker": "D", "color": None, "alpha": 1.0},
                "UMA-m+AP3-LR+D4(S)": {"marker": "D", "color": None, "alpha": 1.0},
                "AP3(SFT)+D4(I)": {"marker": ">", "color": None, "alpha": 1.0},
                "OPLS": {"marker": "^", "color": "grey", "alpha": 1.0},
            }

            # Plot methods (skip AP2 and UMA-m for cleaner plots)
            # skip_methods = ["AP2", "UMA-m"] if v == "bm" else ["AP2"]
            skip_methods = ["AP3", "AP3(SFT)+D4(I)"] if v == "bm" else ["AP3"]

            for method_name, energies in method_energies.items():
                if method_name in skip_methods:
                    continue
                if len(energies) > 0:
                    config = plot_config.get(
                        method_name, {"marker": "o", "color": None, "alpha": 1.0}
                    )
                    ax.plot(
                        ml_sep_distances,
                        energies,
                        marker=config["marker"],
                        color=config["color"],
                        label=method_name,
                        markersize=5,
                        linewidth=1.2,
                        alpha=config["alpha"],
                    )

            # Plot reference
            ax.plot(
                sep_distances,
                ref_energies,
                "-",
                color="red" if v == "apprx" else "black",
                label=ref_label,
                linewidth=2.5,
                alpha=0.8,
            )

            # Add vertical line at 6.0 Å
            ax.axvline(x=6.0, color="gray", linestyle="--", linewidth=1.5, alpha=0.5)

        # Styling
        # make minor ticks inner
        ax.minorticks_on()
        # thicker mjor ticks
        ax.tick_params(which="both", direction="in", top=True, right=True, width=1.2)
        # Only set y-label on left column
        if col == 0:
            ax.set_ylabel("2B CLE (kJ/mol)", fontsize=12)
        ax.set_title(
            f"{crystal_name_mapping[crystal]} (n={len(df_c)}, QM={N}, ref={ref_final_cle:.1f} kJ/mol)",
            fontsize=12,
        )
        ax.tick_params(axis="both", which="major", labelsize=10)

        # Only show x-label on bottom row
        if row == n_rows - 1:
            ax.set_xlabel("Min. Mon. Sep. (Å)", fontsize=12)

        # Only show legend on first plot
        if idx == 0:
            ax.legend(loc="upper right", fontsize=10, framealpha=0.7)

    # Hide unused subplots
    for idx in range(n_crystals, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].axis("off")

    # Overall title
    # fig.suptitle(
    #     f"Crystal Lattice Energies (N={N}, vs {ref_label})",
    #     fontsize=12,
    #     fontweight="bold",
    # )

    # Adjust layout
    # plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.tight_layout()

    # Save
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\nSlide figure saved to: {output_path}")
    plt.close()

    return output_path


def main():
    tl_N = 1000
    uma_cutoff = 3.8

    # Option 1: Use original function (slower, computes everything each time)
    # plot_crystal_lattice_energies_with_N(0, sft=False, tl_N=tl_N, uma_cutoff=uma_cutoff)
    # plot_crystal_lattice_energies_with_N(1, sft=False, tl_N=tl_N, uma_cutoff=uma_cutoff)
    # plot_crystal_lattice_energies_with_N(5, sft=False, tl_N=tl_N, uma_cutoff=uma_cutoff)
    # plot_crystal_lattice_energies_with_N(10, sft=False, tl_N=tl_N, uma_cutoff=uma_cutoff)

    # Option 2: Prepare intermediates once, then plot multiple times (faster)
    print("Preparing crystal data intermediates...")
    # intermediates = prepare_crystal_data_intermediates(
    #     tl_N=tl_N, uma_cutoff=uma_cutoff, sft=False
    # )

    # plot_from_intermediates_slides(
    #     N=0,
    #     sft=False,
    #     v="bm",
    #     crystals=["acetic_acid", "CO2", "pyrazine", "benzene"],
    # )
    # for i in range(0, 10):
    #     plot_from_intermediates_slides(
    #         N=i,
    #         sft=False,
    #         v="bm",
    #         crystals=["acetic_acid", "CO2", "pyrazine", "benzene"],
    #     )
    print("\nPlotting with N=0...")
    plot_from_intermediates(N=0, sft=False)
    print("\nPlotting with N=1...")
    plot_from_intermediates(N=1, sft=False)
    print("\nPlotting with N=5...")
    plot_from_intermediates(N=5, sft=False)
    # print("\nPlotting with N=10...")
    # plot_from_intermediates(N=10, sft=False)

    return


if __name__ == "__main__":
    main()
