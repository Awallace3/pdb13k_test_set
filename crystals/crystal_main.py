import pandas as pd
from pprint import pprint as pp
import qcelemental as qcel
from glob import glob
from pathlib import Path
import subprocess
import numpy as np
from qm_tools_aw import tools
import apnet_pt
import os
from cdsg_plot import error_statistics
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import AutoMinorLocator
import shutil


qcml_model_dir = os.path.expanduser("~/gits/qcmlforge/models")

kcalmol_to_kjmol = qcel.constants.conversion_factor("kcal/mol", "kJ/mol")
ha_to_kjmol = qcel.constants.conversion_factor("hartree", "kJ/mol")

sft_n_epochs = 300
sft_lr = 5e-5

crystal_names_all = [
    "14-cyclohexanedione",
    "acetic_acid",
    # "adamantane",
    "ammonia",
    # "anthracene", # missing becnhmark-cc
    "benzene",
    "cyanamide",
    "cytosine",
    "ethyl_carbamate",
    "formamide",
    "hexamine",
    "ice",
    "imidazole",
    # "naphthalene",
    "oxalic_acid_alpha",
    "oxalic_acid_beta",
    "pyrazine",
    "pyrazole",
    "succinic_acid",
    "triazine",
    "trioxane",
    "uracil",
    "urea",
    "CO2",
]


def ap2_energies(mols, compile=False, finetune_mols=[], finetune_labels=[], pretrained_ap2_path=None, data_dir="data_dir"):
    if len(finetune_mols) > 0 and len(finetune_labels) > 0:
        print("Finetuning to crystal...")
        ds_qcel_molecules = finetune_mols
        ds_energy_labels = finetune_labels
        finetune = True
        ignore_database_null = False
    else:
        ds_qcel_molecules = None
        ds_energy_labels = None
        finetune = False
        ignore_database_null = True

    if os.path.exists("data_dir"):
        shutil.rmtree("data_dir")
    if pretrained_ap2_path is None:
        pretrained_ap2_path = f"{qcml_model_dir}/ap2-fused_ensemble/ap2_1.pt"

    ap2 = apnet_pt.AtomPairwiseModels.apnet2_fused.APNet2_AM_Model(
        ds_root=data_dir,
        atom_model=apnet_pt.AtomModels.ap2_atom_model.AtomModel(
            pre_trained_model_path=f"{qcml_model_dir}/am_ensemble/am_1.pt",
        ).model,
        # pre_trained_model_path=f"{qcml_model_dir}/ap2-fused_ensemble/ap2_1.pt",
        pre_trained_model_path=pretrained_ap2_path,
        ds_qcel_molecules=ds_qcel_molecules,
        ds_energy_labels=ds_energy_labels,
        ds_spec_type=None,
        ignore_database_null=ignore_database_null,
    )
    if compile:
        ap2.compile_model()
    if finetune:
        print(ds_qcel_molecules)
        print(ds_energy_labels)
        ap2.freeze_parameters_except_readouts()
        ap2.train(
            n_epochs=sft_n_epochs,
            lr=sft_lr,
            split_percent=0.99,
            skip_compile=True,
            transfer_learning=True,
        )
    pred = ap2.predict_qcel_mols(mols, batch_size=64)
    return pred


def ap3_d_elst_classical_energies(mols, finetune_mols=[], finetune_labels=[], pretrained_ap3_path=None, data_dir="data_dir"):
    if len(finetune_mols) > 0 and len(finetune_labels) > 0:
        print("Finetuning to crystal...")
        ds_qcel_molecules = finetune_mols
        ds_energy_labels = finetune_labels
        finetune = True
        ignore_database_null = False
    else:
        ds_qcel_molecules = None
        ds_energy_labels = None
        finetune = False
        ignore_database_null = True

    path_to_qcml = os.path.join(os.path.expanduser("~"), "gits/qcmlforge/models")
    am_path = f"{path_to_qcml}/../models/ap3_ensemble/1/am_3.pt"
    at_hf_vw_path = f"{path_to_qcml}/../models/ap3_ensemble/1/am_h+1_3.pt"
    at_elst_path = f"{path_to_qcml}/../models/ap3_ensemble/1/am_elst_h+1_3.pt"
    if pretrained_ap3_path is None:
        pretrained_ap3_path = f"{path_to_qcml}/../models/ap3_ensemble/0/ap3_.pt"
    atom_type_hf_vw_model = apnet_pt.AtomPairwiseModels.mtp_mtp.AtomTypeParamModel(
        ds_root=None,
        use_GPU=False,
        ignore_database_null=True,
        atom_model_pre_trained_path=am_path,
        pre_trained_model_path=at_hf_vw_path,
    )
    atom_type_elst_model = apnet_pt.AtomPairwiseModels.mtp_mtp.AM_DimerParam_Model(
        use_GPU=False,
        n_neuron=64,
        n_params=1,
        ignore_database_null=True,
        atom_model=atom_type_hf_vw_model.model,
        atom_model_type="AtomTypeParamNN",
        model_type="AtomTypeParamNN",
        pre_trained_model_path=at_elst_path,
    )
    if os.path.exists("data_dir"):
        shutil.rmtree("data_dir")

    ap3 = apnet_pt.AtomPairwiseModels.apnet3_fused.APNet3_AtomType_Model(
        ds_root=data_dir,
        atom_type_model=atom_type_hf_vw_model.model,
        dimer_prop_model=atom_type_elst_model.dimer_model,
        pre_trained_model_path=pretrained_ap3_path,
        ds_spec_type=None,
        ignore_database_null=ignore_database_null,
        ds_qcel_molecules=ds_qcel_molecules,
        ds_energy_labels=ds_energy_labels,
    )
    if finetune:
        ap3.freeze_parameters_except_readouts()
        ap3.train(
            n_epochs=sft_n_epochs,
            lr=sft_lr,
            split_percent=0.99,
            transfer_learning=True,
        )
    pred, pair_elst, pair_ind = ap3.predict_qcel_mols(
        mols, batch_size=64, return_classical_pairs=True
    )
    return pred, pair_elst, pair_ind


def ap2_ap3_df_energies(generate=False):
    v = "apprx"
    v = "bm"
    mol_str = "mol " + v
    pkl_fn = f"crystals_ap2_ap3_results_{mol_str.replace(' ', '_')}.pkl"
    if not os.path.exists(pkl_fn) or generate:
        df = pd.read_pickle("./x23_dfs/main_df.pkl")
        df = df.dropna(subset=[mol_str])
        print("AP2 start")
        mols = df[mol_str].tolist()
        pred_ap2 = ap2_energies(mols)
        pred_ap2 *= kcalmol_to_kjmol
        df["AP2 TOTAL"] = np.sum(pred_ap2[:, 0:4], axis=1)
        df["AP2 ELST"] = pred_ap2[:, 0]
        df["AP2 EXCH"] = pred_ap2[:, 1]
        df["AP2 INDU"] = pred_ap2[:, 2]
        df["AP2 DISP"] = pred_ap2[:, 3]
        df.to_pickle(pkl_fn)
        print("AP3 start")
        pred_ap3, pair_elst, pair_ind = ap3_d_elst_classical_energies(mols)
        pred_ap3 *= kcalmol_to_kjmol
        elst_energies = [np.sum(e) * kcalmol_to_kjmol for e in pair_elst]
        ind_energies = [np.sum(e) * kcalmol_to_kjmol for e in pair_ind]
        # df["mtp_elst_energies"] = elst_undamped
        df["ap3_d_elst"] = elst_energies
        df["ap3_classical_ind_energy"] = ind_energies
        df["AP3 TOTAL"] = np.sum(pred_ap3[:, 0:4], axis=1)
        df["AP3 ELST"] = pred_ap3[:, 0]
        df["AP3 EXCH"] = pred_ap3[:, 1]
        df["AP3 INDU"] = pred_ap3[:, 2]
        df["AP3 DISP"] = pred_ap3[:, 3]
        # LE
        non_additive_cols = [
            i for i in df.columns if "Non-Additive MB Energy (kJ/mol)" in i
        ]
        for c in non_additive_cols:
            label = c.split()[-1]
            df[f"{label}_le_contribution"] = df.apply(
                lambda r: r[c]
                * r[f"Num. Rep. (#) {label}"]
                / int(r[f"N-mer Name {v}"][0]),
                axis=1,
            )
        df["ap3_le_contribution"] = df.apply(
            lambda r: r["AP3 TOTAL"]
            * r["Num. Rep. (#) sapt0-dz-aug"]
            / int(r[f"N-mer Name {v}"][0]),
            axis=1,
        )
        df["ap2_le_contribution"] = df.apply(
            lambda r: r["AP2 TOTAL"]
            * r["Num. Rep. (#) sapt0-dz-aug"]
            / int(r[f"N-mer Name {v}"][0]),
            axis=1,
        )
        df.to_pickle(pkl_fn)
    else:
        df = pd.read_pickle(pkl_fn)

    if v == "apprx":
        pp(df.columns.tolist())
        print("\nResults\n")
        df["AP3 total error"] = (
            df["Non-Additive MB Energy (kJ/mol) sapt0-dz-aug"] - df["AP3 TOTAL"]
        )
        df["AP2 total error"] = (
            df["Non-Additive MB Energy (kJ/mol) sapt0-dz-aug"] - df["AP2 TOTAL"]
        )
        mae_total_ap3 = df["AP3 total error"].abs().mean()
        print(f"MAE AP3 Total: {mae_total_ap3:.4f} kJ/mol")
        mae_total_ap2 = df["AP2 total error"].abs().mean()
        print(f"MAE AP2 Total: {mae_total_ap2:.4f} kJ/mol")
        print(
            df[
                [
                    "AP3 total error",
                    "AP2 total error",
                ]
            ].describe()
        )
    return df


def ap2_ap3_df_energies_des370k_tl(v='bm', generate=False, N=100):
    # NOTE: pkl_fn uses _des_, so ensure that other AP energies are already in that df
    mol_str = "mol " + v
    pkl_fn = f"crystals_ap2_ap3_des_results_{mol_str.replace(' ', '_')}.pkl"
    df = pd.read_pickle(f"crystals_ap2_ap3_des_results_{mol_str.replace(' ', '_')}.pkl")
    df = df.dropna(subset=[mol_str])
    print("AP2 start")
    mols = df[mol_str].tolist()
    pred_ap2 = ap2_energies(
        mols,
        pretrained_ap2_path=f"./sft_models/ap2_des370k_{N}.pt",
        data_dir="data_des370k",
        compile=False,
    )
    pred_ap2 *= kcalmol_to_kjmol
    df[f"AP2-des-tl{N} TOTAL"] = np.sum(pred_ap2[:, 0:4], axis=1)
    df[f"AP2-des-tl{N} ELST"] = pred_ap2[:, 0]
    df[f"AP2-des-tl{N} EXCH"] = pred_ap2[:, 1]
    df[f"AP2-des-tl{N} INDU"] = pred_ap2[:, 2]
    df[f"AP2-des-tl{N} DISP"] = pred_ap2[:, 3]
    df.to_pickle(pkl_fn)
    print("AP3 start")
    pred_ap3, pair_elst, pair_ind = ap3_d_elst_classical_energies(
        mols,
        pretrained_ap3_path=f"./sft_models/ap3_des370k_{N}.pt",
        data_dir="data_des370k",
    )
    pred_ap3 *= kcalmol_to_kjmol
    elst_energies = [np.sum(e) * kcalmol_to_kjmol for e in pair_elst]
    ind_energies = [np.sum(e) * kcalmol_to_kjmol for e in pair_ind]
    # df["mtp_elst_energies"] = elst_undamped
    df[f"ap3_d_elst-des-tl{N}"] = elst_energies
    df[f"ap3_classical_ind_energy-des-tl{N}"] = ind_energies
    df[f"AP3-des-tl{N} TOTAL"] = np.sum(pred_ap3[:, 0:4], axis=1)
    df[f"AP3-des-tl{N} ELST"] = pred_ap3[:, 0]
    df[f"AP3-des-tl{N} EXCH"] = pred_ap3[:, 1]
    df[f"AP3-des-tl{N} INDU"] = pred_ap3[:, 2]
    df[f"AP3-des-tl{N} DISP"] = pred_ap3[:, 3]
    # LE
    df[f"ap3-des-tl{N}_le_contribution"] = df.apply(
        lambda r: r[f"AP3-des-tl{N} TOTAL"]
        * r["Num. Rep. (#) sapt0-dz-aug"]
        / int(r[f"N-mer Name {v}"][0]),
        axis=1,
    )
    df[f"ap2-des-tl{N}_le_contribution"] = df.apply(
        lambda r: r[f"AP2-des-tl{N} TOTAL"]
        * r["Num. Rep. (#) sapt0-dz-aug"]
        / int(r[f"N-mer Name {v}"][0]),
        axis=1,
    )
    df.to_pickle(pkl_fn)

    if v == "apprx":
        pp(df.columns.tolist())
        print("\nResults\n")
        df["AP3 total error"] = (
            df["Non-Additive MB Energy (kJ/mol) sapt0-dz-aug"] - df[f"AP3-des-tl{N} TOTAL"]
        )
        df["AP2 total error"] = (
            df["Non-Additive MB Energy (kJ/mol) sapt0-dz-aug"] - df[f"AP2-des-tl{N} TOTAL"]
        )
        mae_total_ap3 = df["AP3 total error"].abs().mean()
        print(f"MAE AP3 Total: {mae_total_ap3:.4f} kJ/mol")
        mae_total_ap2 = df["AP2 total error"].abs().mean()
        print(f"MAE AP2 Total: {mae_total_ap2:.4f} kJ/mol")
        print(
            df[
                [
                    "AP3 total error",
                    "AP2 total error",
                ]
            ].describe()
        )
    return df


def ap2_ap3_df_energies_sft(generate=False, v="apprx", N=10):
    # v = "bm"
    mol_str = "mol " + v
    pkl_fn = f"sft_crystals_ap2_ap3_results_{mol_str.replace(' ', '_')}.pkl"
    new_cols = [
        "AP3 TOTAL",
        "AP3 ELST",
        "AP3 EXCH",
        "AP3 INDU",
        "AP3 DISP",
        "AP2 TOTAL",
        "AP2 ELST",
        "AP2 EXCH",
        "AP2 INDU",
        "AP2 DISP",
    ]
    if not os.path.exists(pkl_fn) or generate:
        df = pd.read_pickle("./x23_dfs/main_df.pkl")
        # df = df.sample(n=20)
        for c in new_cols:
            df[c] = None
        print(df)
        df = df.dropna(subset=[mol_str])
        if v == "apprx":
            mms_col = "Minimum Monomer Separations (A) sapt0-dz-aug"
            target_col = "Non-Additive MB Energy (kJ/mol) sapt0-dz-aug"
        else:
            mms_col = "Minimum Monomer Separations (A) CCSD(T)/CBS"
            target_col = "Non-Additive MB Energy (kJ/mol) CCSD(T)/CBS"
        for n, c in enumerate(crystal_names_all):
            # Get indices for this crystal
            crystal_mask = df[f"crystal {v}"] == c
            crystal_indices = df[crystal_mask].index

            if len(crystal_indices) == 0:
                continue

            # Get sorted crystal dataframe
            df_c_a = df.loc[crystal_indices].copy()
            df_c_a.sort_values(by=mms_col, inplace=True)
            print(f"\nProcessing crystal: {n} {c} with {len(df_c_a)} entries")
            print(df_c_a[[mms_col, target_col]])
            mols = df_c_a[mol_str].tolist()
            print("AP3 start")
            # return
            pred_ap3, pair_elst, pair_ind = ap3_d_elst_classical_energies(
                mols,
                finetune_labels=[
                    i / kcalmol_to_kjmol for i in df_c_a[target_col].to_list()[:N]
                ],
                finetune_mols=mols[:N],
            )
            pred_ap3 *= kcalmol_to_kjmol
            elst_energies = [np.sum(e) * kcalmol_to_kjmol for e in pair_elst]
            ind_energies = [np.sum(e) * kcalmol_to_kjmol for e in pair_ind]

            # Update main dataframe using .loc with the sorted indices
            df.loc[df_c_a.index, "ap3_d_elst"] = elst_energies
            df.loc[df_c_a.index, "ap3_classical_ind_energy"] = ind_energies
            df.loc[df_c_a.index, "AP3 TOTAL"] = np.sum(pred_ap3[:, 0:4], axis=1)
            df.loc[df_c_a.index, "AP3 ELST"] = pred_ap3[:, 0]
            df.loc[df_c_a.index, "AP3 EXCH"] = pred_ap3[:, 1]
            df.loc[df_c_a.index, "AP3 INDU"] = pred_ap3[:, 2]
            df.loc[df_c_a.index, "AP3 DISP"] = pred_ap3[:, 3]

            print("AP2 start")
            pred_ap2 = ap2_energies(
                mols,
                compile=False,
                finetune_labels=[
                    i / kcalmol_to_kjmol for i in df_c_a[target_col].to_list()[:N]
                ],
                finetune_mols=mols[:N],
            )
            pred_ap2 *= kcalmol_to_kjmol

            # Update main dataframe using .loc with the sorted indices
            df.loc[df_c_a.index, "AP2 TOTAL"] = np.sum(pred_ap2[:, 0:4], axis=1)
            df.loc[df_c_a.index, "AP2 ELST"] = pred_ap2[:, 0]
            df.loc[df_c_a.index, "AP2 EXCH"] = pred_ap2[:, 1]
            df.loc[df_c_a.index, "AP2 INDU"] = pred_ap2[:, 2]
            df.loc[df_c_a.index, "AP2 DISP"] = pred_ap2[:, 3]

        # LE
        non_additive_cols = [
            i for i in df.columns if "Non-Additive MB Energy (kJ/mol)" in i
        ]
        for c in non_additive_cols:
            label = c.split()[-1]
            df[f"{label}_le_contribution"] = df.apply(
                lambda r: r[c]
                * r[f"Num. Rep. (#) {label}"]
                / int(r[f"N-mer Name {v}"][0])
                if pd.notnull(r[c]) and pd.notnull(r[f"N-mer Name {v}"])
                else 0,
                axis=1,
            )
        df["ap3_le_contribution"] = df.apply(
            lambda r: r["AP3 TOTAL"]
            * r["Num. Rep. (#) sapt0-dz-aug"]
            / int(r[f"N-mer Name {v}"][0])
            if pd.notnull(r["AP3 TOTAL"]) and pd.notnull(r[f"N-mer Name {v}"])
            else 0,
            axis=1,
        )
        df["ap2_le_contribution"] = df.apply(
            lambda r: r["AP2 TOTAL"]
            * r["Num. Rep. (#) sapt0-dz-aug"]
            / int(r[f"N-mer Name {v}"][0])
            if pd.notnull(r["AP2 TOTAL"]) and pd.notnull(r[f"N-mer Name {v}"])
            else 0,
            axis=1,
        )
        df.to_pickle(pkl_fn)
    else:
        df = pd.read_pickle(pkl_fn)

    if v == "apprx":
        pp(df.columns.tolist())
        print("\nResults\n")
        df["AP3 total error"] = (
            df["Non-Additive MB Energy (kJ/mol) sapt0-dz-aug"] - df["AP3 TOTAL"]
        )
        df["AP2 total error"] = (
            df["Non-Additive MB Energy (kJ/mol) sapt0-dz-aug"] - df["AP2 TOTAL"]
        )
        mae_total_ap3 = df["AP3 total error"].abs().mean()
        print(f"MAE AP3 Total: {mae_total_ap3:.4f} kJ/mol")
        mae_total_ap2 = df["AP2 total error"].abs().mean()
        print(f"MAE AP2 Total: {mae_total_ap2:.4f} kJ/mol")
        print(
            df[
                [
                    "AP3 total error",
                    "AP2 total error",
                ]
            ].describe()
        )
    return df


def dapnet_main_df_crystals(
    delta_model_paths={
        # "b3lyp": "../../dapnet_models/delta_correction_B3LYP-D3_aug-cc-pVDZ_CP_to_CCSDT_CBS_CP_limit_B2PLYP-D3_aug-cc-pVTZ_CP",
        "b3lyp": "/home/awallace43/gits/qcmlforge/models/dapnet2/B3LYP-D3_aug-cc-pVDZ_CP_to_CCSD_LP_T_RP_CBS_CP_0.pt",
        # "pbe-d3-dz-aug": "../../dapnet_models/delta_correction_PBE-D3_aug-cc-pVDZ_CP_to_CCSDT_CBS_CP_limit_B2PLYP-D3_aug-cc-pVTZ_CP",
        # "pbeh3c-590": "../../dapnet_models/delta_correction_PBEh-3c_aug-cc-pVDZ_CP_to_CCSDT_CBS_CP_limit_B2PLYP-D3_aug-cc-pVTZ_CP",
    },
    output_pickle="./x23_dfs/main_df_cbs.pkl",
    compute_delta=False,
    verbose=False,
):
    if compute_delta:
        df = pd.read_pickle("./x23_dfs/main_df.pkl")
    else:
        df = pd.read_pickle(output_pickle)
    crystal_information = {
        k: {
            "bm cle": np.nan,
            "apprx cles": {k: np.nan for k in delta_model_paths.keys()},
            "cle target": {k: np.nan for k in delta_model_paths.keys()},
            "dAPNet2 apprx": {k: np.nan for k in delta_model_paths.keys()},
        }
        for k in crystal_names_all
    }
    for c in crystal_names_all:
        df_c_a = df[df["crystal apprx"] == c]
        df_c_a = df_c_a.dropna(subset=["mol apprx"])
        df_c_a.sort_values(by="local_crystal_index", inplace=True)
        df_c_b = df[df["crystal bm"] == c]
        df_c_b = df_c_b.dropna(subset=["mol bm"])
        df_c_a.sort_values(by="local_crystal_index", inplace=True)
        if verbose:
            print(c, len(df_c_a), len(df_c_b))
        for k, v in delta_model_paths.items():
            basefile = Path(v).stem
            apprx_final = df_c_a.apply(
                lambda r: (
                    (r[f"Non-Additive MB Energy (kJ/mol) {k}"])
                    * r[f"Num. Rep. (#) {k}"]
                    / int(r[f"N-mer Name apprx"][0])
                    if pd.notnull(r["N-mer Name apprx"])
                    else 0
                ),
                axis=1,
            ).sum()
            bm_final = df_c_b.apply(
                lambda r: (
                    (r[f"Non-Additive MB Energy (kJ/mol) CCSD(T)/CBS"])
                    * r[f"Num. Rep. (#) CCSD(T)/CBS"]
                    / int(r[f"N-mer Name bm"][0])
                    if pd.notnull(r["N-mer Name bm"])
                    else 0
                ),
                axis=1,
            ).sum()
            # bm_final = df_c_b[f"Partial Crys. Lattice Ener. (kJ/mol) sapt0-dz-jun bm"].values[-1]
            if bm_final == 0.0:
                bm_final = np.nan
            if apprx_final == 0.0:
                apprx_final = np.nan
            crystal_information[c]["bm cle"] = bm_final
            crystal_information[c]["apprx cles"][k] = apprx_final
            crystal_information[c]["cle target"][k] = bm_final - apprx_final
            if not compute_delta:
                df_c_a[f"dapnet cle {v}"] = df_c_a.apply(
                    lambda r: (
                        r[f"Non-Additive MB Energy (kJ/mol) {k}"]
                        - r[f"dAPNet2 apprx {basefile}"]
                    )
                    * r[f"Num. Rep. (#) {k}"]
                    / int(r[f"N-mer Name apprx"][0]),
                    axis=1,
                )
                crystal_information[c]["dAPNet2 apprx"][k] = df_c_a[
                    f"dapnet cle {v}"
                ].sum()
    if verbose:
        pp(crystal_information)
    # import apnet

    update_progress = 1
    for k, delta_model_path in delta_model_paths.items():
        basefile = Path(delta_model_path).stem
        if verbose:
            print(basefile)
        if compute_delta:
            df[f"dAPNet2 apprx {basefile}"] = None
            df[f"dAPNet2 bm {basefile}"] = None
            # pm = apnet.PairModel(
            #     apnet.AtomModel().pretrained(0), pair_model_type="delta_correction"
            # ).from_file(delta_model_path, pair_model_type="delta_correction")
            print("Model loaded")
            for n, row in df.iterrows():
                if pd.notnull(row["mol apprx"]):
                    # energy = pm.predict([row["mol apprx"]], batch_size=1)
                    # if len(energy) == 0:
                    #     energy = [[0]]
                    energy = dapnet2.predict_qcel_mols([row["mol apprx"]], batch_size=1)
                    df.loc[n, f"dAPNet2 apprx {basefile}"] = (
                        energy[0] * kcalmol_to_kjmol
                    )
                if pd.notnull(row["mol bm"]):
                    # energy = pm.predict([row["mol bm"]], batch_size=1)
                    energy = dapnet2.predict_qcel_mols([row["mol bm"]], batch_size=1)
                    if len(energy) == 0:
                        energy = [[0]]
                    df.loc[n, f"dAPNet2 bm {basefile}"] = (
                        # energy[0][0] * kcalmol_to_kjmol
                        energy[0] * kcalmol_to_kjmol
                    )
                if n % update_progress == 0:
                    energies_sum_crystal_apprx = df[
                        df["crystal apprx"] == row["crystal apprx"]
                    ][f"dAPNet2 apprx {basefile}"].sum()
                    energies_sum_crystal_bm = df[df["crystal bm"] == row["crystal bm"]][
                        f"dAPNet2 bm {basefile}"
                    ].sum()
                    active_crystal = (
                        row["crystal apprx"]
                        if pd.notnull(row["crystal apprx"])
                        else row["crystal bm"]
                    )
                    apprx_error = crystal_information[active_crystal]["cle target"][k]
                    print(
                        f"{n}/{len(df)}, crystal: {active_crystal}, dAP(apprx, bm): ({
                            energies_sum_crystal_apprx: .2f
                        }, {energies_sum_crystal_bm: .2f}), CLE d-target: {
                            apprx_error: .2f
                        }"
                    )
        else:
            if verbose:
                for n, row in df.iterrows():
                    if n % update_progress == 0:
                        energies_sum_crystal_apprx = df[
                            df["crystal apprx"] == row["crystal apprx"]
                        ][f"dAPNet2 apprx {basefile}"].sum()
                        energies_sum_crystal_bm = df[
                            df["crystal bm"] == row["crystal bm"]
                        ][f"dAPNet2 bm {basefile}"].sum()
                        active_crystal = (
                            row["crystal apprx"]
                            if pd.notnull(row["crystal apprx"])
                            else row["crystal bm"]
                        )
                        apprx_error = crystal_information[active_crystal]["cle target"][
                            k
                        ]
                        print(
                            f"{n}/{len(df)}, crystal: {
                                active_crystal
                            }, dAP(apprx, bm): ({energies_sum_crystal_apprx: .2f}, {
                                energies_sum_crystal_bm: .2f
                            }), CLE d-target: {apprx_error: .2f}"
                        )
        if compute_delta:
            Path(output_pickle).parent.mkdir(parents=True, exist_ok=True)
            print(df)
            print("Saving to", output_pickle)
            df.to_pickle(output_pickle)
    return df, crystal_information


def plot_crystal_errors(
    compute_delta=False,
):
    if compute_delta:
        df, ci = dapnet_main_df_crystals(
            compute_delta=compute_delta,
        )
    df, ci = dapnet_main_df_crystals()
    pp(ci)
    data = {}
    for c, v in ci.items():
        data[c] = {}
        for k, v2 in v.items():
            if isinstance(v2, dict):
                for k2, v3 in v2.items():
                    data[c][f"{k} {k2}"] = v3
            else:
                data[c][k] = v2
    df_c = pd.DataFrame(data).T
    # df_c = df_c.dropna(subset=['apprx cles pbeh3c-590'])
    # df_c = df_c.dropna(subset=['apprx cles pbeh3c-590'])
    bounds = [[-10, 10] for i in range(3)]
    dfs = []
    for i in [
        # "pbe-d3-dz-aug", "pbeh3c-590",
        "b3lyp"
    ]:
        df_c[f"{i} error"] = df_c[f"apprx cles {i}"] - df_c[f"bm cle"]
        df_c[f"{i}+dAPNet2 error"] = df_c[f"dAPNet2 apprx {i}"] - df_c[f"bm cle"]
        df_c["diff"] = df_c[f"{i}+dAPNet2 error"] - df_c[f"{i} error"]
        print(df_c[[f"{i} error", f"{i}+dAPNet2 error", "diff"]].describe())
    dfs.append(
        {
            "df": df_c,
            "basis": "",
            "label": "CCSD(T)/CBS",
            "ylim": bounds,
        }
    )
    print(
        df_c[
            ["bm cle", "apprx cles b3lyp", "dAPNet2 apprx b3lyp", "b3lyp+dAPNet2 error"]
        ]
    )
    print(df_c[["bm cle", "apprx cles b3lyp", "b3lyp error", "b3lyp+dAPNet2 error"]])
    df_c.sort_values(by="b3lyp+dAPNet2 error", inplace=True)
    print(
        df_c[
            ["bm cle", "apprx cles b3lyp", "dAPNet2 apprx b3lyp", "b3lyp+dAPNet2 error"]
        ]
    )
    print(df_c[["bm cle", "apprx cles b3lyp", "b3lyp error", "b3lyp+dAPNet2 error"]])
    df_tex = df_c[
        ["bm cle", "apprx cles b3lyp", "b3lyp error", "b3lyp+dAPNet2 error"]
    ].copy()
    # fmt to 2 decimal
    df_tex["abs_error"] = df_tex["b3lyp+dAPNet2 error"].abs()
    df_tex.sort_values(by="abs_error", inplace=True)
    df_tex.drop(columns=["abs_error"], inplace=True)
    df_tex.to_latex(
        "./x23_plots/b3lypd3adz_crystal_errors.tex",
        index=True,
        float_format="{:0.2f}".format,
    )

    error_statistics.violin_plot_table_multi_SAPT_components(
        dfs,
        df_labels_and_columns_total={
            "B3LYP-D3(BJ)\\\\/aDZ": "b3lyp error",
            "B3LYP-D3(BJ)\\\\/aDZ+dAPNet2": "b3lyp+dAPNet2 error",
            # "PBE-D3/aDZ": "pbe-d3-dz-aug error",
            # "PBE-D3/aDZ+dAPNet2": "pbe-d3-dz-aug+dAPNet2 error",
            # "PBEh-3c/aDZ": "pbeh3c-590 error",
            # "PBEh-3c/aDZ+dAPNet2": "pbeh3c-590+dAPNet2 error",
        },
        output_filename="./x23_plots/crystal_errors.png",
        grid_heights=[0.3, 1.0],
        grid_widths=[1],
        legend_loc="upper left",
        annotations_texty=0.3,
        figure_size=(4, 2.5),
        add_title=False,
        name_violin=False,
        ylabel=r"Error (kJ$\cdot$mol$^{-1}$)",
    )
    return


def main_og():
    # ammonia()
    # hexamine(True)
    # imidazole()
    # process_all_crystals(True)
    collect_crystal_data(generate=False)
    plot_crystal_errors(compute_delta=False)

    df, ci = dapnet_main_df_crystals()
    pp(ci)
    data = {}
    for c, v in ci.items():
        data[c] = {}
        for k, v2 in v.items():
            if isinstance(v2, dict):
                for k2, v3 in v2.items():
                    data[c][f"{k} {k2}"] = v3
            else:
                data[c][k] = v2
    df_c = pd.DataFrame(data).T
    print(df_c)
    return
    drude_crystals = [
        "pyrazine",
        "imidazole",
        "acetic_acid",
        "ice",
        "pyrazole",
    ]
    for i in drude_crystals:
        print(i)
        output_data = (
            f"/theoryfs2/ds/csargent/chem/x23-crystals/{i}/sapt0-dz-aug/{i}/{i}.csv"
        )
        if i == "pyrazine":
            output_data = f"/ theoryfs2/ds/csargent/chem/x23-crystals/{i}/sapt0-dz-aug/{
                i
            }/sapt0-dz-aug.csv"
        process_inputs_sapt0_induction(
            output_data=output_data,
            delta_model_path=None,
            data_path=f"/ theoryfs2/ds/csargent/chem/x23-crystals/{i}/sapt0-dz-aug/{
                i
            }/*.out",
            output_csv=f"./sapt0_induction/{i}_sapt0adz.pkl",
        )


def plot_full_crystal_errors():
    """
    Create violin plots for full crystal lattice energy errors.
    Similar to dapnet_main_df_crystals, this sums up contributions per crystal
    and compares AP2/AP3 against reference methods.
    """
    from cdsg_plot import error_statistics

    # Load dataframes
    df1 = pd.read_pickle("./crystals_ap2_ap3_results_mol_apprx.pkl")
    df2 = pd.read_pickle("./crystals_ap2_ap3_results_mol_bm.pkl")

    print("Processing df1 (approximate methods)...")
    print(f"Total rows in df1: {len(df1)}")
    print(f"Crystals in df1: {df1['crystal apprx'].unique()}")

    print("\nProcessing df2 (benchmark)...")
    print(f"Total rows in df2: {len(df2)}")
    print(f"Crystals in df2: {df2['crystal bm'].unique()}")

    # Calculate crystal lattice energies for df1 (comparing against SAPT0)
    crystal_data_df1 = {}
    for crystal in df1["crystal apprx"].dropna().unique():
        df_c = df1[df1["crystal apprx"] == crystal].copy()

        # Sum up lattice energy contributions for SAPT0 reference
        sapt0_le = df_c.apply(
            lambda r: r["Non-Additive MB Energy (kJ/mol) sapt0-dz-aug"]
            * r["Num. Rep. (#) sapt0-dz-aug"]
            / int(r["N-mer Name apprx"][0])
            if pd.notnull(r["N-mer Name apprx"])
            else 0,
            axis=1,
        ).sum()

        # Sum up lattice energy contributions for AP2
        ap2_le = df_c.apply(
            lambda r: r["AP2 TOTAL"]
            * r["Num. Rep. (#) sapt0-dz-aug"]
            / int(r["N-mer Name apprx"][0])
            if pd.notnull(r["N-mer Name apprx"])
            else 0,
            axis=1,
        ).sum()

        # Sum up lattice energy contributions for AP3
        ap3_le = df_c.apply(
            lambda r: r["AP3 TOTAL"]
            * r["Num. Rep. (#) sapt0-dz-aug"]
            / int(r["N-mer Name apprx"][0])
            if pd.notnull(r["N-mer Name apprx"])
            else 0,
            axis=1,
        ).sum()

        # Calculate errors
        crystal_data_df1[crystal] = {
            "sapt0_le": sapt0_le,
            "ap2_le": ap2_le,
            "ap3_le": ap3_le,
            "AP2 error": ap2_le - sapt0_le,
            "AP3 error": ap3_le - sapt0_le,
        }

    # Calculate crystal lattice energies for df2 (comparing against CCSD(T)/CBS)
    crystal_data_df2 = {}
    for crystal in df2["crystal bm"].dropna().unique():
        df_c = df2[df2["crystal bm"] == crystal].copy()

        # Sum up lattice energy contributions for CCSD(T)/CBS reference
        ccsdt_le = df_c.apply(
            lambda r: r["Non-Additive MB Energy (kJ/mol) CCSD(T)/CBS"]
            * r["Num. Rep. (#) CCSD(T)/CBS"]
            / int(r["N-mer Name bm"][0])
            if pd.notnull(r["N-mer Name bm"])
            else 0,
            axis=1,
        ).sum()

        # Sum up lattice energy contributions for AP2
        ap2_le = df_c.apply(
            lambda r: r["AP2 TOTAL"]
            * r["Num. Rep. (#) CCSD(T)/CBS"]
            / int(r["N-mer Name bm"][0])
            if pd.notnull(r["N-mer Name bm"])
            else 0,
            axis=1,
        ).sum()

        # Sum up lattice energy contributions for AP3
        ap3_le = df_c.apply(
            lambda r: r["AP3 TOTAL"]
            * r["Num. Rep. (#) CCSD(T)/CBS"]
            / int(r["N-mer Name bm"][0])
            if pd.notnull(r["N-mer Name bm"])
            else 0,
            axis=1,
        ).sum()

        # Calculate errors
        crystal_data_df2[crystal] = {
            "ccsdt_le": ccsdt_le,
            "ap2_le": ap2_le,
            "ap3_le": ap3_le,
            "AP2 error": ap2_le - ccsdt_le,
            "AP3 error": ap3_le - ccsdt_le,
        }

    # Convert to DataFrames
    df1_crystal_errors = pd.DataFrame(crystal_data_df1).T
    df2_crystal_errors = pd.DataFrame(crystal_data_df2).T

    print("\n=== df1 Crystal Lattice Energy Errors (vs SAPT0) ===")
    print(df1_crystal_errors[["AP2 error", "AP3 error"]])
    print("\nStatistics:")
    print(df1_crystal_errors[["AP2 error", "AP3 error"]].describe())

    print("\n=== df2 Crystal Lattice Energy Errors (vs CCSD(T)/CBS) ===")
    print(df2_crystal_errors[["AP2 error", "AP3 error"]])
    print("\nStatistics:")
    print(df2_crystal_errors[["AP2 error", "AP3 error"]].describe())

    # Prepare data for violin plots
    dfs1 = [
        {
            "df": df1_crystal_errors,
            "basis": "",
            "label": "SAPT0/aDZ Reference",
            "ylim": [[-30, 30]],
        }
    ]

    dfs2 = [
        {
            "df": df2_crystal_errors,
            "basis": "",
            "label": "CCSD(T)/CBS Reference",
            "ylim": [[-30, 30]],
        }
    ]

    # Create violin plot for df1 (approximate methods)
    print("\nCreating violin plot for df1 crystal errors...")
    error_statistics.violin_plot_table_multi_SAPT_components(
        dfs1,
        df_labels_and_columns_total={
            "AP2": "AP2 error",
            "AP3": "AP3 error",
        },
        output_filename="./x23_plots/ap2_ap3_errors_vs_sapt0_violin.png",
        grid_heights=[0.3, 1.0],
        grid_widths=[1],
        legend_loc="upper left",
        annotations_texty=0.3,
        figure_size=(4, 2.5),
        add_title=False,
        name_violin=False,
        ylabel=r"Error (kJ$\cdot$mol$^{-1}$)",
    )
    print("Saved plot to ./x23_plots/ap2_ap3_errors_vs_sapt0_violin.png")

    # Create violin plot for df2 (benchmark comparison)
    print("\nCreating violin plot for df2 crystal errors...")
    error_statistics.violin_plot_table_multi_SAPT_components(
        dfs2,
        df_labels_and_columns_total={
            "AP2": "AP2 error",
            "AP3": "AP3 error",
        },
        output_filename="./x23_plots/ap2_ap3_errors_vs_ccsdt_cbs_violin.png",
        grid_heights=[0.3, 1.0],
        grid_widths=[1],
        legend_loc="upper left",
        annotations_texty=0.3,
        figure_size=(4, 2.5),
        add_title=False,
        name_violin=False,
        ylabel=r"Error (kJ$\cdot$mol$^{-1}$)",
    )
    print("Saved plot to ./x23_plots/ap2_ap3_errors_vs_ccsdt_cbs_violin.png")

    # Print summary statistics
    print("\n=== Summary Statistics (Crystal Lattice Energies) ===")
    print("\ndf1 (vs SAPT0/aDZ):")
    for col in ["AP2 error", "AP3 error"]:
        mae = df1_crystal_errors[col].abs().mean()
        rmse = np.sqrt((df1_crystal_errors[col] ** 2).mean())
        mean = df1_crystal_errors[col].mean()
        print(f"  {col}:")
        print(f"    MAE: {mae:.4f} kJ/mol")
        print(f"    RMSE: {rmse:.4f} kJ/mol")
        print(f"    Mean: {mean:.4f} kJ/mol")

    print("\ndf2 (vs CCSD(T)/CBS):")
    for col in ["AP2 error", "AP3 error"]:
        mae = df2_crystal_errors[col].abs().mean()
        rmse = np.sqrt((df2_crystal_errors[col] ** 2).mean())
        mean = df2_crystal_errors[col].mean()
        print(f"  {col}:")
        print(f"    MAE: {mae:.4f} kJ/mol")
        print(f"    RMSE: {rmse:.4f} kJ/mol")
        print(f"    Mean: {mean:.4f} kJ/mol")

    return df1_crystal_errors, df2_crystal_errors


def plot_all_systems():
    # Load dataframes
    df1 = pd.read_pickle("./crystals_ap2_ap3_results_mol_apprx.pkl")
    df2 = pd.read_pickle("./crystals_ap2_ap3_results_mol_bm.pkl")

    # Print columns to understand available data
    print("df1 columns:")
    pp(
        [
            col
            for col in df1.columns
            if any(x in col for x in ["AP2", "AP3", "CCSD", "Non-Additive", "sapt"])
        ]
    )
    print("\ndf2 columns:")
    pp(
        [
            col
            for col in df2.columns
            if any(x in col for x in ["AP2", "AP3", "CCSD", "Non-Additive"])
        ]
    )

    # Create error columns for df1 (approximate methods vs SAPT0)
    # df1 has multiple methods to compare against SAPT0
    df1["AP2 vs SAPT0 error"] = (
        df1["Non-Additive MB Energy (kJ/mol) sapt0-dz-aug"] - df1["AP2 TOTAL"]
    )
    df1["AP3 vs SAPT0 error"] = (
        df1["Non-Additive MB Energy (kJ/mol) sapt0-dz-aug"] - df1["AP3 TOTAL"]
    )

    # Add other method errors if available
    method_cols = [
        col
        for col in df1.columns
        if "Non-Additive MB Energy (kJ/mol)" in col and "sapt0-dz-aug" not in col
    ]
    for method_col in method_cols[:3]:  # Limit to first 3 additional methods
        method_name = method_col.replace("Non-Additive MB Energy (kJ/mol) ", "")
        df1[f"{method_name} error"] = (
            df1[method_col] - df1["Non-Additive MB Energy (kJ/mol) sapt0-dz-aug"]
        )

    # Create error columns for df2 (comparing against CCSD(T)/CBS)
    df2["AP2 vs CCSD(T)/CBS error"] = (
        df2["Non-Additive MB Energy (kJ/mol) CCSD(T)/CBS"] - df2["AP2 TOTAL"]
    )
    df2["AP3 vs CCSD(T)/CBS error"] = (
        df2["Non-Additive MB Energy (kJ/mol) CCSD(T)/CBS"] - df2["AP3 TOTAL"]
    )

    # Prepare df1 for violin plot
    dfs1 = [
        {
            "df": df1,
            "basis": "",
            "label": "SAPT0/aDZ Reference",
            "ylim": [[-4, 4]],
        }
    ]

    # Labels and columns for df1 (multiple methods)
    df1_labels = {
        "AP2": "AP2 vs SAPT0 error",
        "AP3": "AP3 vs SAPT0 error",
    }

    # Add other methods if available
    for method_col in method_cols[:3]:
        method_name = method_col.replace("Non-Additive MB Energy (kJ/mol) ", "")
        if f"{method_name} error" in df1.columns:
            df1_labels[method_name] = f"{method_name} error"

    # Prepare df2 for violin plot
    dfs2 = [
        {
            "df": df2,
            "basis": "",
            "label": "CCSD(T)/CBS Reference",
            "ylim": [[-4, 4]],
        }
    ]

    # Labels and columns for df2 (only AP2 and AP3 vs CCSD(T)/CBS)
    df2_labels = {
        "AP2": "AP2 vs CCSD(T)/CBS error",
        "AP3": "AP3 vs CCSD(T)/CBS error",
        # "UMA-s": "UMA-s vs CCSD(T)/CBS error",
        # "UMA-m": "UMA-m vs CCSD(T)/CBS error",
    }

    # Create violin plot for df1
    print("\nCreating violin plot for df1 (approximate methods)...")
    error_statistics.violin_plot_table_multi_SAPT_components(
        dfs1,
        df_labels_and_columns_total=df1_labels,
        output_filename="./x23_plots/ap2_ap3_errors_vs_sapt0_all.png",
        grid_heights=[0.3, 1.0],
        grid_widths=[1],
        legend_loc="upper left",
        annotations_texty=0.3,
        figure_size=(6, 2.5),
        add_title=False,
        name_violin=False,
        ylabel=r"Error (kJ$\cdot$mol$^{-1}$)",
    )
    print("Saved plot to ./x23_plots/ap2_ap3_errors_vs_sapt0.png")

    # Create violin plot for df2
    print("\nCreating violin plot for df2 (benchmark comparison)...")
    error_statistics.violin_plot_table_multi_SAPT_components(
        dfs2,
        df_labels_and_columns_total=df2_labels,
        output_filename="./x23_plots/ap2_ap3_errors_vs_ccsdt_cbs_all.png",
        grid_heights=[0.3, 1.0],
        grid_widths=[1],
        legend_loc="upper left",
        annotations_texty=0.3,
        figure_size=(4, 2.5),
        add_title=False,
        name_violin=False,
        ylabel=r"Error (kJ$\cdot$mol$^{-1}$)",
    )
    print("Saved plot to ./x23_plots/ap2_ap3_errors_vs_ccsdt_cbs.png")

    # Print summary statistics
    print("\n=== df1 Summary Statistics ===")
    for label, col in df1_labels.items():
        if col in df1.columns:
            print(f"\n{label}:")
            print(f"  MAE: {df1[col].abs().mean():.4f} kJ/mol")
            print(f"  RMSE: {np.sqrt((df1[col] ** 2).mean()):.4f} kJ/mol")
            print(f"  Mean: {df1[col].mean():.4f} kJ/mol")

    print("\n=== df2 Summary Statistics ===")
    for label, col in df2_labels.items():
        if col in df2.columns:
            print(f"\n{label}:")
            print(f"  MAE: {df2[col].abs().mean():.4f} kJ/mol")
            print(f"  RMSE: {np.sqrt((df2[col] ** 2).mean()):.4f} kJ/mol")
            print(f"  Mean: {df2[col].mean():.4f} kJ/mol")


def plot_switchover_errors():
    """
    For each crystal, plot summed CLE energy errors from all points above X.
    Creates subplots showing AP2/AP3/UMA/AP3+D4 switchover to reference methods.
    """
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    from matplotlib.ticker import AutoMinorLocator

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
    df_apprx = pd.read_pickle("./crystals_ap2_ap3_des_results_mol_apprx.pkl")
    df_bm = pd.read_pickle("./crystals_ap2_ap3_des_results_mol_bm.pkl")

    # df_bm["d4_s IE (kJ/mol)"] *= kcalmol_to_kjmol
    # df_apprx["d4_s IE (kJ/mol)"] *= kcalmol_to_kjmol
    df_bm['ap3+d4'] = df_bm['AP3 TOTAL']
    df_apprx['ap3+d4'] = df_apprx['AP3 TOTAL']

    df_bm['d'] = df_bm["Minimum Monomer Separations (A) CCSD(T)/CBS"]
    df_bm['ref'] = df_bm["Non-Additive MB Energy (kJ/mol) CCSD(T)/CBS"]
    df_apprx['d'] = df_apprx["Minimum Monomer Separations (A) sapt0-dz-aug"]
    df_apprx['ref'] = df_apprx["Non-Additive MB Energy (kJ/mol) sapt0-dz-aug"]
    for i in ["uma-s-1p1", "uma-m-1p1"]:
        df_uma_bm = pd.read_pickle(f"./crystals_ap2_ap3_results_{i}_mol_bm.pkl")
        df_bm[f"{i} IE (kJ/mol)"] = df_uma_bm[f"{i} IE (kJ/mol)"]
        df_uma_apprx = pd.read_pickle(
            f"./crystals_ap2_ap3_results_{i}_mol_apprx.pkl"
        )
        df_apprx[f"{i} IE (kJ/mol)"] = df_uma_apprx[f"{i} IE (kJ/mol)"]
        # where uma-m-1p1 IE (kJ/mol) 0, use ap3_d_elst+ap3_classical_ind_energy
        uma_ap3_lr = []
        df_bm.sort_values(by="Minimum Monomer Separations (A) CCSD(T)/CBS", inplace=True)
        for n, r in df_bm.iterrows():
            if r['Minimum Monomer Separations (A) CCSD(T)/CBS'] > 6.0:
            # if r[f"{i} IE (kJ/mol)"] == 0.0:
                # val = r["ap3_d_elst"] + r["ap3_classical_ind_energy"]
                val = r["ap3_d_elst"] + r["ap3_classical_ind_energy"] + r["d4_s IE (kJ/mol)"]
                uma_ap3_lr.append(val)
            else:
                uma_ap3_lr.append(r[f"{i} IE (kJ/mol)"])
            # if r['AP3 DISP'] == 0.0:
            #     df_bm.loc[n, 'ap3+d4'] += r["d4_s IE (kJ/mol)"] / 2
            #     # print ap3 total, d4, ap3+d4 :4f
            #     print(f"{r['crystal bm']:19s}:{r['d']:.1f}, {r['ref']:+.4f}|{df_bm.loc[n, 'ap3+d4']:+.4f}={r['AP3 TOTAL']:+.4f}+{r['d4_s IE (kJ/mol)']:+.4f}")
        # print(df_bm[['crystal bm', 'd', 'ref', f"{i} IE (kJ/mol)", "ap3_d_elst", "ap3_classical_ind_energy"]])
        # print(df_bm[['crystal bm', 'd', 'ref', f"{i} IE (kJ/mol)", "ap3_d_elst", "d4_s IE (kJ/mol)"]])
        # print(df_bm[['crystal bm', 'd', 'ref', f"{i} IE (kJ/mol)", "ap3_d_elst", "d4_s IE (kJ/mol)"]])
        df_bm[f"{i}+ap3_lr IE (kJ/mol)"] = uma_ap3_lr
        uma_ap3_lr = []
        for n, r in df_apprx.iterrows():
            # if r[f"{i} IE (kJ/mol)"] == 0.0:
            if r['Minimum Monomer Separations (A) CCSD(T)/CBS'] > 6.0:
                val = r["ap3_d_elst"] + r["ap3_classical_ind_energy"] + r["d4_s IE (kJ/mol)"]
                uma_ap3_lr.append(val)
            else:
                uma_ap3_lr.append(r[f"{i} IE (kJ/mol)"])
            # if r['AP3 DISP'] == 0.0:
            #     df_apprx.loc[n, 'ap3+d4'] += r["d4_s IE (kJ/mol)"] / 2
        df_apprx[f"{i}+ap3_lr IE (kJ/mol)"] = uma_ap3_lr

    pd.set_option('display.max_rows', None)
    pd.set_option('display.float_format', '{:.4f}'.format)
    df_bm['ap3+d4'] = df_bm['AP3 TOTAL'] + df_bm["d4_s IE (kJ/mol)"] - df_bm['AP3 DISP']
    df_apprx['ap3+d4'] = df_apprx['AP3 TOTAL'] + df_apprx["d4_s IE (kJ/mol)"] - df_apprx['AP3 DISP']
    print(df_apprx[['crystal apprx', 'd', "d4_s IE (kJ/mol)", 'AP3 DISP']])


    # Get unique crystals
    crystals_apprx = sorted(df_apprx["crystal apprx"].dropna().unique())
    crystals_bm = sorted(df_bm["crystal bm"].dropna().unique())

    # Get unique crystals
    crystals_apprx = sorted(df_apprx["crystal apprx"].dropna().unique())
    crystals_bm = sorted(df_bm["crystal bm"].dropna().unique())

    # Combine and get all unique crystals
    all_crystals = sorted(list(set(crystals_apprx) | set(crystals_bm)))
    N = len(all_crystals)

    print(f"Processing {N} crystals for switchover error plots")

    # Create figure with subplots (2 columns: apprx and bm)
    fig, axes = plt.subplots(N, 2, figsize=(12, N * 2 + 2))
    if N == 1:
        axes = axes.reshape(1, -1)

    # Define separation distance range
    increment = 0.25
    sep_distances = np.arange(1.0, 15.0 + 0.05, increment)

    # Process each crystal
    for idx, crystal in enumerate(all_crystals):
        print(f"\nProcessing crystal {idx + 1}/{N}: {crystal}")

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
                df_c['uma-s-1p1_cle'] = df_c.apply(
                    lambda r: r["uma-s-1p1 IE (kJ/mol)"] * r[num_rep_col] / int(r[nmer_col][0])
                    if pd.notnull(r[nmer_col]) and "uma-s-1p1 IE (kJ/mol)" in r
                    else 0,
                    axis=1,
                )
                df_c['uma-m-1p1_cle'] = df_c.apply(
                    lambda r: r["uma-m-1p1 IE (kJ/mol)"] * r[num_rep_col] / int(r[nmer_col][0])
                    if pd.notnull(r[nmer_col]) and "uma-m-1p1 IE (kJ/mol)" in r
                    else 0,
                    axis=1,
                )
                df_c['ap3_d4_cle'] = df_c.apply(
                    lambda r: r["ap3+d4"] * r[num_rep_col] / int(r[nmer_col][0])
                    if pd.notnull(r[nmer_col]) and "ap3+d4" in r
                    else 0,
                    axis=1,
                )

                # Calculate switchover errors for different cutoffs
                ap2_errors = []
                ap3_errors = []
                uma_s_errors = []
                uma_m_errors = []
                ap3_d4_errors = []
                zeros = []

                for d in sep_distances:
                    # Hybrid: use ML method above d, reference method below d
                    ref_below = df_c[df_c[mms_col] < d]["ref_cle"].sum()
                    ap2_above = df_c[df_c[mms_col] >= d]["ap2_cle"].sum()
                    ap2_hybrid_total = ap2_above + ref_below

                    ap3_above = df_c[df_c[mms_col] >= d]["ap3_cle"].sum()
                    ap3_hybrid_total = ap3_above + ref_below

                    uma_s_above = df_c[df_c[mms_col] >= d]["uma-s-1p1_cle"].sum()
                    uma_s_hybrid_total = uma_s_above + ref_below

                    uma_m_above = df_c[df_c[mms_col] >= d]["uma-m-1p1_cle"].sum()
                    uma_m_hybrid_total = uma_m_above + ref_below

                    ap3_d4_above = df_c[df_c[mms_col] >= d]["ap3_d4_cle"].sum()
                    ap3_d4_hybrid_total = ap3_d4_above + ref_below

                    # Reference total
                    ref_total = df_c["ref_cle"].sum()

                    # Error = hybrid - reference
                    ap2_errors.append(ap2_hybrid_total - ref_total)
                    ap3_errors.append(ap3_hybrid_total - ref_total)
                    uma_s_errors.append(uma_s_hybrid_total - ref_total)
                    uma_m_errors.append(uma_m_hybrid_total - ref_total)
                    ap3_d4_errors.append(ap3_d4_hybrid_total - ref_total)
                    zeros.append(ref_below-ref_total)

                # Plot
                ax_apprx.plot(
                    sep_distances,
                    ap2_errors,
                    "o-",
                    label="AP2",
                    markersize=4,
                    linewidth=1.5,
                    alpha=0.8,
                )
                ax_apprx.plot(
                    sep_distances,
                    ap3_errors,
                    "s-",
                    label="AP3",
                    markersize=4,
                    linewidth=1.5,
                    alpha=0.8,
                )
                ax_apprx.plot(
                    sep_distances,
                    ap3_d4_errors,
                    "^-",
                    label="AP3+D4",
                    markersize=4,
                    linewidth=1.5,
                    alpha=0.8,
                )
                ax_apprx.plot(
                    sep_distances,
                    uma_s_errors,
                    "v-",
                    label="UMA-s",
                    markersize=4,
                    linewidth=1.5,
                    alpha=0.8,
                )
                ax_apprx.plot(
                    sep_distances,
                    uma_m_errors,
                    "d-",
                    label="UMA-m",
                    markersize=4,
                    linewidth=1.5,
                    alpha=0.8,
                )
                ax_apprx.plot(
                    sep_distances,
                    zeros,
                    "-",
                    color='black',
                    label="Pred=0",
                    linewidth=2.0,
                )
                ax_apprx.axhline(
                    y=0, color="black", linestyle="--", linewidth=1.0, alpha=0.5
                )
                ax_apprx.axvline(
                    x=6.0, color="gray", linestyle="--", linewidth=1.0, alpha=0.5
                )
                ax_apprx.set_ylabel("CLE Error (kJ/mol)", fontsize=10)
                ax_apprx.set_title(f"{crystal}\nvs SAPT0/aDZ", fontsize=10)

                if idx == 0:
                    ax_apprx.legend(loc="best", fontsize=8)

                ax_apprx.set_ylim(-5, 5)

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
                df_c['uma-s-1p1_cle'] = df_c.apply(
                    lambda r: r["uma-s-1p1 IE (kJ/mol)"] * r[num_rep_col] / int(r[nmer_col][0])
                    if pd.notnull(r[nmer_col]) and "uma-s-1p1 IE (kJ/mol)" in r
                    else 0,
                    axis=1,
                )
                df_c['uma-m-1p1_cle'] = df_c.apply(
                    lambda r: r["uma-m-1p1 IE (kJ/mol)"] * r[num_rep_col] / int(r[nmer_col][0])
                    if pd.notnull(r[nmer_col]) and "uma-m-1p1 IE (kJ/mol)" in r
                    else 0,
                    axis=1,
                )
                df_c['ap3_d4_cle'] = df_c.apply(
                    lambda r: r["ap3+d4"] * r[num_rep_col] / int(r[nmer_col][0])
                    if pd.notnull(r[nmer_col]) and "ap3+d4" in r
                    else 0,
                    axis=1,
                )

                # Calculate switchover errors for different cutoffs
                ap2_errors = []
                ap3_errors = []
                uma_s_errors = []
                uma_m_errors = []
                ap3_d4_errors = []
                zeros = []

                for d in sep_distances:
                    # Hybrid: use ML method above d, reference method below d
                    ref_below = df_c[df_c[mms_col] < d]["ref_cle"].sum()
                    ap2_above = df_c[df_c[mms_col] >= d]["ap2_cle"].sum()
                    ap2_hybrid_total = ap2_above + ref_below

                    ap3_above = df_c[df_c[mms_col] >= d]["ap3_cle"].sum()
                    ap3_hybrid_total = ap3_above + ref_below

                    uma_s_above = df_c[df_c[mms_col] >= d]["uma-s-1p1_cle"].sum()
                    uma_s_hybrid_total = uma_s_above + ref_below

                    uma_m_above = df_c[df_c[mms_col] >= d]["uma-m-1p1_cle"].sum()
                    uma_m_hybrid_total = uma_m_above + ref_below

                    ap3_d4_above = df_c[df_c[mms_col] >= d]["ap3_d4_cle"].sum()
                    ap3_d4_hybrid_total = ap3_d4_above + ref_below

                    # Reference total
                    ref_total = df_c["ref_cle"].sum()

                    # Error = hybrid - reference
                    ap2_errors.append(ap2_hybrid_total - ref_total)
                    ap3_errors.append(ap3_hybrid_total - ref_total)
                    uma_s_errors.append(uma_s_hybrid_total - ref_total)
                    uma_m_errors.append(uma_m_hybrid_total - ref_total)
                    ap3_d4_errors.append(ap3_d4_hybrid_total - ref_total)
                    zeros.append(ref_below-ref_total)

                # Plot
                ax_bm.plot(
                    sep_distances,
                    ap2_errors,
                    "o-",
                    label="AP2",
                    markersize=4,
                    linewidth=1.5,
                    alpha=0.8,
                )
                ax_bm.plot(
                    sep_distances,
                    ap3_errors,
                    "s-",
                    label="AP3",
                    markersize=4,
                    linewidth=1.5,
                    alpha=0.8,
                )
                ax_bm.plot(
                    sep_distances,
                    ap3_d4_errors,
                    "^-",
                    label="AP3+D4",
                    markersize=4,
                    linewidth=1.5,
                    alpha=0.8,
                )
                ax_bm.plot(
                    sep_distances,
                    uma_s_errors,
                    "v-",
                    label="UMA-s",
                    markersize=4,
                    linewidth=1.5,
                    alpha=0.8,
                )
                ax_bm.plot(
                    sep_distances,
                    uma_m_errors,
                    "d-",
                    label="UMA-m",
                    markersize=4,
                    linewidth=1.5,
                    alpha=0.8,
                )
                ax_bm.plot(
                    sep_distances,
                    zeros,
                    "-",
                    color='black',
                    label="Pred=0",
                    linewidth=2.0,
                )
                ax_bm.axhline(
                    y=0, color="black", linestyle="--", linewidth=1.0, alpha=0.5
                )
                ax_bm.axvline(
                    x=6.0, color="gray", linestyle="--", linewidth=1.0, alpha=0.5
                )
                ax_bm.set_ylabel("CLE Error (kJ/mol)", fontsize=10)
                ax_bm.set_title(f"{crystal}\nvs CCSD(T)/CBS", fontsize=10)

                if idx == 0:
                    ax_bm.legend(loc="best", fontsize=8)

                ax_bm.set_ylim(-5, 5)

        # Style axes
        for ax in [ax_apprx, ax_bm]:
            ax.tick_params(which="major", direction="in", top=True, right=True)
            # no gridlines
            ax.xaxis.set_minor_locator(AutoMinorLocator(n=2))
            ax.yaxis.set_minor_locator(AutoMinorLocator(n=2))
            ax.tick_params(which="minor", direction="in", top=True, right=True, length=4)


            # Only show x-label on bottom row
            if idx == N - 1:
                ax.set_xlabel("Switchover Distance $R^*$ (Å)", fontsize=11)
            else:
                ax.tick_params(axis="x", labelbottom=False)

    # Adjust layout
    plt.tight_layout()

    # Save figure
    output_path = "./x23_plots/switchover_errors_all_crystals.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\nFigure saved to: {output_path}")

    return fig, axes


def plot_crystal_lattice_energies(sft=False):
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

    # Define consistent colors for all methods
    method_colors = {
        "AP2": "#1f77b4",      # blue
        "AP3": "#ff7f0e",      # orange
        "UMA-s": "#2ca02c",    # green
        "UMA-m": "#d62728",    # red
        "ref": "black",         # black for reference
    }

    # Load dataframes
    if sft:
        df_apprx = pd.read_pickle("./sft_crystals_ap2_ap3_results_mol_apprx.pkl")
        df_bm = pd.read_pickle("./sft_crystals_ap2_ap3_results_mol_bm.pkl")
        df_uma_s_bm = pd.read_pickle("./crystals_ap2_ap3_results_uma-s-1p1_mol_bm.pkl")
        print(df_uma_s_bm["uma-s-1p1_le_contribution"])
        output_path = "./x23_plots/CLE_all_crystals_sft.png"
        output_violin_apprx = "./x23_plots/ap2_ap3_errors_vs_sapt0_sft.png"
        output_violin_bm = "./x23_plots/ap2_ap3_errors_vs_ccsdt_cbs_sft.png"
    else:
        df_apprx = pd.read_pickle("./crystals_ap2_ap3_results_mol_apprx.pkl")
        df_bm = pd.read_pickle("./crystals_ap2_ap3_results_mol_bm.pkl")
        output_path = "./x23_plots/CLE_all_crystals.png"
        output_violin_apprx = "./x23_plots/ap2_ap3_errors_vs_sapt0.png"
        output_violin_bm = "./x23_plots/ap2_ap3_errors_vs_ccsdt_cbs.png"

    for i in ["uma-s-1p1", "uma-m-1p1"]:
        df_uma_bm = pd.read_pickle(f"./crystals_ap2_ap3_results_{i}_mol_bm.pkl")
        df_bm[f"{i} IE (kJ/mol)"] = df_uma_bm[f"{i} IE (kJ/mol)"]
        df_uma_apprx = pd.read_pickle(
            f"./crystals_ap2_ap3_results_{i}_mol_apprx.pkl"
        )
        df_apprx[f"{i} IE (kJ/mol)"] = df_uma_apprx[f"{i} IE (kJ/mol)"]

    # Get unique crystals
    crystals_apprx = sorted(df_apprx["crystal apprx"].dropna().unique())
    crystals_bm = sorted(df_bm["crystal bm"].dropna().unique())

    # Combine and get all unique crystals
    all_crystals = sorted(list(set(crystals_apprx) | set(crystals_bm)))
    N = len(all_crystals)

    print(f"Processing {N} crystals for CLE plots")

    # Create figure with subplots (2 columns: apprx and bm)
    fig, axes = plt.subplots(N, 2, figsize=(12, N * 2 + 2))
    if N == 1:
        axes = axes.reshape(1, -1)

    # Define separation distance range
    increment = 0.25
    sep_distances = np.arange(1.0, 15.0 + 0.05, increment)

    ap2_full_cle_errors_sapt0_aDZ = []
    ap3_full_cle_errors_sapt0_aDZ = []
    uma_s_full_cle_errors_sapt0_aDZ = []
    uma_m_full_cle_errors_sapt0_aDZ = []
    ap2_full_cle_errors_ccsd_t_CBS = []
    ap3_full_cle_errors_ccsd_t_CBS = []
    uma_s_full_cle_errors_ccsd_t_CBS = []
    uma_m_full_cle_errors_ccsd_t_CBS = []

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
                df_c['uma-s-1p1_cle'] = df_c.apply(
                    lambda r: r["uma-s-1p1 IE (kJ/mol)"] * r[num_rep_col] / int(r[nmer_col][0])
                    if pd.notnull(r[nmer_col]) and "uma-s-1p1 IE (kJ/mol)" in r
                    else 0,
                    axis=1,
                )
                df_c['uma-m-1p1_cle'] = df_c.apply(
                    lambda r: r["uma-m-1p1 IE (kJ/mol)"] * r[num_rep_col] / int(r[nmer_col][0])
                    if pd.notnull(r[nmer_col]) and "uma-m-1p1 IE (kJ/mol)" in r
                    else 0,
                    axis=1,
                )

                ap2_2b_energies = []
                ap3_2b_energies = []
                uma_s_2b_energies = []
                uma_m_2b_energies = []
                ref_2b_energies = []
                for d in sep_distances:
                    ap2_2b_energies.append(df_c[df_c[mms_col] <= d]["ap2_cle"].sum())
                    ap3_2b_energies.append(df_c[df_c[mms_col] <= d]["ap3_cle"].sum())
                    uma_s_2b_energies.append(df_c[df_c[mms_col] <= d]["uma-s-1p1_cle"].sum())
                    uma_m_2b_energies.append(df_c[df_c[mms_col] <= d]["uma-m-1p1_cle"].sum())
                    ref_2b_energies.append(df_c[df_c[mms_col] <= d]["ref_cle"].sum())


                # Plot
                ax_apprx.plot(
                    sep_distances,
                    ap2_2b_energies,
                    "o-",
                    color=method_colors["AP2"],
                    label="AP2",
                    markersize=4,
                    linewidth=1.5,
                    alpha=0.8,
                )
                ax_apprx.plot(
                    sep_distances,
                    ap3_2b_energies,
                    "s-",
                    color=method_colors["AP3"],
                    label="AP3",
                    markersize=4,
                    linewidth=1.5,
                    alpha=0.8,
                )
                ax_apprx.plot(
                    sep_distances,
                    uma_s_2b_energies,
                    "^-",
                    color=method_colors["UMA-s"],
                    label="UMA-s-1p1",
                    markersize=4,
                    linewidth=1.5,
                    alpha=0.8,
                )
                ax_apprx.plot(
                    sep_distances,
                    uma_m_2b_energies,
                    "v-",
                    color=method_colors["UMA-m"],
                    label="UMA-m-1p1",
                    markersize=4,
                    linewidth=1.5,
                    alpha=0.8,
                )
                ax_apprx.plot(
                    sep_distances,
                    ref_2b_energies,
                    "-",
                    color=method_colors["ref"],
                    label="SAPT0/aDZ",
                    markersize=4,
                    linewidth=1.5,
                    alpha=0.8,
                )
                ax_apprx.axhline(
                    y=0, color="black", linestyle="--", linewidth=1.0, alpha=0.5
                )
                ax_apprx.set_ylabel("CLE Error (kJ/mol)", fontsize=10)
                ax_apprx.set_title(f"{crystal}\nvs SAPT0/aDZ", fontsize=10)

                if idx == 0:
                    ax_apprx.legend(loc="best", fontsize=8)

                # ax_apprx.set_ylim(-5, 5)
                if not abs(ap2_2b_energies[-1]) < 8e-8:
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
                df_c['uma-s-1p1_cle'] = df_c.apply(
                    lambda r: r["uma-s-1p1 IE (kJ/mol)"] * r[num_rep_col] / int(r[nmer_col][0])
                    if pd.notnull(r[nmer_col]) and "uma-s-1p1 IE (kJ/mol)" in r
                    else 0,
                    axis=1,
                )
                df_c['uma-m-1p1_cle'] = df_c.apply(
                    lambda r: r["uma-m-1p1 IE (kJ/mol)"] * r[num_rep_col] / int(r[nmer_col][0])
                    if pd.notnull(r[nmer_col]) and "uma-m-1p1 IE (kJ/mol)" in r
                    else 0,
                    axis=1,
                )


                ap2_2b_energies = []
                ap3_2b_energies = []
                uma_s_2b_energies = []
                uma_m_2b_energies = []
                ref_2b_energies = []
                for d in sep_distances:
                    ap2_2b_energies.append(df_c[df_c[mms_col] <= d]["ap2_cle"].sum())
                    ap3_2b_energies.append(df_c[df_c[mms_col] <= d]["ap3_cle"].sum())
                    uma_s_2b_energies.append(df_c[df_c[mms_col] <= d]["uma-s-1p1_cle"].sum())
                    uma_m_2b_energies.append(df_c[df_c[mms_col] <= d]["uma-m-1p1_cle"].sum())
                    ref_2b_energies.append(df_c[df_c[mms_col] <= d]["ref_cle"].sum())

                # Plot
                ax_bm.plot(
                    sep_distances,
                    ap2_2b_energies,
                    "o-",
                    color=method_colors["AP2"],
                    label="AP2",
                    markersize=4,
                    linewidth=1.5,
                    alpha=0.8,
                )
                ax_bm.plot(
                    sep_distances,
                    uma_s_2b_energies,
                    "^-",
                    color=method_colors["UMA-s"],
                    label="UMA-s-1p1",
                    markersize=4,
                    linewidth=1.5,
                    alpha=0.8,
                )
                ax_bm.plot(
                    sep_distances,
                    uma_m_2b_energies,
                    "v-",
                    color=method_colors["UMA-m"],
                    label="UMA-m-1p1",
                    markersize=4,
                    linewidth=1.5,
                    alpha=0.8,
                )
                ax_bm.plot(
                    sep_distances,
                    ap3_2b_energies,
                    "s-",
                    color=method_colors["AP3"],
                    label="AP3",
                    markersize=4,
                    linewidth=1.5,
                    alpha=0.8,
                )
                ax_bm.plot(
                    sep_distances,
                    ref_2b_energies,
                    "-",
                    color=method_colors["ref"],
                    label="CCSD(T)/CBS",
                    markersize=4,
                    linewidth=1.5,
                    alpha=0.8,
                )
                ax_bm.axhline(
                    y=0, color="black", linestyle="--", linewidth=1.0, alpha=0.5
                )
                ax_bm.set_ylabel("CLE Error (kJ/mol)", fontsize=10)
                ax_bm.set_title(f"{crystal}\nvs CCSD(T)/CBS", fontsize=10)

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
                    # print(
                    #     ap2_2b_energies[-1] - ref_2b_energies[-1],
                    #     ap2_2b_energies[-1],
                    #     ref_2b_energies[-1],
                    # )

                # ax_bm.set_ylim(-5, 5)

        # Style axes
        for ax in [ax_apprx, ax_bm]:
            ax.grid(True, alpha=0.3, linestyle=":")
            ax.tick_params(which="major", direction="in", top=True, right=True)

            # Only show x-label on bottom row
            if idx == N - 1:
                ax.set_xlabel("Min. Mon. Sep. (Å)", fontsize=11)
            else:
                ax.tick_params(axis="x", labelbottom=False)

    # Filter out nans from ap2_full_cle_errors_ccsd_t_CBS
    ap2_full_cle_errors_ccsd_t_CBS = [
        x for x in ap2_full_cle_errors_ccsd_t_CBS if x != 0.0
    ]
    ap3_full_cle_errors_ccsd_t_CBS = [
        x for x in ap3_full_cle_errors_ccsd_t_CBS if x != 0.0
    ]
    ap2_full_cle_errors_sapt0_aDZ = [
        x for x in ap2_full_cle_errors_sapt0_aDZ if x != 0.0
    ]
    ap3_full_cle_errors_sapt0_aDZ = [
        x for x in ap3_full_cle_errors_sapt0_aDZ if x != 0.0
    ]

    me_ap2_sapt = np.sum(np.array(ap2_full_cle_errors_sapt0_aDZ)) / len(
        ap2_full_cle_errors_sapt0_aDZ
    )
    me_ap3_sapt = np.sum(np.array(ap3_full_cle_errors_sapt0_aDZ)) / len(
        ap3_full_cle_errors_sapt0_aDZ
    )
    me_ap2_ccsd = np.sum(np.array(ap2_full_cle_errors_ccsd_t_CBS)) / len(
        ap2_full_cle_errors_ccsd_t_CBS
    )
    me_ap3_ccsd = np.sum(np.array(ap3_full_cle_errors_ccsd_t_CBS)) / len(
        ap3_full_cle_errors_ccsd_t_CBS
    )
    print(f"{me_ap2_sapt=:.4f} kJ/mol")
    print(f"{me_ap3_sapt=:.4f} kJ/mol")
    print(f"{me_ap2_ccsd=:.4f} kJ/mol")
    print(f"{me_ap3_ccsd=:.4f} kJ/mol")
    mae_ap2_sapt = np.sum(np.abs(np.array(ap2_full_cle_errors_sapt0_aDZ))) / len(
        ap2_full_cle_errors_sapt0_aDZ
    )
    mae_ap3_sapt = np.sum(np.abs(np.array(ap3_full_cle_errors_sapt0_aDZ))) / len(
        ap3_full_cle_errors_sapt0_aDZ
    )
    mae_ap2_ccsd = np.sum(np.abs(np.array(ap2_full_cle_errors_ccsd_t_CBS))) / len(
        ap2_full_cle_errors_ccsd_t_CBS
    )
    mae_ap3_ccsd = np.sum(np.abs(np.array(ap3_full_cle_errors_ccsd_t_CBS))) / len(
        ap3_full_cle_errors_ccsd_t_CBS
    )
    print(f"{mae_ap2_sapt=:.4f} kJ/mol")
    print(f"{mae_ap3_sapt=:.4f} kJ/mol")
    print(f"{mae_ap2_ccsd=:.4f} kJ/mol")
    print(f"{mae_ap3_ccsd=:.4f} kJ/mol")

    # Adjust layout
    plt.tight_layout()

    # Save figure
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\nFigure saved to: {output_path}")

    error_df1 = pd.DataFrame(
        {
            "AP2 vs SAPT0 error": ap2_full_cle_errors_sapt0_aDZ,
            "AP3 vs SAPT0 error": ap3_full_cle_errors_sapt0_aDZ,
            "UMA-s vs SAPT0 error": uma_s_full_cle_errors_sapt0_aDZ,
            "UMA-m vs SAPT0 error": uma_m_full_cle_errors_sapt0_aDZ,
        }
    )

    # Prepare df1 for violin plot
    dfs1 = [
        {
            "df": error_df1,
            "basis": "",
            "label": "SAPT0/aDZ Reference",
            "ylim": [[-20, 20]],
        }
    ]

    # Labels and columns for df1 (multiple methods)
    df1_labels = {
        "AP2": "AP2 vs SAPT0 error",
        "AP3": "AP3 vs SAPT0 error",
        "UMA-s": "UMA-s vs SAPT0 error",
        "UMA-m": "UMA-m vs SAPT0 error",
    }

    method_cols = [
        col
        for col in df_apprx.columns
        if "Non-Additive MB Energy (kJ/mol)" in col and "sapt0-dz-aug" not in col
    ]
    # Add other methods if available
    for method_col in method_cols[:3]:
        method_name = method_col.replace("Non-Additive MB Energy (kJ/mol) ", "")
        if f"{method_name} error" in df_apprx.columns:
            df1_labels[method_name] = f"{method_name} error"

    error_df2 = pd.DataFrame(
        {
            "AP2 vs CCSD(T)/CBS error": ap2_full_cle_errors_ccsd_t_CBS,
            "AP3 vs CCSD(T)/CBS error": ap3_full_cle_errors_ccsd_t_CBS,
            "UMA-s vs CCSD(T)/CBS error": uma_s_full_cle_errors_ccsd_t_CBS,
            "UMA-m vs CCSD(T)/CBS error": uma_m_full_cle_errors_ccsd_t_CBS,
        }
    )

    # Prepare df2 for violin plot
    dfs2 = [
        {
            "df": error_df2,
            "basis": "",
            "label": "CCSD(T)/CBS Reference",
            "ylim": [[-20, 20]],
        }
    ]

    # Labels and columns for df2 (only AP2 and AP3 vs CCSD(T)/CBS)
    df2_labels = {
        "AP2": "AP2 vs CCSD(T)/CBS error",
        "AP3": "AP3 vs CCSD(T)/CBS error",
        "UMA-s": "UMA-s vs CCSD(T)/CBS error",
        "UMA-m": "UMA-m vs CCSD(T)/CBS error",
    }

    # Create violin plot for df1
    print("\nCreating violin plot for df1 (approximate methods)...")
    error_statistics.violin_plot_table_multi_SAPT_components(
        dfs1,
        df_labels_and_columns_total=df1_labels,
        output_filename=output_violin_apprx,
        grid_heights=[0.3, 1.0],
        grid_widths=[1],
        legend_loc="upper left",
        annotations_texty=0.3,
        figure_size=(6, 2.5),
        add_title=False,
        name_violin=False,
        ylabel=r"Error (kJ$\cdot$mol$^{-1}$)",
    )
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
    # Print summary statistics
    return fig, axes


def plot_crystal_lattice_energies_with_switchover(switchover=2.5):
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
    df_apprx = pd.read_pickle("./crystals_ap2_ap3_results_mol_apprx.pkl")
    df_bm = pd.read_pickle("./crystals_ap2_ap3_results_mol_bm.pkl")

    # Get unique crystals
    crystals_apprx = sorted(df_apprx["crystal apprx"].dropna().unique())
    crystals_bm = sorted(df_bm["crystal bm"].dropna().unique())

    # Combine and get all unique crystals
    all_crystals = sorted(list(set(crystals_apprx) | set(crystals_bm)))
    N = len(all_crystals)

    print(f"Processing {N} crystals for switchover error plots")

    # Create figure with subplots (2 columns: apprx and bm)
    fig, axes = plt.subplots(N, 2, figsize=(12, N * 2 + 2))
    if N == 1:
        axes = axes.reshape(1, -1)

    # Define separation distance range
    increment = 0.25
    sep_distances = np.arange(1.0, 15.0 + 0.05, increment)

    ap2_full_cle_errors_sapt0_aDZ = []
    ap3_full_cle_errors_sapt0_aDZ = []
    ap2_full_cle_errors_ccsd_t_CBS = []
    ap3_full_cle_errors_ccsd_t_CBS = []

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

                ap2_2b_energies = []
                ap3_2b_energies = []
                ref_2b_energies = []
                for d in sep_distances:
                    ap2_above = df_c[
                        (df_c[mms_col] >= switchover) & (df_c[mms_col] < d)
                    ]["ap2_cle"].sum()
                    ap2_below = df_c[
                        (df_c[mms_col] < switchover) & (df_c[mms_col] < d)
                    ]["ref_cle"].sum()
                    ap2_hybrid_total = ap2_above + ap2_below
                    ap3_above = df_c[
                        (df_c[mms_col] >= switchover) & (df_c[mms_col] < d)
                    ]["ap3_cle"].sum()
                    ap3_below = df_c[
                        (df_c[mms_col] < switchover) & (df_c[mms_col] < d)
                    ]["ref_cle"].sum()
                    ref_below = df_c[df_c[mms_col] < d]["ref_cle"].sum()
                    ap3_hybrid_total = ap3_above + ap3_below
                    ap2_2b_energies.append(ap2_hybrid_total)
                    ap3_2b_energies.append(ap3_hybrid_total)
                    ref_2b_energies.append(ref_below)

                # Plot
                ax_apprx.plot(
                    sep_distances,
                    ap2_2b_energies,
                    "o-",
                    label="AP2",
                    markersize=4,
                    linewidth=1.5,
                    alpha=0.8,
                )
                ax_apprx.plot(
                    sep_distances,
                    ap3_2b_energies,
                    "s-",
                    label="AP3",
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
                ax_apprx.axhline(
                    y=0, color="black", linestyle="--", linewidth=1.0, alpha=0.5
                )
                ax_apprx.set_ylabel("CLE Error (kJ/mol)", fontsize=10)
                ax_apprx.set_title(f"{crystal}\nvs SAPT0/aDZ", fontsize=10)

                if idx == 0:
                    ax_apprx.legend(loc="best", fontsize=8)

                ap2_full_cle_errors_sapt0_aDZ.append(
                    ap2_2b_energies[-1] - ref_2b_energies[-1]
                )
                ap3_full_cle_errors_sapt0_aDZ.append(
                    ap3_2b_energies[-1] - ref_2b_energies[-1]
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

                ap2_2b_energies = []
                ap3_2b_energies = []
                ref_2b_energies = []
                print(
                    f"{crystal=}, cnt below SO: {
                        len(df_c[(df_c[mms_col] < switchover)])
                    }/{len(df_c)}"
                )
                for d in sep_distances:
                    ap2_above = df_c[
                        (df_c[mms_col] >= switchover) & (df_c[mms_col] < d)
                    ]["ap2_cle"].sum()
                    ap2_below = df_c[
                        (df_c[mms_col] < switchover) & (df_c[mms_col] < d)
                    ]["ref_cle"].sum()
                    ap2_hybrid_total = ap2_above + ap2_below
                    ap3_above = df_c[
                        (df_c[mms_col] >= switchover) & (df_c[mms_col] < d)
                    ]["ap3_cle"].sum()
                    ap3_below = df_c[
                        (df_c[mms_col] < switchover) & (df_c[mms_col] < d)
                    ]["ref_cle"].sum()
                    ap3_hybrid_total = ap3_above + ap3_below
                    ref_below = df_c[df_c[mms_col] < d]["ref_cle"].sum()
                    ap2_2b_energies.append(ap2_hybrid_total)
                    ap3_2b_energies.append(ap3_hybrid_total)
                    ref_2b_energies.append(ref_below)

                # Plot
                ax_bm.plot(
                    sep_distances,
                    ap2_2b_energies,
                    "o-",
                    label="AP2",
                    markersize=4,
                    linewidth=1.5,
                    alpha=0.8,
                )
                ax_bm.plot(
                    sep_distances,
                    ap3_2b_energies,
                    "s-",
                    label="AP3",
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
                ax_bm.axhline(
                    y=0, color="black", linestyle="--", linewidth=1.0, alpha=0.5
                )
                ax_bm.set_ylabel("CLE Error (kJ/mol)", fontsize=10)
                ax_bm.set_title(f"{crystal}\nvs CCSD(T)/CBS", fontsize=10)

                if idx == 0:
                    ax_bm.legend(loc="best", fontsize=8)

                ap2_full_cle_errors_ccsd_t_CBS.append(
                    ap2_2b_energies[-1] - ref_2b_energies[-1]
                )
                ap3_full_cle_errors_ccsd_t_CBS.append(
                    ap3_2b_energies[-1] - ref_2b_energies[-1]
                )
                # ax_bm.set_ylim(-5, 5)

        # Style axes
        for ax in [ax_apprx, ax_bm]:
            ax.grid(True, alpha=0.3, linestyle=":")
            ax.tick_params(which="major", direction="in", top=True, right=True)

            # Only show x-label on bottom row
            if idx == N - 1:
                ax.set_xlabel("Min. Mon. Sep. (Å)", fontsize=11)
            else:
                ax.tick_params(axis="x", labelbottom=False)

    # Adjust layout
    plt.tight_layout()
    print("Error CLE statistics")
    mae_ap2_sapt = np.sum(np.array(ap2_full_cle_errors_sapt0_aDZ)) / len(
        ap2_full_cle_errors_sapt0_aDZ
    )
    mae_ap3_sapt = np.sum(np.array(ap3_full_cle_errors_sapt0_aDZ)) / len(
        ap3_full_cle_errors_sapt0_aDZ
    )
    mae_ap2_ccsd = np.sum(np.array(ap2_full_cle_errors_ccsd_t_CBS)) / len(
        ap2_full_cle_errors_ccsd_t_CBS
    )
    mae_ap3_ccsd = np.sum(np.array(ap3_full_cle_errors_ccsd_t_CBS)) / len(
        ap3_full_cle_errors_ccsd_t_CBS
    )
    print(f"{mae_ap2_sapt=:.4f} kJ/mol")
    print(f"{mae_ap3_sapt=:.4f} kJ/mol")
    print(f"{mae_ap2_ccsd=:.4f} kJ/mol")
    print(f"{mae_ap3_ccsd=:.4f} kJ/mol")

    # Save figure
    output_path = "./x23_plots/CLE_all_crystals_switchover.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\nFigure saved to: {output_path}")
    return fig, axes


def plot_crystal_lattice_energies_with_N(N=1, sft=False, tl_N=100):
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

    # df_bm["d4_s IE (kJ/mol)"] *= kcalmol_to_kjmol
    # df_apprx["d4_s IE (kJ/mol)"] *= kcalmol_to_kjmol
    df_bm['ap3+d4'] = df_bm['AP3 TOTAL']
    df_apprx['ap3+d4'] = df_apprx['AP3 TOTAL']

    df_bm['d'] = df_bm["Minimum Monomer Separations (A) CCSD(T)/CBS"]
    df_bm['ref'] = df_bm["Non-Additive MB Energy (kJ/mol) CCSD(T)/CBS"]
    df_apprx['d'] = df_apprx["Minimum Monomer Separations (A) sapt0-dz-aug"]
    df_apprx['ref'] = df_apprx["Non-Additive MB Energy (kJ/mol) sapt0-dz-aug"]
    for i in ["uma-s-1p1", "uma-m-1p1"]:
        df_uma_bm = pd.read_pickle(f"./crystals_ap2_ap3_results_{i}_mol_bm.pkl")
        df_bm[f"{i} IE (kJ/mol)"] = df_uma_bm[f"{i} IE (kJ/mol)"]
        df_uma_apprx = pd.read_pickle(
            f"./crystals_ap2_ap3_results_{i}_mol_apprx.pkl"
        )
        df_apprx[f"{i} IE (kJ/mol)"] = df_uma_apprx[f"{i} IE (kJ/mol)"]
        # where uma-m-1p1 IE (kJ/mol) 0, use ap3_d_elst+ap3_classical_ind_energy
        uma_ap3_lr = []
        df_bm.sort_values(by="Minimum Monomer Separations (A) CCSD(T)/CBS", inplace=True)
        for n, r in df_bm.iterrows():
            if r['Minimum Monomer Separations (A) CCSD(T)/CBS'] > 6.0:
            # if r[f"{i} IE (kJ/mol)"] == 0.0:
                # val = r["ap3_d_elst"] + r["ap3_classical_ind_energy"]
                val = r["ap3_d_elst"] + r["ap3_classical_ind_energy"] + r["d4_s IE (kJ/mol)"]
                uma_ap3_lr.append(val)
            else:
                uma_ap3_lr.append(r[f"{i} IE (kJ/mol)"])
            # if r['AP3 DISP'] == 0.0:
            #     df_bm.loc[n, 'ap3+d4'] += r["d4_s IE (kJ/mol)"] / 2
            #     # print ap3 total, d4, ap3+d4 :4f
            #     print(f"{r['crystal bm']:19s}:{r['d']:.1f}, {r['ref']:+.4f}|{df_bm.loc[n, 'ap3+d4']:+.4f}={r['AP3 TOTAL']:+.4f}+{r['d4_s IE (kJ/mol)']:+.4f}")
        # print(df_bm[['crystal bm', 'd', 'ref', f"{i} IE (kJ/mol)", "ap3_d_elst", "ap3_classical_ind_energy"]])
        # print(df_bm[['crystal bm', 'd', 'ref', f"{i} IE (kJ/mol)", "ap3_d_elst", "d4_s IE (kJ/mol)"]])
        # print(df_bm[['crystal bm', 'd', 'ref', f"{i} IE (kJ/mol)", "ap3_d_elst", "d4_s IE (kJ/mol)"]])
        df_bm[f"{i}+ap3_lr IE (kJ/mol)"] = uma_ap3_lr
        uma_ap3_lr = []
        for n, r in df_apprx.iterrows():
            # if r[f"{i} IE (kJ/mol)"] == 0.0:
            if r['Minimum Monomer Separations (A) CCSD(T)/CBS'] > 6.0:
                val = r["ap3_d_elst"] + r["ap3_classical_ind_energy"] + r["d4_s IE (kJ/mol)"]
                uma_ap3_lr.append(val)
            else:
                uma_ap3_lr.append(r[f"{i} IE (kJ/mol)"])
            # if r['AP3 DISP'] == 0.0:
            #     df_apprx.loc[n, 'ap3+d4'] += r["d4_s IE (kJ/mol)"] / 2
        df_apprx[f"{i}+ap3_lr IE (kJ/mol)"] = uma_ap3_lr

    pd.set_option('display.max_rows', None)
    pd.set_option('display.float_format', '{:.4f}'.format)
    df_bm['ap3+d4'] = df_bm['AP3 TOTAL'] + df_bm["d4_s IE (kJ/mol)"] - df_bm['AP3 DISP']
    df_apprx['ap3+d4'] = df_apprx['AP3 TOTAL'] + df_apprx["d4_s IE (kJ/mol)"] - df_apprx['AP3 DISP']
    print(df_apprx[['crystal apprx', 'd', "d4_s IE (kJ/mol)", 'AP3 DISP']])


    # Get unique crystals
    crystals_apprx = sorted(df_apprx["crystal apprx"].dropna().unique())
    crystals_bm = sorted(df_bm["crystal bm"].dropna().unique())

    # Combine and get all unique crystals
    all_crystals = sorted(list(set(crystals_apprx) | set(crystals_bm)))
    N_crystals = len(all_crystals)

    print(f"Processing {N_crystals} crystals for switchover error plots")

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
                    lambda r: r[f"AP2-des-tl{tl_N} TOTAL"] * r[num_rep_col] / int(r[nmer_col][0])
                    if pd.notnull(r[nmer_col])
                    else 0,
                    axis=1,
                )
                df_c["ap3_des_cle"] = df_c.apply(
                    lambda r: r[f"AP3-des-tl{tl_N} TOTAL"] * r[num_rep_col] / int(r[nmer_col][0])
                    if pd.notnull(r[nmer_col])
                    else 0,
                    axis=1,
                )
                df_c['uma-s-1p1_cle'] = df_c.apply(
                    lambda r: r["uma-s-1p1 IE (kJ/mol)"] * r[num_rep_col] / int(r[nmer_col][0])
                    if pd.notnull(r[nmer_col]) and "uma-s-1p1 IE (kJ/mol)" in r
                    else 0,
                    axis=1,
                )
                df_c['uma-m-1p1_cle'] = df_c.apply(
                    lambda r: r["uma-m-1p1 IE (kJ/mol)"] * r[num_rep_col] / int(r[nmer_col][0])
                    if pd.notnull(r[nmer_col]) and "uma-m-1p1 IE (kJ/mol)" in r
                    else 0,
                    axis=1,
                )
                df_c['uma-s-1p1+ap3_lr_cle'] = df_c.apply(
                    lambda r: r["uma-s-1p1+ap3_lr IE (kJ/mol)"] * r[num_rep_col] / int(r[nmer_col][0])
                    if pd.notnull(r[nmer_col]) and "uma-s-1p1+ap3_lr IE (kJ/mol)" in r
                    else 0,
                    axis=1,
                )
                df_c['uma-m-1p1+ap3_lr_cle'] = df_c.apply(
                    lambda r: r["uma-m-1p1+ap3_lr IE (kJ/mol)"] * r[num_rep_col] / int(r[nmer_col][0])
                    if pd.notnull(r[nmer_col]) and "uma-m-1p1+ap3_lr IE (kJ/mol)" in r
                    else 0,
                    axis=1,
                )
                df_c['ap3_d4_cle'] = df_c.apply(
                    lambda r: r["ap3+d4"] * r[num_rep_col] / int(r[nmer_col][0])
                    if pd.notnull(r[nmer_col]) and "ap3+d4" in r
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
                uma_s_2b_energies = []
                uma_s_ap3lr_2b_energies = []
                uma_m_ap3lr_2b_energies = []
                uma_m_2b_energies = []
                ref_2b_energies = []
                ap2_des_2b_energies = []
                ap3_des_2b_energies = []
                df_c = df_c.sort_values(by=mms_col, ascending=True)
                df_c_N = df_c.iloc[:N]
                df_c_above = df_c.iloc[N:]
                ml_sep_distances = []
                for d in sep_distances:
                    ref_N = df_c_N[df_c_N[mms_col] < d]["ref_cle"].sum()
                    ap2_above = df_c_above[df_c_above[mms_col] < d]["ap2_cle"].sum()
                    ap2_hybrid_total = ap2_above + ref_N

                    ap3_above = df_c_above[df_c_above[mms_col] < d]["ap3_cle"].sum()
                    ap3_hybrid_total = ap3_above + ref_N

                    ap3_d4_above = df_c_above[df_c_above[mms_col] < d]["ap3_d4_cle"].sum()
                    ap3_d4_hybrid_total = ap3_d4_above + ref_N

                    ap2_des_above = df_c_above[df_c_above[mms_col] < d]["ap2_des_cle"].sum()
                    ap2_des_hybrid_total = ap2_des_above + ref_N

                    ap3_des_above = df_c_above[df_c_above[mms_col] < d]["ap3_des_cle"].sum()
                    ap3_des_hybrid_total = ap3_des_above + ref_N

                    uma_s_above = df_c_above[df_c_above[mms_col] < d]["uma-s-1p1_cle"].sum()
                    uma_s_hybrid_total = uma_s_above + ref_N

                    uma_s_ap3lr_above = df_c_above[df_c_above[mms_col] < d]['uma-s-1p1+ap3_lr_cle'].sum()
                    uma_s_ap3lr_hybrid_total = uma_s_ap3lr_above  + ref_N

                    uma_m_ap3lr_above = df_c_above[df_c_above[mms_col] < d]['uma-m-1p1+ap3_lr_cle'].sum()
                    uma_m_ap3lr_hybrid_total = uma_m_ap3lr_above  + ref_N

                    uma_m_above = df_c_above[df_c_above[mms_col] < d]["uma-m-1p1_cle"].sum()
                    uma_m_hybrid_total = uma_m_above + ref_N

                    ref_below = df_c[df_c[mms_col] < d]["ref_cle"].sum()

                    if len(df_c_above[df_c_above[mms_col] < d]) > 0:
                        ml_sep_distances.append(d)
                        ap2_2b_energies.append(ap2_hybrid_total)
                        ap3_2b_energies.append(ap3_hybrid_total)
                        uma_s_2b_energies.append(uma_s_hybrid_total)
                        uma_s_ap3lr_2b_energies.append(uma_s_ap3lr_hybrid_total)
                        uma_m_ap3lr_2b_energies.append(uma_m_ap3lr_hybrid_total)
                        uma_m_2b_energies.append(uma_m_hybrid_total)
                        ap2_des_2b_energies.append(ap2_des_hybrid_total)
                        ap3_des_2b_energies.append(ap3_des_hybrid_total)
                        ap3_d4_2b_energies.append(ap3_d4_hybrid_total)
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
                        label="AP3+D4",
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
                    lambda r: r[f"AP2-des-tl{tl_N} TOTAL"] * r[num_rep_col] / int(r[nmer_col][0])
                    if pd.notnull(r[nmer_col])
                    else 0,
                    axis=1,
                )
                df_c["ap3_des_cle"] = df_c.apply(
                    lambda r: r[f"AP3-des-tl{tl_N} TOTAL"] * r[num_rep_col] / int(r[nmer_col][0])
                    if pd.notnull(r[nmer_col])
                    else 0,
                    axis=1,
                )
                df_c['uma-s-1p1_cle'] = df_c.apply(
                    lambda r: r["uma-s-1p1 IE (kJ/mol)"] * r[num_rep_col] / int(r[nmer_col][0])
                    if pd.notnull(r[nmer_col]) and "uma-s-1p1 IE (kJ/mol)" in r
                    else 0,
                    axis=1,
                )
                df_c['uma-m-1p1_cle'] = df_c.apply(
                    lambda r: r["uma-m-1p1 IE (kJ/mol)"] * r[num_rep_col] / int(r[nmer_col][0])
                    if pd.notnull(r[nmer_col]) and "uma-m-1p1 IE (kJ/mol)" in r
                    else 0,
                    axis=1,
                )
                df_c['uma-s-1p1+ap3_lr_cle'] = df_c.apply(
                    lambda r: r["uma-s-1p1+ap3_lr IE (kJ/mol)"] * r[num_rep_col] / int(r[nmer_col][0])
                    if pd.notnull(r[nmer_col]) and "uma-s-1p1+ap3_lr IE (kJ/mol)" in r
                    else 0,
                    axis=1,
                )
                df_c['uma-m-1p1+ap3_lr_cle'] = df_c.apply(
                    lambda r: r["uma-m-1p1+ap3_lr IE (kJ/mol)"] * r[num_rep_col] / int(r[nmer_col][0])
                    if pd.notnull(r[nmer_col]) and "uma-m-1p1+ap3_lr IE (kJ/mol)" in r
                    else 0,
                    axis=1,
                )
                df_c['ap3_d4_cle'] = df_c.apply(
                    lambda r: r["ap3+d4"] * r[num_rep_col] / int(r[nmer_col][0])
                    if pd.notnull(r[nmer_col]) and "ap3+d4" in r
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
                uma_s_2b_energies = []
                uma_s_ap3lr_2b_energies = []
                uma_m_ap3lr_2b_energies = []
                uma_m_2b_energies = []
                ref_2b_energies = []
                ap2_des_2b_energies = []
                ap3_des_2b_energies = []
                df_c = df_c.sort_values(by=mms_col, ascending=True)
                df_c_N = df_c.iloc[:N]
                df_c_above = df_c.iloc[N:]
                ml_sep_distances = []
                for d in sep_distances:
                    ref_N = df_c_N[df_c_N[mms_col] < d]["ref_cle"].sum()
                    ap2_above = df_c_above[df_c_above[mms_col] < d]["ap2_cle"].sum()
                    ap2_hybrid_total = ap2_above + ref_N

                    ap3_above = df_c_above[df_c_above[mms_col] < d]["ap3_cle"].sum()
                    ap3_hybrid_total = ap3_above + ref_N

                    ap3_d4_above = df_c_above[df_c_above[mms_col] < d]["ap3_d4_cle"].sum()
                    ap3_d4_hybrid_total = ap3_d4_above + ref_N

                    ap2_des_above = df_c_above[df_c_above[mms_col] < d]["ap2_des_cle"].sum()
                    ap2_des_hybrid_total = ap2_des_above + ref_N

                    ap3_des_above = df_c_above[df_c_above[mms_col] < d]["ap3_des_cle"].sum()
                    ap3_des_hybrid_total = ap3_des_above + ref_N

                    uma_s_above = df_c_above[df_c_above[mms_col] < d]["uma-s-1p1_cle"].sum()
                    uma_s_hybrid_total = uma_s_above + ref_N

                    uma_s_ap3lr_above = df_c_above[df_c_above[mms_col] < d]['uma-s-1p1+ap3_lr_cle'].sum()
                    uma_s_ap3lr_hybrid_total = uma_s_ap3lr_above  + ref_N

                    uma_m_ap3lr_above = df_c_above[df_c_above[mms_col] < d]['uma-m-1p1+ap3_lr_cle'].sum()
                    uma_m_ap3lr_hybrid_total = uma_m_ap3lr_above  + ref_N

                    uma_m_above = df_c_above[df_c_above[mms_col] < d]["uma-m-1p1_cle"].sum()
                    uma_m_hybrid_total = uma_m_above + ref_N

                    ref_below = df_c[df_c[mms_col] < d]["ref_cle"].sum()

                    if len(df_c_above[df_c_above[mms_col] < d]) > 0:
                        ml_sep_distances.append(d)
                        ap2_2b_energies.append(ap2_hybrid_total)
                        ap3_2b_energies.append(ap3_hybrid_total)
                        uma_s_2b_energies.append(uma_s_hybrid_total)
                        uma_s_ap3lr_2b_energies.append(uma_s_ap3lr_hybrid_total)
                        uma_m_ap3lr_2b_energies.append(uma_m_ap3lr_hybrid_total)
                        uma_m_2b_energies.append(uma_m_hybrid_total)
                        ap2_des_2b_energies.append(ap2_des_hybrid_total)
                        ap3_des_2b_energies.append(ap3_des_hybrid_total)
                        ap3_d4_2b_energies.append(ap3_d4_hybrid_total)
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
                    ax_bm.plot(
                        ml_sep_distances,
                        ap2_des_2b_energies,
                        "o-",
                        label=f"AP2-DES{tl_N}",
                        markersize=4,
                        linewidth=1.5,
                        alpha=0.8,
                    )
                    ax_bm.plot(
                        ml_sep_distances,
                        ap3_des_2b_energies,
                        "s-",
                        label=f"AP3-DES{tl_N}",
                        markersize=4,
                        linewidth=1.5,
                        alpha=0.8,
                    )
                    ax_bm.plot(
                        ml_sep_distances,
                        uma_s_2b_energies,
                        "^-",
                        label="UMA-s",
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
                        uma_s_ap3lr_2b_energies,
                        "v-",
                        label="UMA-s+AP3-LR",
                        markersize=4,
                        linewidth=1.5,
                        alpha=0.8,
                    )
                    ax_bm.plot(
                        ml_sep_distances,
                        uma_m_ap3lr_2b_energies,
                        "v-",
                        label="UMA-m+AP3-LR",
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
                    # difference between uma_s and uma_s_ap3lr
                    print(f"{crystal:19s} bm|UMA-s: {uma_s_2b_energies[-1]:.4f}, UMA-s+AP3-LR: {uma_s_ap3lr_2b_energies[-1]:.4f}, diff: {uma_s_2b_energies[-1]-uma_s_ap3lr_2b_energies[-1]:.8f}, ref: {ref_2b_energies[-1]:.4f} kJ/mol")
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
    print("Error CLE statistics")
    # Filter out nans from ap2_full_cle_errors_ccsd_t_CBS

    me_ap2_sapt = np.sum(np.array(ap2_full_cle_errors_sapt0_aDZ)) / len(
        ap2_full_cle_errors_sapt0_aDZ
    )
    me_ap3_sapt = np.sum(np.array(ap3_full_cle_errors_sapt0_aDZ)) / len(
        ap3_full_cle_errors_sapt0_aDZ
    )
    me_ap2_ccsd = np.sum(np.array(ap2_full_cle_errors_ccsd_t_CBS)) / len(
        ap2_full_cle_errors_ccsd_t_CBS
    )
    me_ap3_ccsd = np.sum(np.array(ap3_full_cle_errors_ccsd_t_CBS)) / len(
        ap3_full_cle_errors_ccsd_t_CBS
    )
    print(f"{me_ap2_sapt=:.4f} kJ/mol")
    print(f"{me_ap3_sapt=:.4f} kJ/mol")
    print(f"{me_ap2_ccsd=:.4f} kJ/mol")
    print(f"{me_ap3_ccsd=:.4f} kJ/mol")
    mae_ap2_sapt = np.sum(np.abs(np.array(ap2_full_cle_errors_sapt0_aDZ))) / len(
        ap2_full_cle_errors_sapt0_aDZ
    )
    mae_ap3_sapt = np.sum(np.abs(np.array(ap3_full_cle_errors_sapt0_aDZ))) / len(
        ap3_full_cle_errors_sapt0_aDZ
    )
    mae_ap2_ccsd = np.sum(np.abs(np.array(ap2_full_cle_errors_ccsd_t_CBS))) / len(
        ap2_full_cle_errors_ccsd_t_CBS
    )
    mae_ap3_ccsd = np.sum(np.abs(np.array(ap3_full_cle_errors_ccsd_t_CBS))) / len(
        ap3_full_cle_errors_ccsd_t_CBS
    )
    print(f"{mae_ap2_sapt=:.4f} kJ/mol")
    print(f"{mae_ap3_sapt=:.4f} kJ/mol")
    print(f"{mae_ap2_ccsd=:.4f} kJ/mol")
    print(f"{mae_ap3_ccsd=:.4f} kJ/mol")

    # Save figure
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\nFigure saved to: {output_path}")
    error_df1 = pd.DataFrame(
        {
            "AP2 vs SAPT0 error": ap2_full_cle_errors_sapt0_aDZ,
            "AP3 vs SAPT0 error": ap3_full_cle_errors_sapt0_aDZ,
            "AP3-D4 vs SAPT0 error": ap3_d4_full_cle_errors_sapt0_aDZ,
            f"AP2-DES-tl{tl_N} vs SAPT0 error": ap2_des_full_cle_errors_sapt0_aDZ,
            f"AP3-DES-tl{tl_N} vs SAPT0 error": ap3_des_full_cle_errors_sapt0_aDZ,
            "UMA-s vs SAPT0 error": uma_s_full_cle_errors_sapt0_aDZ,
            "UMA-m vs SAPT0 error": uma_m_full_cle_errors_sapt0_aDZ,
            "UMA-s+AP3-LR vs SAPT0 error": uma_s_ap3lr_full_cle_errors_sapt0_aDZ,
            "UMA-m+AP3-LR vs SAPT0 error": uma_m_ap3lr_full_cle_errors_sapt0_aDZ,
        }
    )

    # Prepare df1 for violin plot
    dfs1 = [
        {
            "df": error_df1,
            "basis": "",
            "label": "SAPT0/aDZ Reference",
            "ylim": [[-10, 10]],
        }
    ]

    # Labels and columns for df1 (multiple methods)
    df1_labels = {
        "AP2": "AP2 vs SAPT0 error",
        "AP3": "AP3 vs SAPT0 error",
        "AP3-D4": "AP3-D4 vs SAPT0 error",
        f"AP2-DES-tl{tl_N}": f"AP2-DES-tl{tl_N} vs SAPT0 error",
        f"AP3-DES-tl{tl_N}": f"AP3-DES-tl{tl_N} vs SAPT0 error",
        "UMA-s": "UMA-s vs SAPT0 error",
        "UMA-m": "UMA-m vs SAPT0 error",
        "UMA-s+AP3-LR": "UMA-s+AP3-LR vs SAPT0 error",
        "UMA-m+AP3-LR": "UMA-m+AP3-LR vs SAPT0 error",
    }

    method_cols = [
        col
        for col in df_apprx.columns
        if "Non-Additive MB Energy (kJ/mol)" in col and "sapt0-dz-aug" not in col
    ]
    # Add other methods if available
    for method_col in method_cols[:3]:
        method_name = method_col.replace("Non-Additive MB Energy (kJ/mol) ", "")
        if f"{method_name} error" in df_apprx.columns:
            df1_labels[method_name] = f"{method_name} error"

    error_df2 = pd.DataFrame(
        {
            "AP2 vs CCSD(T)/CBS error": ap2_full_cle_errors_ccsd_t_CBS,
            "AP3 vs CCSD(T)/CBS error": ap3_full_cle_errors_ccsd_t_CBS,
            f"AP2-DES-tl{tl_N} vs CCSD(T)/CBS error": ap2_des_full_cle_errors_ccsd_t_CBS,
            f"AP3-DES-tl{tl_N} vs CCSD(T)/CBS error": ap3_des_full_cle_errors_ccsd_t_CBS,
            "UMA-s vs CCSD(T)/CBS error": uma_s_full_cle_errors_ccsd_t_CBS,
            "UMA-m vs CCSD(T)/CBS error": uma_m_full_cle_errors_ccsd_t_CBS,
            "UMA-s+AP3-LR vs CCSD(T)/CBS error": uma_s_ap3lr_full_cle_errors_ccsd_t_CBS,
            "UMA-m+AP3-LR vs CCSD(T)/CBS error": uma_m_ap3lr_full_cle_errors_ccsd_t_CBS,
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
        f"AP2-DES-tl{tl_N}": f"AP2-DES-tl{tl_N} vs CCSD(T)/CBS error",
        f"AP3-DES-tl{tl_N}": f"AP3-DES-tl{tl_N} vs CCSD(T)/CBS error",
        "UMA-s": "UMA-s vs CCSD(T)/CBS error",
        "UMA-m": "UMA-m vs CCSD(T)/CBS error",
        "UMA-s+AP3-LR": "UMA-s+AP3-LR vs CCSD(T)/CBS error",
        "UMA-m+AP3-LR": "UMA-m+AP3-LR vs CCSD(T)/CBS error",
    }

    # Create violin plot for df1
    print("\nCreating violin plot for df1 (approximate methods)...")
    error_statistics.violin_plot_table_multi_SAPT_components(
        dfs1,
        df_labels_and_columns_total=df1_labels,
        output_filename=output_violin_apprx,
        grid_heights=[0.3, 1.0],
        grid_widths=[1],
        legend_loc="upper left",
        annotations_texty=0.3,
        figure_size=(6, 2.5),
        add_title=False,
        name_violin=False,
        ylabel=r"Error (kJ$\cdot$mol$^{-1}$)",
    )
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
    return


def plot_crystal_violin_errors(N=1, sft=False, tl_N=100):
    """
    Create violin plots showing error distributions for each crystal.
    Uses the same energy computation logic as plot_crystal_lattice_energies_with_N,
    but creates a subplot violin distribution for each crystal.
    
    Parameters:
    -----------
    N : int
        Number of points to use as reference (hybrid approach)
    sft : bool
        Whether to use SFT (supervised fine-tuning) models
    tl_N : int
        Transfer learning parameter N for DES models
    """
    # Load dataframes
    if sft:
        df_apprx = pd.read_pickle("./sft_crystals_ap2_ap3_results_mol_apprx.pkl")
        df_bm = pd.read_pickle("./sft_crystals_ap2_ap3_results_mol_bm.pkl")
        output_path_apprx = f"./x23_plots/N{N}_crystal_violin_errors_apprx_sft.png"
        output_path_bm = f"./x23_plots/N{N}_crystal_violin_errors_bm_sft.png"
    else:
        df_apprx = pd.read_pickle("./crystals_ap2_ap3_des_results_mol_apprx.pkl")
        df_bm = pd.read_pickle("./crystals_ap2_ap3_des_results_mol_bm.pkl")
        output_path_apprx = f"./x23_plots/N{N}_crystal_violin_errors_apprx.png"
        output_path_bm = f"./x23_plots/N{N}_crystal_violin_errors_bm.png"

    # Load UMA data
    for i in ["uma-s-1p1", "uma-m-1p1"]:
        df_uma_bm = pd.read_pickle(f"./crystals_ap2_ap3_results_{i}_mol_bm.pkl")
        df_bm[f"{i} IE (kJ/mol)"] = df_uma_bm[f"{i} IE (kJ/mol)"]
        df_uma_apprx = pd.read_pickle(
            f"./crystals_ap2_ap3_results_{i}_mol_apprx.pkl"
        )
        df_apprx[f"{i} IE (kJ/mol)"] = df_uma_apprx[f"{i} IE (kJ/mol)"]
        
        # Add AP3-LR hybrid for long-range
        uma_ap3_lr = []
        df_bm_sorted = df_bm.sort_values(by="Minimum Monomer Separations (A) CCSD(T)/CBS")
        for n, r in df_bm_sorted.iterrows():
            # if r['Minimum Monomer Separations (A) CCSD(T)/CBS'] > 6.0:
            if r[f"{i} IE (kJ/mol)"] == 0.0:
                val = r["ap3_d_elst"] + r["ap3_classical_ind_energy"]
                uma_ap3_lr.append(val)
            else:
                uma_ap3_lr.append(r[f"{i} IE (kJ/mol)"])
        df_bm[f"{i}+ap3_lr IE (kJ/mol)"] = uma_ap3_lr
        
        uma_ap3_lr = []
        for n, r in df_apprx.iterrows():
            # if r['Minimum Monomer Separations (A) CCSD(T)/CBS'] > 6.0:
            if r[f"{i} IE (kJ/mol)"] == 0.0:
                val = r["ap3_d_elst"] + r["ap3_classical_ind_energy"]
                uma_ap3_lr.append(val)
            else:
                uma_ap3_lr.append(r[f"{i} IE (kJ/mol)"])
        df_apprx[f"{i}+ap3_lr IE (kJ/mol)"] = uma_ap3_lr

    # Get unique crystals
    crystals_apprx = sorted(df_apprx["crystal apprx"].dropna().unique())
    crystals_bm = sorted(df_bm["crystal bm"].dropna().unique())

    # Process apprx (vs SAPT0/aDZ)
    print(f"Processing approximate methods (vs SAPT0/aDZ) with N={N}")
    crystal_errors_apprx = {}
    
    for crystal in crystals_apprx:
        df_c = df_apprx[df_apprx["crystal apprx"] == crystal].copy()
        
        # Reference method
        ref_col = "Non-Additive MB Energy (kJ/mol) sapt0-dz-aug"
        num_rep_col = "Num. Rep. (#) sapt0-dz-aug"
        nmer_col = "N-mer Name apprx"
        mms_col = "Minimum Monomer Separations (A) sapt0-dz-aug"
        
        if ref_col not in df_c.columns or "AP2 TOTAL" not in df_c.columns:
            continue
            
        # Sort by minimum monomer separation
        df_c = df_c.sort_values(by=mms_col, ascending=True)
        
        # Split into N closest points (reference) and rest (ML)
        # Calculate CLE contributions
        for col_suffix, method_col in [
            ("ap2_cle", "AP2 TOTAL"),
            ("ap3_cle", "AP3 TOTAL"),
            ("ap2_des_cle", f"AP2-des-tl{tl_N} TOTAL"),
            ("ap3_des_cle", f"AP3-des-tl{tl_N} TOTAL"),
        ]:
            if method_col in df_c.columns:
                df_c[col_suffix] = df_c.apply(
                    lambda r: r[method_col] * r[num_rep_col] / int(r[nmer_col][0])
                    if pd.notnull(r[nmer_col])
                    else 0,
                    axis=1,
                )
        
        df_c["ref_cle"] = df_c.apply(
            lambda r: r[ref_col] * r[num_rep_col] / int(r[nmer_col][0])
            if pd.notnull(r[nmer_col])
            else 0,
            axis=1,
        )
        
        for uma_method in ["uma-s-1p1", "uma-m-1p1", "uma-s-1p1+ap3_lr"]:
            if f"{uma_method} IE (kJ/mol)" in df_c.columns:
                df_c[f"{uma_method}_cle"] = df_c.apply(
                    lambda r: r[f"{uma_method} IE (kJ/mol)"] * r[num_rep_col] / int(r[nmer_col][0])
                    if pd.notnull(r[nmer_col]) and f"{uma_method} IE (kJ/mol)" in r
                    else 0,
                    axis=1,
                )
        
        df_c_N = df_c.iloc[:N]
        df_c_above = df_c.iloc[N:]
        
        if len(df_c_above) == 0:
            continue
        
        # Calculate errors for each point in df_c_above
        ref_N_total = df_c_N["ref_cle"].sum()
        ref_total = df_c["ref_cle"].sum()
        
        errors = {}
        for method, col in [
            ("AP2", "ap2_cle"),
            ("AP3", "ap3_cle"),
            (f"AP2-DES-tl{tl_N}", "ap2_des_cle"),
            (f"AP3-DES-tl{tl_N}", "ap3_des_cle"),
            ("UMA-s", "uma-s-1p1_cle"),
            ("UMA-m", "uma-m-1p1_cle"),
            ("UMA-s+AP3-LR", "uma-s-1p1+ap3_lr_cle"),
        ]:
            if col in df_c_above.columns:
                # Hybrid total = ML predictions for points > N + reference for points <= N
                ml_total = df_c_above[col]
                ref_total = df_c_above["ref_cle"]
                error = ml_total - ref_total
                errors[method] = error
        
        crystal_errors_apprx[crystal] = errors
    
    # Process bm (vs CCSD(T)/CBS)
    print(f"Processing benchmark (vs CCSD(T)/CBS) with N={N}")
    crystal_errors_bm = {}
    ylims=[-20, 20]
    for crystal in crystals_bm:
        df_c = df_bm[df_bm["crystal bm"] == crystal].copy()
        
        # Reference method
        ref_col = "Non-Additive MB Energy (kJ/mol) CCSD(T)/CBS"
        num_rep_col = "Num. Rep. (#) CCSD(T)/CBS"
        nmer_col = "N-mer Name bm"
        mms_col = "Minimum Monomer Separations (A) CCSD(T)/CBS"
        
        if ref_col not in df_c.columns or "AP2 TOTAL" not in df_c.columns:
            continue
            
        # Sort by minimum monomer separation
        df_c = df_c.sort_values(by=mms_col, ascending=True)
        
        # Split into N closest points (reference) and rest (ML)
        
        if len(df_c_above) == 0:
            continue
        
        # Calculate CLE contributions
        for col_suffix, method_col in [
            ("ap2_cle", "AP2 TOTAL"),
            ("ap3_cle", "AP3 TOTAL"),
            ("ap2_des_cle", f"AP2-des-tl{tl_N} TOTAL"),
            ("ap3_des_cle", f"AP3-des-tl{tl_N} TOTAL"),
        ]:
            if method_col in df_c.columns:
                df_c[col_suffix] = df_c.apply(
                    lambda r: r[method_col] * r[num_rep_col] / int(r[nmer_col][0])
                    if pd.notnull(r[nmer_col])
                    else 0,
                    axis=1,
                )
        
        df_c["ref_cle"] = df_c.apply(
            lambda r: r[ref_col] * r[num_rep_col] / int(r[nmer_col][0])
            if pd.notnull(r[nmer_col])
            else 0,
            axis=1,
        )
        
        for uma_method in ["uma-s-1p1", "uma-m-1p1", "uma-s-1p1+ap3_lr"]:
            if f"{uma_method} IE (kJ/mol)" in df_c.columns:
                df_c[f"{uma_method}_cle"] = df_c.apply(
                    lambda r: r[f"{uma_method} IE (kJ/mol)"] * r[num_rep_col] / int(r[nmer_col][0])
                    if pd.notnull(r[nmer_col]) and f"{uma_method} IE (kJ/mol)" in r
                    else 0,
                    axis=1,
                )
        df_c_N = df_c.iloc[:N]
        df_c_above = df_c.iloc[N:]
        
        errors = {}
        for method, col in [
            ("AP2", "ap2_cle"),
            ("AP3", "ap3_cle"),
            (f"AP2-DES-tl{tl_N}", "ap2_des_cle"),
            (f"AP3-DES-tl{tl_N}", "ap3_des_cle"),
            ("UMA-s", "uma-s-1p1_cle"),
            ("UMA-m", "uma-m-1p1_cle"),
            ("UMA-s+AP3-LR", "uma-s-1p1+ap3_lr_cle"),
        ]:
            if col in df_c_above.columns:
                # Hybrid total = ML predictions for points > N + reference for points <= N
                ml_total = df_c_above[col]
                ref_total = df_c_above["ref_cle"]
                error = ml_total - ref_total
                errors[method] = error
        
        crystal_errors_bm[crystal] = errors
    
    # Plot for apprx
    if crystal_errors_apprx:
        # Convert to DataFrame format expected by violin plot
        all_methods = list(next(iter(crystal_errors_apprx.values())).keys())
        
        fig, axes = plt.subplots(1, len(crystal_errors_apprx), 
                                figsize=(3*len(crystal_errors_apprx), 4),
                                sharey=True)
        if len(crystal_errors_apprx) == 1:
            axes = [axes]
        
        for idx, (crystal, errors) in enumerate(sorted(crystal_errors_apprx.items())):
            ax = axes[idx]
            
            # Create violin plot for this crystal
            positions = list(range(len(all_methods)))
            data_to_plot = [[errors.get(method, 0)] for method in all_methods]
            
            parts = ax.violinplot(data_to_plot, positions=positions, 
                                 showmeans=True, showmedians=True)
            
            ax.set_xticks(positions)
            ax.set_xticklabels(all_methods, rotation=45, ha='right', fontsize=8)
            ax.set_ylim(ylims)
            ax.set_title(crystal, fontsize=10)
            ax.axhline(y=0, color='black', linestyle='--', linewidth=0.5, alpha=0.5)
            ax.grid(True, alpha=0.3, linestyle=':')
            
            if idx == 0:
                ax.set_ylabel(r"Error (kJ$\cdot$mol$^{-1}$)", fontsize=11)
        
        plt.tight_layout()
        plt.savefig(output_path_apprx, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path_apprx}")
        plt.close()
    
    # Plot for bm
    if crystal_errors_bm:
        # Convert to DataFrame format expected by violin plot
        all_methods = list(next(iter(crystal_errors_bm.values())).keys())
        
        fig, axes = plt.subplots(1, len(crystal_errors_bm), 
                                figsize=(3*len(crystal_errors_bm), 4),
                                sharey=True)
        if len(crystal_errors_bm) == 1:
            axes = [axes]
        
        for idx, (crystal, errors) in enumerate(sorted(crystal_errors_bm.items())):
            ax = axes[idx]
            
            # Create violin plot for this crystal
            positions = list(range(len(all_methods)))
            data_to_plot = [[errors.get(method, 0)] for method in all_methods]
            
            parts = ax.violinplot(data_to_plot, positions=positions, 
                                 showmeans=True, showmedians=True)
            
            ax.set_xticks(positions)
            ax.set_xticklabels(all_methods, rotation=45, ha='right', fontsize=8)
            ax.set_title(crystal, fontsize=10)
            ax.axhline(y=0, color='black', linestyle='--', linewidth=0.5, alpha=0.5)
            ax.set_ylim(ylims)
            ax.grid(True, alpha=0.3, linestyle=':')
            
            if idx == 0:
                ax.set_ylabel(r"Error (kJ$\cdot$mol$^{-1}$)", fontsize=11)
        
        plt.tight_layout()
        plt.savefig(output_path_bm, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path_bm}")
        plt.close()
    
    return crystal_errors_apprx, crystal_errors_bm


def main():
    # plot_all_systems()
    # plot_full_crystal_errors()
    # plot_crystal_lattice_energies_with_switchover(2.9)

    # GENERATE DATAFRAMES
    # ap2_ap3_df_energies_sft(
    #     generate=True,
    #     v='apprx'
          # N=10,
    # )
    # ap2_ap3_df_energies_sft(
    #     generate=True,
    #     v='bm'
          # N=10,
    # )
    # plot_crystal_lattice_energies(sft=False)
    # plot_crystal_lattice_energies(sft=True)

    tl_N = 1000
    # plot_crystal_violin_errors(0, sft=False, tl_N=tl_N)

    # ap2_ap3_df_energies_des370k_tl(N=100)
    # for tl_N in [100, 1000, 10000]:
    #     ap2_ap3_df_energies_des370k_tl(v='bm', N=tl_N)
    #     ap2_ap3_df_energies_des370k_tl(v='apprx', N=tl_N)
    # plot_crystal_lattice_energies_with_N(0, sft=False, tl_N=tl_N)
    # plot_crystal_lattice_energies_with_N(0, sft=False, tl_N=tl_N)
    # plot_crystal_lattice_energies_with_N(1, sft=False, tl_N=tl_N)
    # plot_crystal_lattice_energies_with_N(5, sft=False, tl_N=tl_N)
    # plot_crystal_lattice_energies_with_N(10, sft=False, tl_N=tl_N)

    plot_switchover_errors()
    return


if __name__ == "__main__":
    main()
