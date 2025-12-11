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

sft_n_epochs = 50
sft_lr = 5e-4


def ap2_energies(compile=True, finetune_mols=[], finetune_labels=[]):
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

    ap2 = apnet_pt.AtomPairwiseModels.apnet2_fused.APNet2_AM_Model(
        ds_root="data_dir",
        atom_model=apnet_pt.AtomModels.ap2_atom_model.AtomModel(
            pre_trained_model_path=f"{qcml_model_dir}/am_ensemble/am_1.pt",
        ).model,
        pre_trained_model_path=f"{qcml_model_dir}/ap2-fused_ensemble/ap2_1.pt",
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
        # ap2.freeze_parameters_except_readouts()
        ap2.train(
            n_epochs=sft_n_epochs,
            lr=sft_lr,
            split_percent=0.90,
            skip_compile=True,
            transfer_learning=True,
            model_path=f"./sft_models/ap2_des370k_{len(finetune_mols)}.pt",
        )
    return ap2


def ap3_d_elst_classical_energies(finetune_mols=[], finetune_labels=[]):
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
    ap3_path = f"{path_to_qcml}/../models/ap3_ensemble/0/ap3_.pt"
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
        ds_root="data_dir",
        atom_type_model=atom_type_hf_vw_model.model,
        dimer_prop_model=atom_type_elst_model.dimer_model,
        pre_trained_model_path=ap3_path,
        ds_spec_type=None,
        ignore_database_null=ignore_database_null,
        ds_qcel_molecules=ds_qcel_molecules,
        ds_energy_labels=ds_energy_labels,
    )
    if finetune:
        # ap3.freeze_parameters_except_readouts()
        ap3.train(
            n_epochs=sft_n_epochs,
            lr=sft_lr,
            split_percent=0.90,
            transfer_learning=True,
            model_path=f"./sft_models/ap3_des370k_{len(finetune_mols)}.pt",
            
        )
    return ap3

def finetune_sizes():
    df = pd.read_pickle("./data/des370k.pkl")
    pp(df.columns.to_list())
    # get sample of N=10000 random rows
    for n in [100, 1000, 10000, 1000000, -1]:
        if n == -1:
            finetune_mols = df["qcel_mol"].to_list()
            finetune_labels = df["ref"].to_list()
        else:
            df_sample = df.sample(n=n, random_state=42).reset_index(drop=True)
            finetune_mols = df_sample["qcel_mol"].to_list()
            finetune_labels = df_sample["ref"].to_list()
        ap2_model = ap2_energies(
            compile=False,
            finetune_mols=finetune_mols,
            finetune_labels=finetune_labels,
        )
        ap3_model = ap3_d_elst_classical_energies(
            finetune_mols=finetune_mols,
            finetune_labels=finetune_labels,
        )
    return

def main():
    df = pd.read_pickle("./data/des370k.pkl")
    df.head(10).to_pickle("./data/des370k_head10.pkl")
    df = pd.read_pickle("./data/des370k_head10.pkl")
    pp(df.columns.to_list())
    # get sample of N=10000 random rows
    for n in [-1]:
        if n == -1:
            finetune_mols = df["qcel_mol"].to_list()
            finetune_labels = df["ref"].to_list()
            print(finetune_mols[0].to_string("psi4"))
        else:
            df_sample = df.sample(n=n, random_state=42).reset_index(drop=True)
            finetune_mols = df_sample["qcel_mol"].to_list()
            finetune_labels = df_sample["ref"].to_list()
        ap2_model = ap2_energies(
            compile=False,
            finetune_mols=finetune_mols,
            finetune_labels=finetune_labels,
        )

if __name__ == "__main__":
    # main()
    finetune_sizes()
