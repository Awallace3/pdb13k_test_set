import numpy as np
import pandas as pd
from pprint import pprint as pp
from qm_tools_aw import tools
import pandas as pd
from pprint import pprint as pp
import qcelemental as qcel
import apnet_pt
import os
import shutil
import argparse


qcml_model_dir = os.path.expanduser("~/gits/qcmlforge/models")

ha_to_kcalmol = qcel.constants.conversion_factor("hartree", "kcal/mol")


def ap2_energies(
    compile=True,
    finetune_mols=[],
    finetune_labels=[],
    data_dir="data_dir",
    sft_n_epochs=50,
    sft_lr=5e-4,
    transfer_learning=True,
):
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

    if os.path.exists(data_dir):
        shutil.rmtree(data_dir)
    pre_trained_model_path = f"./sft_models/ap2_los2_{len(finetune_mols)}.pt"
    if not os.path.exists(pre_trained_model_path):
        pre_trained_model_path = f"{qcml_model_dir}/ap2-fused_ensemble/ap2_1.pt"

    ap2 = apnet_pt.AtomPairwiseModels.apnet2_fused.APNet2_AM_Model(
        ds_root=data_dir,
        atom_model=apnet_pt.AtomModels.ap2_atom_model.AtomModel(
            pre_trained_model_path=f"{qcml_model_dir}/am_ensemble/am_1.pt",
        ).model,
        pre_trained_model_path=pre_trained_model_path,
        ds_qcel_molecules=ds_qcel_molecules,
        ds_energy_labels=ds_energy_labels,
        ds_spec_type=None,
        ignore_database_null=ignore_database_null,
    )
    if compile:
        ap2.compile_model()
    if finetune:
        # ap2.freeze_parameters_except_readouts()
        ap2.train(
            n_epochs=sft_n_epochs,
            lr=sft_lr,
            split_percent=0.90,
            skip_compile=True,
            transfer_learning=transfer_learning,
            model_path=f"./sft_models/ap2_los2_{len(finetune_mols)}.pt",
        )
    return ap2


def ap3_d_elst_classical_energies(
    finetune_mols=[],
    finetune_labels=[],
    data_dir="data_dir",
    sft_n_epochs=50,
    sft_lr=5e-4,
    transfer_learning=True,
):
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
    ap3_path = f"./sft_models/ap3_los2_{len(finetune_mols)}.pt"
    if not os.path.exists(ap3_path):
        ap3_path = f"{path_to_qcml}/../models/ap3_ensemble/0/ap3_.pt"
        # cp ap3_path to ./sft_models/ap3_los2_{len(finetune_mols)}.pt
        shutil.copyfile(ap3_path, f"./sft_models/ap3_los2_{len(finetune_mols)}.pt")

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
    if os.path.exists(data_dir):
        shutil.rmtree(data_dir)

    ap3 = apnet_pt.AtomPairwiseModels.apnet3_fused.APNet3_AtomType_Model(
        ds_root=data_dir,
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
            transfer_learning=transfer_learning,
            model_path=f"./sft_models/ap3_los2_{len(finetune_mols)}.pt",
        )
    return ap3


def finetune_sizes():
    df = pd.read_pickle("./data/los2.pkl")
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
        ap2_energies(
            compile=False,
            finetune_mols=finetune_mols,
            finetune_labels=finetune_labels,
        )
        ap3_d_elst_classical_energies(
            finetune_mols=finetune_mols,
            finetune_labels=finetune_labels,
        )
    return


def los2_training_sapt(M="ap2", n_epochs=50, lr=5e-4):
    df = pd.read_pickle("combined_df_4569.pkl")
    # df = df.head(5)
    df["qcel_molecule"] = df.apply(
        lambda r: tools.convert_pos_carts_to_mol(
            [
                r["atomic_numbers"][np.array(r["monAs"], dtype=np.int32)],
                r["atomic_numbers"][np.array(r["monBs"], dtype=np.int32)],
            ],
            [
                r["coordinates"][np.array(r["monAs"], dtype=np.int32)],
                r["coordinates"][np.array(r["monBs"], dtype=np.int32)],
            ],
            [
                [r["monA_charge"], r["monA_multiplicity"]],
                [r["monB_charge"], r["monB_multiplicity"]],
            ],
            units="angstrom",
        ),
        axis=1,
    )
    finetune_mols = df["qcel_molecule"].to_list()
    finetune_labels = []
    for idx, row in df.iterrows():
        sapt_energy = np.array(
            [
                row["SAPT2+3(CCD)DMP2 ELST ENERGY atqz"],
                row["SAPT2+3(CCD)DMP2 EXCH ENERGY atqz"],
                row["SAPT2+3(CCD)DMP2 IND ENERGY atqz"],
                row["SAPT2+3(CCD)DMP2 DISP ENERGY atqz"],
            ]
        )
        finetune_labels.append(sapt_energy * ha_to_kcalmol)
    finetune_labels = finetune_labels
    if M.lower() == "ap2":
        ap2_energies(
            compile=False,
            finetune_mols=finetune_mols,
            finetune_labels=finetune_labels,
            data_dir="data_los2_sapt_ap2",
            sft_n_epochs=n_epochs,
            sft_lr=lr,
            transfer_learning=False,
        )
        shutil.copyfile(f"./sft_models/ap2_los2_{len(df)}.pt", "./sft_models/ap2_los2_sapt.pt")
    elif M.lower() == "ap3":
        ap3_d_elst_classical_energies(
            finetune_mols=finetune_mols,
            finetune_labels=finetune_labels,
            data_dir="data_los2_sapt_ap3",
            sft_n_epochs=n_epochs,
            sft_lr=lr,
            transfer_learning=False,
        )
        shutil.copyfile(f"./sft_models/ap3_los2_{len(df)}.pt", "./sft_models/ap3_los2_sapt.pt")

    finetune_labels = df["Benchmark"].to_list()
    if M.lower() == "ap2":
        ap2_energies(
            compile=False,
            finetune_mols=finetune_mols,
            finetune_labels=finetune_labels,
            data_dir="data_los2_sapt_ap2_benchmark",
            sft_n_epochs=n_epochs,
            sft_lr=lr,
            transfer_learning=True,
        )
    elif M.lower() == "ap3":
        ap3_d_elst_classical_energies(
            finetune_mols=finetune_mols,
            finetune_labels=finetune_labels,
            data_dir="data_los2_sapt_ap3_benchmark",
            sft_n_epochs=n_epochs,
            sft_lr=lr,
            transfer_learning=True,
        )
    return


def main():
    parser = argparse.ArgumentParser(
        description="Finetune AP2 or AP3 models with DES370k dataset"
    )
    parser.add_argument(
        "-M",
        type=str,
        required=True,
        choices=["ap2", "ap3", "AP2", "AP3"],
        help="Model type to finetune: ap2 or ap3",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="N epochs int",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=5e-4,
        help="N epochs int",
    )
    args = parser.parse_args()
    los2_training_sapt(args.M, n_epochs=args.epochs, lr=args.lr)
    return


if __name__ == "__main__":
    main()
