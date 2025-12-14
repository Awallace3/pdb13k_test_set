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
        ds_datapoint_storage_n_objects=1,
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
            dataloader_num_workers=8,
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
        ds_datapoint_storage_n_objects=1,
    )
    if finetune:
        # ap3.freeze_parameters_except_readouts()
        ap3.train(
            n_epochs=sft_n_epochs,
            lr=sft_lr,
            split_percent=0.90,
            dataloader_num_workers=8,
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
    df = pd.read_pickle("los2_corrected.pkl")
    # df = df.head(5)
    if "qcel_molecule" not in df.columns:
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
                row["SAPT2+3(CCD)DMP2 ELST ENERGY atz"],
                row["SAPT2+3(CCD)DMP2 EXCH ENERGY atz"],
                row["SAPT2+3(CCD)DMP2 IND ENERGY atz"],
                row["SAPT2+3(CCD)DMP2 DISP ENERGY atz"],
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
        shutil.copyfile(
            f"./sft_models/ap2_los2_{len(df)}.pt", "./sft_models/ap2_los2_sapt.pt"
        )
    elif M.lower() == "ap3":
        ap3_d_elst_classical_energies(
            finetune_mols=finetune_mols,
            finetune_labels=finetune_labels,
            data_dir="data_los2_sapt_ap3",
            sft_n_epochs=n_epochs,
            sft_lr=lr,
            transfer_learning=False,
        )
        shutil.copyfile(
            f"./sft_models/ap3_los2_{len(df)}.pt", "./sft_models/ap3_los2_sapt.pt"
        )

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


def load_ap2_for_inference(model_path=None):
    """Load AP2 model for inference only (no training)."""
    if model_path is None:
        model_path = f"{qcml_model_dir}/ap2-fused_ensemble/ap2_1.pt"

    data_dir = "data_dir_inference_ap2"
    if os.path.exists(data_dir):
        shutil.rmtree(data_dir)

    ap2 = apnet_pt.AtomPairwiseModels.apnet2_fused.APNet2_AM_Model(
        ds_root=data_dir,
        atom_model=apnet_pt.AtomModels.ap2_atom_model.AtomModel(
            pre_trained_model_path=f"{qcml_model_dir}/am_ensemble/am_1.pt",
        ).model,
        pre_trained_model_path=model_path,
        ds_qcel_molecules=None,
        ds_energy_labels=None,
        ds_spec_type=None,
        ignore_database_null=True,
    )
    return ap2


def load_ap3_for_inference(model_path=None):
    """Load AP3 model for inference only (no training)."""
    path_to_qcml = os.path.join(os.path.expanduser("~"), "gits/qcmlforge/models")
    am_path = f"{path_to_qcml}/../models/ap3_ensemble/1/am_3.pt"
    at_hf_vw_path = f"{path_to_qcml}/../models/ap3_ensemble/1/am_h+1_3.pt"
    at_elst_path = f"{path_to_qcml}/../models/ap3_ensemble/1/am_elst_h+1_3.pt"

    if model_path is None:
        model_path = f"{path_to_qcml}/../models/ap3_ensemble/0/ap3_.pt"

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

    data_dir = "data_dir_inference_ap3"
    if os.path.exists(data_dir):
        shutil.rmtree(data_dir)

    ap3 = apnet_pt.AtomPairwiseModels.apnet3_fused.APNet3_AtomType_Model(
        ds_root=data_dir,
        atom_type_model=atom_type_hf_vw_model.model,
        dimer_prop_model=atom_type_elst_model.dimer_model,
        pre_trained_model_path=model_path,
        ds_spec_type=None,
        ignore_database_null=True,
        ds_qcel_molecules=None,
        ds_energy_labels=None,
    )
    return ap3


def run_inference(model_type, qcel_molecules, model_path=None):
    """
    Run inference on a list of qcel molecules.

    Parameters:
    -----------
    model_type : str
        Either 'ap2' or 'ap3'
    qcel_molecules : list
        List of QCElemental molecule objects
    model_path : str, optional
        Path to custom model checkpoint. If None, uses default pretrained model.

    Returns:
    --------
    list : Predicted energies for each molecule
    """
    if model_type.lower() == "ap2":
        print(f"Loading AP2 model from {model_path or 'default pretrained path'}...")
        model = load_ap2_for_inference(model_path)
    elif model_type.lower() == "ap3":
        print(f"Loading AP3 model from {model_path or 'default pretrained path'}...")
        model = load_ap3_for_inference(model_path)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    print(f"Running inference on {len(qcel_molecules)} molecules...")
    pred = model.predict_qcel_mols(qcel_molecules, batch_size=200)
    return pred


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
    parser.add_argument(
        "--inference",
        action="store_true",
        help="Run inference only (no training)",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to model checkpoint for inference",
    )
    parser.add_argument(
        "--sapt-target",
        type=str,
        default="SAPT2+3(CCD)DMP2 atz",
        help="SAPT target for finetuning and prediction errors, e.g., 'SAPT2+3(CCD)DMP2 atz' or 'SAPT0 atz'",
    )
    parser.add_argument(
        "--input-pkl",
        type=str,
        # default="./combined_df_4569.pkl",
        default="./los2_corrected.pkl",
        help="Path to input pickle file with qcel_mol column",
    )
    args = parser.parse_args()
    sapt_method, sapt_basis = args.sapt_target.split()
    if args.inference:
        if args.input_pkl is None:
            raise ValueError("--input-pkl is required for inference mode")

        print(f"Loading data from {args.input_pkl}...")
        df = pd.read_pickle(args.input_pkl)
        if "qcel_mol" in df.columns:
            qcel_molecules = df["qcel_mol"].to_list()
        elif "qcel_molecule" in df.columns:
            qcel_molecules = df["qcel_molecule"].to_list()
        else:
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
            qcel_molecules = df["qcel_molecule"].to_list()

        predictions = run_inference(args.M, qcel_molecules, args.model_path)
        problem_ids = []
        for idx, pred in enumerate(predictions):
            if (
                abs(pred[0] - df[f"{sapt_method} ELST ENERGY {sapt_basis}"].iloc[idx] * ha_to_kcalmol)
                > 400
            ):
                print(
                    f"Prediction {idx}:\n{qcel_molecules[idx].to_string('xyz')}\n{pred}\n{df['SAPT0 ELST ENERGY atz'].iloc[idx] * ha_to_kcalmol:.4f}, {df['SAPT0 EXCH ENERGY atz'].iloc[idx] * ha_to_kcalmol:.4f}, {df['SAPT0 IND ENERGY atz'].iloc[idx] * ha_to_kcalmol:.4f} {df['SAPT0 DISP ENERGY atz'].iloc[idx] * ha_to_kcalmol:.4f}"
                )
                problem_ids.append(idx)
                r = df.iloc[idx]
                r["coordinates"] *= (
                    qcel.constants.conversion_factor("angstrom", "bohr") ** 2
                )
                df.at[idx, "qcel_molecule"] = tools.convert_pos_carts_to_mol(
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
                    units="bohr",
                )
                # print updated geometry xyz
                print(df.at[idx, "qcel_molecule"].to_string("xyz"))
        print(f"Problem IDs: {problem_ids}")
        ap_total = np.sum(
            np.array([[pred[0], pred[1], pred[2], pred[3]] for pred in predictions]), axis=1
        )
        ap_mae_total = np.mean(np.abs(ap_total - df["SAPT0 TOTAL ENERGY atz"] * ha_to_kcalmol))
        ap_mae_elst = np.mean(
            np.abs(
                np.array([pred[0] for pred in predictions])
                - np.array(df["SAPT0 ELST ENERGY atz"]) * ha_to_kcalmol
            )
        )
        ap_mae_exch = np.mean(
            np.abs(
                np.array([pred[1] for pred in predictions])
                - np.array(df["SAPT0 EXCH ENERGY atz"]) * ha_to_kcalmol
            )
        )
        ap_mae_ind = np.mean(
            np.abs(
                np.array([pred[2] for pred in predictions])
                - np.array(df["SAPT0 IND ENERGY atz"]) * ha_to_kcalmol
            )
        )
        ap_mae_disp = np.mean(
            np.abs(
                np.array([pred[3] for pred in predictions])
                - np.array(df["SAPT0 DISP ENERGY atz"]) * ha_to_kcalmol
            )
        )
        print(f"AP {args.M.upper()} SAPT0 MAE TOTAL: {ap_mae_total:.4f} kcal/mol")
        print(f"AP {args.M.upper()} SAPT0 MAE ELST: {ap_mae_elst:.4f} kcal/mol")
        print(f"AP {args.M.upper()} SAPT0 MAE EXCH: {ap_mae_exch:.4f} kcal/mol")
        print(f"AP {args.M.upper()} SAPT0 MAE IND: {ap_mae_ind:.4f} kcal/mol")
        print(f"AP {args.M.upper()} SAPT0 MAE DISP: {ap_mae_disp:.4f} kcal/mol")
        # df.to_pickle("los2_corrected.pkl")

        # Save predictions
    else:
        los2_training_sapt(args.M, n_epochs=args.epochs, lr=args.lr)

    return


if __name__ == "__main__":
    main()
