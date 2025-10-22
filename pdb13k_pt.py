from apnet_pt.pretrained_models import apnet2_model_predict
import apnet_pt
from apnet_pt.AtomPairwiseModels.apnet2 import APNet2Model
from apnet_pt.AtomPairwiseModels.apnet2_fused import APNet2_AM_Model
from apnet_pt.AtomModels.ap2_atom_model import AtomModel
import qcelemental as qcel
from glob import glob
import pickle
import pandas as pd
from pprint import pprint as pp
from pathlib import Path
import os
import numpy as np
from qm_tools_aw.molecular_visualization import visualize_molecule

mol_dimer = qcel.models.Molecule.from_data("""
0 1
O 0.000000 0.000000  0.000000
H 0.758602 0.000000  0.504284
H 0.260455 0.000000 -0.872893
--
0 1
O 3.000000 0.500000  0.000000
H 3.758602 0.500000  0.504284
H 3.260455 0.500000 -0.872893
""")


def example_ensemble():
    mols = [mol_dimer for _ in range(3)]
    interaction_energies = apnet2_model_predict(
        mols,
        compile=False,
        batch_size=2,
    )
    print()
    return


def pdb13k_df_og():
    data = glob("./data/2021-bms-drugdimer/*.pkl")
    df_data = {
        "system_id": [],
        "qcel_molecule": [],
        "SAPT0 TOTAL ENERGY": [],
        "SAPT0 ELST ENERGY": [],
        "SAPT0 EXCH ENERGY": [],
        "SAPT0 IND ENERGY": [],
        "SAPT0 DISP ENERGY": [],
    }
    cols = [
        'SAPT0 TOTAL ENERGY',
        'SAPT0 ELST ENERGY',
        'SAPT0 EXCH ENERGY',
        'SAPT0 IND ENERGY',
        'SAPT0 DISP ENERGY',
    ]
    for n, file in enumerate(data):
        with open(file, "rb") as f:
            row = pickle.load(f)
        fn = str(Path(file).stem).replace(".prot", "")
        print(fn)
        with open(f"./data/2021-bms-drugdimer/{fn}.cx.out", "r") as f:
            xyz_data = f.read().split("mol {")[-1].split("}")[0]
        mol = qcel.models.Molecule.from_data(xyz_data)
        print(mol)
        df_data['system_id'].append(fn)
        df_data['qcel_molecule'].append(mol)
        for key in cols:
            df_data[key].append(row[key])
        if n == 10:
            break
    df = pd.DataFrame(df_data)
    df.to_pickle("pdb13k_errors.pkl")
    print(df)
    return


elem_to_z = {
    'H': 1,
    'B': 3,
    'C': 6,
    'N': 7,
    'O': 8,
    'F': 9,
    'NA': 11,
    'Na': 11,
    'P': 15,
    'S': 16,
    'CL': 17,
    'Cl': 17,
    'BR': 35,
    'Br': 35,
}


def psi4_output_to_qcel_mol(file):
    with open(f"./data/2021-bms-drugdimer/{file}", "r") as f:
        xyz_data = f.read().split("mol {")[-1].split("}")[0]
    xyz_data = xyz_data.replace("0 0 0 1", "0 1")
    mol = qcel.models.Molecule.from_data(xyz_data)
    ZA = mol.symbols[mol.fragments[0]]
    ZB = mol.symbols[mol.fragments[1]]
    try:
        ZA = np.array([elem_to_z[za] for za in ZA])
        ZB = np.array([elem_to_z[zb] for zb in ZB])
    except KeyError:
        print(f"Error: {file}\n{xyz_data}\n{ZA} {ZB}")
    return mol


def psi4_output_to_qcel_mol(file):
    with open(f"./data/2021-bms-drugdimer/{file}", "r") as f:
        xyz_data = f.read().split("mol {")[-1].split("}")[0]
    xyz_data.replace("0 0 0 1", "0 1")
    return qcel.models.Molecule.from_data(xyz_data)


def pdb13k_df():
    df = pd.read_csv(
        "./data/2021-bms-drugdimer/SAPT0-ADZ-NRGS-COMPONENTS.txt",
        sep=","
    )
    # df = df.head()
    df['qcel_molecule'] = df['Jobname'].apply(psi4_output_to_qcel_mol)
    df.to_pickle("pdb13k_errors-1.pkl")
    return


def pdb13k_errors_ensemble():
    pkl_fn = "pdb13k_errors_pt-ap2-t2.pkl"
    if not os.path.exists(pkl_fn):
        df = pd.read_pickle("pdb13k_errors-1.pkl")
        print(df)
        mols = df['qcel_molecule'].tolist()
        interaction_energies = apnet2_model_predict(
            mols,
            compile=True,
            batch_size=1000,
            ensemble_model_dir="./models/"
        )
        df['pt-ap2'] = [ie for ie in interaction_energies]
        df['PT-AP2 TOTAL'] = df['pt-ap2'].apply(lambda x: x[0])
        df['PT-AP2 ELST'] = df['pt-ap2'].apply(lambda x: x[1])
        df['PT-AP2 EXCH'] = df['pt-ap2'].apply(lambda x: x[2])
        df['PT-AP2 IND'] = df['pt-ap2'].apply(lambda x:  x[3])
        df['PT-AP2 DISP'] = df['pt-ap2'].apply(lambda x: x[4])
        df.to_pickle(pkl_fn)
    else:
        df = pd.read_pickle(pkl_fn)
    print(df.isna().sum())

    df['total error'] = df['Total(kcal)'] - df['PT-AP2 TOTAL']
    df['elst error'] = df['Electrostatic'] - df['PT-AP2 ELST']
    df['exch error'] = df['Exchange'] - df['PT-AP2 EXCH']
    df['ind error'] = df['Induction'] - df['PT-AP2 IND']
    df['disp error'] = df['Dispersion'] - df['PT-AP2 DISP']
    mae_total = df['total error'].abs().mean()
    mae_elst = df['elst error'].abs().mean()
    mae_exch = df['exch error'].abs().mean()
    mae_ind = df['ind error'].abs().mean()
    mae_disp = df['disp error'].abs().mean()
    print(df[['total error', 'elst error', 'exch error', 'ind error', 'disp error']].describe())
    print(f"MAE Total: {mae_total}")
    print(f"MAE Elst: {mae_elst}")
    print(f"MAE Exch: {mae_exch}")
    print(f"MAE Ind: {mae_ind}")
    print(f"MAE Disp: {mae_disp}")
    print(df[['Total(kcal)', 'Electrostatic', 'Exchange', 'Induction', 'Dispersion']].describe())
    return

def pdb13k_errors_single_model():
    pkl_fn = "pdb13k_errors_pt-ap2_single-t5.pkl"
    if not os.path.exists(pkl_fn):
        df = pd.read_pickle("pdb13k_errors-1.pkl")
        ap2 = APNet2_AM_Model(
            atom_model=AtomModel(
                pre_trained_model_path="/home/amwalla3/gits/qcmlforge/models/am_ensemble/am_1.pt",
            ).model,
            pre_trained_model_path="/home/amwalla3/gits/qcmlforge/models/ap2-fused_ensemble/ap2_1.pt",
        )
        # ap2 = APNet2Model(
        #     atom_model=AtomModel(
        #         pre_trained_model_path="/home/amwalla3/gits/qcmlforge/models/am_pbe0_ensemble/am_0.pt",
        #     ).model,
        #     pre_trained_model_path="/home/amwalla3/gits/qcmlforge/models/ap2_pbe0_ensemble/ap2_t2_0.pt",
        # )
        ap2.compile_model()
        print(df)
        mols = df['qcel_molecule'].tolist()
        interaction_energies = ap2.predict_qcel_mols(
            mols,
            batch_size=200,
            verbose=True,
        )
        print(interaction_energies)
        df['pt-ap2'] = [ie for ie in interaction_energies]
        df['PT-AP2 ELST'] = df['pt-ap2'].apply(lambda x: x[0])
        df['PT-AP2 EXCH'] = df['pt-ap2'].apply(lambda x: x[1])
        df['PT-AP2 IND'] = df['pt-ap2'].apply(lambda x:  x[2])
        df['PT-AP2 DISP'] = df['pt-ap2'].apply(lambda x: x[3])
        df['PT-AP2 TOTAL'] = df.apply(lambda x: x['PT-AP2 ELST'] + x['PT-AP2 EXCH'] + x['PT-AP2 IND'] + x['PT-AP2 DISP'], axis=1)
        df.to_pickle(pkl_fn)
    else:
        df = pd.read_pickle(pkl_fn)
    print(df.isna().sum())

    df['total error'] = df['Total(kcal)'] - df['PT-AP2 TOTAL']
    df['elst error'] = df['Electrostatic'] - df['PT-AP2 ELST']
    df['exch error'] = df['Exchange'] - df['PT-AP2 EXCH']
    df['ind error'] = df['Induction'] - df['PT-AP2 IND']
    df['disp error'] = df['Dispersion'] - df['PT-AP2 DISP']
    mae_total = df['total error'].abs().mean()
    mae_elst = df['elst error'].abs().mean()
    mae_exch = df['exch error'].abs().mean()
    mae_ind = df['ind error'].abs().mean()
    mae_disp = df['disp error'].abs().mean()
    print(df[['total error', 'elst error', 'Electrostatic', 'PT-AP2 ELST']])
    print(df[['total error', 'elst error', 'exch error', 'ind error', 'disp error']].describe())
    print(f"MAE Total: {mae_total}")
    print(f"MAE Elst: {mae_elst}")
    print(f"MAE Exch: {mae_exch}")
    print(f"MAE Ind: {mae_ind}")
    print(f"MAE Disp: {mae_disp}")

    print(df[['Total(kcal)', 'Electrostatic', 'Exchange', 'Induction', 'Dispersion']].describe())
    return

def isolate_pt_top_errors_compared_to_tf():
    df_tf = pd.read_pickle("pdb13k_errors_tf-ap2.pkl")
    df_pt = pd.read_pickle("pdb13k_errors_pt-ap2_single.pkl")
    df_ap3 = pd.read_pickle("pdb13k_errors_pt-ap3_single.pkl")
    df = pd.merge(df_tf, df_pt, on='Jobname', suffixes=('_tf', '_pt'))
    df = pd.merge(df, df_ap3, on='Jobname', suffixes=('', '_ap3'))
    df['TF-AP2 ELST'] = df['PT-AP2 ELST_tf']
    df['PT-AP2 ELST'] = df['PT-AP2 ELST_pt']
    df['TF-AP2 IND'] = df['PT-AP2 IND_tf']
    df['PT-AP2 IND'] = df['PT-AP2 IND_pt']
    df['TF-AP2 EXCH'] = df['PT-AP2 EXCH_tf']
    df['PT-AP2 EXCH'] = df['PT-AP2 EXCH_pt']
    df['TF-AP2 DISP'] = df['PT-AP2 DISP_tf']
    df['PT-AP2 DISP'] = df['PT-AP2 DISP_pt']
    df.drop(columns=['qcel_molecule_tf', 'qcel_molecule_pt'], inplace=True)
    pp(df.columns.tolist())
    # ELST
    df['PT elst error'] = abs(df['PT-AP2 ELST_pt'] - df['Electrostatic_pt'])
    df['TF elst error'] = abs(df['PT-AP2 ELST_tf'] - df['Electrostatic_tf'])
    df['AP3 elst error'] = abs(df['AP3 ELST'] - df['Electrostatic_pt'])
    df.sort_values(by='PT elst error', ascending=False, inplace=True)
    print(df[['Jobname', 'PT elst error', 'TF elst error', 'AP3 elst error']])
    print(df[['Jobname', 'Electrostatic_pt', 'PT-AP2 ELST', 'TF-AP2 ELST', 'AP3 ELST']])
    # IND
    df['PT ind error'] = abs(df['PT-AP2 IND_pt'] - df['Induction_pt'])
    df['TF ind error'] = abs(df['PT-AP2 IND_tf'] - df['Induction_tf'])
    df['AP3 ind error'] = abs(df['AP3 INDU'] - df['Induction_pt'])
    df.sort_values(by='PT ind error', ascending=False, inplace=True)
    print(df[['Jobname', 'Induction_pt', 'PT-AP2 IND', 'TF-AP2 IND', 'AP3 INDU']]);
    pd.set_option('display.max_rows', None)
    print(df[['Jobname', 'PT ind error', 'TF ind error', 'AP3 ind error']])
    df.reset_index(inplace=True, drop=True)
    for n, i in df.iterrows():
        if n > 1000:
            break
        mol = i['qcel_molecule']
        monA = mol.get_fragment(0)
        monB = mol.get_fragment(1)
        # print(n, i['Jobname'], f"\n   PT-AP2={i['PT-AP2 ELST']:.2f}, TF-AP2={i['TF-AP2 ELST']:.2f}, PT error={i['PT elst error']:.2f}, TF error={i['TF elst error']:.2f}")
        # print(f"\n   {monA.molecular_charge}, {monB.molecular_charge}")
        # continue
        # if n > 10:
        #     continue
        if False:
            visualize_molecule(
                i['qcel_molecule'],
                # title=f"{i['Jobname']}\nPT-AP2={i['PT-AP2 ELST']:.2f}, TF-AP2={i['TF-AP2 ELST']:.2f}, PT error={i['PT elst error']:.2f}, TF error={i['TF elst error']:.2f}",
                title=f"{monA.molecular_charge}::{monB.molecular_charge}, PT:{i['PT-AP2 ELST']:.2f} TF:{i['TF-AP2 ELST']:.2f},Elst:{i['Electrostatic_pt']:.2f}",
                temp_filename=f"./mol_viz/{n}.html"
            )
    pd.set_option('display.max_rows', None)
    print(df[['Electrostatic_pt']].head(1000))
    df.to_pickle("pdb13k_errors_pt_top_errors.pkl")
    df.head(1000).to_pickle("pdb13k_errors_pt_top_1000_errors.pkl")
    df.to_pickle("pdb13k_errors_ap2_ap3.pkl")
    return

def ap3_d_elst_classical_energies(mols):
    path_to_qcml = os.path.join(os.path.expanduser("~"), "gits/qcmlforge/models")
    am_path = f"{path_to_qcml}/../models/ap3_ensemble/1/am_3.pt"
    at_hf_vw_path = f"{path_to_qcml}/../models/ap3_ensemble/1/am_h+1_3.pt"
    at_elst_path = f"{path_to_qcml}/../models/ap3_ensemble/1/am_elst_h+1_3.pt"
    ap3_path = f"{path_to_qcml}/../models/ap3_ensemble/3/ap3_.pt"
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
        # model_type="AtomTypeParamMPNN",
        # pre_trained_model_path=at_elst_path_mpnn,
        pre_trained_model_path=at_elst_path,
    )
    # atom_type_elst_undamped_model = (
    #     apnet_pt.AtomPairwiseModels.mtp_mtp.AM_DimerParam_Model(
    #         use_GPU=False,
    #         n_neuron=64,
    #         n_params=1,
    #         ignore_database_null=True,
    #         atom_model=atom_type_hf_vw_model.model,
    #         atom_model_type="AtomTypeParamNN",
    #         pre_trained_model_path=at_elst_path,
    #         dimer_eval_type="elst",
    #     )
    # )
    ap3 = apnet_pt.AtomPairwiseModels.apnet3_fused.APNet3_AtomType_Model(
        ds_root=None,
        atom_type_model=atom_type_hf_vw_model.model,
        dimer_prop_model=atom_type_elst_model.dimer_model,
        pre_trained_model_path=ap3_path,
    )
    pred, pair_elst, pair_ind = ap3.predict_qcel_mols(
        mols, batch_size=16, return_classical_pairs=True
    )
    # elst_undamped = atom_type_elst_undamped_model.predict_qcel_mols_dimer(
    #     mols, batch_size=16
    # )
    return pred, pair_elst, pair_ind


def pdb13k_errors_single_model_ap3():
    pkl_fn = "pdb13k_errors_pt-ap3_single.pkl"
    if not os.path.exists(pkl_fn):
        df = pd.read_pickle("pdb13k_errors-1.pkl")

        print("AP3 start")
        pred, pair_elst, pair_ind = ap3_d_elst_classical_energies(
            df["qcel_molecule"].tolist()
        )
        elst_energies = [np.sum(e) for e in pair_elst]
        ind_energies = [np.sum(e) for e in pair_ind]
        # df["mtp_elst_energies"] = elst_undamped
        df["ap3_d_elst"] = elst_energies
        df["ap3_classical_ind_energy"] = ind_energies
        df["AP3 TOTAL"] = np.sum(pred[:, 0:4], axis=1)
        df["AP3 ELST"] = pred[:, 0]
        df["AP3 EXCH"] = pred[:, 1]
        df["AP3 INDU"] = pred[:, 2]
        df["AP3 DISP"] = pred[:, 3]
        df.to_pickle(pkl_fn)
    else:
        df = pd.read_pickle(pkl_fn)
    print(df.isna().sum())

    df['AP3 total error'] = df['Total(kcal)'] - df['AP3 TOTAL']
    df['AP3 elst error'] = df['Electrostatic'] - df['AP3 ELST']
    df['AP3 exch error'] = df['Exchange'] - df['AP3 EXCH']
    df['AP3 ind error'] = df['Induction'] - df['AP3 INDU']
    df['AP3 disp error'] = df['Dispersion'] - df['AP3 DISP']
    mae_total = df['AP3 total error'].abs().mean()
    mae_elst = df['AP3 elst error'].abs().mean()
    mae_exch = df['AP3 exch error'].abs().mean()
    mae_ind = df['AP3 ind error'].abs().mean()
    mae_disp = df['AP3 disp error'].abs().mean()
    print(df[['AP3 total error', 'AP3 elst error', 'AP3 exch error', 'AP3 ind error', 'AP3 disp error']].describe())
    print(f"MAE Total: {mae_total}")
    print(f"MAE Elst: {mae_elst}")
    print(f"MAE Exch: {mae_exch}")
    print(f"MAE Ind: {mae_ind}")
    print(f"MAE Disp: {mae_disp}")

    print(df[['Total(kcal)', 'Electrostatic', 'Exchange', 'Induction', 'Dispersion']].describe())
    return


def main():
    # pdb13k_df()
    # pdb13k_errors_ensemble()
    # pdb13k_errors_single_model()
    # pdb13k_errors_single_model_ap3()
    isolate_pt_top_errors_compared_to_tf()
    return


if __name__ == "__main__":
    main()
