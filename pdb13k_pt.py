from apnet_pt.pretrained_models import apnet2_model_predict
import qcelemental as qcel
from glob import glob
import pickle
import pandas as pd
from pprint import pprint as pp
from pathlib import Path
import os
import numpy as np

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


def pdb13k_errors():
    pkl_fn = "pdb13k_errors_pt-ap2.pkl"
    if not os.path.exists(pkl_fn):
        df = pd.read_pickle("pdb13k_errors-1.pkl")
        print(df)
        mols = df['qcel_molecule'].tolist()
        interaction_energies = apnet2_model_predict(
            mols,
            compile=True,
            batch_size=1000,
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
    return


def main():
    pdb13k_df()
    pdb13k_errors()
    return


if __name__ == "__main__":
    main()
