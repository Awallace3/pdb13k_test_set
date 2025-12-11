import qcelemental as qcel
from ase import Atoms
import os
import pandas as pd
import numpy as np
from time import time
import subprocess
import json

eV_to_kcalmol = qcel.constants.conversion_factor("eV", "kcal/mol")
eV_to_kJmol = qcel.constants.conversion_factor("eV", "kJ/mol")
kcalmol_to_kjmol = qcel.constants.conversion_factor("kcal/mol", "kJ/mol")

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


def write_xyz_from_np(atom_numbers, carts, outfile="dat.xyz", charges=[0, 1]) -> None:
    """
    write_xyz_from_np
    """
    with open(outfile, "w") as f:
        f.write(str(len(carts)) + "\n\n")
        for n, i in enumerate(carts):
            el = str(int(atom_numbers[n]))
            v = "    ".join(["%.16f" % k for k in i])
            line = "%s    %s\n" % (el, v)
            f.write(line)
    return


def calc_dftd4_c6_c8_pairDisp2(
    atom_numbers: np.array,
    carts: np.array,  # angstroms
    charges: np.array,
    input_xyz: str = "dat.xyz",
    dftd4_bin: str = "/theoryfs2/ds/amwalla3/.local/bin/dftd4",
    p: [] = [1.0, 1.61679827, 0.44959224, 3.35743605],
    s9=0.0,
    C6s_ATM=False,
):
    """
    Ensure that dftd4 binary is from compiling git@github.com:Awallace3/dftd4
        - this is used to generate more decimal places on values for c6, c8,
          and pairDisp2
    """

    write_xyz_from_np(
        atom_numbers,
        carts,
        outfile=input_xyz,
        charges=charges,
    )
    args = [
        dftd4_bin,
        input_xyz,
        "--property",
        "--param",
        str(p[0]),
        str(p[1]),
        str(p[2]),
        str(p[3]),
        "--mbdscale",
        f"{s9}",
        "-c",
        str(charges[0]),
        "--pair-resolved",
    ]
    # print(" ".join(args))
    v = subprocess.call(
        args=args,
        shell=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
    )
    assert v == 0
    output_json = "C_n.json"
    with open(output_json) as f:
        cs = json.load(f)
    C6s = np.array(cs["c6"], dtype=np.float64)
    C8s = np.array(cs["c8"], dtype=np.float64)
    output_json = "pairs.json"
    with open(output_json) as f:
        pairs = json.load(f)
        pairs = np.array(pairs["pairs2"])
    with open(".EDISP", "r") as f:
        e = float(f.read())
    os.remove(input_xyz)
    os.remove("C_n.json")
    os.remove("pairs.json")
    os.remove(".EDISP")
    if C6s_ATM:
        with open("C_n_ATM.json") as f:
            cs = json.load(f)
        C6s_ATM = np.array(cs["c6_ATM"], dtype=np.float64)
        os.remove("C_n_ATM.json")
        return C6s, C8s, pairs, e, C6s_ATM
    else:
        return C6s, C8s, pairs, e

def calc_dftd4_c6_for_d_a_b(
    cD,
    pD,
    pA,
    cA,
    pB,
    cB,
    charges: np.array,
    input_xyz: str = "dat.xyz",
    dftd4_bin: str = "/theoryfs2/ds/amwalla3/.local/bin/dftd4",
    p: [] = [1.0, 1.61679827, 0.44959224, 3.35743605],
    s9=0.0,
):
    C6s_dimer, _, _, df_c_e = calc_dftd4_c6_c8_pairDisp2(
        pD,
        cD,
        charges[0],
        p=p,
        dftd4_bin=dftd4_bin,
    )
    C6s_mA, _, _, _ = calc_dftd4_c6_c8_pairDisp2(
        pA,
        cA,
        charges[1],
        p=p,
        dftd4_bin=dftd4_bin,
    )
    C6s_mB, _, _, _ = calc_dftd4_c6_c8_pairDisp2(
        pB,
        cB,
        charges[2],
        p=p,
        dftd4_bin=dftd4_bin,
    )
    return C6s_dimer, C6s_mA, C6s_mB


def uma_df_energies_sft(generate=False, v="apprx", dftd4_type="d4_i"):
    # v = "bm"
    mol_str = "mol " + v
    pkl_fn = f"crystals_ap2_ap3_results_{dftd4_type}_{mol_str.replace(' ', '_')}.pkl"
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

    def energy(mol):
        symbols = mol.symbols
        positions = mol.geometry * qcel.constants.bohr2angstroms
        atoms = Atoms(
            symbols,
            positions=positions,
            info={
                "charge": int(mol.molecular_charge),
                "spin": int(mol.molecular_multiplicity),
            },
        )
        atoms.calc = calc
        energy = atoms.get_potential_energy()
        return energy

    def interaction_energy(mol):
        monA = mol.get_fragment(0)
        monB = mol.get_fragment(1)
        eAB = energy(mol) * eV_to_kcalmol
        eA = energy(monA) * eV_to_kcalmol
        eB = energy(monB) * eV_to_kcalmol
        int_energy = eAB - (eA + eB)
        return int_energy, eAB, eA, eB

    if not os.path.exists(pkl_fn) or generate:
        t1 = time()
        df = pd.read_pickle(f"./crystals_ap2_ap3_results_mol_{v}.pkl")
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
            uma_energies = []
            cnt = 0
            for i, row in df_c_a.iterrows():
                print(f"{cnt:4d} / {len(df_c_a):4d}, {time() - t1:.2f} seconds", end="\r")
                cnt += 1
                mol = row[mol_str]
                int_energy, eAB, eA, eB = interaction_energy(mol)
                uma_energies.append(
                    {
                        "index": i,
                        "int_energy": int_energy * kcalmol_to_kjmol,
                        "eAB": eAB * kcalmol_to_kjmol,
                        "eA": eA * kcalmol_to_kjmol,
                        "eB": eB * kcalmol_to_kjmol,
                    }
                )
            df.loc[df_c_a.index, f"{dftd4_type} IE (kJ/mol)"] = [
                ue["int_energy"] for ue in uma_energies
            ]
            df.loc[df_c_a.index, f"{dftd4_type} dimer (kJ/mol)"] = [
                ue["eAB"] for ue in uma_energies
            ]
            df.loc[df_c_a.index, f"{dftd4_type} monomer A (kJ/mol)"] = [
                ue["eA"] for ue in uma_energies
            ]
            df.loc[df_c_a.index, f"{dftd4_type} monomer B (kJ/mol)"] = [
                ue["eB"] for ue in uma_energies
            ]
            print(f"Completed crystal in {time() - t1:.2f} seconds")

        # LE
        df[f"{dftd4_type}_le_contribution"] = df.apply(
            lambda r: r[f"{dftd4_type} IE (kJ/mol)"]
            * r["Num. Rep. (#) sapt0-dz-aug"]
            / int(r[f"N-mer Name {v}"][0])
            if pd.notnull(r[f"{dftd4_type} IE (kJ/mol)"])
            and pd.notnull(r[f"N-mer Name {v}"])
            else 0,
            axis=1,
        )
        df.to_pickle(pkl_fn)
    else:
        df = pd.read_pickle(pkl_fn)
    return df


def main():
    # df_uma = uma_df_energies_sft(generate=True, v="apprx", uma_type="uma-s-1p1")
    # print(df_uma.head())
    # df_uma = uma_df_energies_sft(generate=True, v="bm", uma_type="uma-s-1p1")
    # print(df_uma.head())

    df_uma = uma_df_energies_sft(generate=True, v="apprx", dftd4_type="uma-m-1p1")
    print(df_uma.head())
    df_uma = uma_df_energies_sft(generate=True, v="bm", dftd4_type="uma-m-1p1")
    print(df_uma.head())
    # uma-m-1p1 took ~23K s on apprx and ~48K on bm


if __name__ == "__main__":
    main()
