import qcelemental as qcel
import os
import pandas as pd
import numpy as np
from time import time
import subprocess
import json
from qm_tools_aw import tools
import pydispersion
from pprint import pprint as pp

eV_to_kcalmol = qcel.constants.conversion_factor("eV", "kcal/mol")
eV_to_kJmol = qcel.constants.conversion_factor("eV", "kJ/mol")
kcalmol_to_kjmol = qcel.constants.conversion_factor("kcal/mol", "kJ/mol")
ha_to_kjmol = qcel.constants.conversion_factor("hartree", "kJ/mol")

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
            el = qcel.periodictable.to_E(str(int(atom_numbers[n])))
            v = "    ".join(["%.16f" % k for k in i])
            line = "%s    %s\n" % (el, v)
            f.write(line)
    return


def calc_dftd4_c6_c8_pairDisp2(
    atom_numbers: np.array,
    carts: np.array,  # angstroms
    charges: np.array,
    input_xyz: str = "dat.xyz",
    dftd4_bin: str = "/home/awallace43/.local/bin/dftd4",
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
    print(" ".join(args))
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
    # os.remove(input_xyz)
    # os.remove("C_n.json")
    # os.remove("pairs.json")
    # os.remove(".EDISP")
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
    dftd4_bin: str = "/home/awallace43/.local/bin/dftd4",
    p: [] = [1.0, 1.61679827, 0.44959224, 3.35743605],
    s9=0.0,
):
    C6s_mA, _, _, _ = calc_dftd4_c6_c8_pairDisp2(
        pA,
        cA,
        charges[1],
        p=p,
        dftd4_bin=dftd4_bin,
        input_xyz="monA.xyz",
    )
    C6s_mB, _, _, _ = calc_dftd4_c6_c8_pairDisp2(
        pB,
        cB,
        charges[2],
        p=p,
        dftd4_bin=dftd4_bin,
        input_xyz="monB.xyz",
    )
    C6s_dimer, _, _, df_c_e = calc_dftd4_c6_c8_pairDisp2(
        pD,
        cD,
        charges[0],
        p=p,
        dftd4_bin=dftd4_bin,
        input_xyz="dimer.xyz",
    )
    return C6s_dimer, C6s_mA, C6s_mB


def dftd4_df_c6s(generate=False, v="apprx", dftd4_type="d4_i"):
    print(f"DFTD4 C6s for {v} - {dftd4_type}")
    # v = "bm"
    mol_str = "mol " + v
    pkl_fn = f"crystals_c6s_{dftd4_type}_{mol_str.replace(' ', '_')}.pkl"
    df = pd.read_pickle(f"./crystals_ap2_ap3_des_results_mol_{v}.pkl")
    table = {
        "charges": [],
        "monAs": [],
        "monBs": [],
        "C6s": [],
        "C6_A": [],
        "C6_B": [],
    }
    # use tqdm for progress bar
    from tqdm import tqdm

    for n, r in tqdm(
        df.iterrows(),
        total=len(df),
        desc="DFTD4 C6s",
        ascii=True,
        bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}",
    ):
        geom, pD, cD, ma, mb, charges = tools.mol_to_pos_carts_ma_mb(r[mol_str], units_angstroms=True)
        table["monAs"].append(ma)
        table["monBs"].append(mb)
        pA, cA = pD[ma], cD[ma, :]
        pB, cB = pD[mb], cD[mb, :]
        C6s_dimer, C6s_mA, C6s_mB = calc_dftd4_c6_for_d_a_b(
            cD, pD, pA, cA, pB, cB, charges
        )
        table["C6s"].append(C6s_dimer)
        table["C6_A"].append(C6s_mA)
        table["C6_B"].append(C6s_mB)
        table["charges"].append(charges)
    for k, v in table.items():
        df[k] = v
    df.to_pickle(pkl_fn)
    return df


def dftd4_df_energies_supermolecular(v="apprx"):
    print(f"DFTD4 Energies Supermolecular for {v}")
    # v = "bm"
    dftd4_type = "d4_s"
    mol_str = "mol " + v
    pkl_fn = f"crystals_c6s_d4_i_{mol_str.replace(' ', '_')}.pkl"
    df = pd.read_pickle(pkl_fn)
    params = np.array([1.0, 0.829861, 0.706055, 1.123903], dtype=np.float64)
    # params = np.array([1.0, 1.61679827, 0.44959224, 3.35743605], dtype=np.float64)
    def interaction_energy(mol, c6s, c6s_A, c6s_B):
        geom, pD, cD, ma, mb, charges = tools.mol_to_pos_carts_ma_mb(
            mol, units_angstroms=False
        )
        pA, cA = pD[ma], cD[ma, :]
        pB, cB = pD[mb], cD[mb, :]
        eAB = pydispersion.calculate_dispersion_energy(
            pD,
            cD,
            c6s,
            params=params,
        )
        eA = pydispersion.calculate_dispersion_energy(
            pA,
            cA,
            c6s_A,
            params=params,
        )
        eB = pydispersion.calculate_dispersion_energy(
            pB,
            cB,
            c6s_B,
            params=params,
        )
        int_energy = eAB - (eA + eB)
        return int_energy, eAB, eA, eB

    t1 = time()
    df = df.dropna(subset=[mol_str])
    if v == "apprx":
        mms_col = "Minimum Monomer Separations (A) sapt0-dz-aug"
    else:
        mms_col = "Minimum Monomer Separations (A) CCSD(T)/CBS"
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
        dftd4_energies = []
        cnt = 0
        for i, row in df_c_a.iterrows():
            print(f"{cnt:4d} / {len(df_c_a):4d}, {time() - t1:.2f} seconds", end="\r")
            cnt += 1
            mol = row[mol_str]
            c6s = row["C6s"]
            c6s_A = row["C6_A"]
            c6s_B = row["C6_B"]
            int_energy, eAB, eA, eB = interaction_energy(mol, c6s, c6s_A, c6s_B)
            dftd4_energies.append(
                {
                    "index": i,
                    "int_energy": int_energy * ha_to_kjmol,
                    "eAB": eAB * ha_to_kjmol,
                    "eA": eA * ha_to_kjmol,
                    "eB": eB * ha_to_kjmol,
                }
            )
        df.loc[df_c_a.index, f"{dftd4_type} IE (kJ/mol)"] = [
            ue["int_energy"] for ue in dftd4_energies
        ]
        df.loc[df_c_a.index, f"{dftd4_type} dimer (kJ/mol)"] = [
            ue["eAB"] for ue in dftd4_energies
        ]
        df.loc[df_c_a.index, f"{dftd4_type} monomer A (kJ/mol)"] = [
            ue["eA"] for ue in dftd4_energies
        ]
        df.loc[df_c_a.index, f"{dftd4_type} monomer B (kJ/mol)"] = [
            ue["eB"] for ue in dftd4_energies
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
    return df


def merge_dftd4_results(v="apprx"):
    df = pd.read_pickle(f"./crystals_ap2_ap3_des_results_mol_{v}.pkl")
    pkl_fn_d4_i = f"crystals_c6s_d4_i_mol_{v}.pkl"
    df_d4_i = pd.read_pickle(pkl_fn_d4_i)
    # merge by index
    df.drop(
        ['d4_s IE (kJ/mol)', 'd4_s dimer (kJ/mol)', 'd4_s monomer A (kJ/mol)', 'd4_s monomer B (kJ/mol)', 'd4_s_le_contribution'],
        axis=1,
        inplace=True,
    )
    df = df.join(
        df_d4_i[
            [
                "d4_s IE (kJ/mol)",
                "d4_s dimer (kJ/mol)",
                "d4_s monomer A (kJ/mol)",
                "d4_s monomer B (kJ/mol)",
                "d4_s_le_contribution",
            ]
        ],
    )
    # pd print .4f 
    print_options = pd.get_option("display.float_format")
    pd.set_option("display.float_format", "{:.4f}".format)
    df['d'] = df["Minimum Monomer Separations (A) sapt0-dz-aug"]
    print(df[[f'crystal {v}', 'd', 'd4_s IE (kJ/mol)', 'AP3 DISP']])
    print(df[['d', 'd4_s IE (kJ/mol)', 'AP3 DISP', 'd4_s dimer (kJ/mol)', 'd4_s monomer A (kJ/mol)', 'd4_s monomer B (kJ/mol)']])
    print(df)
    df.to_pickle(f"./crystals_ap2_ap3_des_results_mol_{v}.pkl")
    return df


def main():
    # df_d4 = dftd4_df_c6s(generate=True, v="apprx", dftd4_type="d4_i")
    df_d4 = dftd4_df_energies_supermolecular(v="apprx")
    df_merged = merge_dftd4_results('apprx')

    # print("BM")
    # df_d4 = dftd4_df_c6s(generate=True, v="bm", dftd4_type="d4_i")
    df_d4 = dftd4_df_energies_supermolecular(v="bm")
    df_merged = merge_dftd4_results('bm')


if __name__ == "__main__":
    main()
