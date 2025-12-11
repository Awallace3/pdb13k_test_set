import qcelemental as qcel
from ase import Atoms
import os
import pandas as pd
import numpy as np
from time import time
import subprocess
import json
from qm_tools_aw import tools

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


def generate_D4_data(df):
    xyzs = df["Geometry"].to_list()
    monAs = df["monAs"].to_list()
    monBs = df["monBs"].to_list()
    charges = df["charges"].to_list()
    (
        C6s,
        C6_A,
        C6_B,
        C6_ATMs,
        C6_ATM_A,
        C6_ATM_B,
        disp_d,
        disp_a,
        disp_b,
    ) = calc_c6s_c8s_pairDisp2_for_df(xyzs, monAs, monBs, charges)
    df["C6s"] = C6s
    df["C6_A"] = C6_A
    df["C6_B"] = C6_B
    df["C6_ATM"] = C6_ATMs
    df["C6_ATM_A"] = C6_ATM_A
    df["C6_ATM_B"] = C6_ATM_B
    df["disp_d"] = disp_d
    df["disp_a"] = disp_a
    df["disp_b"] = disp_b
    return df


def calc_c6s_c8s_pairDisp2_for_df(xyzs, monAs, monBs, charges) -> ([], [], []):
    """
    runs pairDisp2 for all xyzs to accumulate C6s
    """
    C6s = [np.array([]) for i in range(len(xyzs))]
    C6_A = [np.array([]) for i in range(len(xyzs))]
    C6_B = [np.array([]) for i in range(len(xyzs))]
    C6_ATMs = [np.array([]) for i in range(len(xyzs))]
    C6_ATM_A = [np.array([]) for i in range(len(xyzs))]
    C6_ATM_B = [np.array([]) for i in range(len(xyzs))]
    disp_d = [np.array([]) for i in range(len(xyzs))]
    disp_a = [np.array([]) for i in range(len(xyzs))]
    disp_b = [np.array([]) for i in range(len(xyzs))]
    for n, c in enumerate(
        tqdm(
            xyzs[:],
            desc="DFTD4 Props",
            ascii=True,
            bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}",
        )
    ):
        g3 = np.array(c)
        pos = g3[:, 0]
        carts = g3[:, 1:]
        c = charges[n]
        C6, _, _, dispd, C6_ATM = locald4.calc_dftd4_c6_c8_pairDisp2(
            pos, carts, c[0], C6s_ATM=True
        )
        C6s[n] = C6
        C6_ATMs[n] = C6_ATM
        disp_d[n] = dispd

        Ma = monAs[n]
        mon_pa, mon_ca = create_mon_geom(pos, carts, Ma)
        C6a, _, _, dispa, C6_ATMa = locald4.calc_dftd4_c6_c8_pairDisp2(
            mon_pa, mon_ca, c[1], C6s_ATM=True
        )
        C6_A[n] = C6a
        C6_ATM_A[n] = C6_ATMa
        disp_a[n] = dispa

        Mb = monBs[n]
        mon_pb, mon_cb = create_mon_geom(pos, carts, Mb)
        C6b, _, _, dispb, C6_ATMb = locald4.calc_dftd4_c6_c8_pairDisp2(
            mon_pb, mon_cb, c[2], C6s_ATM=True
        )
        C6_B[n] = C6b
        C6_ATM_B[n] = C6_ATMb
        disp_b[n] = dispb
    return C6s, C6_A, C6_B, C6_ATMs, C6_ATM_A, C6_ATM_B, disp_d, disp_a, disp_b


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


def dftd4_df_c6s(generate=False, v="apprx", dftd4_type="d4_i"):
    # v = "bm"
    mol_str = "mol " + v
    pkl_fn = f"crystals_c6s_{dftd4_type}_{mol_str.replace(' ', '_')}.pkl"
    df = pd.read_pickle(f"./crystals_ap2_ap3_des_results_mol_{v}.pkl")
    table = {
        "Geometry": [],
        "charges": [],
        "Geometry_bohr": [],
        "monAs": [],
        "monBs": [],
        "C6s": [],
        "C6_A": [],
        "C6_B": [],
    }
    ang_to_bohr = qcel.constants.conversion_factor("angstrom", "bohr")
    for n, r in df.iterrows():
        geom, pD, cD, ma, mb, charges = tools.mol_to_pos_carts_ma_mb(r[mol_str])
        table["Geometry_bohr"].append(geom.copy())
        geom[:, 1:] *= ang_to_bohr
        table["Geometry"].append(geom)
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


def dftd4_df_energies(generate=False, v="apprx", dftd4_type="d4_i"):
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
                print(
                    f"{cnt:4d} / {len(df_c_a):4d}, {time() - t1:.2f} seconds", end="\r"
                )
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
    df_d4 = dftd4_df_c6s(generate=True, v="apprx", dftd4_type="d4_i")
    df_d4 = dftd4_df_c6s(generate=True, v="bm", dftd4_type="d4_i")

    # df_uma = dftd4_df_energies(generate=True, v="apprx", dftd4_type="d4_i")
    # print(df_uma.head())
    # df_uma = dftd4_df_energies(generate=True, v="bm", dftd4_type="d4_i")
    # print(df_uma.head())
    # uma-m-1p1 took ~23K s on apprx and ~48K on bm


if __name__ == "__main__":
    main()
