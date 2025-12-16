import qcelemental as qcel
from ase import Atoms
import os
import pandas as pd
import numpy as np
from time import time
from sevenn.calculator import SevenNetCalculator
import qcelemental as qcel
from ase import Atoms
import ase
from pprint import pprint as pp

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
#
# mol_water = qcel.models.Molecule.from_data("""
# 0 1
# O 0.000000 0.000000  0.000000
# H 0.758602 0.000000  0.504284
# H 0.260455 0.000000 -0.872893
# --
# 0 1
# O 3.000000 0.500000  0.000000
# H 3.758602 0.500000  0.504284
# H 3.260455 0.500000 -0.872893
# """)

# int_energy, eAB, eA, eB = interaction_energy(mol_water)
# print(f"IE : {int_energy:.6f} kcal/mol")
# print(f"dimer    : {eAB:.6f} kcal/mol")
# print(f"monomer A: {eA:.6f} kcal/mol")
# print(f"monomer B: {eB:.6f} kcal/mol")


def sevennet_df_energies_sft(generate=False, v="apprx", sevennet_type="mpa"):
    calc = SevenNetCalculator(
        # model="./checkpoint_sevennet_mf_ompa.pth",
        model="7net-omni",
        modal=sevennet_type,
        enable_cueq=False,
        enable_flash=False,
    )

    # v = "bm"
    mol_str = "mol " + v
    pkl_fn = f"crystals_ap2_ap3_results_{sevennet_type}_{mol_str.replace(' ', '_')}.pkl"
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
            sevennet_energies = []
            cnt = 0
            for i, row in df_c_a.iterrows():
                if row[mms_col] > 6.0:
                    sevennet_energies.append(
                        {
                            "index": i,
                            "int_energy": 0,
                            "eAB": 0,
                            "eA": 0,
                            "eB": 0,
                        }
                    )
                    continue
                print(
                    f"{cnt:4d} / {len(df_c_a):4d}, {time() - t1:.2f} seconds", end="\r"
                )
                cnt += 1
                mol = row[mol_str]
                int_energy, eAB, eA, eB = interaction_energy(mol)
                sevennet_energies.append(
                    {
                        "index": i,
                        "int_energy": int_energy * kcalmol_to_kjmol,
                        "eAB": eAB * kcalmol_to_kjmol,
                        "eA": eA * kcalmol_to_kjmol,
                        "eB": eB * kcalmol_to_kjmol,
                    }
                )
                pp(sevennet_energies[-1])
                print(row[target_col], "at", row[mms_col])
                if row[mms_col] > 6.0:
                    assert abs(sevennet_energies[-1]["int_energy"]) < 1e-8, (
                        f"failed at {row[mms_col]}, {sevennet_energies[-1]['int_energy']}"
                    )
                # print
            df.loc[df_c_a.index, f"{sevennet_type} IE (kJ/mol)"] = [
                ue["int_energy"] for ue in sevennet_energies
            ]
            df.loc[df_c_a.index, f"{sevennet_type} dimer (kJ/mol)"] = [
                ue["eAB"] for ue in sevennet_energies
            ]
            df.loc[df_c_a.index, f"{sevennet_type} monomer A (kJ/mol)"] = [
                ue["eA"] for ue in sevennet_energies
            ]
            df.loc[df_c_a.index, f"{sevennet_type} monomer B (kJ/mol)"] = [
                ue["eB"] for ue in sevennet_energies
            ]
            print(f"Completed crystal in {time() - t1:.2f} seconds")

        # LE
        df[f"{sevennet_type}_le_contribution"] = df.apply(
            lambda r: r[f"{sevennet_type} IE (kJ/mol)"]
            * r["Num. Rep. (#) sapt0-dz-aug"]
            / int(r[f"N-mer Name {v}"][0])
            if pd.notnull(r[f"{sevennet_type} IE (kJ/mol)"])
            and pd.notnull(r[f"N-mer Name {v}"])
            else 0,
            axis=1,
        )
        df.to_pickle(pkl_fn)
    else:
        df = pd.read_pickle(pkl_fn)
    return df


def main():
    # df_sevennet = sevennet_df_energies_sft(generate=True, v="apprx", sevennet_type="sevennet-s-1p1")
    # print(df_sevennet.head())
    # df_sevennet = sevennet_df_energies_sft(generate=True, v="bm", sevennet_type="sevennet-s-1p1")
    # print(df_sevennet.head())

    df_sevennet = sevennet_df_energies_sft(
        generate=True, v="apprx", sevennet_type="omol25_low"
    )
    print(df_sevennet.head())
    # sevennet-m-1p1 took ~23K s on apprx and ~48K on bm


if __name__ == "__main__":
    main()
