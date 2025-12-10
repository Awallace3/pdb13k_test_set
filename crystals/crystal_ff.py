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
import subprocess
from subprocess import check_output


qcml_model_dir = os.path.expanduser("~/gits/qcmlforge/models")
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


def generate_paramaters():
    if not os.path.exists("./ff_params/"):
        os.makedirs("./ff_params/")
    df = pd.read_pickle("./crystals_ap2_ap3_results_mol_apprx.pkl")
    for c in crystal_names_all:
        c_path = f"ff_params/{c}"
        if not os.path.exists(c_path):
            os.makedirs(c_path)
        mol = df[df["crystal apprx"] == c]["mol apprx"].iloc[0]
        mol.get_fragment(0).to_file(f"{c_path}/{c}_mol.xyz")
        cmd = f"obabel -ixyz {c_path}/{c}_mol.xyz -osmi -O {c_path}/{c}_mol.smi"
        cmd = f"obabel -ixyz {c_path}/{c}_mol.xyz -osmi"
        result = check_output(cmd, shell=True)
        smiles = result.decode("utf-8").split()[0]
        print(f"{smiles = }")
        ligpargen_cmd = f'''docker run --rm -v $(pwd)/{c_path}:/opt/output awallace43/ligpargen bash -c "ligpargen -s '{smiles}'"'''
        result = subprocess.run(ligpargen_cmd, shell=True)
        print(result)
    return


def run_ff_dimer():
    df = pd.read_pickle("./crystals_ap2_ap3_results_mol_apprx.pkl")
    for c in ['ice']:
        c_path = f"ff_params/{c}"
        if not os.path.exists(c_path):
            os.makedirs(c_path)
        mol = df[df["crystal apprx"] == c]["mol apprx"].iloc[0]
        print(mol.to_string("psi4"))
        break
    return


def main():
    # generate_paramaters()
    run_ff_dimer()
    return


if __name__ == "__main__":
    main()
