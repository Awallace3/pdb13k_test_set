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


def create_mols(
    data_path="./ammonia-pbe-d3-dz-aug/ammonia/*.p4str",
):
    result = {}
    for d in glob(data_path):
        basename = Path(d).stem
        with open(d, "r") as f:
            data = f.read()
            data += "\nunits au\n"
        mol = qcel.models.Molecule.from_data(data)
        result[basename] = [mol]
    return result


def process_inputs(
    output_data="./ammonia-pbe-d3-dz-aug/ammonia/ammonia.csv",
    delta_model_path="../../apnet_delta_correction/delta_correction_PBE_aug-cc-pVTZ_CP",
    data_path="./ammonia-pbe-d3-dz-aug/ammonia/*.p4str",
    output_csv="./ammonia-pbe-d3-dz-aug/ammonia/ammonia_dapnet2.csv",
    verbose=False,
):
    # Load data
    df_lt = pd.read_csv(output_data)
    if data_path.endswith(".p4str"):
        data = create_mols()
    elif data_path.endswith("*.in") or data_path.endswith("*.out"):
        # need to process data from path more interactively
        data = {}
        p4input_files = glob(data_path)
        for i in p4input_files:
            geom = subprocess.check_output(
                f"sed -n '/molecule/,/units = au/ {{/^  [A-Za-z]/p; /^--/p}}' {i}",
                shell=True,
            )
            # decode geom
            geom = geom.decode("utf-8")
            geom += "\nunits au\n"
            if verbose:
                print(i)
                print(geom)
            mol = qcel.models.Molecule.from_data(geom)
            data[Path(i).stem] = [mol]
    else:
        if verbose:
            print("Unrecognized filetype. Skipping mols")
        data = {}
        # raise ValueError("Unrecognized filetype")

    if delta_model_path is None:
        # need to save Geometry to df_lt['mol']
        RAs, RBs, ZAs, ZBs = [], [], [], []
        if len(data) > 0:
            mol_len = len(data[list(data.keys())[0]][0].fragments[0])
            df_lt["mol"] = [
                np.array([[0.0, 0.0, 0.0] for j in range(mol_len)])
                for i in range(len(df_lt))
            ]
            df_lt["RA"] = [
                np.array([[0.0, 0.0, 0.0] for j in range(mol_len)])
                for i in range(len(df_lt))
            ]
            df_lt["RB"] = [
                np.array([[0.0, 0.0, 0.0] for j in range(mol_len)])
                for i in range(len(df_lt))
            ]
            df_lt["ZA"] = [
                np.array([[0.0, 0.0, 0.0] for j in range(mol_len)])
                for i in range(len(df_lt))
            ]
            df_lt["ZB"] = [
                np.array([[0.0, 0.0, 0.0] for j in range(mol_len)])
                for i in range(len(df_lt))
            ]
            df_lt["RA"] = df_lt["RA"].astype(object)
            df_lt["RB"] = df_lt["RB"].astype(object)
            df_lt["ZA"] = df_lt["ZA"].astype(object)
            df_lt["ZB"] = df_lt["ZB"].astype(object)
            for i, (k, v) in enumerate(data.items()):
                dimer = v[0]
                df_lt.loc[df_lt["N-mer Name"] == k, "mol"] = v
                RA = (
                    np.array(dimer.geometry[dimer.fragments[0]], dtype=np.float32)
                    * qcel.constants.bohr2angstroms
                )
                RB = (
                    np.array(dimer.geometry[dimer.fragments[1]], dtype=np.float32)
                    * qcel.constants.bohr2angstroms
                )
                ZA = np.array(dimer.atomic_numbers[dimer.fragments[0]], dtype=np.int32)
                ZB = np.array(dimer.atomic_numbers[dimer.fragments[1]], dtype=np.int32)
                for idx in df_lt.loc[df_lt["N-mer Name"] == k].index:
                    df_lt.at[idx, "RA"] = RA
                    df_lt.at[idx, "RB"] = RB
                    df_lt.at[idx, "ZA"] = ZA
                    df_lt.at[idx, "ZB"] = ZB
        df_lt.to_pickle(output_csv.replace("csv", "pkl"))
        print(df_lt[["N-mer Name", "RA", "ZA"]])
        return df_lt

    df_lt["dAPNet2"] = None
    for k, v in data.items():
        df_lt.loc[df_lt["N-mer Name"] == k, "dAPNet2"] = 0.0
    print(dapnet2)
    for i, (k, v) in enumerate(data.items()):
        print(k, v)
        energy = dapnet2.predict_qcel_mols(v, batch_size=1)
        print(energy)
        df_lt.loc[df_lt["N-mer Name"] == k, "dAPNet2"] = energy[0]
    # import apnet
    #
    # pm = apnet.PairModel(
    #     apnet.AtomModel().pretrained(0), pair_model_type="delta_correction"
    # ).from_file(delta_model_path, pair_model_type="delta_correction")
    # for i, (k, v) in enumerate(data.items()):
    #     print(k, v)
    #     energy = pm.predict(v, batch_size=1)
    #     print(energy)
    #     if len(energy) == 0:
    #         energy = [[0]]
    #     df_lt.loc[df_lt["N-mer Name"] == k, "dAPNet2"] = energy[0][0]

    # ensure base path of output_csv exists and create if not
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    df_lt.to_csv(output_csv, index=False)
    return


def process_inputs_sapt0_induction(
    output_data="./ammonia-pbe-d3-dz-aug/ammonia/ammonia.csv",
    delta_model_path="../../apnet_delta_correction/delta_correction_PBE_aug-cc-pVTZ_CP",
    data_path="./ammonia-pbe-d3-dz-aug/ammonia/*.p4str",
    output_csv="./ammonia-pbe-d3-dz-aug/ammonia/ammonia_dapnet2.csv",
    verbose=False,
):
    # Load data
    df_lt = pd.read_csv(output_data)
    if data_path.endswith(".p4str"):
        data = create_mols()
    elif data_path.endswith("*.in") or data_path.endswith("*.out"):
        # need to process data from path more interactively
        data = {}
        p4input_files = glob(data_path)
        for i in p4input_files:
            geom = subprocess.check_output(
                f"sed -n '/molecule/,/units = au/ {{/^  [A-Za-z]/p; /^--/p}}' {i}",
                shell=True,
            )
            # decode geom
            geom = geom.decode("utf-8")
            geom += "\nunits au\n"
            Induction = subprocess.check_output(
                f"grep 'Induction       ' {i} | awk '{{ print $6 }}'",
                shell=True,
            )
            Induction = float(Induction.decode("utf-8"))
            Electrostatics = subprocess.check_output(
                f"grep 'Electrostatics               ' {i} | awk '{{ print $6 }}'",
                shell=True,
            )
            Electrostatics = float(Electrostatics.decode("utf-8"))
            Exchange = subprocess.check_output(
                f"grep 'Exchange               ' {i} | awk '{{ print $6 }}'",
                shell=True,
            )
            Exchange = float(Exchange.decode("utf-8"))
            Dispersion = subprocess.check_output(
                f"grep 'Dispersion               ' {i} | awk '{{ print $6 }}'",
                shell=True,
            )
            Dispersion = float(Dispersion.decode("utf-8"))
            if verbose:
                print(i)
                print(geom)
            mol = qcel.models.Molecule.from_data(geom)
            data[Path(i).stem] = [mol, Electrostatics, Exchange, Induction, Dispersion]
    else:
        if verbose:
            print("Unrecognized filetype. Skipping mols")
        data = {}
        # raise ValueError("Unrecognized filetype")

    if delta_model_path is None:
        # need to save Geometry to df_lt['mol']
        RAs, RBs, ZAs, ZBs = [], [], [], []
        if len(data) > 0:
            mol_len = len(data[list(data.keys())[0]][0].fragments[0])
            df_lt["mol"] = [
                np.array([[0.0, 0.0, 0.0] for j in range(mol_len)])
                for i in range(len(df_lt))
            ]
            df_lt["RA"] = [
                np.array([[0.0, 0.0, 0.0] for j in range(mol_len)])
                for i in range(len(df_lt))
            ]
            df_lt["RB"] = [
                np.array([[0.0, 0.0, 0.0] for j in range(mol_len)])
                for i in range(len(df_lt))
            ]
            df_lt["ZA"] = [
                np.array([[0.0, 0.0, 0.0] for j in range(mol_len)])
                for i in range(len(df_lt))
            ]
            df_lt["ZB"] = [
                np.array([[0.0, 0.0, 0.0] for j in range(mol_len)])
                for i in range(len(df_lt))
            ]
            df_lt["RA"] = df_lt["RA"].astype(object)
            df_lt["RB"] = df_lt["RB"].astype(object)
            df_lt["ZA"] = df_lt["ZA"].astype(object)
            df_lt["ZB"] = df_lt["ZB"].astype(object)
            df_lt["SAPT0 Electrostatics (kJ/mol)"] = [np.nan for i in range(len(df_lt))]
            df_lt["SAPT0 Exchange (kJ/mol)"] = [np.nan for i in range(len(df_lt))]
            df_lt["SAPT0 Induction (kJ/mol)"] = [np.nan for i in range(len(df_lt))]
            df_lt["SAPT0 Dispersion (kJ/mol)"] = [np.nan for i in range(len(df_lt))]
            for i, (k, v) in enumerate(data.items()):
                dimer = v[0]
                df_lt.loc[df_lt["N-mer Name"] == k, "mol"] = [v[0]]
                df_lt.loc[df_lt["N-mer Name"] == k, "SAPT0 Electrostatics (kJ/mol)"] = [
                    v[1]
                ]
                df_lt.loc[df_lt["N-mer Name"] == k, "SAPT0 Exchange (kJ/mol)"] = [v[2]]
                df_lt.loc[df_lt["N-mer Name"] == k, "SAPT0 Induction (kJ/mol)"] = [v[3]]
                df_lt.loc[df_lt["N-mer Name"] == k, "SAPT0 Dispersion (kJ/mol)"] = [
                    v[4]
                ]
                RA = (
                    np.array(dimer.geometry[dimer.fragments[0]], dtype=np.float32)
                    * qcel.constants.bohr2angstroms
                )
                RB = (
                    np.array(dimer.geometry[dimer.fragments[1]], dtype=np.float32)
                    * qcel.constants.bohr2angstroms
                )
                ZA = np.array(dimer.atomic_numbers[dimer.fragments[0]], dtype=np.int32)
                ZB = np.array(dimer.atomic_numbers[dimer.fragments[1]], dtype=np.int32)
                for idx in df_lt.loc[df_lt["N-mer Name"] == k].index:
                    df_lt.at[idx, "RA"] = RA
                    df_lt.at[idx, "RB"] = RB
                    df_lt.at[idx, "ZA"] = ZA
                    df_lt.at[idx, "ZB"] = ZB
        df_lt.to_pickle(output_csv.replace("csv", "pkl"))
        r1 = df_lt.iloc[0]
        tools.print_cartesians_pos_carts(r1["ZA"], r1["RA"])
        print(
            df_lt[
                [
                    "N-mer Name",
                    "SAPT0 Induction (kJ/mol)",
                    "SAPT0 Electrostatics (kJ/mol)",
                ]
            ]
        )
        return df_lt

    df_lt["dAPNet2"] = None
    for k, v in data.items():
        df_lt.loc[df_lt["N-mer Name"] == k, "dAPNet2"] = 0.0
    import apnet

    pm = apnet.PairModel(
        apnet.AtomModel().pretrained(0), pair_model_type="delta_correction"
    ).from_file(delta_model_path, pair_model_type="delta_correction")
    for i, (k, v) in enumerate(data.items()):
        print(k, v)
        energy = pm.predict(v, batch_size=1)
        print(energy)
        if len(energy) == 0:
            energy = [[0]]
        df_lt.loc[df_lt["N-mer Name"] == k, "dAPNet2"] = energy[0][0]
    # ensure base path of output_csv exists and create if not
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    df_lt.to_csv(output_csv, index=False)
    return


def analyze_results(
    data_path="./ammonia-pbe-d3-dz-aug/ammonia/ammonia_dapnet2.csv",
    bm_data_path="./ammonia-benchmark-cc/ammonia/ammonia.csv",
):
    df = pd.read_csv(data_path)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_rows", None)
    df["le_contribution"] = df.apply(
        lambda r: r["Non-Additive MB Energy (kJ/mol)"]
        * r["Num. Rep. (#)"]
        / int(r["N-mer Name"][0]),
        axis=1,
    )
    df["dAPNet2_le_contribution"] = df.apply(
        lambda r: (r["Non-Additive MB Energy (kJ/mol)"] + r["dAPNet2"])
        * r["Num. Rep. (#)"]
        / int(r["N-mer Name"][0]),
        axis=1,
    )
    df.rename(
        columns={
            "Minimum Monomer Separations (A)": "min_sep",
            "Partial Crys. Lattice Ener. (kJ/mol)": "lattice_energy",
            "Non-Additive MB Energy (kJ/mol)": "mb_energy",
        },
        inplace=True,
    )
    df["mb_energy_norm"] = df["mb_energy"] * 3
    # df['le_prev'] = df['lattice_energy'] - df['mb_energy'] * 3
    base_energy = df.iloc[0]["lattice_energy"]
    le_prev = []
    dAPNet2_le = []
    for i, row in df.iterrows():
        if i == 0:
            le_prev.append(base_energy)
            dAPNet2_le.append(row["dAPNet2_le_contribution"])
            continue
        le_prev.append(le_prev[-1] + row["le_contribution"])
        dAPNet2_le.append(dAPNet2_le[-1] + row["dAPNet2_le_contribution"])
    df["dAPNet2_le"] = dAPNet2_le
    df["le_prev"] = le_prev
    df_bm = pd.read_csv(bm_data_path)
    df_bm_le_first = df_bm.iloc[0]["Partial Crys. Lattice Ener. (kJ/mol)"]
    df_bm_le_last = df_bm.iloc[-1]["Partial Crys. Lattice Ener. (kJ/mol)"]
    df_le_first = df.iloc[0]["lattice_energy"]
    df_le_last = df.iloc[-1]["lattice_energy"]
    df_dapnet2_le_first = df.iloc[0]["dAPNet2_le"]
    df_dapnet2_le_last = df.iloc[-1]["dAPNet2_le"]
    print(f"   BASE LE First: {df_le_first:.4f}, LE Last: {df_le_last:.4f}")
    print(
        f"dAPNet2 LE First: {df_dapnet2_le_first:.4f}, LE Last: {df_dapnet2_le_last:.4f}"
    )
    print(f"     BM LE First: {df_bm_le_first:.4f}, LE Last: {df_bm_le_last:.4f}")
    df_bm.rename(
        columns={
            "Minimum Monomer Separations (A)": "min_sep",
            "Partial Crys. Lattice Ener. (kJ/mol)": "lattice_energy",
            "Non-Additive MB Energy (kJ/mol)": "mb_energy",
        },
        inplace=True,
    )
    bm_le = [0 for i in range(len(df))]
    bm_le[0] = df_bm_le_first
    bm_le[-1] = df_bm_le_last
    df["ccsd(t)_le"] = bm_le
    df.to_csv(data_path.replace(".csv", "_out.csv"), index=False)
    return


def ammonia(reprocess=False):
    if reprocess:
        process_inputs()
    analyze_results()
    return


def imidazole(reprocess=False):
    if reprocess:
        process_inputs(
            output_data="./imidazole-pbe-d3-dz-aug/imidazole/imidazole.csv",
            delta_model_path="../../apnet_delta_correction/delta_correction_PBE_aug-cc-pVTZ_CP",
        )
    analyze_results(
        data_path="./imidazole-pbe-d3-dz-aug/imidazole/imidazole.csv",
        bm_data_path="./imidazole-benchmark-cc/imidazole/imidazole.csv",
    )
    return


def hexamine(reprocess=False):
    if reprocess:
        process_inputs(
            output_data="/theoryfs2/ds/csargent/chem/x23-crystals/hexamine/pbe-d3-dz-aug/hexamine/hexamine.csv",
            data_path="/theoryfs2/ds/csargent/chem/x23-crystals/hexamine/pbe-d3-dz-aug/hexamine/*.in",
            delta_model_path="../../apnet_delta_correction/delta_correction_PBE_aug-cc-pVTZ_CP",
            output_csv="./ouputs/hexamine/pbe-d3-dz-aug/hexamine/hexamine_dapnet2.csv",
        )
    analyze_results(
        data_path="./ouputs/hexamine/pbe-d3-dz-aug/hexamine/hexamine_dapnet2.csv",
        bm_data_path="/theoryfs2/ds/csargent/chem/x23-crystals/hexamine/benchmark-cc/hexamine/hexamine.csv",
    )
    return


def process_all_crystals(compute=False):
    base_path = "/theoryfs2/ds/csargent/chem/x23-crystals/"
    crystal_names = [
        "14-cyclohexanedione",
        "acetic_acid",
        "adamantane",
        "ammonia",
        "anthracene",
        "benzene",
        "cyanamide",
        "cytosine",
        "ethyl_carbamate",
        "formamide",
        "hexamine",
        "ice",
        "imidazole",
        "naphthalene",
        "oxalic_acid_alpha",
        "oxalic_acid_beta",
        "pyrazine",
        "pyrazole",
        "succinic_acid",
        "triazine",
        "trioxane",
        "uracil",
        "urea",
    ]
    level_of_theories = [
        # [
        #     "pbe-d3-dz-aug",
        #     # "../../apnet_delta_correction/delta_correction_PBE-D3BJ_aug-cc-pVDZ_CP",
        #     "../../dapnet_models/delta_correction_PBE-D3_aug-cc-pVDZ_CP_to_CCSDT_CBS_CP",
        # ],
        [
            "b3lyp",  # actually is b3lyp-d3bj-adz
            # "../../apnet_delta_correction/delta_correction_B3LYP-D3BJ_aug-cc-pVDZ_CP",
            # "../../dapnet_models/delta_correction_B3LYP-D3_aug-cc-pVDZ_CP_to_CCSDT_CBS_CP",
            "/home/awallace43/gits/qcmlforge/models/dapnet2/B3LYP-D3_aug-cc-pVDZ_CP_to_CCSD_LP_T_RP_CBS_CP_0.pt",
        ],
        # [
        #     "pbeh3c",
        #     # "../../apnet_delta_correction/delta_correction_PBEh-3c_aug-cc-pVDZ_CP",
        #     "../../dapnet_models/delta_correction_PBEh-3c_aug-cc-pVDZ_CP_to_CCSDT_CBS_CP",
        # ],
        # [
        #     "b97-d3bj-dz-aug",
        #     # "../../apnet_delta_correction/delta_correction_B97-D3_aug-cc-pVTZ_CP",
        #     "../../apnet_delta_correction/delta_correction_B97"
        # ],
        # [
        #     "mp2-dz-aug",
        #     "../../apnet_delta_correction/delta_correction_MP2_aug-cc-pVDZ_CP",
        # ],
    ]
    crystals = []
    for c in crystal_names:
        for lt in level_of_theories:
            crystals.append(
                {
                    "name": c,
                    "level_of_theory": lt[0],
                    "delta_model_path": lt[1],
                }
            )
    print("Crystal Plan:")
    pp(crystals)
    for c in crystals:
        try:
            print(f"\nProcessing {c['name']} crystal at {c['level_of_theory']}")
            pp(c)
            if compute:
                process_inputs(
                    output_data=f"{base_path}{c['name']}/{c['level_of_theory']}/{c['name']}/{c['name']}.csv",
                    data_path=f"{base_path}{c['name']}/{c['level_of_theory']}/{c['name']}/*.in",
                    delta_model_path=c["delta_model_path"],
                    output_csv=f"./outputs/{c['name']}/{c['level_of_theory']}/{c['name']}/{c['name']}_dapnet2.csv",
                )
            analyze_results(
                data_path=f"./outputs/{c['name']}/{c['level_of_theory']}/{c['name']}/{c['name']}_dapnet2.csv",
                bm_data_path=f"{base_path}{c['name']}/benchmark-cc/{c['name']}/{c['name']}.csv",
            )
        except Exception as e:
            print(f"Error processing {c['name']} crystal at {c['level_of_theory']}")
            print(e)
            print("Continue processing other crystals...\n\n")
            # raise e
    return


def collect_crystal_data(generate=False):
    base_path = "/theoryfs2/ds/csargent/chem/x23-crystals/"
    crystal_names = [
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
    level_of_theories = [
        "b3lyp",
        "b97-d3bj-dz-aug",
        "b97d-dz-aug",
        "hf3c",
        "mp2-dz-aug",
        "mp2-dz-jun",
        "mp2_5-dz-aug",
        "mp2_5-dz-jun",
        "mp2d-dz-aug",
        "mp2d-dz-jun",
        "pbe-d3-def2",
        "pbe-d3-dz-aug",
        "pbeh3c-590",
        "sapt0-dz-aug",
        "sapt0-dz-jun",
    ]
    main_frames = []
    for cname in crystal_names:
        frames = []
        if generate:
            for n, lt in enumerate(level_of_theories):
                c = {
                    "name": cname,
                    "level_of_theory": lt,
                }
                try:
                    print(f"\nProcessing {c['name']} crystal at {c['level_of_theory']}")
                    pp(c)
                    if n == 0:
                        data_path = f"{base_path}{c['name']}/{c['level_of_theory']}/{c['name']}/*.in"
                        print(data_path)
                    else:
                        data_path = ""
                    # data_path=""
                    df = process_inputs(
                        output_data=f"{base_path}{c['name']}/{c['level_of_theory']}/{c['name']}/{c['name']}.csv",
                        data_path=data_path,
                        delta_model_path=None,
                        output_csv=f"./x23_dfs/{c['name']}_{c['level_of_theory']}_cle.csv",
                    )
                    df.rename(
                        columns={
                            k: f"{k} {lt}"
                            for k in df.columns
                            if k != "N-mer Name" and k != "mol"
                        },
                        inplace=True,
                    )
                    df[f"output {c['level_of_theory']}"] = df["N-mer Name"].apply(
                        lambda x: data_path.replace("*", x)
                    )
                    df = df.dropna(subset=["N-mer Name"])
                    frames.append(df)
                except Exception as e:
                    print(
                        f"Error processing {c['name']} crystal at {c['level_of_theory']}"
                    )
                    print(e)
                    print("Continue processing other crystals...\n\n")
            for n, f in enumerate(frames):
                if n == 0:
                    df = f
                else:
                    df = df.merge(f, on="N-mer Name")
            if "mol" not in df.columns:
                df["mol"] = None
            df["crystal"] = [cname for _ in range(len(df))]
            print(df[["N-mer Name", "mol"]])
            df = df.dropna(subset=["N-mer Name"])
            print("Null mols:", df["mol"].isna().sum())
            df.to_pickle(f"./x23_dfs/{cname}_apprx_method.pkl")
            df_apprx = df.copy()
        else:
            df_apprx = pd.read_pickle(f"./x23_dfs/{cname}_apprx_method.pkl")
            print("Null mols:", df_apprx["mol"].isna().sum())
        c = {
            "name": cname,
            "level_of_theory": "benchmark-cc",
        }
        pp(c)
        # if cname in ['formamide', 'oxalic_acid_beta']:
        #     output_data=f"{base_path}{c['name']}/{c['level_of_theory']}/{c['name']}/0-benchmark-cc.csv"
        # elif cname in ['naphthalene']:
        #     output_data=f"{base_path}{c['name']}/{c['level_of_theory']}/0-benchmark-cc.csv"
        # elif cname in ['oxalic_acid_alpha', 'pyrazine']:
        #     output_data=f"{base_path}{c['name']}/{c['level_of_theory']}/{c['name']}/benchmark-cc.csv"
        output_data = f"{base_path}{c['name']}/{c['level_of_theory']}/{c['name']}/{c['name']}-fp.csv"
        if cname in ["CO2"]:
            output_data = f"{base_path}{c['name']}/{c['level_of_theory']}/carbon_dioxide/CO2-fp.csv"
        # else:

        base_parent = Path(output_data).parent
        data_path = f"{base_parent}/*.out"
        # data_path=""
        if generate:
            df = process_inputs(
                output_data=output_data,
                data_path=data_path,
                delta_model_path=None,
                output_csv=f"./x23_dfs/{c['name']}_{c['level_of_theory']}_cle.csv",
                verbose=False,
            )
            df.rename(
                columns={
                    k: f"{k} CCSD(T)/CBS"
                    for k in df.columns
                    if k != "N-mer Name" and k != "mol"
                },
                inplace=True,
            )
            df[f"output {c['level_of_theory']}"] = df["N-mer Name"].apply(
                lambda x: data_path.replace("*", x)
            )
            df["crystal"] = [cname for _ in range(len(df))]
            if "mol" not in df.columns:
                df["mol"] = None
            print(df[["N-mer Name", "mol"]])
            print("Null mols:", df["mol"].isna().sum())
            df.to_pickle(f"./x23_dfs/{cname}_benchmark.pkl")
            df_bm = df.copy()
        else:
            df_bm = pd.read_pickle(f"./x23_dfs/{cname}_benchmark.pkl")
            print("Null mols:", df_bm["mol"].isna().sum())
        df_apprx.sort_values(by="N-mer Name", inplace=True)
        df_bm.sort_values(by="N-mer Name", inplace=True)
        df_apprx["hash"] = df_apprx["mol"].apply(lambda x: x.get_hash())
        df_bm["hash"] = df_bm["mol"].apply(lambda x: x.get_hash())
        # print(df_apprx[['N-mer Name', 'mol']])
        # print(df_bm[['N-mer Name', 'mol']])
        df = df_apprx.merge(df_bm, on="hash", how="outer", suffixes=(" apprx", " bm"))
        print(
            "lengths:", len(df_apprx), len(df_bm), len(df), len(df_apprx) + len(df_bm)
        )
        print(df)
        df.to_pickle(f"./x23_dfs/{cname}_all.pkl")
        main_frames.append(df)
    # for i in range(len(main_frames)):
    #     if i == 0:
    #         columns = main_frames[i].columns.tolist()
    #     else:
    #         columns += main_frames[i].columns.tolist()
    # for i in range(len(main_frames)):
    #     for c in columns:
    #         if c not in main_frames[i].columns:
    #             main_frames[i][c] = None
    main_df = pd.concat(main_frames)
    main_df.reset_index(drop=False, inplace=True)
    main_df.rename(columns={"index": "local_crystal_index"}, inplace=True)
    pp(main_df.columns.tolist())
    print(main_df)
    print(main_df[["N-mer Name apprx", "crystal apprx", "crystal bm"]])
    main_df.to_pickle("./x23_dfs/main_df.pkl")
    return


def ap2_energies(mols, compile=True):
    ap2 = apnet_pt.AtomPairwiseModels.apnet2_fused.APNet2_AM_Model(
        atom_model=apnet_pt.AtomModels.ap2_atom_model.AtomModel(
            pre_trained_model_path="/home/awallace43/gits/qcmlforge/models/am_ensemble/am_1.pt",
        ).model,
        pre_trained_model_path="/home/awallace43/gits/qcmlforge/models/ap2-fused_ensemble/ap2_1.pt",
    )
    if compile:
        ap2.compile_model()
    pred = ap2.predict_qcel_mols(mols, batch_size=64)
    return pred


def ap3_d_elst_classical_energies(mols):
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
    ap3 = apnet_pt.AtomPairwiseModels.apnet3_fused.APNet3_AtomType_Model(
        ds_root=None,
        atom_type_model=atom_type_hf_vw_model.model,
        dimer_prop_model=atom_type_elst_model.dimer_model,
        pre_trained_model_path=ap3_path,
    )
    pred, pair_elst, pair_ind = ap3.predict_qcel_mols(
        mols, batch_size=64, return_classical_pairs=True
    )
    return pred, pair_elst, pair_ind


def ap2_ap3_df_energies(generate=False):
    v = "apprx"
    v = "bm"
    mol_str = "mol " + v
    pkl_fn = f"crystals_ap2_ap3_results_{mol_str.replace(' ', '_')}.pkl"
    if not os.path.exists(pkl_fn) or generate:
        df = pd.read_pickle("./x23_dfs/main_df.pkl")
        print(df)
        df = df.dropna(subset=[mol_str])
        # df = df.dropna(subset=["mol bm"])
        print(df)
        # df = df.head()
        pp(df.columns.tolist())
        print("AP2 start")
        mols = df[mol_str].tolist()
        pred_ap2 = ap2_energies(mols)
        pred_ap2 *= kcalmol_to_kjmol
        df["AP2 TOTAL"] = np.sum(pred_ap2[:, 0:4], axis=1)
        df["AP2 ELST"] = pred_ap2[:, 0]
        df["AP2 EXCH"] = pred_ap2[:, 1]
        df["AP2 INDU"] = pred_ap2[:, 2]
        df["AP2 DISP"] = pred_ap2[:, 3]
        df.to_pickle(pkl_fn)
        print("AP3 start")
        pred_ap3, pair_elst, pair_ind = ap3_d_elst_classical_energies(mols)
        pred_ap3 *= kcalmol_to_kjmol
        elst_energies = [np.sum(e) * kcalmol_to_kjmol for e in pair_elst]
        ind_energies = [np.sum(e) * kcalmol_to_kjmol for e in pair_ind]
        # df["mtp_elst_energies"] = elst_undamped
        df["ap3_d_elst"] = elst_energies
        df["ap3_classical_ind_energy"] = ind_energies
        df["AP3 TOTAL"] = np.sum(pred_ap3[:, 0:4], axis=1)
        df["AP3 ELST"] = pred_ap3[:, 0]
        df["AP3 EXCH"] = pred_ap3[:, 1]
        df["AP3 INDU"] = pred_ap3[:, 2]
        df["AP3 DISP"] = pred_ap3[:, 3]
        # LE
        non_additive_cols = [
            i for i in df.columns if "Non-Additive MB Energy (kJ/mol)" in i
        ]
        for c in non_additive_cols:
            label = c.split()[-1]
            df[f"{label}_le_contribution"] = df.apply(
                lambda r: r[c]
                * r[f"Num. Rep. (#) {label}"]
                / int(r[f"N-mer Name {v}"][0]),
                axis=1,
            )
        df["ap3_le_contribution"] = df.apply(
            lambda r: r["AP3 TOTAL"]
            * r["Num. Rep. (#) sapt0-dz-aug"]
            / int(r[f"N-mer Name {v}"][0]),
            axis=1,
        )
        df["ap2_le_contribution"] = df.apply(
            lambda r: r["AP2 TOTAL"]
            * r["Num. Rep. (#) sapt0-dz-aug"]
            / int(r[f"N-mer Name {v}"][0]),
            axis=1,
        )
        df.to_pickle(pkl_fn)
    else:
        df = pd.read_pickle(pkl_fn)

    if v == "apprx":
        pp(df.columns.tolist())
        print("\nResults\n")
        df["AP3 total error"] = (
            df["Non-Additive MB Energy (kJ/mol) sapt0-dz-aug"] - df["AP3 TOTAL"]
        )
        df["AP2 total error"] = (
            df["Non-Additive MB Energy (kJ/mol) sapt0-dz-aug"] - df["AP2 TOTAL"]
        )
        mae_total_ap3 = df["AP3 total error"].abs().mean()
        print(f"MAE AP3 Total: {mae_total_ap3:.4f} kJ/mol")
        mae_total_ap2 = df["AP2 total error"].abs().mean()
        print(f"MAE AP2 Total: {mae_total_ap2:.4f} kJ/mol")
        print(
            df[
                [
                    "AP3 total error",
                    "AP2 total error",
                ]
            ].describe()
        )
    return df


def dapnet_main_df_crystals(
    delta_model_paths={
        # "b3lyp": "../../dapnet_models/delta_correction_B3LYP-D3_aug-cc-pVDZ_CP_to_CCSDT_CBS_CP_limit_B2PLYP-D3_aug-cc-pVTZ_CP",
        "b3lyp": "/home/awallace43/gits/qcmlforge/models/dapnet2/B3LYP-D3_aug-cc-pVDZ_CP_to_CCSD_LP_T_RP_CBS_CP_0.pt",
        # "pbe-d3-dz-aug": "../../dapnet_models/delta_correction_PBE-D3_aug-cc-pVDZ_CP_to_CCSDT_CBS_CP_limit_B2PLYP-D3_aug-cc-pVTZ_CP",
        # "pbeh3c-590": "../../dapnet_models/delta_correction_PBEh-3c_aug-cc-pVDZ_CP_to_CCSDT_CBS_CP_limit_B2PLYP-D3_aug-cc-pVTZ_CP",
    },
    output_pickle="./x23_dfs/main_df_cbs.pkl",
    compute_delta=False,
    verbose=False,
):
    if compute_delta:
        df = pd.read_pickle("./x23_dfs/main_df.pkl")
    else:
        df = pd.read_pickle(output_pickle)
    crystal_information = {
        k: {
            "bm cle": np.nan,
            "apprx cles": {k: np.nan for k in delta_model_paths.keys()},
            "cle target": {k: np.nan for k in delta_model_paths.keys()},
            "dAPNet2 apprx": {k: np.nan for k in delta_model_paths.keys()},
        }
        for k in crystal_names_all
    }
    for c in crystal_names_all:
        df_c_a = df[df["crystal apprx"] == c]
        df_c_a = df_c_a.dropna(subset=["mol apprx"])
        df_c_a.sort_values(by="local_crystal_index", inplace=True)
        df_c_b = df[df["crystal bm"] == c]
        df_c_b = df_c_b.dropna(subset=["mol bm"])
        df_c_a.sort_values(by="local_crystal_index", inplace=True)
        if verbose:
            print(c, len(df_c_a), len(df_c_b))
        for k, v in delta_model_paths.items():
            basefile = Path(v).stem
            apprx_final = df_c_a.apply(
                lambda r: (
                    (r[f"Non-Additive MB Energy (kJ/mol) {k}"])
                    * r[f"Num. Rep. (#) {k}"]
                    / int(r[f"N-mer Name apprx"][0])
                    if pd.notnull(r["N-mer Name apprx"])
                    else 0
                ),
                axis=1,
            ).sum()
            bm_final = df_c_b.apply(
                lambda r: (
                    (r[f"Non-Additive MB Energy (kJ/mol) CCSD(T)/CBS"])
                    * r[f"Num. Rep. (#) CCSD(T)/CBS"]
                    / int(r[f"N-mer Name bm"][0])
                    if pd.notnull(r["N-mer Name bm"])
                    else 0
                ),
                axis=1,
            ).sum()
            # bm_final = df_c_b[f"Partial Crys. Lattice Ener. (kJ/mol) sapt0-dz-jun bm"].values[-1]
            if bm_final == 0.0:
                bm_final = np.nan
            if apprx_final == 0.0:
                apprx_final = np.nan
            crystal_information[c]["bm cle"] = bm_final
            crystal_information[c]["apprx cles"][k] = apprx_final
            crystal_information[c]["cle target"][k] = bm_final - apprx_final
            if not compute_delta:
                df_c_a[f"dapnet cle {v}"] = df_c_a.apply(
                    lambda r: (
                        r[f"Non-Additive MB Energy (kJ/mol) {k}"]
                        - r[f"dAPNet2 apprx {basefile}"]
                    )
                    * r[f"Num. Rep. (#) {k}"]
                    / int(r[f"N-mer Name apprx"][0]),
                    axis=1,
                )
                crystal_information[c]["dAPNet2 apprx"][k] = df_c_a[
                    f"dapnet cle {v}"
                ].sum()
    if verbose:
        pp(crystal_information)
    # import apnet

    update_progress = 1
    for k, delta_model_path in delta_model_paths.items():
        basefile = Path(delta_model_path).stem
        if verbose:
            print(basefile)
        if compute_delta:
            df[f"dAPNet2 apprx {basefile}"] = None
            df[f"dAPNet2 bm {basefile}"] = None
            # pm = apnet.PairModel(
            #     apnet.AtomModel().pretrained(0), pair_model_type="delta_correction"
            # ).from_file(delta_model_path, pair_model_type="delta_correction")
            print("Model loaded")
            for n, row in df.iterrows():
                if pd.notnull(row["mol apprx"]):
                    # energy = pm.predict([row["mol apprx"]], batch_size=1)
                    # if len(energy) == 0:
                    #     energy = [[0]]
                    energy = dapnet2.predict_qcel_mols([row["mol apprx"]], batch_size=1)
                    df.loc[n, f"dAPNet2 apprx {basefile}"] = (
                        energy[0] * kcalmol_to_kjmol
                    )
                if pd.notnull(row["mol bm"]):
                    # energy = pm.predict([row["mol bm"]], batch_size=1)
                    energy = dapnet2.predict_qcel_mols([row["mol bm"]], batch_size=1)
                    if len(energy) == 0:
                        energy = [[0]]
                    df.loc[n, f"dAPNet2 bm {basefile}"] = (
                        # energy[0][0] * kcalmol_to_kjmol
                        energy[0] * kcalmol_to_kjmol
                    )
                if n % update_progress == 0:
                    energies_sum_crystal_apprx = df[
                        df["crystal apprx"] == row["crystal apprx"]
                    ][f"dAPNet2 apprx {basefile}"].sum()
                    energies_sum_crystal_bm = df[df["crystal bm"] == row["crystal bm"]][
                        f"dAPNet2 bm {basefile}"
                    ].sum()
                    active_crystal = (
                        row["crystal apprx"]
                        if pd.notnull(row["crystal apprx"])
                        else row["crystal bm"]
                    )
                    apprx_error = crystal_information[active_crystal]["cle target"][k]
                    print(
                        f"{n}/{len(df)}, crystal: {active_crystal}, dAP (apprx, bm): ({energies_sum_crystal_apprx:.2f}, {energies_sum_crystal_bm:.2f}), CLE d-target: {apprx_error:.2f}"
                    )
        else:
            if verbose:
                for n, row in df.iterrows():
                    if n % update_progress == 0:
                        energies_sum_crystal_apprx = df[
                            df["crystal apprx"] == row["crystal apprx"]
                        ][f"dAPNet2 apprx {basefile}"].sum()
                        energies_sum_crystal_bm = df[
                            df["crystal bm"] == row["crystal bm"]
                        ][f"dAPNet2 bm {basefile}"].sum()
                        active_crystal = (
                            row["crystal apprx"]
                            if pd.notnull(row["crystal apprx"])
                            else row["crystal bm"]
                        )
                        apprx_error = crystal_information[active_crystal]["cle target"][
                            k
                        ]
                        print(
                            f"{n}/{len(df)}, crystal: {active_crystal}, dAP (apprx, bm): ({energies_sum_crystal_apprx:.2f}, {energies_sum_crystal_bm:.2f}), CLE d-target: {apprx_error:.2f}"
                        )
        if compute_delta:
            Path(output_pickle).parent.mkdir(parents=True, exist_ok=True)
            print(df)
            print("Saving to", output_pickle)
            df.to_pickle(output_pickle)
    return df, crystal_information


def plot_crystal_errors(
    compute_delta=False,
):
    if compute_delta:
        df, ci = dapnet_main_df_crystals(
            compute_delta=compute_delta,
        )
    df, ci = dapnet_main_df_crystals()
    pp(ci)
    data = {}
    for c, v in ci.items():
        data[c] = {}
        for k, v2 in v.items():
            if isinstance(v2, dict):
                for k2, v3 in v2.items():
                    data[c][f"{k} {k2}"] = v3
            else:
                data[c][k] = v2
    df_c = pd.DataFrame(data).T
    # df_c = df_c.dropna(subset=['apprx cles pbeh3c-590'])
    # df_c = df_c.dropna(subset=['apprx cles pbeh3c-590'])
    bounds = [[-10, 10] for i in range(3)]
    dfs = []
    for i in [
        # "pbe-d3-dz-aug", "pbeh3c-590",
        "b3lyp"
    ]:
        df_c[f"{i} error"] = df_c[f"apprx cles {i}"] - df_c[f"bm cle"]
        df_c[f"{i}+dAPNet2 error"] = df_c[f"dAPNet2 apprx {i}"] - df_c[f"bm cle"]
        df_c["diff"] = df_c[f"{i}+dAPNet2 error"] - df_c[f"{i} error"]
        print(df_c[[f"{i} error", f"{i}+dAPNet2 error", "diff"]].describe())
    dfs.append(
        {
            "df": df_c,
            "basis": "",
            "label": "CCSD(T)/CBS",
            "ylim": bounds,
        }
    )
    print(
        df_c[
            ["bm cle", "apprx cles b3lyp", "dAPNet2 apprx b3lyp", "b3lyp+dAPNet2 error"]
        ]
    )
    print(df_c[["bm cle", "apprx cles b3lyp", "b3lyp error", "b3lyp+dAPNet2 error"]])
    df_c.sort_values(by="b3lyp+dAPNet2 error", inplace=True)
    print(
        df_c[
            ["bm cle", "apprx cles b3lyp", "dAPNet2 apprx b3lyp", "b3lyp+dAPNet2 error"]
        ]
    )
    print(df_c[["bm cle", "apprx cles b3lyp", "b3lyp error", "b3lyp+dAPNet2 error"]])
    df_tex = df_c[
        ["bm cle", "apprx cles b3lyp", "b3lyp error", "b3lyp+dAPNet2 error"]
    ].copy()
    # fmt to 2 decimal
    df_tex["abs_error"] = df_tex["b3lyp+dAPNet2 error"].abs()
    df_tex.sort_values(by="abs_error", inplace=True)
    df_tex.drop(columns=["abs_error"], inplace=True)
    df_tex.to_latex(
        "./x23_plots/b3lypd3adz_crystal_errors.tex",
        index=True,
        float_format="{:0.2f}".format,
    )
    from cdsg_plot import error_statistics

    error_statistics.violin_plot_table_multi_SAPT_components(
        dfs,
        df_labels_and_columns_total={
            "B3LYP-D3(BJ)\\\\/aDZ": "b3lyp error",
            "B3LYP-D3(BJ)\\\\/aDZ+dAPNet2": "b3lyp+dAPNet2 error",
            # "PBE-D3/aDZ": "pbe-d3-dz-aug error",
            # "PBE-D3/aDZ+dAPNet2": "pbe-d3-dz-aug+dAPNet2 error",
            # "PBEh-3c/aDZ": "pbeh3c-590 error",
            # "PBEh-3c/aDZ+dAPNet2": "pbeh3c-590+dAPNet2 error",
        },
        output_filename="./x23_plots/crystal_errors.png",
        grid_heights=[0.3, 1.0],
        grid_widths=[1],
        legend_loc="upper left",
        annotations_texty=0.3,
        figure_size=(4, 2.5),
        add_title=False,
    )
    return


def main_og():
    # ammonia()
    # hexamine(True)
    # imidazole()
    # process_all_crystals(True)
    collect_crystal_data(generate=False)
    plot_crystal_errors(compute_delta=False)

    df, ci = dapnet_main_df_crystals()
    pp(ci)
    data = {}
    for c, v in ci.items():
        data[c] = {}
        for k, v2 in v.items():
            if isinstance(v2, dict):
                for k2, v3 in v2.items():
                    data[c][f"{k} {k2}"] = v3
            else:
                data[c][k] = v2
    df_c = pd.DataFrame(data).T
    print(df_c)
    return
    drude_crystals = [
        "pyrazine",
        "imidazole",
        "acetic_acid",
        "ice",
        "pyrazole",
    ]
    for i in drude_crystals:
        print(i)
        output_data = (
            f"/theoryfs2/ds/csargent/chem/x23-crystals/{i}/sapt0-dz-aug/{i}/{i}.csv"
        )
        if i == "pyrazine":
            output_data = f"/theoryfs2/ds/csargent/chem/x23-crystals/{i}/sapt0-dz-aug/{i}/sapt0-dz-aug.csv"
        process_inputs_sapt0_induction(
            output_data=output_data,
            delta_model_path=None,
            data_path=f"/theoryfs2/ds/csargent/chem/x23-crystals/{i}/sapt0-dz-aug/{i}/*.out",
            output_csv=f"./sapt0_induction/{i}_sapt0adz.pkl",
        )


def plot_full_crystal_errors():
    """
    Create violin plots for full crystal lattice energy errors.
    Similar to dapnet_main_df_crystals, this sums up contributions per crystal
    and compares AP2/AP3 against reference methods.
    """
    from cdsg_plot import error_statistics

    # Load dataframes
    df1 = pd.read_pickle("./crystals_ap2_ap3_results_mol_apprx.pkl")
    df2 = pd.read_pickle("./crystals_ap2_ap3_results_mol_bm.pkl")

    print("Processing df1 (approximate methods)...")
    print(f"Total rows in df1: {len(df1)}")
    print(f"Crystals in df1: {df1['crystal apprx'].unique()}")

    print("\nProcessing df2 (benchmark)...")
    print(f"Total rows in df2: {len(df2)}")
    print(f"Crystals in df2: {df2['crystal bm'].unique()}")

    # Calculate crystal lattice energies for df1 (comparing against SAPT0)
    crystal_data_df1 = {}
    for crystal in df1["crystal apprx"].dropna().unique():
        df_c = df1[df1["crystal apprx"] == crystal].copy()

        # Sum up lattice energy contributions for SAPT0 reference
        sapt0_le = df_c.apply(
            lambda r: r["Non-Additive MB Energy (kJ/mol) sapt0-dz-aug"]
            * r["Num. Rep. (#) sapt0-dz-aug"]
            / int(r["N-mer Name apprx"][0])
            if pd.notnull(r["N-mer Name apprx"])
            else 0,
            axis=1,
        ).sum()

        # Sum up lattice energy contributions for AP2
        ap2_le = df_c.apply(
            lambda r: r["AP2 TOTAL"]
            * r["Num. Rep. (#) sapt0-dz-aug"]
            / int(r["N-mer Name apprx"][0])
            if pd.notnull(r["N-mer Name apprx"])
            else 0,
            axis=1,
        ).sum()

        # Sum up lattice energy contributions for AP3
        ap3_le = df_c.apply(
            lambda r: r["AP3 TOTAL"]
            * r["Num. Rep. (#) sapt0-dz-aug"]
            / int(r["N-mer Name apprx"][0])
            if pd.notnull(r["N-mer Name apprx"])
            else 0,
            axis=1,
        ).sum()

        # Calculate errors
        crystal_data_df1[crystal] = {
            "sapt0_le": sapt0_le,
            "ap2_le": ap2_le,
            "ap3_le": ap3_le,
            "AP2 error": ap2_le - sapt0_le,
            "AP3 error": ap3_le - sapt0_le,
        }

    # Calculate crystal lattice energies for df2 (comparing against CCSD(T)/CBS)
    crystal_data_df2 = {}
    for crystal in df2["crystal bm"].dropna().unique():
        df_c = df2[df2["crystal bm"] == crystal].copy()

        # Sum up lattice energy contributions for CCSD(T)/CBS reference
        ccsdt_le = df_c.apply(
            lambda r: r["Non-Additive MB Energy (kJ/mol) CCSD(T)/CBS"]
            * r["Num. Rep. (#) CCSD(T)/CBS"]
            / int(r["N-mer Name bm"][0])
            if pd.notnull(r["N-mer Name bm"])
            else 0,
            axis=1,
        ).sum()

        # Sum up lattice energy contributions for AP2
        ap2_le = df_c.apply(
            lambda r: r["AP2 TOTAL"]
            * r["Num. Rep. (#) CCSD(T)/CBS"]
            / int(r["N-mer Name bm"][0])
            if pd.notnull(r["N-mer Name bm"])
            else 0,
            axis=1,
        ).sum()

        # Sum up lattice energy contributions for AP3
        ap3_le = df_c.apply(
            lambda r: r["AP3 TOTAL"]
            * r["Num. Rep. (#) CCSD(T)/CBS"]
            / int(r["N-mer Name bm"][0])
            if pd.notnull(r["N-mer Name bm"])
            else 0,
            axis=1,
        ).sum()

        # Calculate errors
        crystal_data_df2[crystal] = {
            "ccsdt_le": ccsdt_le,
            "ap2_le": ap2_le,
            "ap3_le": ap3_le,
            "AP2 error": ap2_le - ccsdt_le,
            "AP3 error": ap3_le - ccsdt_le,
        }

    # Convert to DataFrames
    df1_crystal_errors = pd.DataFrame(crystal_data_df1).T
    df2_crystal_errors = pd.DataFrame(crystal_data_df2).T

    print("\n=== df1 Crystal Lattice Energy Errors (vs SAPT0) ===")
    print(df1_crystal_errors[["AP2 error", "AP3 error"]])
    print("\nStatistics:")
    print(df1_crystal_errors[["AP2 error", "AP3 error"]].describe())

    print("\n=== df2 Crystal Lattice Energy Errors (vs CCSD(T)/CBS) ===")
    print(df2_crystal_errors[["AP2 error", "AP3 error"]])
    print("\nStatistics:")
    print(df2_crystal_errors[["AP2 error", "AP3 error"]].describe())

    # Prepare data for violin plots
    dfs1 = [
        {
            "df": df1_crystal_errors,
            "basis": "",
            "label": "SAPT0/aDZ Reference",
            "ylim": [[-30, 30]],
        }
    ]

    dfs2 = [
        {
            "df": df2_crystal_errors,
            "basis": "",
            "label": "CCSD(T)/CBS Reference",
            "ylim": [[-30, 30]],
        }
    ]

    # Create violin plot for df1 (approximate methods)
    print("\nCreating violin plot for df1 crystal errors...")
    error_statistics.violin_plot_table_multi_SAPT_components(
        dfs1,
        df_labels_and_columns_total={
            "AP2": "AP2 error",
            "AP3": "AP3 error",
        },
        output_filename="./x23_plots/ap2_ap3_errors_vs_sapt0_violin.png",
        grid_heights=[0.3, 1.0],
        grid_widths=[1],
        legend_loc="upper left",
        annotations_texty=0.3,
        figure_size=(4, 2.5),
        add_title=False,
    )
    print("Saved plot to ./x23_plots/ap2_ap3_errors_vs_sapt0_violin.png")

    # Create violin plot for df2 (benchmark comparison)
    print("\nCreating violin plot for df2 crystal errors...")
    error_statistics.violin_plot_table_multi_SAPT_components(
        dfs2,
        df_labels_and_columns_total={
            "AP2": "AP2 error",
            "AP3": "AP3 error",
        },
        output_filename="./x23_plots/ap2_ap3_errors_vs_ccsdt_cbs_violin.png",
        grid_heights=[0.3, 1.0],
        grid_widths=[1],
        legend_loc="upper left",
        annotations_texty=0.3,
        figure_size=(4, 2.5),
        add_title=False,
    )
    print("Saved plot to ./x23_plots/ap2_ap3_errors_vs_ccsdt_cbs_violin.png")

    # Print summary statistics
    print("\n=== Summary Statistics (Crystal Lattice Energies) ===")
    print("\ndf1 (vs SAPT0/aDZ):")
    for col in ["AP2 error", "AP3 error"]:
        mae = df1_crystal_errors[col].abs().mean()
        rmse = np.sqrt((df1_crystal_errors[col] ** 2).mean())
        mean = df1_crystal_errors[col].mean()
        print(f"  {col}:")
        print(f"    MAE: {mae:.4f} kJ/mol")
        print(f"    RMSE: {rmse:.4f} kJ/mol")
        print(f"    Mean: {mean:.4f} kJ/mol")

    print("\ndf2 (vs CCSD(T)/CBS):")
    for col in ["AP2 error", "AP3 error"]:
        mae = df2_crystal_errors[col].abs().mean()
        rmse = np.sqrt((df2_crystal_errors[col] ** 2).mean())
        mean = df2_crystal_errors[col].mean()
        print(f"  {col}:")
        print(f"    MAE: {mae:.4f} kJ/mol")
        print(f"    RMSE: {rmse:.4f} kJ/mol")
        print(f"    Mean: {mean:.4f} kJ/mol")

    return df1_crystal_errors, df2_crystal_errors


def plot_all_systems():
    # Load dataframes
    df1 = pd.read_pickle("./crystals_ap2_ap3_results_mol_apprx.pkl")
    df2 = pd.read_pickle("./crystals_ap2_ap3_results_mol_bm.pkl")

    # Print columns to understand available data
    print("df1 columns:")
    pp(
        [
            col
            for col in df1.columns
            if any(x in col for x in ["AP2", "AP3", "CCSD", "Non-Additive", "sapt"])
        ]
    )
    print("\ndf2 columns:")
    pp(
        [
            col
            for col in df2.columns
            if any(x in col for x in ["AP2", "AP3", "CCSD", "Non-Additive"])
        ]
    )

    # Create error columns for df1 (approximate methods vs SAPT0)
    # df1 has multiple methods to compare against SAPT0
    df1["AP2 vs SAPT0 error"] = (
        df1["Non-Additive MB Energy (kJ/mol) sapt0-dz-aug"] - df1["AP2 TOTAL"]
    )
    df1["AP3 vs SAPT0 error"] = (
        df1["Non-Additive MB Energy (kJ/mol) sapt0-dz-aug"] - df1["AP3 TOTAL"]
    )

    # Add other method errors if available
    method_cols = [
        col
        for col in df1.columns
        if "Non-Additive MB Energy (kJ/mol)" in col and "sapt0-dz-aug" not in col
    ]
    for method_col in method_cols[:3]:  # Limit to first 3 additional methods
        method_name = method_col.replace("Non-Additive MB Energy (kJ/mol) ", "")
        df1[f"{method_name} error"] = (
            df1[method_col] - df1["Non-Additive MB Energy (kJ/mol) sapt0-dz-aug"]
        )

    # Create error columns for df2 (comparing against CCSD(T)/CBS)
    df2["AP2 vs CCSD(T)/CBS error"] = (
        df2["Non-Additive MB Energy (kJ/mol) CCSD(T)/CBS"] - df2["AP2 TOTAL"]
    )
    df2["AP3 vs CCSD(T)/CBS error"] = (
        df2["Non-Additive MB Energy (kJ/mol) CCSD(T)/CBS"] - df2["AP3 TOTAL"]
    )

    # Prepare df1 for violin plot
    dfs1 = [
        {
            "df": df1,
            "basis": "",
            "label": "SAPT0/aDZ Reference",
            "ylim": [[-4, 4]],
        }
    ]

    # Labels and columns for df1 (multiple methods)
    df1_labels = {
        "AP2": "AP2 vs SAPT0 error",
        "AP3": "AP3 vs SAPT0 error",
    }

    # Add other methods if available
    for method_col in method_cols[:3]:
        method_name = method_col.replace("Non-Additive MB Energy (kJ/mol) ", "")
        if f"{method_name} error" in df1.columns:
            df1_labels[method_name] = f"{method_name} error"

    # Prepare df2 for violin plot
    dfs2 = [
        {
            "df": df2,
            "basis": "",
            "label": "CCSD(T)/CBS Reference",
            "ylim": [[-4, 4]],
        }
    ]

    # Labels and columns for df2 (only AP2 and AP3 vs CCSD(T)/CBS)
    df2_labels = {
        "AP2": "AP2 vs CCSD(T)/CBS error",
        "AP3": "AP3 vs CCSD(T)/CBS error",
    }

    # Create violin plot for df1
    print("\nCreating violin plot for df1 (approximate methods)...")
    error_statistics.violin_plot_table_multi_SAPT_components(
        dfs1,
        df_labels_and_columns_total=df1_labels,
        output_filename="./x23_plots/ap2_ap3_errors_vs_sapt0.png",
        grid_heights=[0.3, 1.0],
        grid_widths=[1],
        legend_loc="upper left",
        annotations_texty=0.3,
        figure_size=(6, 2.5),
        add_title=False,
    )
    print("Saved plot to ./x23_plots/ap2_ap3_errors_vs_sapt0.png")

    # Create violin plot for df2
    print("\nCreating violin plot for df2 (benchmark comparison)...")
    error_statistics.violin_plot_table_multi_SAPT_components(
        dfs2,
        df_labels_and_columns_total=df2_labels,
        output_filename="./x23_plots/ap2_ap3_errors_vs_ccsdt_cbs.png",
        grid_heights=[0.3, 1.0],
        grid_widths=[1],
        legend_loc="upper left",
        annotations_texty=0.3,
        figure_size=(4, 2.5),
        add_title=False,
    )
    print("Saved plot to ./x23_plots/ap2_ap3_errors_vs_ccsdt_cbs.png")

    # Print summary statistics
    print("\n=== df1 Summary Statistics ===")
    for label, col in df1_labels.items():
        if col in df1.columns:
            print(f"\n{label}:")
            print(f"  MAE: {df1[col].abs().mean():.4f} kJ/mol")
            print(f"  RMSE: {np.sqrt((df1[col] ** 2).mean()):.4f} kJ/mol")
            print(f"  Mean: {df1[col].mean():.4f} kJ/mol")

    print("\n=== df2 Summary Statistics ===")
    for label, col in df2_labels.items():
        if col in df2.columns:
            print(f"\n{label}:")
            print(f"  MAE: {df2[col].abs().mean():.4f} kJ/mol")
            print(f"  RMSE: {np.sqrt((df2[col] ** 2).mean()):.4f} kJ/mol")
            print(f"  Mean: {df2[col].mean():.4f} kJ/mol")


def plot_switchover_errors():
    """
    For each crystal, plot summed CLE energy errors from all points above X.
    Creates subplots showing AP2/AP3 switchover to reference methods.
    """
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    from matplotlib.ticker import AutoMinorLocator

    # Set up plot styling
    mpl.rcParams["figure.figsize"] = (10, 6)
    mpl.rcParams["font.size"] = 10
    mpl.rcParams["axes.labelsize"] = 12
    mpl.rcParams["xtick.labelsize"] = 10
    mpl.rcParams["ytick.labelsize"] = 10
    mpl.rcParams["legend.fontsize"] = 10
    mpl.rcParams["lines.linewidth"] = 2.0
    mpl.rcParams["lines.markersize"] = 6

    # Load dataframes
    df_apprx = pd.read_pickle("./crystals_ap2_ap3_results_mol_apprx.pkl")
    df_bm = pd.read_pickle("./crystals_ap2_ap3_results_mol_bm.pkl")

    # Get unique crystals
    crystals_apprx = sorted(df_apprx["crystal apprx"].dropna().unique())
    crystals_bm = sorted(df_bm["crystal bm"].dropna().unique())

    # Combine and get all unique crystals
    all_crystals = sorted(list(set(crystals_apprx) | set(crystals_bm)))
    N = len(all_crystals)

    print(f"Processing {N} crystals for switchover error plots")

    # Create figure with subplots (2 columns: apprx and bm)
    fig, axes = plt.subplots(N, 2, figsize=(12, N * 2 + 2))
    if N == 1:
        axes = axes.reshape(1, -1)

    # Define separation distance range
    increment = 0.25
    sep_distances = np.arange(1.0, 15.0 + 0.05, increment)

    # Process each crystal
    for idx, crystal in enumerate(all_crystals):
        print(f"\nProcessing crystal {idx + 1}/{N}: {crystal}")

        # Left plot: apprx (vs SAPT0/aDZ)
        ax_apprx = axes[idx, 0]
        if crystal in crystals_apprx:
            df_c = df_apprx[df_apprx["crystal apprx"] == crystal].copy()

            # Reference method
            ref_col = "Non-Additive MB Energy (kJ/mol) sapt0-dz-aug"
            num_rep_col = "Num. Rep. (#) sapt0-dz-aug"
            nmer_col = "N-mer Name apprx"
            mms_col = "Minimum Monomer Separations (A) sapt0-dz-aug"

            if ref_col in df_c.columns and "AP2 TOTAL" in df_c.columns:
                # Calculate CLE contributions
                df_c["ref_cle"] = df_c.apply(
                    lambda r: r[ref_col] * r[num_rep_col] / int(r[nmer_col][0])
                    if pd.notnull(r[nmer_col])
                    else 0,
                    axis=1,
                )
                df_c["ap2_cle"] = df_c.apply(
                    lambda r: r["AP2 TOTAL"] * r[num_rep_col] / int(r[nmer_col][0])
                    if pd.notnull(r[nmer_col])
                    else 0,
                    axis=1,
                )
                df_c["ap3_cle"] = df_c.apply(
                    lambda r: r["AP3 TOTAL"] * r[num_rep_col] / int(r[nmer_col][0])
                    if pd.notnull(r[nmer_col])
                    else 0,
                    axis=1,
                )

                # Calculate switchover errors for different cutoffs
                ap2_errors = []
                ap3_errors = []

                for d in sep_distances:
                    # Hybrid: use ML method above d, reference method above d
                    ap2_above = df_c[df_c[mms_col] >= d]["ap2_cle"].sum()
                    ap2_below = df_c[df_c[mms_col] < d]["ref_cle"].sum()
                    ap2_hybrid_total = ap2_above + ap2_below

                    ap3_above = df_c[df_c[mms_col] >= d]["ap3_cle"].sum()
                    ap3_below = df_c[df_c[mms_col] < d]["ref_cle"].sum()
                    ap3_hybrid_total = ap3_above + ap3_below

                    # Reference total (all ref)
                    ref_total = df_c["ref_cle"].sum()

                    # Error = hybrid - reference
                    ap2_errors.append(ap2_hybrid_total - ref_total)
                    ap3_errors.append(ap3_hybrid_total - ref_total)

                print("APPRX")
                print(df_c[['ap2_cle', 'ap3_cle', 'ref_cle']])
                print(df_c[['AP2 TOTAL', 'AP3 TOTAL', ref_col]])
                # Plot
                ax_apprx.plot(
                    sep_distances,
                    ap2_errors,
                    "o-",
                    label="AP2",
                    markersize=4,
                    linewidth=1.5,
                    alpha=0.8,
                )
                ax_apprx.plot(
                    sep_distances,
                    ap3_errors,
                    "s-",
                    label="AP3",
                    markersize=4,
                    linewidth=1.5,
                    alpha=0.8,
                )
                ax_apprx.axhline(
                    y=0, color="black", linestyle="--", linewidth=1.0, alpha=0.5
                )
                ax_apprx.set_ylabel("CLE Error (kJ/mol)", fontsize=10)
                ax_apprx.set_title(f"{crystal}\nvs SAPT0/aDZ", fontsize=10)

                if idx == 0:
                    ax_apprx.legend(loc="best", fontsize=8)

                ax_apprx.set_ylim(-5, 5)

        # Right plot: bm (vs CCSD(T)/CBS)
        ax_bm = axes[idx, 1]
        if crystal in crystals_bm:
            df_c = df_bm[df_bm["crystal bm"] == crystal].copy()

            # Reference method
            ref_col = "Non-Additive MB Energy (kJ/mol) CCSD(T)/CBS"
            num_rep_col = "Num. Rep. (#) CCSD(T)/CBS"
            nmer_col = "N-mer Name bm"
            mms_col = "Minimum Monomer Separations (A) CCSD(T)/CBS"
            df_c.sort_values(by=mms_col, inplace=True)

            if ref_col in df_c.columns and "AP2 TOTAL" in df_c.columns:
                # Calculate CLE contributions
                df_c["ref_cle"] = df_c.apply(
                    lambda r: r[ref_col] * r[num_rep_col] / int(r[nmer_col][0])
                    if pd.notnull(r[nmer_col])
                    else 0,
                    axis=1,
                )
                df_c["ap2_cle"] = df_c.apply(
                    lambda r: r["AP2 TOTAL"] * r[num_rep_col] / int(r[nmer_col][0])
                    if pd.notnull(r[nmer_col])
                    else 0,
                    axis=1,
                )
                df_c["ap3_cle"] = df_c.apply(
                    lambda r: r["AP3 TOTAL"] * r[num_rep_col] / int(r[nmer_col][0])
                    if pd.notnull(r[nmer_col])
                    else 0,
                    axis=1,
                )
                print("BM")
                print(df_c[['ap2_cle', 'ref_cle']])

                # Calculate switchover errors for different cutoffs
                ap2_errors = []
                ap3_errors = []

                for d in sep_distances:
                    # Hybrid: use ML method above d, reference method above d
                    ap2_above = df_c[df_c[mms_col] >= d]["ap2_cle"].sum()
                    ap2_below = df_c[df_c[mms_col] < d]["ref_cle"].sum()
                    ap2_hybrid_total = ap2_above + ap2_below

                    ap3_above = df_c[df_c[mms_col] >= d]["ap3_cle"].sum()
                    ap3_below = df_c[df_c[mms_col] < d]["ref_cle"].sum()
                    ap3_hybrid_total = ap3_above + ap3_below

                    # Reference total
                    ref_total = df_c["ref_cle"].sum()

                    # Error = hybrid - reference
                    ap2_errors.append(ap2_hybrid_total - ref_total)
                    ap3_errors.append(ap3_hybrid_total - ref_total)

                # Plot
                ax_bm.plot(
                    sep_distances,
                    ap2_errors,
                    "o-",
                    label="AP2",
                    markersize=4,
                    linewidth=1.5,
                    alpha=0.8,
                )
                ax_bm.plot(
                    sep_distances,
                    ap3_errors,
                    "s-",
                    label="AP3",
                    markersize=4,
                    linewidth=1.5,
                    alpha=0.8,
                )
                ax_bm.axhline(
                    y=0, color="black", linestyle="--", linewidth=1.0, alpha=0.5
                )
                ax_bm.set_ylabel("CLE Error (kJ/mol)", fontsize=10)
                ax_bm.set_title(f"{crystal}\nvs CCSD(T)/CBS", fontsize=10)

                if idx == 0:
                    ax_bm.legend(loc="best", fontsize=8)

                ax_bm.set_ylim(-5, 5)

        # Style axes
        for ax in [ax_apprx, ax_bm]:
            ax.grid(True, alpha=0.3, linestyle=":")
            ax.tick_params(which="major", direction="in", top=True, right=True)

            # Only show x-label on bottom row
            if idx == N - 1:
                ax.set_xlabel("Switchover Distance $R^*$ (Å)", fontsize=11)
            else:
                ax.tick_params(axis="x", labelbottom=False)

    # Adjust layout
    plt.tight_layout()

    # Save figure
    output_path = "./x23_plots/switchover_errors_all_crystals.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\nFigure saved to: {output_path}")

    return fig, axes


def plot_crystal_lattice_energies():
    """
    For each crystal, plot rolling sum of CLE energy errors from all points
    above X. Creates subplots showing AP2/AP3
    """
    # Set up plot styling
    mpl.rcParams["figure.figsize"] = (10, 6)
    mpl.rcParams["font.size"] = 10
    mpl.rcParams["axes.labelsize"] = 12
    mpl.rcParams["xtick.labelsize"] = 10
    mpl.rcParams["ytick.labelsize"] = 10
    mpl.rcParams["legend.fontsize"] = 10
    mpl.rcParams["lines.linewidth"] = 2.0
    mpl.rcParams["lines.markersize"] = 6

    # Load dataframes
    df_apprx = pd.read_pickle("./crystals_ap2_ap3_results_mol_apprx.pkl")
    df_bm = pd.read_pickle("./crystals_ap2_ap3_results_mol_bm.pkl")

    # Get unique crystals
    crystals_apprx = sorted(df_apprx["crystal apprx"].dropna().unique())
    crystals_bm = sorted(df_bm["crystal bm"].dropna().unique())

    # Combine and get all unique crystals
    all_crystals = sorted(list(set(crystals_apprx) | set(crystals_bm)))
    N = len(all_crystals)

    print(f"Processing {N} crystals for switchover error plots")

    # Create figure with subplots (2 columns: apprx and bm)
    fig, axes = plt.subplots(N, 2, figsize=(12, N * 2 + 2))
    if N == 1:
        axes = axes.reshape(1, -1)

    # Define separation distance range
    increment = 0.25
    sep_distances = np.arange(1.0, 15.0 + 0.05, increment)

    # Process each crystal
    for idx, crystal in enumerate(all_crystals):
        print(f"\nProcessing crystal {idx + 1}/{N}: {crystal}")

        # Left plot: apprx (vs SAPT0/aDZ)
        ax_apprx = axes[idx, 0]
        if crystal in crystals_apprx:
            df_c = df_apprx[df_apprx["crystal apprx"] == crystal].copy()

            # Reference method
            ref_col = "Non-Additive MB Energy (kJ/mol) sapt0-dz-aug"
            num_rep_col = "Num. Rep. (#) sapt0-dz-aug"
            nmer_col = "N-mer Name apprx"
            mms_col = "Minimum Monomer Separations (A) sapt0-dz-aug"

            if ref_col in df_c.columns and "AP2 TOTAL" in df_c.columns:
                # Calculate CLE contributions
                df_c["ref_cle"] = df_c.apply(
                    lambda r: r[ref_col] * r[num_rep_col] / int(r[nmer_col][0])
                    if pd.notnull(r[nmer_col])
                    else 0,
                    axis=1,
                )
                df_c["ap2_cle"] = df_c.apply(
                    lambda r: r["AP2 TOTAL"] * r[num_rep_col] / int(r[nmer_col][0])
                    if pd.notnull(r[nmer_col])
                    else 0,
                    axis=1,
                )
                df_c["ap3_cle"] = df_c.apply(
                    lambda r: r["AP3 TOTAL"] * r[num_rep_col] / int(r[nmer_col][0])
                    if pd.notnull(r[nmer_col])
                    else 0,
                    axis=1,
                )

                ap2_2b_energies = []
                ap3_2b_energies = []
                ref_2b_energies = []
                for d in sep_distances:
                    ap2_2b_energies.append(df_c[df_c[mms_col] <= d]["ap2_cle"].sum())
                    ap3_2b_energies.append(df_c[df_c[mms_col] <= d]["ap3_cle"].sum())
                    ref_2b_energies.append(df_c[df_c[mms_col] <= d]['ref_cle'].sum())

                # Plot
                ax_apprx.plot(
                    sep_distances,
                    ap2_2b_energies,
                    "o-",
                    label="AP2",
                    markersize=4,
                    linewidth=1.5,
                    alpha=0.8,
                )
                ax_apprx.plot(
                    sep_distances,
                    ap3_2b_energies,
                    "s-",
                    label="AP3",
                    markersize=4,
                    linewidth=1.5,
                    alpha=0.8,
                )
                ax_apprx.plot(
                    sep_distances,
                    ref_2b_energies,
                    "-",
                    color='red',
                    label="SAPT0/aDZ",
                    markersize=4,
                    linewidth=1.5,
                    alpha=0.8,
                )
                ax_apprx.axhline(
                    y=0, color="black", linestyle="--", linewidth=1.0, alpha=0.5
                )
                ax_apprx.set_ylabel("CLE Error (kJ/mol)", fontsize=10)
                ax_apprx.set_title(f"{crystal}\nvs SAPT0/aDZ", fontsize=10)

                if idx == 0:
                    ax_apprx.legend(loc="best", fontsize=8)

                # ax_apprx.set_ylim(-5, 5)

        # Right plot: bm (vs CCSD(T)/CBS)
        ax_bm = axes[idx, 1]
        if crystal in crystals_bm:
            df_c = df_bm[df_bm["crystal bm"] == crystal].copy()

            # Reference method
            ref_col = "Non-Additive MB Energy (kJ/mol) CCSD(T)/CBS"
            num_rep_col = "Num. Rep. (#) CCSD(T)/CBS"
            nmer_col = "N-mer Name bm"
            mms_col = "Minimum Monomer Separations (A) CCSD(T)/CBS"
            df_c.sort_values(by=mms_col, inplace=True)

            if ref_col in df_c.columns and "AP2 TOTAL" in df_c.columns:
                # Calculate CLE contributions
                df_c["ref_cle"] = df_c.apply(
                    lambda r: r[ref_col] * r[num_rep_col] / int(r[nmer_col][0])
                    if pd.notnull(r[nmer_col])
                    else 0,
                    axis=1,
                )
                df_c["ap2_cle"] = df_c.apply(
                    lambda r: r["AP2 TOTAL"] * r[num_rep_col] / int(r[nmer_col][0])
                    if pd.notnull(r[nmer_col])
                    else 0,
                    axis=1,
                )
                df_c["ap3_cle"] = df_c.apply(
                    lambda r: r["AP3 TOTAL"] * r[num_rep_col] / int(r[nmer_col][0])
                    if pd.notnull(r[nmer_col])
                    else 0,
                    axis=1,
                )

                ap2_2b_energies = []
                ap3_2b_energies = []
                ref_2b_energies = []
                for d in sep_distances:
                    ap2_2b_energies.append(df_c[df_c[mms_col] <= d]["ap2_cle"].sum())
                    ap3_2b_energies.append(df_c[df_c[mms_col] <= d]["ap3_cle"].sum())
                    ref_2b_energies.append(df_c[df_c[mms_col] <= d]['ref_cle'].sum())

                # Plot
                ax_bm.plot(
                    sep_distances,
                    ap2_2b_energies,
                    "o-",
                    label="AP2",
                    markersize=4,
                    linewidth=1.5,
                    alpha=0.8,
                )
                ax_bm.plot(
                    sep_distances,
                    ap3_2b_energies,
                    "s-",
                    label="AP3",
                    markersize=4,
                    linewidth=1.5,
                    alpha=0.8,
                )
                ax_bm.plot(
                    sep_distances,
                    ref_2b_energies,
                    "k-",
                    label="CCSD(T)/CBS",
                    markersize=4,
                    linewidth=1.5,
                    alpha=0.8,
                )
                ax_bm.axhline(
                    y=0, color="black", linestyle="--", linewidth=1.0, alpha=0.5
                )
                ax_bm.set_ylabel("CLE Error (kJ/mol)", fontsize=10)
                ax_bm.set_title(f"{crystal}\nvs CCSD(T)/CBS", fontsize=10)

                if idx == 0:
                    ax_bm.legend(loc="best", fontsize=8)

                # ax_bm.set_ylim(-5, 5)

        # Style axes
        for ax in [ax_apprx, ax_bm]:
            ax.grid(True, alpha=0.3, linestyle=":")
            ax.tick_params(which="major", direction="in", top=True, right=True)

            # Only show x-label on bottom row
            if idx == N - 1:
                ax.set_xlabel("Min. Mon. Sep. (Å)", fontsize=11)
            else:
                ax.tick_params(axis="x", labelbottom=False)

    # Adjust layout
    plt.tight_layout()

    # Save figure
    output_path = "./x23_plots/CLE_all_crystals.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\nFigure saved to: {output_path}")
    return fig, axes


def plot_crystal_lattice_energies_with_switchover(switchover=2.5):
    """
    For each crystal, plot rolling sum of CLE energy errors from all points
    above X. Creates subplots showing AP2/AP3
    """
    # Set up plot styling
    mpl.rcParams["figure.figsize"] = (10, 6)
    mpl.rcParams["font.size"] = 10
    mpl.rcParams["axes.labelsize"] = 12
    mpl.rcParams["xtick.labelsize"] = 10
    mpl.rcParams["ytick.labelsize"] = 10
    mpl.rcParams["legend.fontsize"] = 10
    mpl.rcParams["lines.linewidth"] = 2.0
    mpl.rcParams["lines.markersize"] = 6

    # Load dataframes
    df_apprx = pd.read_pickle("./crystals_ap2_ap3_results_mol_apprx.pkl")
    df_bm = pd.read_pickle("./crystals_ap2_ap3_results_mol_bm.pkl")

    # Get unique crystals
    crystals_apprx = sorted(df_apprx["crystal apprx"].dropna().unique())
    crystals_bm = sorted(df_bm["crystal bm"].dropna().unique())

    # Combine and get all unique crystals
    all_crystals = sorted(list(set(crystals_apprx) | set(crystals_bm)))
    N = len(all_crystals)

    print(f"Processing {N} crystals for switchover error plots")

    # Create figure with subplots (2 columns: apprx and bm)
    fig, axes = plt.subplots(N, 2, figsize=(12, N * 2 + 2))
    if N == 1:
        axes = axes.reshape(1, -1)

    # Define separation distance range
    increment = 0.25
    sep_distances = np.arange(1.0, 15.0 + 0.05, increment)

    ap2_full_cle_errors_sapt0_aDZ = []
    ap3_full_cle_errors_sapt0_aDZ = []
    ap2_full_cle_errors_ccsd_t_CBS = []
    ap3_full_cle_errors_ccsd_t_CBS = []

    # Process each crystal
    for idx, crystal in enumerate(all_crystals):
        print(f"\nProcessing crystal {idx + 1}/{N}: {crystal}")

        # Left plot: apprx (vs SAPT0/aDZ)
        ax_apprx = axes[idx, 0]
        if crystal in crystals_apprx:
            df_c = df_apprx[df_apprx["crystal apprx"] == crystal].copy()

            # Reference method
            ref_col = "Non-Additive MB Energy (kJ/mol) sapt0-dz-aug"
            num_rep_col = "Num. Rep. (#) sapt0-dz-aug"
            nmer_col = "N-mer Name apprx"
            mms_col = "Minimum Monomer Separations (A) sapt0-dz-aug"

            if ref_col in df_c.columns and "AP2 TOTAL" in df_c.columns:
                # Calculate CLE contributions
                df_c["ref_cle"] = df_c.apply(
                    lambda r: r[ref_col] * r[num_rep_col] / int(r[nmer_col][0])
                    if pd.notnull(r[nmer_col])
                    else 0,
                    axis=1,
                )
                df_c["ap2_cle"] = df_c.apply(
                    lambda r: r["AP2 TOTAL"] * r[num_rep_col] / int(r[nmer_col][0])
                    if pd.notnull(r[nmer_col])
                    else 0,
                    axis=1,
                )
                df_c["ap3_cle"] = df_c.apply(
                    lambda r: r["AP3 TOTAL"] * r[num_rep_col] / int(r[nmer_col][0])
                    if pd.notnull(r[nmer_col])
                    else 0,
                    axis=1,
                )

                ap2_2b_energies = []
                ap3_2b_energies = []
                ref_2b_energies = []
                for d in sep_distances:
                    ap2_above = df_c[(df_c[mms_col] >= switchover) & (df_c[mms_col] < d)]["ap2_cle"].sum()
                    ap2_below = df_c[(df_c[mms_col] < switchover) & (df_c[mms_col] < d)]["ref_cle"].sum()
                    ap2_hybrid_total = ap2_above + ap2_below
                    ap3_above = df_c[(df_c[mms_col] >= switchover) & (df_c[mms_col] < d)]["ap3_cle"].sum()
                    ap3_below = df_c[(df_c[mms_col] < switchover) & (df_c[mms_col] < d)]["ref_cle"].sum()
                    ref_below = df_c[df_c[mms_col] < d]["ref_cle"].sum()
                    ap3_hybrid_total = ap3_above + ap3_below
                    ap2_2b_energies.append(ap2_hybrid_total)
                    ap3_2b_energies.append(ap3_hybrid_total)
                    ref_2b_energies.append(ref_below)

                # Plot
                ax_apprx.plot(
                    sep_distances,
                    ap2_2b_energies,
                    "o-",
                    label="AP2",
                    markersize=4,
                    linewidth=1.5,
                    alpha=0.8,
                )
                ax_apprx.plot(
                    sep_distances,
                    ap3_2b_energies,
                    "s-",
                    label="AP3",
                    markersize=4,
                    linewidth=1.5,
                    alpha=0.8,
                )
                ax_apprx.plot(
                    sep_distances,
                    ref_2b_energies,
                    "-",
                    color='red',
                    label="SAPT0/aDZ",
                    markersize=4,
                    linewidth=1.5,
                    alpha=0.8,
                )
                ax_apprx.axhline(
                    y=0, color="black", linestyle="--", linewidth=1.0, alpha=0.5
                )
                ax_apprx.set_ylabel("CLE Error (kJ/mol)", fontsize=10)
                ax_apprx.set_title(f"{crystal}\nvs SAPT0/aDZ", fontsize=10)

                if idx == 0:
                    ax_apprx.legend(loc="best", fontsize=8)

                ap2_full_cle_errors_sapt0_aDZ.append(ap2_2b_energies[-1] - ref_2b_energies[-1])
                ap3_full_cle_errors_sapt0_aDZ.append(ap3_2b_energies[-1] - ref_2b_energies[-1])
                # ax_apprx.set_ylim(-5, 5)

        # Right plot: bm (vs CCSD(T)/CBS)
        ax_bm = axes[idx, 1]
        if crystal in crystals_bm:
            df_c = df_bm[df_bm["crystal bm"] == crystal].copy()

            # Reference method
            ref_col = "Non-Additive MB Energy (kJ/mol) CCSD(T)/CBS"
            num_rep_col = "Num. Rep. (#) CCSD(T)/CBS"
            nmer_col = "N-mer Name bm"
            mms_col = "Minimum Monomer Separations (A) CCSD(T)/CBS"
            df_c.sort_values(by=mms_col, inplace=True)

            if ref_col in df_c.columns and "AP2 TOTAL" in df_c.columns:
                # Calculate CLE contributions
                df_c["ref_cle"] = df_c.apply(
                    lambda r: r[ref_col] * r[num_rep_col] / int(r[nmer_col][0])
                    if pd.notnull(r[nmer_col])
                    else 0,
                    axis=1,
                )
                df_c["ap2_cle"] = df_c.apply(
                    lambda r: r["AP2 TOTAL"] * r[num_rep_col] / int(r[nmer_col][0])
                    if pd.notnull(r[nmer_col])
                    else 0,
                    axis=1,
                )
                df_c["ap3_cle"] = df_c.apply(
                    lambda r: r["AP3 TOTAL"] * r[num_rep_col] / int(r[nmer_col][0])
                    if pd.notnull(r[nmer_col])
                    else 0,
                    axis=1,
                )

                ap2_2b_energies = []
                ap3_2b_energies = []
                ref_2b_energies = []
                print(f"{crystal = }, cnt below SO: {len(df_c[(df_c[mms_col] < switchover)])}/{len(df_c)}")
                for d in sep_distances:
                    ap2_above = df_c[(df_c[mms_col] >= switchover) & (df_c[mms_col] < d)]["ap2_cle"].sum()
                    ap2_below = df_c[(df_c[mms_col] < switchover) & (df_c[mms_col] < d)]["ref_cle"].sum()
                    ap2_hybrid_total = ap2_above + ap2_below
                    ap3_above = df_c[(df_c[mms_col] >= switchover) & (df_c[mms_col] < d)]["ap3_cle"].sum()
                    ap3_below = df_c[(df_c[mms_col] < switchover) & (df_c[mms_col] < d)]["ref_cle"].sum()
                    ap3_hybrid_total = ap3_above + ap3_below
                    ref_below = df_c[df_c[mms_col] < d]["ref_cle"].sum()
                    ap2_2b_energies.append(ap2_hybrid_total)
                    ap3_2b_energies.append(ap3_hybrid_total)
                    ref_2b_energies.append(ref_below)

                # Plot
                ax_bm.plot(
                    sep_distances,
                    ap2_2b_energies,
                    "o-",
                    label="AP2",
                    markersize=4,
                    linewidth=1.5,
                    alpha=0.8,
                )
                ax_bm.plot(
                    sep_distances,
                    ap3_2b_energies,
                    "s-",
                    label="AP3",
                    markersize=4,
                    linewidth=1.5,
                    alpha=0.8,
                )
                ax_bm.plot(
                    sep_distances,
                    ref_2b_energies,
                    "k-",
                    label="CCSD(T)/CBS",
                    markersize=4,
                    linewidth=1.5,
                    alpha=0.8,
                )
                ax_bm.axhline(
                    y=0, color="black", linestyle="--", linewidth=1.0, alpha=0.5
                )
                ax_bm.set_ylabel("CLE Error (kJ/mol)", fontsize=10)
                ax_bm.set_title(f"{crystal}\nvs CCSD(T)/CBS", fontsize=10)

                if idx == 0:
                    ax_bm.legend(loc="best", fontsize=8)

                ap2_full_cle_errors_ccsd_t_CBS.append(ap2_2b_energies[-1] - ref_2b_energies[-1])
                ap3_full_cle_errors_ccsd_t_CBS.append(ap3_2b_energies[-1] - ref_2b_energies[-1])
                # ax_bm.set_ylim(-5, 5)

        # Style axes
        for ax in [ax_apprx, ax_bm]:
            ax.grid(True, alpha=0.3, linestyle=":")
            ax.tick_params(which="major", direction="in", top=True, right=True)

            # Only show x-label on bottom row
            if idx == N - 1:
                ax.set_xlabel("Min. Mon. Sep. (Å)", fontsize=11)
            else:
                ax.tick_params(axis="x", labelbottom=False)

    # Adjust layout
    plt.tight_layout()
    print("Error CLE statistics")
    mae_ap2_sapt = np.sum(np.array(ap2_full_cle_errors_sapt0_aDZ)) / len(ap2_full_cle_errors_sapt0_aDZ)
    mae_ap3_sapt = np.sum(np.array(ap3_full_cle_errors_sapt0_aDZ)) / len(ap3_full_cle_errors_sapt0_aDZ)
    mae_ap2_ccsd = np.sum(np.array(ap2_full_cle_errors_ccsd_t_CBS)) / len(ap2_full_cle_errors_ccsd_t_CBS)
    mae_ap3_ccsd = np.sum(np.array(ap3_full_cle_errors_ccsd_t_CBS)) / len(ap3_full_cle_errors_ccsd_t_CBS)
    print(f"{mae_ap2_sapt=:.4f} kJ/mol")
    print(f"{mae_ap3_sapt=:.4f} kJ/mol")
    print(f"{mae_ap2_ccsd=:.4f} kJ/mol")
    print(f"{mae_ap3_ccsd=:.4f} kJ/mol")

    # Save figure
    output_path = "./x23_plots/CLE_all_crystals_switchover.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\nFigure saved to: {output_path}")
    return fig, axes


def plot_crystal_lattice_energies_with_N(N=1):
    """
    For each crystal, plot rolling sum of CLE energy errors from all points
    above X. Creates subplots showing AP2/AP3
    """
    # Set up plot styling
    mpl.rcParams["figure.figsize"] = (10, 6)
    mpl.rcParams["font.size"] = 10
    mpl.rcParams["axes.labelsize"] = 12
    mpl.rcParams["xtick.labelsize"] = 10
    mpl.rcParams["ytick.labelsize"] = 10
    mpl.rcParams["legend.fontsize"] = 10
    mpl.rcParams["lines.linewidth"] = 2.0
    mpl.rcParams["lines.markersize"] = 6

    # Load dataframes
    df_apprx = pd.read_pickle("./crystals_ap2_ap3_results_mol_apprx.pkl")
    df_bm = pd.read_pickle("./crystals_ap2_ap3_results_mol_bm.pkl")

    # Get unique crystals
    crystals_apprx = sorted(df_apprx["crystal apprx"].dropna().unique())
    crystals_bm = sorted(df_bm["crystal bm"].dropna().unique())

    # Combine and get all unique crystals
    all_crystals = sorted(list(set(crystals_apprx) | set(crystals_bm)))
    N = len(all_crystals)

    print(f"Processing {N} crystals for switchover error plots")

    # Create figure with subplots (2 columns: apprx and bm)
    fig, axes = plt.subplots(N, 2, figsize=(12, N * 2 + 2))
    if N == 1:
        axes = axes.reshape(1, -1)

    # Define separation distance range
    increment = 0.25
    sep_distances = np.arange(1.0, 15.0 + 0.05, increment)

    ap2_full_cle_errors_sapt0_aDZ = []
    ap3_full_cle_errors_sapt0_aDZ = []
    ap2_full_cle_errors_ccsd_t_CBS = []
    ap3_full_cle_errors_ccsd_t_CBS = []

    # Process each crystal
    for idx, crystal in enumerate(all_crystals):
        print(f"\nProcessing crystal {idx + 1}/{N}: {crystal}")

        # Left plot: apprx (vs SAPT0/aDZ)
        ax_apprx = axes[idx, 0]
        if crystal in crystals_apprx:
            df_c = df_apprx[df_apprx["crystal apprx"] == crystal].copy()

            # Reference method
            ref_col = "Non-Additive MB Energy (kJ/mol) sapt0-dz-aug"
            num_rep_col = "Num. Rep. (#) sapt0-dz-aug"
            nmer_col = "N-mer Name apprx"
            mms_col = "Minimum Monomer Separations (A) sapt0-dz-aug"

            if ref_col in df_c.columns and "AP2 TOTAL" in df_c.columns:
                # Calculate CLE contributions
                df_c["ref_cle"] = df_c.apply(
                    lambda r: r[ref_col] * r[num_rep_col] / int(r[nmer_col][0])
                    if pd.notnull(r[nmer_col])
                    else 0,
                    axis=1,
                )
                df_c["ap2_cle"] = df_c.apply(
                    lambda r: r["AP2 TOTAL"] * r[num_rep_col] / int(r[nmer_col][0])
                    if pd.notnull(r[nmer_col])
                    else 0,
                    axis=1,
                )
                df_c["ap3_cle"] = df_c.apply(
                    lambda r: r["AP3 TOTAL"] * r[num_rep_col] / int(r[nmer_col][0])
                    if pd.notnull(r[nmer_col])
                    else 0,
                    axis=1,
                )

                ap2_2b_energies = []
                ap3_2b_energies = []
                ref_2b_energies = []
                df_c = df_c.sort_values(by=mms_col, ascending=True)
                df_c_N = df_c.iloc[:N]
                df_c_above = df_c.iloc[N:]
                for d in sep_distances:
                    ap2_above = df_c_above[df_c_above[mms_col] > d]["ap2_cle"].sum()
                    ap2_below = df_c_N[df_c_N[mms_col] < d]["ref_cle"].sum()
                    ap2_hybrid_total = ap2_above + ap2_below

                    ap3_above = df_c_above[df_c_above[mms_col] > d]["ap3_cle"].sum()
                    ap3_below = df_c_N[df_c_N[mms_col] < d]["ref_cle"].sum()
                    ap3_hybrid_total = ap3_above + ap3_below

                    ref_below = df_c[df_c[mms_col] < d]["ref_cle"].sum()
                    ap3_hybrid_total = ap3_above + ap3_below

                    ap2_2b_energies.append(ap2_hybrid_total)
                    ap3_2b_energies.append(ap3_hybrid_total)
                    ref_2b_energies.append(ref_below)

                # Plot
                if ref_2b_energies[-1] != 0.0:
                    ax_apprx.plot(
                        sep_distances,
                        ap2_2b_energies,
                        "o-",
                        label="AP2",
                        markersize=4,
                        linewidth=1.5,
                        alpha=0.8,
                    )
                    ax_apprx.plot(
                        sep_distances,
                        ap3_2b_energies,
                        "s-",
                        label="AP3",
                        markersize=4,
                        linewidth=1.5,
                        alpha=0.8,
                    )
                    ax_apprx.plot(
                        sep_distances,
                        ref_2b_energies,
                        "-",
                        color='red',
                        label="SAPT0/aDZ",
                        markersize=4,
                        linewidth=1.5,
                        alpha=0.8,
                    )
                ax_apprx.axhline(
                    y=0, color="black", linestyle="--", linewidth=1.0, alpha=0.5
                )
                ax_apprx.set_ylabel("CLE Error (kJ/mol)", fontsize=10)
                ax_apprx.set_title(f"{crystal}\nvs SAPT0/aDZ", fontsize=10)

                if idx == 0:
                    ax_apprx.legend(loc="best", fontsize=8)

                ap2_full_cle_errors_sapt0_aDZ.append(ap2_2b_energies[-1] - ref_2b_energies[-1])
                ap3_full_cle_errors_sapt0_aDZ.append(ap3_2b_energies[-1] - ref_2b_energies[-1])
                # ax_apprx.set_ylim(-5, 5)

        # Right plot: bm (vs CCSD(T)/CBS)
        ax_bm = axes[idx, 1]
        if crystal in crystals_bm:
            df_c = df_bm[df_bm["crystal bm"] == crystal].copy()

            # Reference method
            ref_col = "Non-Additive MB Energy (kJ/mol) CCSD(T)/CBS"
            num_rep_col = "Num. Rep. (#) CCSD(T)/CBS"
            nmer_col = "N-mer Name bm"
            mms_col = "Minimum Monomer Separations (A) CCSD(T)/CBS"
            df_c.sort_values(by=mms_col, inplace=True)

            if ref_col in df_c.columns and "AP2 TOTAL" in df_c.columns:
                # Calculate CLE contributions
                df_c["ref_cle"] = df_c.apply(
                    lambda r: r[ref_col] * r[num_rep_col] / int(r[nmer_col][0])
                    if pd.notnull(r[nmer_col])
                    else 0,
                    axis=1,
                )
                df_c["ap2_cle"] = df_c.apply(
                    lambda r: r["AP2 TOTAL"] * r[num_rep_col] / int(r[nmer_col][0])
                    if pd.notnull(r[nmer_col])
                    else 0,
                    axis=1,
                )
                df_c["ap3_cle"] = df_c.apply(
                    lambda r: r["AP3 TOTAL"] * r[num_rep_col] / int(r[nmer_col][0])
                    if pd.notnull(r[nmer_col])
                    else 0,
                    axis=1,
                )

                ap2_2b_energies = []
                ap3_2b_energies = []
                ref_2b_energies = []
                df_c = df_c.sort_values(by=mms_col, ascending=True)
                df_c_N = df_c.iloc[:N]
                df_c_above = df_c.iloc[N:]
                for d in sep_distances:
                    ap2_above = df_c_above[df_c_above[mms_col] > d]["ap2_cle"].sum()
                    ap2_below = df_c_N[df_c_N[mms_col] < d]["ref_cle"].sum()
                    ap2_hybrid_total = ap2_above + ap2_below

                    ap3_above = df_c_above[df_c_above[mms_col] > d]["ap3_cle"].sum()
                    ap3_below = df_c_N[df_c_N[mms_col] < d]["ref_cle"].sum()
                    ap3_hybrid_total = ap3_above + ap3_below

                    ref_below = df_c[df_c[mms_col] < d]["ref_cle"].sum()
                    ap3_hybrid_total = ap3_above + ap3_below
                    ap2_2b_energies.append(ap2_hybrid_total)
                    ap3_2b_energies.append(ap3_hybrid_total)
                    ref_2b_energies.append(ref_below)

                # Plot
                if ref_2b_energies[-1] != 0.0:
                    ax_bm.plot(
                        sep_distances,
                        ap2_2b_energies,
                        "o-",
                        label="AP2",
                        markersize=4,
                        linewidth=1.5,
                        alpha=0.8,
                    )
                    ax_bm.plot(
                        sep_distances,
                        ap3_2b_energies,
                        "s-",
                        label="AP3",
                        markersize=4,
                        linewidth=1.5,
                        alpha=0.8,
                    )
                    ax_bm.plot(
                        sep_distances,
                        ref_2b_energies,
                        "k-",
                        label="CCSD(T)/CBS",
                        markersize=4,
                        linewidth=1.5,
                        alpha=0.8,
                    )
                    ax_bm.axhline(
                        y=0, color="black", linestyle="--", linewidth=1.0, alpha=0.5
                    )
                ax_bm.set_ylabel("CLE Error (kJ/mol)", fontsize=10)
                ax_bm.set_title(f"{crystal}\nvs CCSD(T)/CBS", fontsize=10)

                if idx == 0:
                    ax_bm.legend(loc="best", fontsize=8)

                ap2_full_cle_errors_ccsd_t_CBS.append(ap2_2b_energies[-1] - ref_2b_energies[-1])
                ap3_full_cle_errors_ccsd_t_CBS.append(ap3_2b_energies[-1] - ref_2b_energies[-1])
                # ax_bm.set_ylim(-5, 5)

        # Style axes
        for ax in [ax_apprx, ax_bm]:
            ax.grid(True, alpha=0.3, linestyle=":")
            ax.tick_params(which="major", direction="in", top=True, right=True)

            # Only show x-label on bottom row
            if idx == N - 1:
                ax.set_xlabel("Min. Mon. Sep. (Å)", fontsize=11)
            else:
                ax.tick_params(axis="x", labelbottom=False)

    # Adjust layout
    plt.tight_layout()
    print("Error CLE statistics")
    # Filter out nans from ap2_full_cle_errors_ccsd_t_CBS
    ap2_full_cle_errors_ccsd_t_CBS = [x for x in ap2_full_cle_errors_ccsd_t_CBS if x != 0.0 ]
    ap3_full_cle_errors_ccsd_t_CBS = [x for x in ap3_full_cle_errors_ccsd_t_CBS if x != 0.0 ]
    ap2_full_cle_errors_sapt0_aDZ = [x for x in ap2_full_cle_errors_sapt0_aDZ if x != 0.0 ]
    ap3_full_cle_errors_sapt0_aDZ = [x for x in ap3_full_cle_errors_sapt0_aDZ if x != 0.0 ]

    mae_ap2_sapt = np.sum(np.array(ap2_full_cle_errors_sapt0_aDZ)) / len(ap2_full_cle_errors_sapt0_aDZ)
    mae_ap3_sapt = np.sum(np.array(ap3_full_cle_errors_sapt0_aDZ)) / len(ap3_full_cle_errors_sapt0_aDZ)
    mae_ap2_ccsd = np.sum(np.array(ap2_full_cle_errors_ccsd_t_CBS)) / len(ap2_full_cle_errors_ccsd_t_CBS)
    mae_ap3_ccsd = np.sum(np.array(ap3_full_cle_errors_ccsd_t_CBS)) / len(ap3_full_cle_errors_ccsd_t_CBS)
    print(f"{mae_ap2_sapt=:.4f} kJ/mol")
    print(f"{mae_ap3_sapt=:.4f} kJ/mol")
    print(f"{mae_ap2_ccsd=:.4f} kJ/mol")
    print(f"{mae_ap3_ccsd=:.4f} kJ/mol")

    # Save figure
    output_path = "./x23_plots/CLE_all_crystals_N.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\nFigure saved to: {output_path}")
    return fig, axes


def main():
    # plot_full_crystal_errors()
    # plot_switchover_errors()
    # plot_crystal_lattice_energies()
    # plot_crystal_lattice_energies_with_switchover(2.9)
    plot_crystal_lattice_energies_with_N(1)
    return


if __name__ == "__main__":
    main()
