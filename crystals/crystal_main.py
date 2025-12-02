import pandas as pd
from pprint import pprint as pp
import qcelemental as qcel
from glob import glob
from pathlib import Path
import subprocess
import numpy as np
from qm_tools_aw import tools
import apnet_pt


apnet2_model = apnet_pt.AtomPairwiseModels.apnet2.APNet2Model().set_pretrained_model(model_id=0)
apnet2_model.model.return_hidden_states = True
dapnet2 = apnet_pt.AtomPairwiseModels.dapnet2.dAPNet2Model(
    apnet2_model,
    pre_trained_model_path="/home/awallace43/gits/qcmlforge/models/dapnet2/B3LYP-D3_aug-cc-pVDZ_CP_to_CCSD_LP_T_RP_CBS_CP_0.pt",
)


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
            df_lt["mol"] = [np.array([[0., 0., 0.] for j in range(mol_len)]) for i in range(len(df_lt))]
            df_lt["RA"] =  [np.array([[0., 0., 0.] for j in range(mol_len)]) for i in range(len(df_lt))]
            df_lt["RB"] =  [np.array([[0., 0., 0.] for j in range(mol_len)]) for i in range(len(df_lt))]
            df_lt["ZA"] =  [np.array([[0., 0., 0.] for j in range(mol_len)]) for i in range(len(df_lt))]
            df_lt["ZB"] =  [np.array([[0., 0., 0.] for j in range(mol_len)]) for i in range(len(df_lt))]
            df_lt['RA'] = df_lt['RA'].astype(object)
            df_lt['RB'] = df_lt['RB'].astype(object)
            df_lt['ZA'] = df_lt['ZA'].astype(object)
            df_lt['ZB'] = df_lt['ZB'].astype(object)
            for i, (k, v) in enumerate(data.items()):
                dimer = v[0]
                df_lt.loc[df_lt["N-mer Name"] == k, "mol"] = v
                RA = np.array(dimer.geometry[dimer.fragments[0]], dtype=np.float32) * qcel.constants.bohr2angstroms
                RB = np.array(dimer.geometry[dimer.fragments[1]], dtype=np.float32) * qcel.constants.bohr2angstroms
                ZA = np.array(dimer.atomic_numbers[dimer.fragments[0]], dtype=np.int32)
                ZB = np.array(dimer.atomic_numbers[dimer.fragments[1]], dtype=np.int32)
                for idx in df_lt.loc[df_lt["N-mer Name"] == k].index:
                    df_lt.at[idx, "RA"] = RA
                    df_lt.at[idx, "RB"] = RB
                    df_lt.at[idx, "ZA"] = ZA
                    df_lt.at[idx, "ZB"] = ZB
        df_lt.to_pickle(output_csv.replace("csv", "pkl"))
        print(df_lt[['N-mer Name', 'RA', 'ZA']])
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
            df_lt["mol"] = [np.array([[0., 0., 0.] for j in range(mol_len)]) for i in range(len(df_lt))]
            df_lt["RA"] =  [np.array([[0., 0., 0.] for j in range(mol_len)]) for i in range(len(df_lt))]
            df_lt["RB"] =  [np.array([[0., 0., 0.] for j in range(mol_len)]) for i in range(len(df_lt))]
            df_lt["ZA"] =  [np.array([[0., 0., 0.] for j in range(mol_len)]) for i in range(len(df_lt))]
            df_lt["ZB"] =  [np.array([[0., 0., 0.] for j in range(mol_len)]) for i in range(len(df_lt))]
            df_lt['RA'] = df_lt['RA'].astype(object)
            df_lt['RB'] = df_lt['RB'].astype(object)
            df_lt['ZA'] = df_lt['ZA'].astype(object)
            df_lt['ZB'] = df_lt['ZB'].astype(object)
            df_lt['SAPT0 Electrostatics (kJ/mol)'] = [np.nan for i in range(len(df_lt))]
            df_lt['SAPT0 Exchange (kJ/mol)'] = [np.nan for i in range(len(df_lt))]
            df_lt['SAPT0 Induction (kJ/mol)'] = [np.nan for i in range(len(df_lt))]
            df_lt['SAPT0 Dispersion (kJ/mol)'] = [np.nan for i in range(len(df_lt))]
            for i, (k, v) in enumerate(data.items()):
                dimer = v[0]
                df_lt.loc[df_lt["N-mer Name"] == k, "mol"] = [v[0]]
                df_lt.loc[df_lt["N-mer Name"] == k, "SAPT0 Electrostatics (kJ/mol)"] = [v[1]]
                df_lt.loc[df_lt["N-mer Name"] == k, "SAPT0 Exchange (kJ/mol)"] = [v[2]]
                df_lt.loc[df_lt["N-mer Name"] == k, "SAPT0 Induction (kJ/mol)"] = [v[3]]
                df_lt.loc[df_lt["N-mer Name"] == k, "SAPT0 Dispersion (kJ/mol)"] = [v[4]]
                RA = np.array(dimer.geometry[dimer.fragments[0]], dtype=np.float32) * qcel.constants.bohr2angstroms
                RB = np.array(dimer.geometry[dimer.fragments[1]], dtype=np.float32) * qcel.constants.bohr2angstroms
                ZA = np.array(dimer.atomic_numbers[dimer.fragments[0]], dtype=np.int32)
                ZB = np.array(dimer.atomic_numbers[dimer.fragments[1]], dtype=np.int32)
                for idx in df_lt.loc[df_lt["N-mer Name"] == k].index:
                    df_lt.at[idx, "RA"] = RA
                    df_lt.at[idx, "RB"] = RB
                    df_lt.at[idx, "ZA"] = ZA
                    df_lt.at[idx, "ZB"] = ZB
        df_lt.to_pickle(output_csv.replace("csv", "pkl"))
        r1 = df_lt.iloc[0]
        tools.print_cartesians_pos_carts(r1['ZA'], r1['RA'])
        print(df_lt[['N-mer Name', 'SAPT0 Induction (kJ/mol)', 'SAPT0 Electrostatics (kJ/mol)']])
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
                    df[f'output {c["level_of_theory"]}'] = df["N-mer Name"].apply(
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
            df[f'output {c["level_of_theory"]}'] = df["N-mer Name"].apply(
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
                    energy = dapnet2.predict_qcel_mols([row['mol apprx']], batch_size=1)
                    df.loc[n, f"dAPNet2 apprx {basefile}"] = (
                        energy[0] * kcalmol_to_kjmol
                    )
                if pd.notnull(row["mol bm"]):
                    # energy = pm.predict([row["mol bm"]], batch_size=1)
                    energy = dapnet2.predict_qcel_mols([row['mol bm']], batch_size=1)
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
                        f"{n}/{len(df)}, crystal: {active_crystal }, dAP (apprx, bm): ({energies_sum_crystal_apprx:.2f}, {energies_sum_crystal_bm:.2f}), CLE d-target: {apprx_error:.2f}"
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
                            f"{n}/{len(df)}, crystal: {active_crystal }, dAP (apprx, bm): ({energies_sum_crystal_apprx:.2f}, {energies_sum_crystal_bm:.2f}), CLE d-target: {apprx_error:.2f}"
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


def main():
    # ammonia()
    # hexamine(True)
    # imidazole()
    # process_all_crystals(True)
    # collect_crystal_data(True)
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
        output_data=f"/theoryfs2/ds/csargent/chem/x23-crystals/{i}/sapt0-dz-aug/{i}/{i}.csv"
        if i == "pyrazine":
            output_data = f"/theoryfs2/ds/csargent/chem/x23-crystals/{i}/sapt0-dz-aug/{i}/sapt0-dz-aug.csv"
        process_inputs_sapt0_induction(
            output_data=output_data,
            delta_model_path=None,
            data_path=f"/theoryfs2/ds/csargent/chem/x23-crystals/{i}/sapt0-dz-aug/{i}/*.out",
            output_csv=f"./sapt0_induction/{i}_sapt0adz.pkl",
        )
    return


if __name__ == "__main__":

    main()
