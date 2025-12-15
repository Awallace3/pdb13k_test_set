import numpy as np
import pandas as pd
from pprint import pprint as pp
import qcelemental as qcel
import psi4

psi4.set_memory("50 GB")
psi4.set_num_threads(12)


def compute_psi4_SAPT(p4string, method="SAPT0", basis="aug-cc-pVDZ"):
    mol = psi4.geometry(p4string)
    psi4.core.set_output_file("output.dat", False)
    psi4.set_options({"basis": basis, "scf_type": "df", "freeze_core": "true"})
    _ = psi4.energy(f"{method}", molecule=mol, return_wfn=True)
    vars = psi4.core.variables()
    sapt_terms = {
        "TOTAL": vars["SAPT TOTAL ENERGY"],
        "ELST": vars["SAPT ELST ENERGY"],
        "EXCH": vars["SAPT EXCH ENERGY"],
        "IND": vars["SAPT IND ENERGY"],
        "DISP": vars["SAPT DISP ENERGY"],
    }
    psi4.core.clean()
    return sapt_terms


def compute_psi4_BM(p4string, lot="mp2/cc-pv[tq]z + D:ccsd(t)/cc-pvdz"):
    mol = psi4.geometry(p4string)
    psi4.set_options({"scf_type": "df", "freeze_core": "true"})
    return psi4.energy(f"{lot}", molecule=mol, return_wfn=True, bsse_type="cp")


def run_psi4_calc():
    # df_compute = pd.read_pickle("../los2_corrected.pkl")
    # pp(df_compute.columns.tolist())
    df = pd.read_pickle("./co2_nh3_geometries.pkl")
    sapt_terms = ["TOTAL", "ELST", "EXCH", "IND", "DISP"]
    sapt_methods = ["SAPT0", "SAPT2+3(CCD)DMP2"]
    basis_set_abbr = {"aug-cc-pVDZ": "aDZ", "aug-cc-pVTZ": "aTZ"}
    Benchmark = "mp2/cc-pv[tq]z + D:ccsd(t)/cc-pvdz"
    data = {
        "system_id": [],
        "qcel_molecule": [],
        "Benchmark": [],
        "System Label": [],
        "R": [],
    }
    for t in sapt_terms:
        for m in sapt_methods:
            if m == "SAPT0":
                b = basis_set_abbr["aug-cc-pVDZ"]
            else:
                b = basis_set_abbr["aug-cc-pVTZ"]
            col_name = f"SAPT_{m}_{t}_{b}"
            data[col_name] = []
    pp(data.keys())
    # pp(df.columns.tolist())
    for n, r in df.iterrows():
        print(r["system_id"], r["molecule"].to_string("xyz"), sep="\n")
        data["system_id"].append(r["system_id"])
        data["qcel_molecule"].append(r["molecule"])
        data["System Label"].append(r["orientation"])
        data["R"].append(r["dimer_distance"])
        # sapt calcs
        p4string = r['molecule'].to_string("psi4")
        sapt_terms = compute_psi4_SAPT(
            p4string=p4string,
            method="SAPT0",
            basis="aug-cc-pVDZ",
        )
        print("sapt_terms SAPT0 aug-cc-pVDZ:")
        pp(sapt_terms)
        for t in sapt_terms:
            col_name = f"SAPT_SAPT0_{t}_aDZ"
            data[col_name].append(sapt_terms[t])
        sapt_terms = compute_psi4_SAPT(
            p4string=p4string,
            method="SAPT2+3(CCD)DMP2",
            basis="aug-cc-pVTZ",
        )
        for t in sapt_terms:
            col_name = f"SAPT_SAPT2+3(CCD)DMP2_{t}_aTZ"
            data[col_name].append(sapt_terms[t])
        print("sapt_terms SAPT2+3(CCD)DMP2 aug-cc-pVTZ:")
        pp(sapt_terms)
        # benchmark calc
        bm_energy = compute_psi4_BM(
            p4string=p4string,
            lot=Benchmark,
        )
        print("Benchmark energy:")
        pp(bm_energy)
        data["Benchmark"].append(bm_energy)
        break

    df_results = pd.DataFrame(data)
    print(df_results)
    df_results.to_pickle("./co2_nh3_sapt_results.pkl")
    return


def main():
    run_psi4_calc()
    return


if __name__ == "__main__":
    main()
