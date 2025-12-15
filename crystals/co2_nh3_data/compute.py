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

    # Stage 1: Run all SAPT0 calculations and checkpoint
    print("\n" + "=" * 80)
    print("STAGE 1: Running all SAPT0 calculations")
    print("=" * 80 + "\n")
    for n, r in df.iterrows():
        print(f"\nSAPT0 for system {n}: {r['system_id']}")
        print(r["molecule"].to_string("xyz"))
        data["system_id"].append(r["system_id"])
        data["qcel_molecule"].append(r["molecule"])
        data["System Label"].append(r["orientation"])
        data["R"].append(r["dimer_distance"])

        p4string = r["molecule"].to_string("psi4")
        sapt_terms_result = compute_psi4_SAPT(
            p4string=p4string,
            method="SAPT0",
            basis="aug-cc-pVDZ",
        )
        print("sapt_terms SAPT0 aug-cc-pVDZ:")
        pp(sapt_terms_result)
        for t in sapt_terms_result:
            col_name = f"SAPT_SAPT0_{t}_aDZ"
            data[col_name].append(sapt_terms_result[t])

        # Initialize SAPT2 and Benchmark columns with None for now
        for t in sapt_terms:
            col_name = f"SAPT_SAPT2+3(CCD)DMP2_{t}_aTZ"
            data[col_name].append(None)
        data["Benchmark"].append(None)

    # Save SAPT0 checkpoint
    df_checkpoint = pd.DataFrame(data)
    df_checkpoint.to_pickle("./co2_nh3_sapt_results_sapt0.pkl")
    print("\n" + "=" * 80)
    print("CHECKPOINT: SAPT0 results saved to co2_nh3_sapt_results_sapt0.pkl")
    print("=" * 80 + "\n")

    # Stage 2: Run all SAPT2 calculations and checkpoint
    print("\n" + "=" * 80)
    print("STAGE 2: Running all SAPT2+3(CCD)DMP2 calculations")
    print("=" * 80 + "\n")
    for n, r in df.iterrows():
        print(f"\nSAPT2+3(CCD)DMP2 for system {n}: {r['system_id']}")
        p4string = r["molecule"].to_string("psi4")
        sapt_terms_result = compute_psi4_SAPT(
            p4string=p4string,
            method="SAPT2+3(CCD)DMP2",
            basis="aug-cc-pVTZ",
        )
        print("sapt_terms SAPT2+3(CCD)DMP2 aug-cc-pVTZ:")
        pp(sapt_terms_result)
        for t in sapt_terms_result:
            col_name = f"SAPT_SAPT2+3(CCD)DMP2_{t}_aTZ"
            data[col_name][n] = sapt_terms_result[t]

    # Save SAPT2 checkpoint
    df_checkpoint = pd.DataFrame(data)
    df_checkpoint.to_pickle("./co2_nh3_sapt_results_sapt2.pkl")
    print("\n" + "=" * 80)
    print("CHECKPOINT: SAPT2 results saved to co2_nh3_sapt_results_sapt2.pkl")
    print("=" * 80 + "\n")

    # Stage 3: Run all benchmark calculations
    print("\n" + "=" * 80)
    print("STAGE 3: Running all Benchmark calculations")
    print("=" * 80 + "\n")
    for n, r in df.iterrows():
        print(f"\nBenchmark for system {n}: {r['system_id']}")
        p4string = r["molecule"].to_string("psi4")
        bm_energy = compute_psi4_BM(
            p4string=p4string,
            lot=Benchmark,
        )
        print("Benchmark energy:")
        pp(bm_energy)
        data["Benchmark"][n] = bm_energy

    # Save final results
    df_results = pd.DataFrame(data)
    print("\n" + "=" * 80)
    print("Final Results:")
    print("=" * 80)
    print(df_results)
    df_results.to_pickle("./co2_nh3_sapt_results.pkl")
    print("\n" + "=" * 80)
    print("FINAL: All results saved to co2_nh3_sapt_results.pkl")
    print("=" * 80 + "\n")
    return


def main():
    run_psi4_calc()
    return


if __name__ == "__main__":
    main()
