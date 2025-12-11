import pandas as pd
import qcelemental as qcel
import subprocess
from subprocess import check_output
import numpy as np
import os
import openmm
from openmm import app
from openmm import unit
import omm_utils as utils


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


def create_openmm_system(xml_file, pdb_file):
    """
    Create OpenMM System from force field XML and PDB topology files.

    Parameters
    ----------
    xml_file : str
        Path to OpenMM force field XML file
    pdb_file : str
        Path to PDB file with topology

    Returns
    -------
    system : openmm.System
        OpenMM System object with forces
    topology : openmm.app.Topology
        OpenMM Topology object
    positions : list
        Atomic positions in nm
    """
    # Load force field from XML
    forcefield = app.ForceField(xml_file)

    # Load PDB file to get topology and positions
    pdb = app.PDBFile(pdb_file)

    # Create system with no cutoff for gas-phase calculations
    system = forcefield.createSystem(
        pdb.topology, nonbondedMethod=app.NoCutoff, constraints=None, rigidWater=False
    )

    return system, pdb.topology, pdb.positions


def calculate_energy(system, positions):
    """
    Calculate potential energy for given system and positions.

    Parameters
    ----------
    system : openmm.System
        OpenMM System object
    positions : list or array
        Atomic positions in nm

    Returns
    -------
    energy : float
        Potential energy in kJ/mol
    """
    # Create integrator (dummy, just needed for Context)
    integrator = openmm.VerletIntegrator(0.001 * unit.picoseconds)

    # Create context
    context = openmm.Context(system, integrator)

    # Set positions
    context.setPositions(positions)

    # Get energy
    state = context.getState(getEnergy=True)
    energy = state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)

    # Clean up
    del context
    del integrator

    return energy


def calculate_dimer_interaction_energy(qcel_mol, xml_file, pdb_file_dimer):
    """
    Calculate dimer interaction energy using OpenMM force field.

    E_interaction = E_dimer - (E_monomer1 + E_monomer2)

    Parameters
    ----------
    qcel_mol : qcelemental.models.Molecule
        QCElemental molecule with 2 fragments (dimer)
    xml_file : str
        Path to OpenMM force field XML file
    pdb_file_dimer : str
        Path to PDB file with dimer topology

    Returns
    -------
    interaction_energy : float
        Interaction energy in kJ/mol
    e_dimer : float
        Dimer energy in kJ/mol
    e_mon1 : float
        Monomer 1 energy in kJ/mol
    e_mon2 : float
        Monomer 2 energy in kJ/mol
    """
    # Verify we have a dimer
    if len(qcel_mol.fragments) != 2:
        raise ValueError(f"Expected 2 fragments (dimer), got {len(qcel_mol.fragments)}")

    # Create system for dimer
    system_dimer, topology_dimer, positions_dimer = create_openmm_system(
        xml_file, pdb_file_dimer
    )

    # Calculate dimer energy
    e_dimer = calculate_energy(system_dimer, positions_dimer)

    # Extract monomer geometries from QCElemental molecule
    # Positions need to be converted from Bohr to nanometers
    bohr_to_nm = qcel.constants.conversion_factor("bohr", "nm")

    # Get monomer 1 (fragment 0)
    mol_mon1 = qcel_mol.get_fragment(0)
    positions_mon1 = mol_mon1.geometry * bohr_to_nm

    # Get monomer 2 (fragment 1)
    mol_mon2 = qcel_mol.get_fragment(1)
    positions_mon2 = mol_mon2.geometry * bohr_to_nm

    # Create PDB files for monomers (temporary)
    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", suffix=".pdb", delete=False) as tmp1:
        pdb_mon1 = tmp1.name
        utils._molecule_to_pdb_file(mol_mon1, pdb_mon1, "MOL", None)
        utils._add_CONECT(pdb_mon1)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".pdb", delete=False) as tmp2:
        pdb_mon2 = tmp2.name
        utils._molecule_to_pdb_file(mol_mon2, pdb_mon2, "MOL", None)
        utils._add_CONECT(pdb_mon2)

    try:
        # Create systems for monomers
        system_mon1, topology_mon1, _ = create_openmm_system(xml_file, pdb_mon1)
        system_mon2, topology_mon2, _ = create_openmm_system(xml_file, pdb_mon2)

        # Convert positions to OpenMM format (list of Vec3 with units)
        positions_mon1_omm = [
            openmm.Vec3(x, y, z) * unit.nanometers for x, y, z in positions_mon1
        ]
        positions_mon2_omm = [
            openmm.Vec3(x, y, z) * unit.nanometers for x, y, z in positions_mon2
        ]

        # Calculate monomer energies
        e_mon1 = calculate_energy(system_mon1, positions_mon1_omm)
        e_mon2 = calculate_energy(system_mon2, positions_mon2_omm)

    finally:
        # Clean up temporary files
        os.unlink(pdb_mon1)
        os.unlink(pdb_mon2)

    # Calculate interaction energy
    interaction_energy = e_dimer - (e_mon1 + e_mon2)

    return interaction_energy, e_dimer, e_mon1, e_mon2


def qcel_molecule_to_openmm_for_energy(qcel_mol, **kwargs):
    pdb_file = kwargs.get("pdb_file", None)
    xml_file = kwargs.get("xml_file", None)
    atom_types_map = kwargs.get("atom_types_map", None)

    # fix topological issues in QCElemental (if any)
    qcel_mol = utils._fix_topological_order(qcel_mol)

    # update pdb_file with correct qcel_mol "topology"
    pdb_file = utils._create_topology(qcel_mol, pdb_file, atom_types_map)
    xmlmd = utils.XmlMD(qcel_mol=qcel_mol, atom_types_map=atom_types_map)
    xmlmd.parse_xml(xml_file)
    print(xmlmd)
    return xmlmd


def run_ff_dimer():
    """
    Calculate OpenMM force field dimer interaction energies for crystal systems.

    For each crystal, this function:
    1. Loads dimer geometries from the results dataframe
    2. Uses OpenMM to calculate interaction energies with the OPLS force field
    3. Prints the interaction energy components (E_dimer, E_mon1, E_mon2, E_int)
    """
    df = pd.read_pickle("./crystals_ap2_ap3_results_mol_apprx.pkl")
    mms_col = "Minimum Monomer Separations (A) sapt0-dz-aug"

    # Process ice crystal for now
    for c in crystal_names_all:
        c_path = f"ff_params/{c}"
        if not os.path.exists(c_path + f"/mol_A.openmm_tmp.pdb"):
            continue
        df_c = df[df["crystal apprx"] == c]
        print(f"Processing crystal: {c}, {len(df_c)} dimers")
        df_c = df_c.sort_values(by=mms_col, ascending=True)
        ies, e_dimers, e_mon1s, e_mon2s = [], [], [], []
        for n, r in df_c.iterrows():
            mol = r["mol apprx"]
            interaction_energy, e_dimer, e_mon1, e_mon2 = (
                calculate_dimer_interaction_energy(
                    mol,
                    xml_file=f"{c_path}/mol_A.openmm.xml",
                    pdb_file_dimer=f"{c_path}/mol_A.openmm_tmp.pdb",
                )
            )
            ies.append(interaction_energy)
            e_dimers.append(e_dimer)
            e_mon1s.append(e_mon1)
            e_mon2s.append(e_mon2)

            # print(f"\nEnergy Results (kJ/mol):")
            # print(f"  E_dimer:      {e_dimer:12.6f}")
            # print(f"  E_monomer1:   {e_mon1:12.6f}")
            # print(f"  E_monomer2:   {e_mon2:12.6f}")
            print(f"  E_interaction:{interaction_energy:12.6f}")
            # print("\nReference SAPT0 Results (kJ/mol):")
            # print(r1['Non-Additive MB Energy (kJ/mol) sapt0-dz-aug'])
        df.loc[df_c.index, "OPLS Interaction Energy (kJ/mol)"] = ies
        df.loc[df_c.index, "OPLS E_dimer (kJ/mol)"] = e_dimers
        df.loc[df_c.index, "OPLS E_mon1 (kJ/mol)"] = e_mon1s
        df.loc[df_c.index, "OPLS E_mon2 (kJ/mol)"] = e_mon2s
    df.to_pickle("./crystals_ap2_ap3_results_mol_apprx_opls.pkl")
    return


def main():
    # generate_paramaters()
    run_ff_dimer()
    return


if __name__ == "__main__":
    main()
