import pandas as pd
import qcelemental as qcel
import subprocess
from subprocess import check_output
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

        openmm_pdb = f"{c_path}/mol_A.openmm.pdb"
        debug_pdb = f"{c_path}/mol_A-debug.pdb"
        merged_pdb = f"{c_path}/mol_A.openmm_merged.pdb"

        if not os.path.exists(openmm_pdb):
            print(f"Skipping {c}: {openmm_pdb} not found")
            continue

        if not os.path.exists(debug_pdb):
            print(f"Skipping {c}: {debug_pdb} not found")
            continue

        try:
            merge_pdb_files(openmm_pdb, debug_pdb, merged_pdb)
            print(f"Created: {merged_pdb}")
        except Exception as e:
            print(f"Error processing {c}: {e}")
    return


def merge_pdb_files(openmm_pdb, debug_pdb, output_pdb):
    """
    Merge atom types from openmm PDB with connectivity from debug PDB.

    Parameters
    ----------
    openmm_pdb : str
        Path to mol_A.openmm.pdb with correct atom types
    debug_pdb : str
        Path to mol_A-debug.pdb with CONECT records
    output_pdb : str
        Path to output merged PDB file

    Returns
    -------
    output_pdb : str
        Path to the merged PDB file
    """
    # Read atom lines from openmm PDB
    with open(openmm_pdb, "r") as f:
        openmm_lines = f.readlines()

    # Read CONECT records from debug PDB
    with open(debug_pdb, "r") as f:
        debug_lines = f.readlines()

    # Extract CONECT and END records
    conect_lines = [line for line in debug_lines if line.startswith("CONECT")]
    end_lines = [line for line in debug_lines if line.startswith("END")]

    # Write merged file
    with open(output_pdb, "w") as f:
        # Write atom lines from openmm PDB
        for line in openmm_lines:
            if line.startswith(("ATOM", "HETATM")):
                f.write(line)
        # Write CONECT records from debug PDB
        for line in conect_lines:
            f.write(line)
        # Write END record
        for line in end_lines:
            f.write(line)

    return output_pdb


def create_openmm_system(xml_file, pdb_file, residueTemplates=None):
    """
    Create OpenMM System from force field XML and PDB topology files.

    Parameters
    ----------
    xml_file : str
        Path to OpenMM force field XML file
    pdb_file : str
        Path to PDB file with topology
    residueTemplates : str, optional
        Which residue templates to use ('all', 'first', or None for default)

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
    try:
        system = forcefield.createSystem(
            pdb.topology,
            nonbondedMethod=app.NoCutoff,
            constraints=None,
            rigidWater=False,
            residueTemplates=residueTemplates,
        )
    except Exception as e:
        # If residueTemplates doesn't work, try without it
        system = forcefield.createSystem(
            pdb.topology,
            nonbondedMethod=app.NoCutoff,
            constraints=None,
            rigidWater=False,
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


def add_conect_from_xml(pdb_file, xml_file, output_pdb):
    """
    Add CONECT records to PDB based on bonds defined in XML force field.

    Parameters
    ----------
    pdb_file : str
        Input PDB file without CONECT records
    xml_file : str
        OpenMM XML force field with bond definitions
    output_pdb : str
        Output PDB file with CONECT records

    Returns
    -------
    output_pdb : str
        Path to output PDB file
    """
    import xml.etree.ElementTree as ET

    # Parse XML to get bond information
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # Find bonds in residue template
    bonds = []
    for residue in root.findall(".//Residue[@name='MOL']"):
        for bond in residue.findall("Bond"):
            from_idx = int(bond.get("from"))
            to_idx = int(bond.get("to"))
            bonds.append((from_idx, to_idx))

    # Read PDB file
    with open(pdb_file, "r") as f:
        pdb_lines = f.readlines()

    # Write output with CONECT records
    with open(output_pdb, "w") as f:
        # Write atom records
        for line in pdb_lines:
            if line.startswith(("ATOM", "HETATM")):
                f.write(line)

        # Write CONECT records based on XML bonds
        for from_idx, to_idx in bonds:
            # PDB uses 1-based indexing
            f.write(f"CONECT{from_idx + 1:5d}{to_idx + 1:5d}\n")

        f.write("END\n")

    return output_pdb


def reorder_qcel_mol_to_match_pdb(qcel_mol, pdb_file, xml_file):
    """
    Reorder qcel_mol atoms to match PDB file atom ordering using topology matching.

    This function uses graph isomorphism to match atoms based on their connectivity,
    not just their element type. This is critical for molecules where atoms of the
    same element have different bonding patterns (e.g., different carbons).

    Parameters
    ----------
    qcel_mol : qcelemental.models.Molecule
        QCElemental molecule to reorder
    pdb_file : str
        PDB file with reference atom ordering
    xml_file : str
        XML force field file with bond topology

    Returns
    -------
    reordered_geom : ndarray
        Geometry reordered to match PDB
    """
    import networkx as nx
    from MDAnalysis import Universe
    import tempfile

    # Create graph from qcel_mol using MDAnalysis bond guessing
    with tempfile.NamedTemporaryFile(suffix=".xyz", delete=True) as tmp:
        qcel_mol.to_file(tmp.name, dtype="xyz")
        u_qcel = Universe(tmp.name, format="xyz", to_guess=["bonds"])

    G_qcel = nx.Graph()
    for i, symbol in enumerate(qcel_mol.symbols):
        G_qcel.add_node(i, symbol=symbol)
    for bond in u_qcel.bonds:
        i, j = bond.indices
        G_qcel.add_edge(i, j)

    # Create graph from PDB + XML bonds
    pdb = app.PDBFile(pdb_file)
    pdb_atoms = list(pdb.topology.atoms())
    pdb_symbols = [atom.element.symbol for atom in pdb_atoms]

    G_pdb = nx.Graph()
    for i, symbol in enumerate(pdb_symbols):
        G_pdb.add_node(i, symbol=symbol)

    # Get bonds from XML
    import xml.etree.ElementTree as ET

    tree = ET.parse(xml_file)
    root = tree.getroot()
    for residue_el in root.find("Residues").findall("Residue"):
        for bond_el in residue_el.findall("Bond"):
            i = int(bond_el.get("from"))
            j = int(bond_el.get("to"))
            G_pdb.add_edge(i, j)

    # Use graph isomorphism to find mapping
    def node_match(n1, n2):
        return n1["symbol"] == n2["symbol"]

    GM = nx.isomorphism.GraphMatcher(G_pdb, G_qcel, node_match=node_match)
    if not GM.is_isomorphic():
        raise ValueError("Molecular graphs are not isomorphic!")

    # Get mapping: pdb_idx -> qcel_idx
    mapping = GM.mapping

    # Create reordering: for each PDB position, get corresponding qcel geometry
    import numpy as np

    reorder_map = [mapping[i] for i in range(len(pdb_symbols))]
    reordered_geom = qcel_mol.geometry[reorder_map]

    return reordered_geom


def create_dimer_system(qcel_mol, xml_file, monomer_pdb):
    """
    Create OpenMM System for a dimer using Modeller to combine monomers.

    Parameters
    ----------
    qcel_mol : qcelemental.models.Molecule
        QCElemental molecule with 2 fragments (dimer)
    xml_file : str
        Path to OpenMM force field XML file
    monomer_pdb : str
        Path to monomer PDB with correct atom types

    Returns
    -------
    system : openmm.System
        OpenMM System object for dimer
    topology : openmm.app.Topology
        Topology object for dimer
    """
    import tempfile
    from openmm.app import Modeller

    # Create monomer PDB with correct CONECT records from XML
    with tempfile.NamedTemporaryFile(mode="w", suffix=".pdb", delete=False) as tmp:
        monomer_with_conect = tmp.name
    add_conect_from_xml(monomer_pdb, xml_file, monomer_with_conect)

    try:
        # Load force field
        forcefield = app.ForceField(xml_file)

        # Load monomer PDB to get template topology
        monomer_pdb_obj = app.PDBFile(monomer_with_conect)

        # Create modeller with first monomer
        modeller = Modeller(monomer_pdb_obj.topology, monomer_pdb_obj.positions)

        # Add second monomer
        modeller.add(monomer_pdb_obj.topology, monomer_pdb_obj.positions)

        # Create system from combined topology
        system = forcefield.createSystem(
            modeller.topology,
            nonbondedMethod=app.NoCutoff,
            constraints=None,
            rigidWater=False,
        )

        return system, modeller.topology
    finally:
        # Clean up temporary file
        if os.path.exists(monomer_with_conect):
            os.unlink(monomer_with_conect)


def calculate_dimer_interaction_energy(qcel_mol, xml_file, monomer_pdb):
    """
    Calculate dimer interaction energy using OpenMM force field.

    E_interaction = E_dimer - (E_monomer1 + E_monomer2)

    Parameters
    ----------
    qcel_mol : qcelemental.models.Molecule
        QCElemental molecule with 2 fragments (dimer)
    xml_file : str
        Path to OpenMM force field XML file
    monomer_pdb : str
        Path to PDB file with monomer topology and atom types

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
    geom_hash : str
        MD5 hash (first 8 chars) of geometry for verification
    """
    import tempfile
    import hashlib

    # Verify we have a dimer
    if len(qcel_mol.fragments) != 2:
        raise ValueError(f"Expected 2 fragments (dimer), got {len(qcel_mol.fragments)}")

    # Get monomer fragments
    mol_mon1 = qcel_mol.get_fragment(0)
    mol_mon2 = qcel_mol.get_fragment(1)

    # Create dimer system using monomer PDB (has PDB atom ordering)
    system_dimer, topology_dimer = create_dimer_system(qcel_mol, xml_file, monomer_pdb)

    # Reorder qcel_mol geometry to match PDB ordering using topology-aware matching
    import numpy as np

    bohr_to_nm = qcel.constants.conversion_factor("bohr", "nm")

    reordered_geom_mon1 = reorder_qcel_mol_to_match_pdb(mol_mon1, monomer_pdb, xml_file)
    reordered_geom_mon2 = reorder_qcel_mol_to_match_pdb(mol_mon2, monomer_pdb, xml_file)

    # Concatenate to form dimer geometry in PDB ordering
    reordered_dimer_geom = np.vstack([reordered_geom_mon1, reordered_geom_mon2])

    positions_dimer_from_qcel = reordered_dimer_geom * bohr_to_nm
    positions_dimer_omm = [
        openmm.Vec3(x, y, z) * unit.nanometers for x, y, z in positions_dimer_from_qcel
    ]

    # Debug: Print geometry hash to verify it changes between dimers
    geom_hash = hashlib.md5(qcel_mol.geometry.tobytes()).hexdigest()[:8]

    # Calculate dimer energy with updated geometry
    e_dimer = calculate_energy(system_dimer, positions_dimer_omm)

    # For monomer energies, use ORIGINAL (unreordered) geometries
    # because the temporary PDB files we create have qcel atom ordering
    positions_mon1 = mol_mon1.geometry * bohr_to_nm
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

    return interaction_energy, e_dimer, e_mon1, e_mon2, geom_hash


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


def run_ff_dimer(v="apprx"):
    """
    Calculate OpenMM force field dimer interaction energies for crystal systems.

    For each crystal, this function:
    1. Loads dimer geometries from the results dataframe
    2. Uses OpenMM to calculate interaction energies with the OPLS force field
    3. Prints the interaction energy components (E_dimer, E_mon1, E_mon2, E_int)
    """
    df = pd.read_pickle(f"./crystals_ap2_ap3_des_results_mol_{v}.pkl")
    # from pprint import pprint as pp
    # pp(df.columns.tolist())
    mms_col = "Minimum Monomer Separations (A) sapt0-dz-aug"

    # Process ice crystal for now
    for c in crystal_names_all:
        try:
            c_path = f"ff_params/{c}"
            if not os.path.exists(f"{c_path}/mol_A.openmm.pdb"):
                print(f"Skipping crystal {c}, missing PDB file")
                continue

            # Use original OpenMM PDB - connectivity will be inferred from XML
            pdb_file_to_use = f"{c_path}/mol_A.openmm.pdb"
            print(f"Using PDB: {pdb_file_to_use}")

            df_c = df[df[f"crystal {v}"] == c]
            print(f"Processing crystal: {c}, {len(df_c)} dimers")
            df_c = df_c.sort_values(by=mms_col, ascending=True)
            ies, e_dimers, e_mon1s, e_mon2s = [], [], [], []
            dimer_idx = 0
            for n, r in df_c.iterrows():
                mol = r[f"mol {v}"]
                interaction_energy, e_dimer, e_mon1, e_mon2, geom_hash = (
                    calculate_dimer_interaction_energy(
                        mol,
                        xml_file=f"{c_path}/mol_A.openmm.xml",
                        monomer_pdb=pdb_file_to_use,
                    )
                )
                ies.append(interaction_energy)
                e_dimers.append(e_dimer)
                e_mon1s.append(e_mon1)
                e_mon2s.append(e_mon2)

                print(
                    f"  Dimer {dimer_idx} [geom:{geom_hash}]: "
                    f"E_int={interaction_energy:12.6f}, "
                    f"E_dimer={e_dimer:12.6f}, E_mon1={e_mon1:12.6f}, "
                    f"E_mon2={e_mon2:12.6f}, MMS={r[mms_col]:.3f}"
                    f"E_SAPT0={r['Non-Additive MB Energy (kJ/mol) sapt0-dz-aug']:12.6f}"
                )
                dimer_idx += 1

            # Verify energies are unique
            unique_e_int = len(set(ies))
            unique_e_dimer = len(set(e_dimers))
            print(f"\n  Summary: {len(ies)} dimers processed")
            print(f"  Unique interaction energies: {unique_e_int}/{len(ies)}")
            print(f"  Unique dimer energies: {unique_e_dimer}/{len(ies)}")
            if unique_e_int < len(ies):
                print("  WARNING: Some interaction energies are identical!")
            if unique_e_dimer < len(ies):
                print("  WARNING: Some dimer energies are identical!")
            print()

            df.loc[df_c.index, "OPLS Interaction Energy (kJ/mol)"] = ies
            df.loc[df_c.index, "OPLS E_dimer (kJ/mol)"] = e_dimers
            df.loc[df_c.index, "OPLS E_mon1 (kJ/mol)"] = e_mon1s
            df.loc[df_c.index, "OPLS E_mon2 (kJ/mol)"] = e_mon2s
        except Exception as e:
            print(f"Error processing crystal {c}: {e}")
            continue
    if v == "apprx":
        print(
            df[
                [
                    f"crystal {v}",
                    "OPLS Interaction Energy (kJ/mol)",
                    "Non-Additive MB Energy (kJ/mol) sapt0-dz-aug",
                ]
            ]
        )
    # else:
    #     print(
    #         df[
    #             [
    #                 f"crystal {v}",
    #                 "OPLS Interaction Energy (kJ/mol)",
    #                 "Non-Additive MB Energy (kJ/mol) CCSD(T)/CBS",
    #             ]
    #         ]
    #     )

    df.to_pickle(f"./crystals_ap2_ap3_des_results_mol_{v}.pkl")
    return


def main():
    # generate_paramaters()
    # run_ff_dimer(v='apprx')
    run_ff_dimer(v='bm')
    return


if __name__ == "__main__":
    main()
