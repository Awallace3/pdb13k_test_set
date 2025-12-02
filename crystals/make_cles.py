import os

#  sed -i 's/cc_type df/cc_type df\n  dft_spherical_points 590\n  dft_radial_points 99 \n  dft_pruning_scheme robust/g' *.in

def get_cle(name):


    return f"""cif_input       = ./cif/{name}.cif
cif_output      = {name}.xyz
cif_a           = 0
cif_b           = 0
cif_c           = 0
nmers_up_to     = 2
r_cut_com       = 1000
r_cut_monomer   = 0
r_cut_dimer     = 20
cle_run_type    = psithon
psi4_method     = b3lyp-d3bj/aug-cc-pvdz
psi4_bsse       = cp
psi4_memory     = 7 GB
verbose         = 2"""



for cif_file in os.listdir("cif"):

    print(cif_file) 

    cif_name = cif_file[:-4]
    
    if not os.path.exists(f"{cif_name}/b3lyp"):
        os.makedirs(f"{cif_name}/b3lyp")

    cle_data = get_cle(cif_name)

    with open(f"{cif_name}/b3lyp/{cif_name}.cle", "w") as cle_file:
        cle_file.write(cle_data)


