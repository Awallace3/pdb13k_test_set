import pandas as pd
import qcelemental as qcel
import os
import numpy as np

#file_to_mol = {'14-cyclohexanedione':'14-Cyclohexanedione', 'ammonia':'Ammonia', 'acetic_acid':'Acetic Acid', 'adamantane':'Adamantane', 'benzene':'Benzene', 'carbon_dioxide':'Carbon Dioxide', 'cyanamide':'Cyanamide', 'cytosine':'Cytosine', 'ethyl_carbamate':'Ethyl Carbamate', 'formamide':'Formamide', 'hexamine':'Hexamine', 'ice':'Ice','imidazole':'Imidazole', 'naphthalene':'Naphthalene', 'oxalic_acid_alpha':'Oxalic Acid Alpha', 'oxalic_acid_beta':'Oxalic Acid Beta', 'pyrazine':'Pyrazine', 'pyrazole':'Pyrazole','triazine':'Triazine','trioxane':'Trioxane', 'succinic_acid':'Succinic Acid', 'uracil':'Uracil', 'urea':'Urea'}
#molecules=file_to_mol.keys()
#molecules = list(molecules)[14:]
molecules = ['naphthalene']

for molecule in molecules:
    OUTS_PATH = f'/theoryfs2/ds/metcalf/crystals/cc_ss_2b_x23/{molecule}/'
    filenames= [x for x in os.listdir(OUTS_PATH) if x.endswith('.out')]
    print(molecule)
    data = pd.DataFrame(columns=['N-mer Name','Non-Additive MB Energy (kJ/mol)','Num. Rep. (#)','N-mer Contribution (kJ/mol)','Avg COM Separation (A)','Minimum Monomer Separations (A)'])
    
    for filename in filenames:
        with open(OUTS_PATH+filename, 'r') as outfile:
            lines = outfile.readlines()
            MP2CBSe = []
            # get replica number and MP2/CBS energies
            for line in lines:
                if line.startswith('# Number of replicas:'):
                    rep_num = int(line.split()[-1])
                elif line.startswith('# Minimum COM separations:'):
                    min_COM_sep = line.split()[-1]
                elif line.startswith('# Minimum monomer separations:'):
                    min_mon_sep = line.split()[-1]
                elif line.startswith('   @Extrapolated MP2/(T,Q)'):
                    MP2CBSe.append(float(line.split()[-1])) #Eh
            # calc MP2/CBS inte and convert to kJ/mol
            MP2CBSinteH = MP2CBSe[2] - MP2CBSe[1] - MP2CBSe[0] #Eh
            HtokJmol = qcel.constants.conversion_factor("hartree","kJ/mol")
            MP2CBSinte = MP2CBSinteH * HtokJmol #to kJ/mol
            # determine N-mer contribution 
            contrib = MP2CBSinte / 2 * rep_num
            cle_line = [filename, "{:.8f}".format(MP2CBSinte), rep_num, "{:.8f}".format(contrib),  min_COM_sep, float(min_mon_sep)]
            data.loc[len(data)] = cle_line
    
    # Sort df by min_mon_sep
    data = data.sort_values(by=['Minimum Monomer Separations (A)'])
    data = data.reset_index()
    
    # Make PCLE column (cumulative PCLE as next shortest separated dimer is added)
    data['Partial Crys. Lattice Ener. (kJ/mol)'] = np.nan
    PCLE = 0
    for index, row in data.iterrows():
        contrib = float(row['N-mer Contribution (kJ/mol)'])
        cumPCLE = PCLE + contrib
        data.at[index,'Partial Crys. Lattice Ener. (kJ/mol)'] = "{:.8f}".format(cumPCLE)
        PCLE = cumPCLE
    
    # reorder columns to match CrystaLattE
    data = data[['N-mer Name','Non-Additive MB Energy (kJ/mol)','Num. Rep. (#)','N-mer Contribution (kJ/mol)','Partial Crys. Lattice Ener. (kJ/mol)','Avg COM Separation (A)','Minimum Monomer Separations (A)']]
    print(data)
    
    data.to_csv(f'{molecule}_MP2aTQZ.csv')
