'''
After running log.py [molecule], this takes the output .txt file and graphs the errors of each molecule per  method
'''


import os
import numpy as np
import glob
import matplotlib.pyplot as plt
import math
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('method', type = str)
args = parser.parse_args()

molecule_file_location = os.path.join('matrices','by_method','new_data','*.txt')
molecule_filenames = sorted(glob.glob(molecule_file_location))
print(molecule_filenames)

benzene_file = os.path.join('matrices','by_method','new_data','benzene.txt')

methods = []
with open(benzene_file, 'r') as bf:
    for ln in bf:
        if ln.startswith('benzene'):
            methods.append(ln)
methods = [x.split('/')[-1][:-5] for x in methods]

print(methods)

for_matrix = []
for n,mol in enumerate(molecule_filenames):
    print(mol)
    with open(mol, 'r') as molfile:
        filelines = molfile.readlines()
        for numln,ln in enumerate(filelines):
            if ln.startswith(f'{args.method}'):
                per_molecule = filelines[numln+1]
                per_molecule = [np.float64(x) for x in per_molecule.split(',')]
    print(len(per_molecule))
    if len(per_molecule) > 17:
        print(f" molecule {mol} too long!")
        per_molecule = per_molecule[-17:]
    for_matrix.append(per_molecule)


benchmark_data = np.genfromtxt(fname=f'benzene/analyze/matrix/benchmark/0-benchmark-cc.csv', delimiter=',', skip_header=1, usecols=(4,-1), dtype='unicode')

range_x=list(range(3,20))

matrix = np.array(for_matrix)
print(matrix)
#matrix = np.transpose(matrix)
#matrix = np.abs(matrix)
#matrix = np.log10(matrix)


#molecule_filenames = [x.split('/')[-1][:-4] for x in molecule_filenames]
molecule_filenames = ['14-Cyclohexanedione', 'Acetic Acid', 'Adamantane', 'Ammonia', 'Benzene','CO\u2082','Cyanamide', 'Cytosine', 'Ethyl Carbamate','Formamide', 'Hexamine', 'Ice','Imidazole','Naphthalene', 'Oxalic Acid Alpha','Oxalic Acid Beta','Pyrazine','Pyrazole','Triazine','Trioxane', 'Succinic Acid', 'Uracil', 'Urea']

dictionary = {'b3lyp':'B3LYP-D3BJ/aDZ','b97-d3bj-dz-aug':'B97-D3BJ/aDZ','b97d-dz-aug':'B97-D/aDZ','hf3c':'HF-3c/MINIX', 'mp2-dz-aug':'MP2/aDZ', 'mp2-dz-jun':'MP2/jDZ','mp2_5-dz-aug':'MP2.5/aDZ','mp2_5-dz-jun':'MP2.5/jDZ','mp2d-dz-aug':'MP2-D/aDZ','mp2d-dz-jun':'MP2-D/jDZ','pbe-d3-def2':'PBE-D3BJ/def2-tzvp','pbe-d3-dz-aug':'PBE-D3BJ/aDZ','pbeh3c-590':'PBEh-3c/def2-msvp','pbeh3c':'PBEh-3c/def2-msvp','sapt0-dz-aug':'SAPT0/aDZ','sapt0-dz-jun':'SAPT0/jDZ'}

fig,ax = plt.subplots()
im = ax.imshow(matrix, cmap='gnuplot2',vmin=-2, vmax=1.5)
ax.set_title(f'{dictionary[args.method]}', fontsize=9)
ax.set_xticks(np.arange(len(range_x)))
#plt.tick_params(left=False)
ax.set_yticks(np.arange(len(molecule_filenames)))
ax.set_xticklabels(range_x, fontsize=7)
ax.set_yticklabels(molecule_filenames, fontsize=7)
plt.xlabel(f'Minimum monomer separation at which {dictionary[args.method]} is used rather than CCSD(T)/CBS', fontsize=7)
cbar = fig.colorbar(im)
cbar.set_label('Error in kJ/mol (log scale)', rotation=270, labelpad=20, fontsize=7)
cbar.ax.set_yticklabels([-2.0, -1.5, -1.0, -0.5, 0, 0.5, 1.0, 1.5], fontsize=7)
plt.show()
#plt.savefig(f'matrices/by_method/june22022/{args.method}.pdf', dpi=300, bbox_inches='tight', transparent=True)


