"""
This script goes into molecule/analyze/output.csv and makes 
.csv files of errors for each method for one range of dimer sep.
"""

import os
import numpy as np

print(os.listdir())

molecules_to_analyze = []

for directory in os.listdir():
    if os.path.exists(os.path.join(directory, 'analyze', 'output.csv')):
        molecules_to_analyze.append(directory)

print(molecules_to_analyze)
molecular_csvs=[]

for molecule in molecules_to_analyze:
    molecular_csv = os.path.join(molecule, 'analyze', 'output.csv')
    molecular_csvs.append(molecular_csv)
print(molecular_csvs)

methods = np.genfromtxt(fname='./ammonia/analyze/output.csv', delimiter=',', max_rows=1, dtype='unicode')
methods = ','.join(methods) + '\n'

range_data=[]

for r in range(2,15):
    outfile = open(f'{r}.csv','w+')
    outfile.write(f'{r}\n')
    outfile.write(methods)
    for f in molecular_csvs:
        f = open(f, 'r')
        for line in f:
            if line.startswith(f' {r}'):
                range_data.append(line)
                range_data_string = ','.join(range_data)
                outfile.write(f'{range_data_string}\n')
                range_data.clear()
    outfile.close()
