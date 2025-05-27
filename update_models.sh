#!/usr/bin/bash

cp ~/gits/qcmlforge/models/ap2_ensemble/ap2_t2_0.pt ./models/ap2_ensemble/ap2_0.pt
cp ~/gits/qcmlforge/models/ap2_ensemble/ap2_t2_1.pt ./models/ap2_ensemble/ap2_1.pt
cp ~/gits/qcmlforge/models/ap2_ensemble/ap2_t2_2.pt ./models/ap2_ensemble/ap2_2.pt;
cp ~/gits/qcmlforge/models/ap2_ensemble/ap2_t2_3.pt ./models/ap2_ensemble/ap2_3.pt
cp ~/gits/qcmlforge/models/ap2_ensemble/ap2_t2_4.pt ./models/ap2_ensemble/ap2_4.pt
mv ./pdb13k_errors_pt-ap2-t2.pkl ./pdb13k_errors_pt-ap2-t2.pkl.bak
python ./pdb13k_pt.py
