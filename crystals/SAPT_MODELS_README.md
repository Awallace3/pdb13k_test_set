# SAPT Model Inference and Visualization

This document describes the new functions added to `crystal_main.py` for running inference with SAPT-trained models and visualizing crystal lattice energy errors.

## New Functions

### 1. `ap2_ap3_df_energies_sapt_models(generate=False, v='bm')`

Computes AP2/AP3 predictions using four SAPT-trained model variants and saves results to a dataframe.

**Models evaluated:**
- `SAPT0_adz`: Base model trained on SAPT0/aug-cc-pVDZ data
- `SAPT0_adz_tl`: Transfer learning version (fine-tuned)
- `SAPT2p3CCDDMP2_atz`: Model trained on SAPT2+3(CCD)DMP2/aug-cc-pVTZ data  
- `SAPT2p3CCDDMP2_atz_tl`: Transfer learning version (fine-tuned)

**Parameters:**
- `generate` (bool): If True, regenerate predictions. If False, load existing pickle.
- `v` (str): Version to use - 'bm' for benchmark or 'apprx' for approximate

**Returns:**
- DataFrame with predictions for all model variants

**Output file:**
- `crystals_ap2_ap3_sapt_results_mol_{v}.pkl`

**Example usage:**
```python
from crystal_main import ap2_ap3_df_energies_sapt_models

# Generate predictions for benchmark version
df = ap2_ap3_df_energies_sapt_models(generate=True, v='bm')

# Load existing results
df = ap2_ap3_df_energies_sapt_models(generate=False, v='bm')
```

### 2. `plot_crystal_lattice_energies_sapt_models(v='bm', N_values=[0, 1, 5, 10, 20])`

Creates violin plots visualizing crystal lattice energy errors for SAPT-trained models across different N-body truncations.

**Parameters:**
- `v` (str): Version to use - 'bm' for benchmark or 'apprx' for approximate
- `N_values` (list): N-body truncation values to plot (0 = no truncation)

**Outputs:**
- Violin plots: `./x23_plots/sapt_models_cle_errors_{v}_N{N}.png`
- CSV files: `./x23_plots/sapt_models_cle_errors_{v}_N{N}.csv`

**Example usage:**
```python
from crystal_main import plot_crystal_lattice_energies_sapt_models

# Create plots for default N values
plot_crystal_lattice_energies_sapt_models(v='bm')

# Custom N values
plot_crystal_lattice_energies_sapt_models(v='bm', N_values=[0, 5, 10])
```

## Quick Start with test_sapt_models.py

A test script is provided for easy usage:

```bash
# Generate predictions and create plots
python test_sapt_models.py --both --version bm

# Just generate predictions
python test_sapt_models.py --generate --version bm

# Just create plots (requires existing predictions)
python test_sapt_models.py --plot --version bm --N-values 0 1 5 10 20

# Use approximate version
python test_sapt_models.py --both --version apprx
```

## Model File Locations

Models are expected in `./sft_models/`:
- `ap2_los2_SAPT0_adz.pt`
- `ap2_los2_SAPT0_adz_tl.pt`  
- `ap2_los2_SAPT2p3CCDDMP2_atz.pt`
- `ap2_los2_SAPT2p3CCDDMP2_atz_tl.pt`
- `ap3_los2_SAPT0_adz.pt`
- `ap3_los2_SAPT0_adz_tl.pt`
- `ap3_los2_SAPT2p3CCDDMP2_atz.pt`
- `ap3_los2_SAPT2p3CCDDMP2_atz_tl.pt`

## Output Data Structure

The generated dataframe includes the following columns for each model:

For each model variant (e.g., `SAPT0_adz`):
- `AP2-{model}_TOTAL`: Total AP2 energy
- `AP2-{model}_ELST`: AP2 electrostatics component
- `AP2-{model}_EXCH`: AP2 exchange component
- `AP2-{model}_INDU`: AP2 induction component
- `AP2-{model}_DISP`: AP2 dispersion component
- `AP3-{model}_TOTAL`: Total AP3 energy
- `AP3-{model}_ELST`: AP3 electrostatics component
- `AP3-{model}_EXCH`: AP3 exchange component
- `AP3-{model}_INDU`: AP3 induction component
- `AP3-{model}_DISP`: AP3 dispersion component
- `ap2-{model}_le_contribution`: AP2 lattice energy contribution
- `ap3-{model}_le_contribution`: AP3 lattice energy contribution
- `ap3_d_elst-{model}`: AP3 damped electrostatics
- `ap3_classical_ind_energy-{model}`: AP3 classical induction energy

## Dependencies

- pandas
- numpy
- apnet_pt (for model inference)
- cdsg_plot.error_statistics (for visualization)
- matplotlib
- qcelemental

## Notes

- The functions automatically handle unit conversions (kcal/mol to kJ/mol)
- Progress is saved after each model to prevent data loss
- Temporary data directories are created per model (`data_{model_name}`)
- Crystal lattice energies are computed with proper accounting for symmetry replicas
