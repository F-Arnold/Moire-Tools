# Moire-Tools
Scripts to help with the analysis of moiré systems created from 2D materials
* `plot_interlayer_distance.py`: Plotting the interlayer distance landscape in twisted bilayer MoS2 (also published in https://zenodo.org/records/7243735)
* `analyze_bond_lengths.py`: statistical analysis of bond lengths, plotting results as histograms
* `displace_oop_FHI-aims.py`: displace atoms randomly out-of-plane to break symmetry in geometry files for FHI-aims
* `W90_remap_xsf.py`: reduce files in xsf format created by Wannier90, if they had multiple k-points in one spatial direction, by mapping the Wannier function back to the primitive unit cell
