# SEA
Scripts and graphics used for the publication of "Macroscopic Order in Block Copolymer Gyroid Films by Solvent Evaporation Annealing"

## Files
distorted_test_L, distorted_test_R <br />
Distorted cross-sections created with 20240623_create_crosssections.py (PyCharm). The cross-section relations are front and left of a cube (F90, F24, L90, L24) for the distorted_test_L and the front right of a cube (R90, R24, F90, F24) for the distorted_test_L. Distortion parameters are added as .csv files in the folder. 
<br />
<br />
rot_test_L <br />
Cross-sections with zero rotation or distortion. The cross-section relations are front and left of a cube (F90, F24, L90, L24)c. Distortion parameters are added as .csv files in the folder.
<br />
<br />
b11_l_1 <br />
FIB SEM cross-sections of a gyroid sample.
<br />
<br />
example_plot <br />
Example plots coming from 20260611_conv2d_loop_pycharm_cleanup_pass.py (PyCharm), single loop using the initial values.
<br />
<br />
## Code Descriptions
20240623_create_crosssections.py (PyCharm) <br />
Creating sample cross-sections with specific scaling, rotation, shearing, and fill fraction. The cross-section relations are either front and left of a cube (F90, F24, L90, L24) or at the front right of a cube (R90, R24, F90, F24). Check the chapter in the thesis for more information. PDF of the chapter is uploaded here as 20260624_distortion_analysis_tpms.pdf. 
<br />
<br />
20240713_crosssections_array_data.csv <br />
 <br />
20260531_crosscorrelation_loop_pycharm_multiprocessing_optimise_test_rot.py (PyCharm) <br />
Multiprocessing of the cross-correlation of matrices only using rotation as a variable parameter. 
 <br />
 <br />
 20260531_crosscorrelation_loop_pycharm_multiprocessing_optimise_test_rot_sca_she.py <br />
 Multiprocessing of the cross-correlation using scaling, rotation, and shearing as a variable parameters. 
 <br />
 <br />
 20260531_crosscorrelation_loop_spyder_optimise.py (Spyder) <br />
 Optimisation loop of the cross-correlation.
 <br />
 <br />
 20260531_scatterplot_topsurface_plot_best_rot_test.py (Spyder) <br />
 Scatterplot of the top surface with the corresponding correlation.
 <br />
 <br />
 20260611_conv2d_loop_pycharm_cleanup_pass.py (PyCharm) <br />
Multiprocessing of the cross-correlation turning the matrices and images into tensors, with the choice of choosing 'cpu' or 'cuda'. 
 <br />
 <br />
 20260624_scatterplot_topsurface_plot_pycharm_results_spyder.py (Spyder) <br />
 Scatterplot of the top surface with the corresponding correlation of the results calculated with PyCharm.
 <br />
 <br />
 
## Python environment for Spyder (Conda: anaconda3 (Python 3.13.5)):
Python 3.13.5 | packaged by Anaconda, Inc. | 
(main, Jun 12 2025, 16:37:03) [MSC v.1929 64 bit (AMD64)]
Type "copyright", "credits" or "license" for more information.

IPython 8.30.0 -- An enhanced Interactive Python. Type '?' for help.

## Python environment for PyCharm 
Requirements needed for the files <br />
20260531_crosscorrelation_loop_pycharm_multiprocessing_optimise_test_rot_sca_she.py <br />
20260531_crosscorrelation_loop_pycharm_multiprocessing_optimise_test_rot.py <br />
20240623_create_crosssections.py <br />

are listed in "pycharm_venv_requirements_python_3.13.txt" for Python version 3.13.

