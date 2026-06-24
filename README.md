# SEA
Scripts and graphics used for the publication of "Macroscopic Order in Block Copolymer Gyroid Films by Solvent Evaporation Annealing"

## Files
### distorted_test_L, distorted_test_R
Distorted cross-sections created with 20240623_create_crosssections.py (PyCharm). The cross-section relations are front and left of a cube (F90, F24, L90, L24) for the distorted_test_L and the front right of a cube (R90, R24, F90, F24) for the distorted_test_L. Distortion parameters are added as .csv files in the folder. 
### rot_test_L
Cross-sections with zero rotation or distortion. The cross-section relations are front and left of a cube (F90, F24, L90, L24)c. Distortion parameters are added as .csv files in the folder.
### b11_l_1
FIB SEM cross-sections of a gyroid sample.
### initial_values
Initial values for the cross-correlation used for 20260611_conv2d_loop_pycharm_cleanup_pass.py (PyCharm).
### example_plot
Example plots coming from 20260611_conv2d_loop_pycharm_cleanup_pass.py (PyCharm), showing one of the problems still occuring in the code. <br />
20260614_conv2d_plot_L1_optimize.png --> single loop using the initial values <br />
20260624_conv2d_plot_L_0_10_optimize.png --> multiple loop using the initial values <br />
20260624_conv2d_plot_L_3_100_optimize.png --> multiple loop using random initial values <br />
20240623_test_12x200_iterations_topsurface_R_binary.png --> scatterplot showing the top surface orientation on a sphere with the coloration depicting the correlation. Yellow dots represent initial values, whereas orange and pink represent basic vectors. Data was created with Spyder, cross-correlation from PyCharm doesn't show the initial values yet. 

## Code Descriptions
### 20240623_create_crosssections.py (PyCharm)
Creating sample cross-sections with specific scaling, rotation, shearing, and fill fraction. The cross-section relations are either front and left of a cube (F90, F24, L90, L24) or at the front right of a cube (R90, R24, F90, F24). Check the chapter in the thesis for more information. PDF of the chapter is uploaded here as 20260624_distortion_analysis_tpms.pdf. 

### 20240713_crosssections_array_data.csv 
20260531_crosscorrelation_loop_pycharm_multiprocessing_optimise_test_rot.py (PyCharm) <br />
Multiprocessing of the cross-correlation of matrices only using rotation as a variable parameter. 

### 20260531_crosscorrelation_loop_pycharm_multiprocessing_optimise_test_rot_sca_she.py 
Multiprocessing of the cross-correlation using scaling, rotation, and shearing as a variable parameters. 

### 20260531_crosscorrelation_loop_spyder_optimise.py (Spyder) 
Optimisation loop of the cross-correlation.

### 20260531_scatterplot_topsurface_plot_best_rot_test.py (Spyder) 
Scatterplot of the top surface with the corresponding correlation.

### 20260611_conv2d_loop_pycharm_cleanup_pass.py (PyCharm) 
Multiprocessing of the cross-correlation turning the matrices and images into tensors, with the choice of choosing 'cpu' or 'cuda'. 

### 20260624_scatterplot_topsurface_plot_pycharm_results_spyder.py (Spyder)
Scatterplot of the top surface with the corresponding correlation of the results calculated with PyCharm.

 
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

