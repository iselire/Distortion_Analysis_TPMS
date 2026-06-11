# -*- coding: utf-8 -*-
"""
Created on Sun May 31 17:43:53 2026

@author: IseliRe
"""

import numpy as np
from numpy import random
import matplotlib.pyplot as plt
import os
import math
import csv

# import imageio
import imageio.v2 as imageio

# scaling the image matrix
import cv2

# to stop script if conditions are not met
import sys

# plotting arrow
from mpl_toolkits.mplot3d import Axes3D

# array to image
from PIL import Image
#text color
import seaborn as sns

from datetime import datetime
import time

##############################################################################    
##############################################################################    
##############################################################################    
##############################################################################

def dir_exists(directory):
    # Check if the directory already exists
    if not os.path.exists(directory):
        # Create the directory
        os.makedirs(directory)
        print("Directory created successfully!")
    else:
        print("Directory already exists!")
        
def read_images_from_folder(folder_path):
    """
    Read 2D images from a folder and return a list of images.

    Parameters:
    - folder_path: Path to the folder containing cross-section images.

    Returns:
    - List of 2D NumPy arrays representing the images.
    """
    image_list = []
    images = []
    for filename in sorted(os.listdir(folder_path)):
        if filename.endswith(".png"):  # Assuming images are in PNG format
            image_path = os.path.join(folder_path, filename)
            img = imageio.imread(image_path, mode='F').astype(np.uint8)
            image_list.append(image_path)
            images.append(img)
    return image_list, images
        
# Function to read data from CSV file and store in arrays
def read_csv(filename):
    # Initialize empty lists to store data
    column_values = []
    row_arr = []
    # Open the CSV file for reading
    with open(filename, 'r') as file:
        # Create a CSV reader object
        csv_reader = csv.reader(file)
        # Iterate over each row in the CSV file
        skip_line = 9
        for row in csv_reader:
            # Assuming the CSV file has two columns
            # Append values from each row to respective lists
            if skip_line == 8 or skip_line == 7 or skip_line == 6 or skip_line == 3:
                print(row)
                row_des = str(row[0]).split(',')[0]
                row_arr = eval(str(row[0]).split('"')[1])
                column_values.append(row_arr)  # Convert to appropriate data type if needed
                skip_line = skip_line - 1
                print(skip_line)
            else:
                skip_line = skip_line - 1
                print(skip_line)
    return column_values

def save_data_csv(tops, topr, topsh, surface_on_top, singlec, topc, runs, time_sf):
    # Specify the file path
    gmt = time.gmtime() 
    file_path = cs + f'/{gmt[0]}{gmt[1]:02}{gmt[2]:02}_crosscorr_arrays_data_{corner}_{num_iterations}_optimize.csv'
    
    # Define column headers
    columns = ['Description', 'Array']
    
    # Define the data to be written
    data = [
        {'Description': 'scaling', 'Array': tops},
        {'Description': 'rotation', 'Array': topr}, 
        {'Description': 'shearing', 'Array': topsh},
        {'Description': 'topsurface', 'Array': surface_on_top},
        {'Description': 'single_correlation', 'Array': singlec},
        {'Description': 'overall_correlation', 'Array': topc},
        {'Description': 'runs_needed', 'Array': runs},
        {'Description': 'time needed', 'Array': time_sf}
    ]
    
    # Write data to CSV file
    with open(file_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=columns)
        writer.writeheader()
        for row in data:
            writer.writerow(row)
            

##############################################################################
# create levelsets, matrices
##############################################################################

def transform_normal(vector, matrix):
    # Extend the 3x1 vector to homogeneous coordinates by adding a fourth dimension with a value of 1
    extended_vector = np.hstack([vector, [1]])
    extended_vector = np.transpose(extended_vector)
    
    # Apply the affine transformation matrix to the extended vector
    m_want = np.linalg.inv(matrix).T
    transformed_vector = np.dot(m_want, extended_vector)
    
    # Extract the rotated 3x1 vector from the transformed vector
    deformed_vector = transformed_vector[:3]
    deformed_vector = deformed_vector/np.linalg.norm(deformed_vector)
    
    return deformed_vector

# Function to find the normal vector to a plane given Miller indices
def miller_indices_to_normal(h, k, l):
    # Find the normal vector to the plane using Miller indices
    normal_vector = np.array([h, k, l])
    return normal_vector / np.linalg.norm(normal_vector)

def make_binary(matrix):
    binary = np.where(matrix < t, 1, 0) * 255
    return binary
 
def level_set_equation_matrix(M):
    
    level_set_matrix0 = np.sin(M[0]) * np.cos(M[1]) + np.sin(M[1]) * np.cos(M[2]) + np.sin(M[2]) * np.cos(M[0])
    level_set_matrix1 = (np.sin(M[4]) * np.cos(M[5]) + np.sin(M[5]) * np.cos(M[6]) + np.sin(M[6]) * np.cos(M[4]))
    level_set_matrix2 = (np.sin(M[8]) * np.cos(M[9]) + np.sin(M[9]) * np.cos(M[10]) + np.sin(M[10]) * np.cos(M[8]))
    level_set_matrix3 = (np.sin(M[12]) * np.cos(M[13]) + np.sin(M[13]) * np.cos(M[14]) + np.sin(M[14]) * np.cos(M[12]))
    level_set_matrix4 = (np.sin(M[16]) * np.cos(M[17]) + np.sin(M[17]) * np.cos(M[18]) + np.sin(M[18]) * np.cos(M[16]))
    level_set_matrix5 = (np.sin(M[20]) * np.cos(M[21]) + np.sin(M[21]) * np.cos(M[22]) + np.sin(M[22]) * np.cos(M[20]))
    level_set_matrix = (level_set_matrix0, level_set_matrix1, level_set_matrix2, level_set_matrix3, level_set_matrix4, level_set_matrix5)
    return level_set_matrix

def level_set_equation_matrix_L(M):
    level_set_matrix0 = np.sin(M[0]) * np.cos(M[1]) + np.sin(M[1]) * np.cos(M[2]) + np.sin(M[2]) * np.cos(M[0])
    level_set_matrix1 = (np.sin(M[4]) * np.cos(M[5]) + np.sin(M[5]) * np.cos(M[6]) + np.sin(M[6]) * np.cos(M[4]))
    level_set_matrix2 = (np.sin(M[8]) * np.cos(M[9]) + np.sin(M[9]) * np.cos(M[10]) + np.sin(M[10]) * np.cos(M[8]))
    level_set_matrix3 = (np.sin(M[12]) * np.cos(M[13]) + np.sin(M[13]) * np.cos(M[14]) + np.sin(M[14]) * np.cos(M[12]))
    level_set_matrix = (level_set_matrix0, level_set_matrix1, level_set_matrix2, level_set_matrix3)
    return level_set_matrix

def level_set_equation_matrix_R(M):
    
    level_set_matrix0 = np.sin(M[0]) * np.cos(M[1]) + np.sin(M[1]) * np.cos(M[2]) + np.sin(M[2]) * np.cos(M[0])
    level_set_matrix1 = (np.sin(M[4]) * np.cos(M[5]) + np.sin(M[5]) * np.cos(M[6]) + np.sin(M[6]) * np.cos(M[4]))
    level_set_matrix4 = (np.sin(M[16]) * np.cos(M[17]) + np.sin(M[17]) * np.cos(M[18]) + np.sin(M[18]) * np.cos(M[16]))
    level_set_matrix5 = (np.sin(M[20]) * np.cos(M[21]) + np.sin(M[21]) * np.cos(M[22]) + np.sin(M[22]) * np.cos(M[20]))
    level_set_matrix = (level_set_matrix0, level_set_matrix1, level_set_matrix2, level_set_matrix3, level_set_matrix4, level_set_matrix5)
    return level_set_matrix
   
# Define rotation matrix function
def rotation_matrix(angle_x, angle_y, angle_z):
    angle_x = float(angle_x)/360 * 2 * math.pi
    angle_y = float(angle_y)/360 * 2 * math.pi
    angle_z = float(angle_z)/360 * 2 * math.pi
    Rx = np.array([[1, 0, 0, 0],
                   [0, np.cos(angle_x), -np.sin(angle_x), 0],
                   [0, np.sin(angle_x), np.cos(angle_x), 0],
                   [0, 0, 0, 1]])
    
    Ry = np.array([[np.cos(angle_y), 0, np.sin(angle_y), 0],
                   [0, 1, 0, 0],
                   [-np.sin(angle_y), 0, np.cos(angle_y), 0],
                   [0, 0, 0, 1]])
    
    Rz = np.array([[np.cos(angle_z), -np.sin(angle_z), 0, 0],
                   [np.sin(angle_z), np.cos(angle_z), 0, 0],
                   [0, 0, 1, 0],
                   [0, 0, 0, 1]])
    
    return Rx.dot(Ry).dot(Rz)

# Define translation matrix function
def translation_matrix(tx, ty, tz):
    return np.array([[1, 0, 0, tx],
                     [0, 1, 0, ty],
                     [0, 0, 1, tz],
                     [0, 0, 0, 1]])

# Define scaling matrix function
def scaling_matrix(sx, sy, sz):
    return np.diag([sx, sy, sz, 1])

# Define shearing matrix function
def shearing_matrix(sxy, sxz, syx, syz, szx, szy):
    return np.array([[1, sxy, sxz, 0],
                     [syx, 1, syz, 0],
                     [szx, szy, 1, 0],
                     [0, 0, 0, 1]])

def create_affine_matrix(angles, translation, scaling, shearing):
    angle_x, angle_y, angle_z = angles      
    tx, ty, tz = translation  
    sx, sy, sz = scaling  
    sxy, sxz, syx, syz, szx, szy = shearing  
    
    # Generate transformation matrices
    R = rotation_matrix(angle_x, angle_y, angle_z)
    T = translation_matrix(tx, ty, tz)
    S = scaling_matrix(sx, sy, sz)
    SH = shearing_matrix(sxy, sxz, syx, syz, szx, szy)
    
    # Combine transformation matrices
    M = T.dot(S).dot(SH).dot(T).dot(R)
    
    return M

def create_4cs_L(v_scaling, v_rotations, v_shearing, grid_size, unitcell, miller):

    # Combine transformation matrices
    # M = T.dot(S).dot(SH).dot(R)
    # scaling
    M1 = create_affine_matrix(angles = (0, 0, 0), translation = (0, 0, 0), scaling = v_scaling, shearing = (0, 0, 0, 0, 0, 0))
    # rotation
    M2 = create_affine_matrix(angles = v_rotations, translation = (0, 0, 0), scaling = (1, 1, 1), shearing = (0, 0, 0, 0, 0, 0))
    # shearing
    M3 = create_affine_matrix(angles = (0, 0, 0), translation = (0, 0, 0), scaling = (1, 1, 1), shearing = v_shearing)
    
    M = (M3).dot(M2).dot(M1)
    
    # Define grid points
    x = np.linspace(-math.pi*grid_size[0]/(unitcell), math.pi*grid_size[0]/(unitcell), grid_size[0])
    y = np.linspace(-math.pi*grid_size[1]/(unitcell), math.pi*grid_size[1]/(unitcell), grid_size[1])
    z = np.linspace(-math.pi*grid_size[2]/(unitcell), math.pi*grid_size[2]/(unitcell), grid_size[2])
    X, Y, Z = np.meshgrid(x, y, z)
    
    
    ##############################################################################
    # Rotation matrix
    ##############################################################################
    
    R2 = create_affine_matrix(angles = (0, 66, 0), translation = (0, 0, 0), scaling = (1, 1, 1), shearing = (0, 0, 0, 0, 0, 0))
    R3 = create_affine_matrix(angles = (0, 0, 90), translation = (0, 0, 0), scaling = (1, 1, 1), shearing = (0, 0, 0, 0, 0, 0))
    R4 = create_affine_matrix(angles = (0, 66, 0), translation = (0, 0, 0), scaling = (1, 1, 1), shearing = (0, 0, 0, 0, 0, 0))
    M4 = np.dot(R3, M)
    
    rot_matrix = np.block([
    [M,np.zeros((4,12))],
    [np.zeros((4,4)),np.dot(R2, M),np.zeros((4,8))],
    [np.zeros((4,8)),np.dot(R3, M),np.zeros((4,4))],
    [np.zeros((4,12)),np.dot(R4, M4)]
        ])
    
    
    ##############################################################################
    # grid matrix
    ##############################################################################
    grid_matrix = np.block([
    [X.flatten()], [Y.flatten()], [Z.flatten()], [np.ones_like(X.flatten())],
    [X.flatten()], [Y.flatten()], [Z.flatten()], [np.ones_like(X.flatten())],
    [X.flatten()], [Y.flatten()], [Z.flatten()], [np.ones_like(X.flatten())],
    [X.flatten()], [Y.flatten()], [Z.flatten()], [np.ones_like(X.flatten())]
    ])
    
    ##############################################################################
    # transformed grid matrix
    ##############################################################################
   
    A = np.dot(rot_matrix, grid_matrix)
    B = level_set_equation_matrix_L(A)
    
    level_set_transformed0 = B[0].reshape(X.shape)
    level_set_transformed1 = B[1].reshape(X.shape)
    level_set_transformed2 = B[2].reshape(X.shape)
    level_set_transformed3 = B[3].reshape(X.shape)
    
    bin_im = (make_binary(level_set_transformed0), make_binary(level_set_transformed1), make_binary(level_set_transformed2), make_binary(level_set_transformed3))
    
    ##############################################################################
    # levelset F090
    label1 = 'F090'
    # Apply transformation to grid points
    norm1 = miller_indices_to_normal(miller[0], miller[1], miller[2])
    norm1 = transform_normal(norm1, M2)
    norm1 = miller_indices_to_normal(norm1[0], norm1[1], norm1[2])
    
    ##############################################################################
    # levelset F024
    label2 = 'F024'
    norm2 = transform_normal(norm1, R2)
    norm2 = miller_indices_to_normal(norm2[0], norm2[1], norm2[2])

    
    ##############################################################################
    # levelset L090
    label3 = 'L090'
    norm3 = transform_normal(norm1, R3)
    norm3 = miller_indices_to_normal(norm3[0], norm3[1], norm3[2])
    
    ##############################################################################
    # levelset L024
    label4 = 'L024'
    norm4 = transform_normal(norm3, R4)
    norm4 = miller_indices_to_normal(norm4[0], norm4[1], norm4[2])
    
    Rtsn = create_affine_matrix(angles = (0, 90, 0), translation = (0, 0, 0), scaling = (1, 1, 1), shearing = (0, 0, 0, 0, 0, 0))
    topsurf_norm = transform_normal(norm1, Rtsn)
    topsurf_norm = miller_indices_to_normal(topsurf_norm[0], topsurf_norm[1], topsurf_norm[2])
    
    norm_images = (norm1, norm2, norm3, norm4)
    label_images = (label1, label2, label3, label4)
    
    return (bin_im, norm_images, label_images, M, A, topsurf_norm)


##############################################################################
# wobbling of the parameters for optimisation
##############################################################################

def wobble(wob_sc, wob_ro, wob_sh, ig):
    # Initialize the best solution and best objective value
    initial_parameter = ig
    wobbled_parameter = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    #print(wobbled_solution)

    # Perform random wobbling iterations
    wobbling_array = (pow(-1, random.randint(10)) * np.random.rand(100)/100 * wob_sc,
    pow(-1, random.randint(10)) * np.random.rand(100)/100 * wob_sc,
    pow(-1, random.randint(10)) * np.random.rand(100)/100 * wob_sc,
    pow(-1, random.randint(10)) * np.random.rand(100)/100 * wob_ro,
    pow(-1, random.randint(10)) * np.random.rand(100)/100 * wob_ro,
    pow(-1, random.randint(10)) * np.random.rand(100)/100 * wob_ro,
    pow(-1, random.randint(10)) * np.random.rand(100)/100 * wob_sh,
    pow(-1, random.randint(10)) * np.random.rand(100)/100 * wob_sh,
    pow(-1, random.randint(10)) * np.random.rand(100)/100 * wob_sh,
    pow(-1, random.randint(10)) * np.random.rand(100)/100 * wob_sh,
    pow(-1, random.randint(10)) * np.random.rand(100)/100 * wob_sh,
    pow(-1, random.randint(10)) * np.random.rand(100)/100 * wob_sh)

    # Apply wobbling to the current solution
    for o in range(0, len(initial_parameter)):
        wobbled_parameter[o] = initial_parameter[o] + wobbling_array[o][0]
        
    return   wobbled_parameter

##############################################################################
# cross correlation
##############################################################################

# Converts a float64 array into a uint8 array without resolution loss
def matrix_to_image(matrix):
    # Scale the float64 image to the range [0, 255]
    image = Image.fromarray(matrix.astype(np.uint8))
    numerator = image - np.min(image)
    denominator = (np.max(image) - np.min(image))
    while denominator < 0:
        denominator = denominator + 1
        print(f'den {denominator}')
    scaled_image = ((numerator / denominator) * 255).astype(np.uint8)
    return scaled_image

def perform_crosscorrelation_left(samples_dir, scaling_square, square_shift, bin_im, 
                                  norm_images, tsn, sample_L24_crop, sample_L90_crop, 
                                  sample_F24_crop, sample_F90_crop, M, v_scaling, 
                                  v_rots, v_shearing, v_translations): 
    
    samples = read_images_from_folder(samples_dir)
    sample_F90 = imageio.imread(samples[0][1], mode='F').astype(np.uint8)
    sample_F90_crop = sample_F90[square_shift[2][0]:(scaling_square + square_shift[2][0]), square_shift[2][1]:(scaling_square + square_shift[2][1])]
    sample_F24 = imageio.imread(samples[0][0], mode='F').astype(np.uint8)
    sample_F24_crop = sample_F24[square_shift[3][0]:(scaling_square + square_shift[3][0]), square_shift[3][1]:(scaling_square + square_shift[3][1])]
    sample_L90 = imageio.imread(samples[0][3], mode='F').astype(np.uint8)
    sample_L90_crop = sample_L90[square_shift[0][0]:(scaling_square + square_shift[0][0]), square_shift[0][1]:(scaling_square + square_shift[0][1])]
    sample_L24 = imageio.imread(samples[0][2], mode='F').astype(np.uint8)
    sample_L24_crop = sample_L24[square_shift[1][0]:(scaling_square + square_shift[1][0]), square_shift[1][1]:(scaling_square + square_shift[1][1])]
      

    top_F90_temp = []
    top_F90_scores_temp = []
    top_F24_temp = []
    top_F24_scores_temp = []
    top_L90_temp = []
    top_L90_scores_temp = []
    top_L24_temp = []
    top_L24_scores_temp = []
    
    top_v_tra = v_translations
    top_v_rot = v_rots
    top_v_sca = v_scaling
    top_v_she = v_shearing
    
    model_matrices_F90_temp = [0]
    model_matrices_F24_temp = [0]
    model_matrices_L90_temp = [0]
    model_matrices_L24_temp = [0]
    
    max_location_F90_temp = [0]
    max_location_F24_temp = [0]
    max_location_L90_temp = [0]
    max_location_L24_temp = [0] 

        
    # Iterate through the translation of the binary matrices left and front
    for k in range(0, norm_translation):
  
        # converting matrices to comparable format
        model_F90 = matrix_to_image(bin_im[0][k])
        model_F24 = matrix_to_image(bin_im[1][k])
        model_L90 = matrix_to_image(bin_im[2][k])
        model_L24 = matrix_to_image(bin_im[3][k])

        result_F90 = cv2.matchTemplate(sample_F90_crop, model_F90, cv2.TM_CCOEFF_NORMED)
        result_F24 = cv2.matchTemplate(sample_F24_crop, model_F24, cv2.TM_CCOEFF_NORMED)
        result_L90 = cv2.matchTemplate(sample_L90_crop, model_L90, cv2.TM_CCOEFF_NORMED)
        result_L24 = cv2.matchTemplate(sample_L24_crop, model_L24, cv2.TM_CCOEFF_NORMED)
        min_val_F90, max_val_F90, min_loc_F90, max_loc_F90 = cv2.minMaxLoc(result_F90)
        min_val_F24, max_val_F24, min_loc_F24, max_loc_F24 = cv2.minMaxLoc(result_F24)
        min_val_L90, max_val_L90, min_loc_L90, max_loc_L90 = cv2.minMaxLoc(result_L90)
        min_val_L24, max_val_L24, min_loc_L24, max_loc_L24 = cv2.minMaxLoc(result_L24)
        
        
        # Check if this translation provides a better match than the top translations
        if k == 0 or max_val_F90 > top_F90_scores_temp:
            top_F90_temp = [M, k]
            top_F90_scores_temp = max_val_F90
            max_location_F90_temp = max_loc_F90
            model_matrices_F90_temp = model_F90
                
        if k == 0 or max_val_F24 > top_F24_scores_temp:
            top_F24_temp = [M, k]
            top_F24_scores_temp = max_val_F24
            max_location_F24_temp = max_loc_F24
            model_matrices_F24_temp = model_F24
    
        if k == 0 or max_val_L90 > top_L90_scores_temp:
            top_L90_temp = [M, k]
            top_L90_scores_temp = max_val_L90
            max_location_L90_temp = max_loc_L90
            model_matrices_L90_temp = model_L90
                
        if k == 0 or max_val_L24 > top_L24_scores_temp:
            top_L24_temp = [M, k]
            top_L24_scores_temp = max_val_L24
            max_location_L24_temp = max_loc_L24
            model_matrices_L24_temp = model_L24

    top_c = (1/np.sqrt(cs_per_volume))*np.sqrt(pow(top_F90_scores_temp, 2) + pow(top_F24_scores_temp, 2) + pow(top_L90_scores_temp, 2) + pow(top_L24_scores_temp, 2))
    top_transf_matrices_tmp = (top_F90_temp, top_F24_temp, top_L90_temp, top_L24_temp)
    top_score_temp = (top_F90_scores_temp, top_F24_scores_temp, top_L90_scores_temp, top_L24_scores_temp)
    max_location = (max_location_F90_temp, max_location_F24_temp, max_location_L90_temp, max_location_L24_temp)
    bin_im = (model_matrices_F90_temp, model_matrices_F24_temp, model_matrices_L90_temp, model_matrices_L24_temp)
    cropped_samples = (sample_F90_crop, sample_F24_crop, sample_L90_crop, sample_L24_crop)
    
    return    top_transf_matrices_tmp, top_score_temp, max_location, bin_im, top_v_sca, top_v_rot, top_v_she, cropped_samples, tsn, top_c
 
    

##############################################################################
# plotting
##############################################################################

def rounddown_0_01(x):
    return int(math.floor(x / 0.01)) * 0.01

def print_top_correlations_left(max_color, min_color, color_palette, tscore, bin_im, cropped_samples, top_v_sc, top_v_ro, top_v_sh, ts, transparency, transparency_sample):
    
    color_scaling = int(round(100*(max_color-min_color), 2))
    color_max = round(rounddown_0_01(max_color), 2)
    color_min = round(rounddown_0_01(min_color), 2)
    # textbox color according to correlation value
    text_values = np.linspace(color_min, color_max, color_scaling)
    # Create a seaborn color palette
    palette = sns.color_palette(color_palette, n_colors=len(text_values))
    ###########################################################################################

    fig_corr, axs_corr = plt.subplots(cs_per_volume, number_attempts, figsize=(30, 40)) 
    fig_corr.suptitle(f'Correlation of {samples_dir}', fontsize=16) 
        
    # Display the original image
    axs_corr = axs_corr.flatten()
    axs_corr[0].set_title('L90')
    axs_corr[1].set_title('L24')
    axs_corr[2].set_title('F90')
    axs_corr[3].set_title('F24') 
    axs_corr[4].set_title('text')    
    
    # Display the "best_combos" 
    for i in range(0, number_attempts):
        
        h, w = bin_im[i][0].shape
        
        iL90 = i*4 + 0
        iL24 = i*4 + 1
        iF90 = i*4 + 2
        iF24 = i*4 + 3
        itext = i*4 + 4
        
        print(f'i is {i}')
        print(color_scaling)
    
        overlay_L90 = cropped_samples[2].copy()  
        overlay_L90 = np.uint8((transparency_sample)*overlay_L90[:,:])
        overlay_L90[final_max_location[i][2][1]:final_max_location[i][2][1]+h, final_max_location[i][2][0]:final_max_location[i][2][0]+w] = np.uint8(transparency*bin_im[i][2] + (1-transparency)*overlay_L90[final_max_location[i][2][1]:final_max_location[i][2][1]+h, final_max_location[i][2][0]:final_max_location[i][2][0]+w])
        axs_corr[iL90].imshow(overlay_L90, cmap='gray')
                    
        if tscore[i][2] <= color_min:
            axs_corr[iL90].text(overlay_L90.shape[1] + round(overlay_L90.shape[1]/20), overlay_L90.shape[0] - round(overlay_L90.shape[0]/20), "%.2f" %tscore[i][2], bbox=dict(facecolor=palette[0], alpha=0.3))
        elif tscore[i][2] >= color_max:
            axs_corr[iL90].text(overlay_L90.shape[1] + round(overlay_L90.shape[1]/20), overlay_L90.shape[0] - round(overlay_L90.shape[0]/20), "%.2f" %tscore[i][2], bbox=dict(facecolor=palette[int(color_scaling - 1)], alpha=0.3))
        else:
            axs_corr[iL90].text(overlay_L90.shape[1] + round(overlay_L90.shape[1]/20), overlay_L90.shape[0] - round(overlay_L90.shape[0]/20), "%.2f" %tscore[i][2], bbox=dict(facecolor=palette[int(rounddown_0_01(tscore[i][2] - color_min)*100)], alpha=0.3))
    
        overlay_L24 = cropped_samples[3].copy()
        overlay_L24[:, :] = np.uint8((transparency_sample)*overlay_L24[:,:])
        overlay_L24[final_max_location[i][3][1]:final_max_location[i][3][1]+h, final_max_location[i][3][0]:final_max_location[i][3][0]+w] = np.uint8(transparency*bin_im[i][3][:,:] + (1-transparency)*overlay_L24[final_max_location[i][3][1]:final_max_location[i][3][1]+h,final_max_location[i][3][0]:final_max_location[i][3][0]+w])
        axs_corr[iL24].imshow(overlay_L24, cmap='gray')
        
        if tscore[i][3] <= color_min:
            axs_corr[iL24].text(overlay_L24.shape[1] + round(overlay_L24.shape[1]/20), overlay_L24.shape[0] - round(overlay_L24.shape[0]/20), "%.2f" %tscore[i][3], bbox=dict(facecolor=palette[0], alpha=0.3))
        elif tscore[i][3] >= color_max:
            axs_corr[iL24].text(overlay_L24.shape[1] + round(overlay_L24.shape[1]/20), overlay_L24.shape[0] - round(overlay_L24.shape[0]/20), "%.2f" %tscore[i][3], bbox=dict(facecolor=palette[int(color_scaling - 1)], alpha=0.3))
        else:
            axs_corr[iL24].text(overlay_L24.shape[1] + round(overlay_L24.shape[1]/20), overlay_L24.shape[0] - round(overlay_L24.shape[0]/20), "%.2f" %tscore[i][3], bbox=dict(facecolor=palette[int(rounddown_0_01(tscore[i][3] - color_min)*100)], alpha=0.3))
    
    
        overlay_F90 = cropped_samples[0].copy()
        overlay_F90[:, :] = np.uint8((transparency_sample)*overlay_F90[:,:])
        overlay_F90[final_max_location[i][0][1]:final_max_location[i][0][1]+h, final_max_location[i][0][0]:final_max_location[i][0][0]+w] = np.uint8(transparency*bin_im[i][0][:,:] + (1-transparency)*overlay_F90[final_max_location[i][0][1]:final_max_location[i][0][1]+h,final_max_location[i][0][0]:final_max_location[i][0][0]+w])
        axs_corr[iF90].imshow(overlay_F90, cmap='gray')
 
        if tscore[i][0] <= color_min:
            axs_corr[iF90].text(overlay_F90.shape[1] + round(overlay_F90.shape[1]/20), overlay_F90.shape[0] - round(overlay_F90.shape[0]/20), "%.2f" %tscore[i][0], bbox=dict(facecolor=palette[0], alpha=0.3))
        elif tscore[i][0] >= color_max:
            axs_corr[iF90].text(overlay_F90.shape[1] + round(overlay_F90.shape[1]/20), overlay_F90.shape[0] - round(overlay_F90.shape[0]/20), "%.2f" %tscore[i][0], bbox=dict(facecolor=palette[int(color_scaling - 1)], alpha=0.3))
        else:
            axs_corr[iF90].text(overlay_F90.shape[1] + round(overlay_F90.shape[1]/20), overlay_F90.shape[0] - round(overlay_F90.shape[0]/20), "%.2f" %tscore[i][0], bbox=dict(facecolor=palette[int(rounddown_0_01(tscore[i][0] - color_min)*100)], alpha=0.3))
    
        
        overlay_F24 = cropped_samples[1].copy()
        overlay_F24[:, :] = np.uint8((transparency_sample)*overlay_F24[:,:])
        overlay_F24[final_max_location[i][1][1]:final_max_location[i][1][1]+h, final_max_location[i][1][0]:final_max_location[i][1][0]+w] = np.uint8(transparency*bin_im[i][1][:,:] + (1-transparency)*overlay_F24[final_max_location[i][1][1]:final_max_location[i][1][1]+h,final_max_location[i][1][0]:final_max_location[i][1][0]+w])
        axs_corr[iF24].imshow(overlay_F24, cmap='gray')
        
        if tscore[i][1] <= color_min:
            axs_corr[iF24].text(overlay_F24.shape[1] + round(overlay_F24.shape[1]/20), overlay_F24.shape[0] - round(overlay_F24.shape[0]/20), "%.2f" %tscore[i][1], bbox=dict(facecolor=palette[0], alpha=0.3))
        elif tscore[i][1] >= color_max:
            axs_corr[iF24].text(overlay_F24.shape[1] + round(overlay_F24.shape[1]/20), overlay_F24.shape[0] - round(overlay_F24.shape[0]/20), "%.2f" %tscore[i][1], bbox=dict(facecolor=palette[int(color_scaling - 1)], alpha=0.3))
        else:
            axs_corr[iF24].text(overlay_F24.shape[1] + round(overlay_F24.shape[1]/20), overlay_F24.shape[0] - round(overlay_F24.shape[0]/20), "%.2f" %tscore[i][1], bbox=dict(facecolor=palette[int(rounddown_0_01(tscore[i][1] - color_min)*100)], alpha=0.3))
        
        top_corr = 1/2*np.sqrt(pow(tscore[i][2], 2) + pow(tscore[i][3], 2) + pow(tscore[i][0], 2) + pow(tscore[i][1], 2))
        axs_corr[iF24].text(overlay_F24.shape[1] + round(overlay_F24.shape[0]/20), overlay_F24.shape[0] - round(overlay_F24.shape[0]/2.8), 
                            f'scaling \n {np.round(top_v_sc[i][0], 2)} \n {np.round(top_v_sc[i][1], 2)} \n {np.round(top_v_sc[i][2], 2)}' + 
        f'\nrotations \n {np.round(top_v_ro[i][0], 2)} \n {np.round(top_v_ro[i][1], 2)} \n {np.round(top_v_ro[i][2], 2)}')
        axs_corr[iF24].text(overlay_F24.shape[1] + round(overlay_F24.shape[0]/20)+100, overlay_F24.shape[0] - 5, 
                            f'shearing \n {np.round(top_v_sh[i][0], 2)} \n {np.round(top_v_sh[i][1], 2)} \n {np.round(top_v_sh[i][2], 2)}'+
        f'\n {np.round(top_v_sh[i][3], 2)} \n {np.round(top_v_sh[i][4], 2)} \n {np.round(top_v_sh[i][5], 2)}'+
        f'\ntopsurface \nh {rounddown_0_01(ts[i][0])} \nk {rounddown_0_01(ts[i][1])} \nl {rounddown_0_01(ts[i][2])}'+
        f'\ntotal corr {np.round(top_corr, 2)}')
        
        print(ts)
        if top_corr <= color_min:
            axs_corr[iF24].text(overlay_F24.shape[1] + round(overlay_F24.shape[0]/20), overlay_F24.shape[0] - round(overlay_F24.shape[0]/5), "%.2f" %top_corr, bbox=dict(facecolor=palette[0], alpha=0.3))
        elif top_corr >= color_max:
            axs_corr[iF24].text(overlay_F24.shape[1] + round(overlay_F24.shape[0]/20), overlay_F24.shape[0] - round(overlay_F24.shape[0]/5), "%.2f" %top_corr, bbox=dict(facecolor=palette[int(color_scaling - 1)], alpha=0.3))
        else:
            axs_corr[iF24].text(overlay_F24.shape[1] + round(overlay_F24.shape[0]/20), overlay_F24.shape[0] - round(overlay_F24.shape[0]/5), "%.2f" %top_corr, bbox=dict(facecolor=palette[int(round(tscore[i][1] - color_min, 2)*100)-1], alpha=0.3))

    # record current timestamp
    gmt = time.gmtime()        
    plt.savefig(directory + f'/{gmt[0]}{gmt[1]:02}{gmt[2]:02}_crosscorr_plot_{corner}_{num_iterations}_optimize' + '.png', dpi = 300)
    plt.savefig(directory + f'/{gmt[0]}{gmt[1]:02}{gmt[2]:02}_crosscorr_plot_{corner}_{num_iterations}_optimize' + '.svg', dpi = 900)
 
def plot_4cs(bin_im, label_images, directory, u):

    bin_im_np_orig = [0, 0, 0, 0]

    bin_im_np_orig[0] = bin_im[0]
    # plt.imshow(bin_im_np_orig[0])
    cv2.imwrite(directory + f'/{label_images[0]}_crosscorr_match_{u}_{num_iterations}_optimize.png', bin_im_np_orig[0])
    
    bin_im_np_orig[1] = bin_im[1]
    # plt.imshow(bin_im_np_orig[1])
    cv2.imwrite(directory + f'/{label_images[1]}_crosscorr_match_{u}_{num_iterations}_optimize.png', bin_im_np_orig[1])
    
    bin_im_np_orig[2] = bin_im[2]
    # plt.imshow(bin_im_np_orig[2])
    cv2.imwrite(directory + f'/{label_images[2]}_crosscorr_match_{u}_{num_iterations}_optimize.png', bin_im_np_orig[2])

    bin_im_np_orig[3] = bin_im[3]
    # plt.imshow(bin_im_np_orig[3])
    cv2.imwrite(directory + f'/{label_images[3]}_crosscorr_match_{u}_{num_iterations}_optimize.png', bin_im_np_orig[3])
  
    

##############################################################################    
##############################################################################    
##############################################################################    
##############################################################################


# directory for saving the files
cs = "C:/Users/iselire/OneDrive/studium/unifr/solvent_evaporation_annealing/codes/morphology/spyder/tests/20260529_test"
directory = cs
# samples should be named sample_F24 etc. or F24 etc.
samples_dir = "C:/Users/iselire/OneDrive/studium/unifr/solvent_evaporation_annealing/codes/morphology/spyder/b11_L_test"
dir_exists(cs)

# where initial values are stored
initial_values = "C:/Users/iselire/OneDrive/studium/unifr/solvent_evaporation_annealing/codes/morphology/spyder/tests/20260529_test/20260528_crosscorr_arrays_inital_values.csv"

# Fill fraction, max value of levelset is 1.4990768364186535
ff = 0.40
if ff >= 0.5:
    t = 0 + (math.sin((ff-0.5) * math.pi/2)) * 1.4990768364186535
elif ff <= 0.5:
    t = 0 - (math.sin((ff) * math.pi/2)) * 1.4990768364186535 
    
elif ff > 1 or ff < 0:
    print('The fill fraction needs to be between 0 and 1')
    sys.exit()
    
threshold = 0.5

# Defining the original basis vectors (example values)
original_basis_vectors = np.array([[1, 0, 0],
                                   [0, 1, 0],
                                   [0, 0, 1]])

miller = (1, 0, 0)
unitcell = 40
grid_size = (80, 80, 80)
norm_translation = 40

best_combos = 1
number_attempts = 4
cs_per_volume = 4

time_start = time.time() 
   
############################################################################## 
# Sample images settings   
##############################################################################

# corner = input("left (L) or right (R) corner of the sample? ") 
corner = 'L'
print(corner) 

# x, y - shift per image
# L24, L90, F24, F90 or F24, F90, R24, R90
square_shift = [(0, 0), (0, 0), (0, 0), (0, 0)]

# Read in sample image
samples = read_images_from_folder(samples_dir)
scaling_square = 200

image_F90 = Image.open(samples[0][1])
sample_F90_crop = image_F90.crop((square_shift[2][0], square_shift[2][1], (scaling_square + square_shift[2][0]),
                               (scaling_square + square_shift[2][1])))

image_F24 = Image.open(samples[0][0])
sample_F24_crop = image_F24.crop((square_shift[3][0], square_shift[3][1], (scaling_square + square_shift[3][0]),
                               (scaling_square + square_shift[3][1])))

image_L90 = Image.open(samples[0][3])
sample_L90_crop = image_L90.crop((square_shift[0][0], square_shift[0][1], (scaling_square + square_shift[0][0]),
                               (scaling_square + square_shift[0][1])))

image_L24 = Image.open(samples[0][2])
sample_L24_crop = image_L24.crop((square_shift[1][0], square_shift[1][1], (scaling_square + square_shift[1][0]),
                               (scaling_square + square_shift[1][1])))


############################################################################## 
# Starting values for the iteration and arrays for the correlation parameters
##############################################################################

iv_arrays = read_csv(initial_values)

top_F90 = [0]
top_F90_scores = [0]
top_F24 = [0]
top_F24_scores = [0]
top_L90 = [0]
top_L90_scores = [0]
top_L24 = [0]
top_L24_scores = [0]
top_R90 = [0]
top_R90_scores = [0]
top_R24 = [0]
top_R24_scores = [0]

max_location_F90 = [0]
max_location_F24 = [0]
max_location_L90 = [0]
max_location_L24 = [0]
max_location_R90 = [0]
max_location_R24 = [0]

model_matrices_F90 = [0]
model_matrices_F24 = [0]
model_matrices_L90 = [0]
model_matrices_L24 = [0]
model_matrices_R90 = [0]
model_matrices_R24 = [0]

top_v_rotations = []
top_v_translations = []
top_v_scaling = []
top_v_shearing = []
top_corr = [0, 0, 0, 0]

top_label = [0]

temp_translations = [0, 0, 0]
temp_scaling = [0, 0, 0]
temp_rotations = [0, 0, 0]
temp_shear = [0, 0, 0, 0, 0, 0]


# Define the number of iterations and the wobbling factor
num_iterations = 100
wob_scale = 1
wob_rot = 20
wob_shear = 0.2

############################################################################## 
# cross correlation
##############################################################################

finalscale = []
finalrot = []
finalshear = []
final_model_matrices = []
final_binary_images = []
final_norm_images = []
final_topsurfaces = []
final_M = []
final_max_location = []
final_top_transf_matrices = []
final_top_scores = []
final_top_correlation = []

for i in range(0, number_attempts):      
    iv_input_loop = [iv_arrays[0][0 + i][0], iv_arrays[0][0 + i][1], iv_arrays[0][0 + i][2], iv_arrays[1][0 + i][0], iv_arrays[1][0 + i][1], iv_arrays[1][0 + i][2],
    iv_arrays[2][0 + i][0], iv_arrays[2][0 + i][1], iv_arrays[2][0 + i][2], iv_arrays[2][0 + i][3], iv_arrays[2][0 + i][4], iv_arrays[2][0 + i][5]]

    temp_scaling = iv_input_loop[:-9]
    temp_rotations = iv_input_loop[3:-6]
    temp_shear = iv_input_loop[6:]
    
    for j in range(0, num_iterations):
        wobbled_parameters = wobble(wob_scale, wob_rot, wob_shear, iv_input_loop)
        
        temp_scaling = wobbled_parameters[:-9]
        temp_rotations = wobbled_parameters[3:-6]
        temp_shear = wobbled_parameters[6:]
        
        binary_images, norm_images, labels, M, A, topsurface = create_4cs_L(temp_scaling, temp_rotations, temp_shear, grid_size,
                                                                                unitcell, miller)
        top_transf_matrices, top_score, max_location, model_matrices, top_v_scaling, top_v_rotations, top_v_shearing, cropped_samples, topsurfaces, top_corr_a = perform_crosscorrelation_left(
            samples_dir, scaling_square, square_shift, binary_images, norm_images, topsurface, sample_L24_crop, sample_L90_crop, sample_F24_crop, sample_F90_crop, M, temp_scaling, temp_rotations, temp_shear, temp_translations)
        
        if top_corr_a > top_corr[i]:
            for j in range(0, 3):
                iv_input_loop[j] = float(top_v_scaling[j])
                iv_input_loop[j + 3] = float(top_v_rotations[j])
            for j in range(0, 6):
                iv_input_loop[j + 6] = float(top_v_shearing[j])
            top_corr[i] = [float(top_corr_a)]
        
        
        percentage = rounddown_0_01((i*num_iterations + j)/(number_attempts*num_iterations)*100)
        time_end = time.time()
    
        # Calculate the elapsed time
        elapsed_time_seconds = time_end - time_start
    
        # Convert elapsed time to days, hours, minutes, and seconds
        days = int(elapsed_time_seconds // (24 * 3600))
        hours = int((elapsed_time_seconds % (24 * 3600)) // 3600)
        minutes = int((elapsed_time_seconds % 3600) // 60)
        seconds = int(elapsed_time_seconds % 60)
    
        time_sofar_used = f'elapsed time: {days}d, {hours}h, {minutes}min, {seconds}s'
        
        estimated_seconds = elapsed_time_seconds/(percentage + 0.01)*100
        # Convert elapsed time to days, hours, minutes, and seconds
        days = int(estimated_seconds // (24 * 3600))
        hours = int((estimated_seconds % (24 * 3600)) // 3600)
        minutes = int((estimated_seconds % 3600) // 60)
        seconds = int(estimated_seconds % 60)
    
        estimated_time = f'estimated time: {days}d, {hours}h, {minutes}min, {seconds}s'
        print(str(percentage) + '%')
            
    temp_scaling = iv_input_loop[:-9]
    temp_rotations = iv_input_loop[3:-6]
    temp_shear = iv_input_loop[6:]
            
    # performing crosscorrelation of the model with the 4 FIB SEM cuts
    finalscale.append(temp_scaling)
    finalrot.append(temp_rotations)
    finalshear.append(temp_shear)
    final_model_matrices.append(model_matrices)
    final_topsurfaces.append([float(topsurface[0]), float(topsurface[1]), float(topsurface[2])])
    final_max_location.append(max_location)
    final_top_transf_matrices.append(top_transf_matrices)
    final_top_correlation.append(top_corr[i])
    final_top_scores.append([top_score[0], top_score[1], top_score[2], top_score[3]])
    
    time_end = time.time()
    time_sofar = [time_start, time_end]
    
    # Calculate the elapsed time
    elapsed_time_seconds = time_end - time_start
    
    # Convert elapsed time to days, hours, minutes, and seconds
    days = int(elapsed_time_seconds // (24 * 3600))
    hours = int((elapsed_time_seconds % (24 * 3600)) // 3600)
    minutes = int((elapsed_time_seconds % 3600) // 60)
    seconds = int(elapsed_time_seconds % 60)
    
    time_sofar_used = f'{days}d, {hours}h, {minutes}min, {seconds}s'
    
    save_data_csv(finalscale, finalrot, finalshear, final_topsurfaces, final_top_scores, final_top_correlation, time_sofar_used, num_iterations)
    print(f'corr is: {final_top_correlation}')
    print(f'{time_sofar_used} of {estimated_time}')   

############################################################################## 
# plotting 
##############################################################################

# color coding
max_color = 1
min_color = 0
color_palette = ("viridis")
transparency_overlay = 0.1
transparency_sample = 0.2

if corner == 'L':  
    print_top_correlations_left(max_color, min_color, color_palette, final_top_scores, final_model_matrices, cropped_samples, finalscale, finalrot, finalshear, final_topsurfaces, transparency_overlay, transparency_sample)

for i in range(0, number_attempts):
    plot_4cs(final_model_matrices[i], labels, directory, i)
