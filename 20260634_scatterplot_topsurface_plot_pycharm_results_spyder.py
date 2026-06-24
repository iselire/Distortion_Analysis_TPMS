# -*- coding: utf-8 -*-
"""
Created on Fri May 31 00:50:59 2024

@author: IseliRe
"""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

import os
import pandas as pd
import csv
import ast
import time
import math

# import imageio
import imageio.v2 as imageio

# array to image
from PIL import Image

# scaling the image matrix
import cv2

#text color
import seaborn as sns
########################################################

def read_csv(filename):
    # Initialize empty lists to store data
    column_values1 = []
    column_values2 = []
    column_values3 = []
    column_values4 = []
    column_values5 = []
    column_values6 = []
    column_values7 = []
    column_values8 = []
    row_arr = []
    # Open the CSV file for reading
    with open(filename, 'r') as file:
        # Create a CSV reader object
        csv_reader = csv.reader(file)
        # Iterate over each row in the CSV file
        skip_line = 0
        for row in csv_reader:
            # Assuming the CSV file has two columns
            # Append values from each row to respective lists
            if skip_line == 1:
                # print(row)
                row_des = row[0]
                row_arr = row[1]
                column_values5.append(row_arr)  # Convert to appropriate data type if needed
                skip_line = skip_line + 1
                # print(skip_line)
                
                
            elif skip_line == 2:
                # print(row)
                row_des = row[0]
                row_arr = row[1]
                column_values6.append(row_arr)  # Convert to appropriate data type if needed
                skip_line = skip_line + 1
                # print(skip_line)
                
                
            elif skip_line == 3:
                # print(row)
                row_des = row[0]
                row_arr = row[1]
                column_values7.append(row_arr)  # Convert to appropriate data type if needed
                skip_line = skip_line + 1
                # print(skip_line)
                
                
            elif skip_line == 4:
                # print(row)
                row_des = row[0]
                row_arr = row[1]
                column_values1.append(row_arr)  # Convert to appropriate data type if needed
                skip_line = skip_line + 1
                # print(skip_line)
                
            elif skip_line == 6:
                # print(row)
                row_des = row[0]
                row_arr = row[1]
                column_values2.append(row_arr)  # Convert to appropriate data type if needed
                skip_line = skip_line + 1
                # print(skip_line)
            elif skip_line == 9:
                # print(row)
                row_des = row[0]
                row_arr = row[1]
                column_values3.append(row_arr)  # Convert to appropriate data type if needed
                skip_line = skip_line + 1
                # print(skip_line)
                
            elif skip_line == 5:
                # print(row)
                row_des = row[0]
                row_arr = row[1]
                column_values4.append(row_arr)  # Convert to appropriate data type if needed
                skip_line = skip_line + 1
                # print(skip_line)
                

            elif skip_line == 8:
                # print(row)
                row_des = row[0]
                row_arr = row[1]
                column_values8.append(-0.8)  # Convert to appropriate data type if needed
                skip_line = skip_line + 1
                # print(skip_line)

            # elif skip_line == 10:
            #     # print(row)
            #     row_des = row[0]
            #     row_arr = row[1]
            #     column_values8.append(row_arr)  # Convert to appropriate data type if needed
            #     skip_line = skip_line + 1
            #     # print(skip_line)
                
            else:
                skip_line = skip_line + 1
                # print(skip_line)
    return column_values1, column_values2, column_values3, column_values4, column_values5, column_values6, column_values7, column_values8


def dir_exists(directory):
    # Check if the directory already exists
    if not os.path.exists(directory):
        # Create the directory
        os.makedirs(directory)
        print("Directory created successfully!")
    else:
        print("Directory already exists!")
        
        
def read_all_csv_files(root_folder):
    all_dataframes_1 = []
    all_dataframes_2 = []
    all_dataframes_3 = []
    all_dataframes_4 = []
    all_dataframes_5 = []
    all_dataframes_6 = []
    all_dataframes_7 = []
    all_dataframes_8 = []
    
    # Walk through all subdirectories
    for subdir, _, files in os.walk(root_folder):
        for file in files:
            if file.endswith('.csv'):
                file_path = os.path.join(subdir, file)
                # print(file_path)
                arr_1, arr_2, arr_3, arr_4, arr_5, arr_6, arr_7, arr_8 = read_csv(file_path)
                all_dataframes_1.append((file_path, arr_1))
                all_dataframes_2.append((file_path, arr_2))
                all_dataframes_3.append((file_path, arr_3))
                all_dataframes_4.append((file_path, arr_4))
                all_dataframes_5.append((file_path, arr_5))
                all_dataframes_6.append((file_path, arr_6))
                all_dataframes_7.append((file_path, arr_7))
                all_dataframes_8.append((file_path, arr_8))
    
    return all_dataframes_1, all_dataframes_2, all_dataframes_3, all_dataframes_4, all_dataframes_5, all_dataframes_6, all_dataframes_7, all_dataframes_8


def topsurface_to_array(array_string):
    if isinstance(array_string, str):
        # Clean the string to make it evaluable
        clean_string = array_string.replace("array('[", "[").replace("])']", "]]").replace("array(", "").replace(")", "")
        
        # Use ast.literal_eval to evaluate the cleaned string as a Python expression
        parsed_list = ast.literal_eval(clean_string)
        
        # Convert the parsed list of lists to a list of NumPy arrays
        array_list = [np.array(arr) for arr in parsed_list]
        
        return array_list
    elif isinstance(array_string, np.ndarray):
        return array_string
    else:
        raise ValueError("Input must be a string or a numpy array")
        
def overallcorr_to_array(array_string, skip_first):
    if isinstance(array_string, str):
        
        # print('now')
        # print(array_string)
        # print(array_string[4:-1])
        # Clean the string to make it evaluable
        clean_string = [array_string]
        
        # Use ast.literal_eval to evaluate the cleaned string as a Python expression
        parsed_list = ast.literal_eval(array_string)
        # parsed_list = (str(clean_string).split(', '))
        
        if skip_first == 'yes':
            parsed_list = parsed_list[1:]
        
        # print(parsed_list)
        # Convert the parsed list of lists to a list of NumPy arrays
        array_list = [np.array(arr) for arr in parsed_list]
        
        return array_list
    elif isinstance(array_string, np.ndarray):
        return array_string
    else:
        raise ValueError("Input must be a string or a numpy array")
        
def vectorstring_to_array(array_string):
    if isinstance(array_string, str):
        # Clean the string to make it evaluable
        clean_string = array_string.replace("])", ']').replace("array([", '[').replace("'[", '[').replace("]'", "]").replace("[", "(").replace("]", ")")
        
        # Use ast.literal_eval to evaluate the cleaned string as a Python expression
        parsed_list = ast.literal_eval(clean_string)
        
        # Convert the parsed list of lists to a list of NumPy arrays
        array_list = [np.array(arr) for arr in parsed_list]
        
        return array_list
    elif isinstance(array_string, np.ndarray):
        return array_string
    else:
        raise ValueError("Input must be a string or a numpy array")
        
def string_to_array(s):
    return np.array(ast.literal_eval(s.replace('array', '')))


def perform_crosscorrelation_left(samples_dir, scaling_square, square_shift, bin_im, norm_images, tsn, M, 
                                  tF90s, tmF90, mlF90, mmF90,
                                  tF24s, tmF24, mlF24, mmF24,
                                  tL90s, tmL90, mlL90, mmL90,
                                  tL24s, tmL24, mlL24, mmL24, 
                                  tvro, tvtr, tvsc, tvsh, tl, tcarr,
                                  vro, vtr, vsc, vsh): 
    # Read in sample and model image
    samples = read_images_from_folder(samples_dir)
    sample_F90 = imageio.imread(samples[0][1], as_gray=True).astype(np.uint8)
    sample_F90_crop = sample_F90[square_shift[2][0]:(scaling_square + square_shift[2][0]), square_shift[2][1]:(scaling_square + square_shift[2][1])]
    sample_F24 = imageio.imread(samples[0][0], as_gray=True).astype(np.uint8)
    sample_F24_crop = sample_F24[square_shift[3][0]:(scaling_square + square_shift[3][0]), square_shift[3][1]:(scaling_square + square_shift[3][1])]
    sample_L90 = imageio.imread(samples[0][3], as_gray=True).astype(np.uint8)
    sample_L90_crop = sample_L90[square_shift[0][0]:(scaling_square + square_shift[0][0]), square_shift[0][1]:(scaling_square + square_shift[0][1])]
    sample_L24 = imageio.imread(samples[0][2], as_gray=True).astype(np.uint8)
    sample_L24_crop = sample_L24[square_shift[1][0]:(scaling_square + square_shift[1][0]), square_shift[1][1]:(scaling_square + square_shift[1][1])]
      
    top_parameters_temp = [0]
    top_parameters_scores_temp = [0]
    top_translation_temp = [0]
    top_translation_scores_temp = [0]
    top_F90_temp = [0]
    top_F90_scores_temp = [0]
    top_F24_temp = [0]
    top_F24_scores_temp = [0]
    top_L90_temp = [0]
    top_L90_scores_temp = [0]
    top_L24_temp = [0]
    top_L24_scores_temp = [0]
    top_R90_temp = [0]
    top_R90_scores_temp = [0]
    top_R24_temp = [0]
    top_R24_scores_temp = [0]
    
    model_matrices_F90_temp = [0]
    model_matrices_F24_temp = [0]
    model_matrices_L90_temp = [0]
    model_matrices_L24_temp = [0]
    model_matrices_R90_temp = [0]
    model_matrices_R24_temp = [0]
    
    max_location_F90_temp = [0]
    max_location_F24_temp = [0]
    max_location_L90_temp = [0]
    max_location_L24_temp = [0] 
    max_location_R90_temp = [0]
    max_location_R24_temp = [0]
    

        
    # Iterate through the translation of the binary matrices left and front
    for k in range(0, translations_per_cut):
  
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
        if max_val_F90 > min(top_F90_scores_temp):
            top_F90_temp.append([M, k])
            top_F90_scores_temp.append(max_val_F90)
            max_location_F90_temp.append(max_loc_F90)
            model_matrices_F90_temp.append(model_F90)
            
            # Keep only the top three rotations
            if len(top_F90_scores_temp) > best_combos:
                min_index = top_F90_scores_temp.index(min(top_F90_scores_temp))
                del top_F90_temp[min_index]
                del top_F90_scores_temp[min_index]
                del max_location_F90_temp[min_index]
                del model_matrices_F90_temp[min_index]
                
        if max_val_F24 > min(top_F24_scores_temp):
            top_F24_temp.append([M, k])
            top_F24_scores_temp.append(max_val_F24)
            max_location_F24_temp.append(max_loc_F24)
            model_matrices_F24_temp.append(model_F24)
            
            # Keep only the top three rotations
            if len(top_F24_scores_temp) > best_combos:
                min_index = top_F24_scores_temp.index(min(top_F24_scores_temp))
                del top_F24_temp[min_index]
                del top_F24_scores_temp[min_index]
                del max_location_F24_temp[min_index]
                del model_matrices_F24_temp[min_index]
    
        if max_val_L90 > min(top_L90_scores_temp):
            top_L90_temp.append([M, k])
            top_L90_scores_temp.append(max_val_L90)
            max_location_L90_temp.append(max_loc_L90)
            model_matrices_L90_temp.append(model_L90)
            
            # Keep only the top three rotations
            if len(top_L90_scores_temp) > best_combos:
                min_index = top_L90_scores_temp.index(min(top_L90_scores_temp))
                del top_L90_temp[min_index]
                del top_L90_scores_temp[min_index]
                del max_location_L90_temp[min_index]
                del model_matrices_L90_temp[min_index]
                
        if max_val_L24 > min(top_L24_scores_temp):
            top_L24_temp.append([M, k])
            top_L24_scores_temp.append(max_val_L24)
            max_location_L24_temp.append(max_loc_L24)
            model_matrices_L24_temp.append(model_L24)
            
            # Keep only the top three rotations
            if len(top_L24_scores_temp) > best_combos:
                min_index = top_L24_scores_temp.index(min(top_L24_scores_temp))
                del top_L24_temp[min_index]
                del top_L24_scores_temp[min_index]
                del max_location_L24_temp[min_index]
                del model_matrices_L24_temp[min_index]
     
    max_F90 = top_F90_scores_temp.index(max(top_F90_scores_temp)) 
    max_F24 = top_F24_scores_temp.index(max(top_F24_scores_temp))
    max_L90 = top_L90_scores_temp.index(max(top_L90_scores_temp)) 
    max_L24 = top_L24_scores_temp.index(max(top_L24_scores_temp))          
    a = (1/np.sqrt(cs_per_volume))*np.sqrt(pow(top_F90_scores_temp[max_F90], 2) + pow(top_F24_scores_temp[max_F24], 2) + pow(top_L90_scores_temp[max_L90], 2) + pow(top_L24_scores_temp[max_L24], 2))
 
    if len(tF90s) <= best_combos and a > max(tcarr): 
        
        max_F90 = top_F90_scores_temp.index(max(top_F90_scores_temp))      
        tmF90.append(top_F90_temp[max_F90])
        tF90s.append(top_F90_scores_temp[max_F90])
        mlF90.append(max_location_F90_temp[max_F90])
        mmF90.append(model_matrices_F90_temp[max_F90]) 
        
        max_F24 = top_F24_scores_temp.index(max(top_F24_scores_temp))
        tmF24.append(top_F24_temp[max_F24])
        tF24s.append(top_F24_scores_temp[max_F24])
        mlF24.append(max_location_F24_temp[max_F24])
        mmF24.append(model_matrices_F24_temp[max_F24]) 
        
        max_L90 = top_L90_scores_temp.index(max(top_L90_scores_temp))
        tmL90.append(top_L90_temp[max_L90])
        tL90s.append(top_L90_scores_temp[max_L90])
        mlL90.append(max_location_L90_temp[max_L90])
        mmL90.append(model_matrices_L90_temp[max_L90]) 
        
        max_L24 = top_L24_scores_temp.index(max(top_L24_scores_temp))
        tmL24.append(top_L24_temp[max_L24])
        tL24s.append(top_L24_scores_temp[max_L24])
        mlL24.append(max_location_L24_temp[max_L24])
        mmL24.append(model_matrices_L24_temp[max_L24]) 
        
        tvro.append(vro)
        tvtr.append(vtr)
        tvsc.append(vsc)
        tvsh.append(vsh)
        tl.append(tsn)
        
        a = (1/np.sqrt(cs_per_volume))*np.sqrt(pow(top_F90_scores_temp[max_F90], 2) + pow(top_F24_scores_temp[max_F24], 2) + pow(top_L90_scores_temp[max_L90], 2) + pow(top_L24_scores_temp[max_L24], 2))
        tcarr.append(a)
        
        
    if len(tF90s) > best_combos:            
        # Keep only the top combos
        min_index = tcarr.index(min(tcarr))
        del tmF90[min_index]
        del tF90s[min_index]
        del mlF90[min_index]
        del mmF90[min_index]
        
        del tmF24[min_index]
        del tF24s[min_index]
        del mlF24[min_index]
        del mmF24[min_index]
        
        del tmL90[min_index]
        del tL90s[min_index]
        del mlL90[min_index]
        del mmL90[min_index]
        
        del tmL24[min_index]
        del tL24s[min_index]
        del mlL24[min_index]
        del mmL24[min_index]
        
        del tvro[min_index]
        del tvtr[min_index]
        del tvsc[min_index]
        del tvsh[min_index]
        del tl[min_index]
        del tcarr[min_index]
        

    top_c = (1/np.sqrt(cs_per_volume))*np.sqrt(pow(max(top_F90_scores_temp), 2) + pow(max(top_F24_scores_temp), 2) + pow(max(top_L90_scores_temp), 2) + pow(max(top_L24_scores_temp), 2))
    top_transf_matrices = (tmF90, tmF24, tmL90, tmL24)
    top_score = (tF90s, tF24s, tL90s, tL24s)
    max_location = (mlF90, mlF24, mlL90, mlL24)
    bin_im = (mmF90, mmF24, mmL90, mmL24)
    cropped_samples = (sample_F90_crop, sample_F24_crop, sample_L90_crop, sample_L24_crop)
    
    return    top_transf_matrices, top_score, max_location, bin_im, tvsc, tvro, tvsh, tl, cropped_samples, tcarr



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
            img = imageio.imread(image_path, as_gray=True).astype(np.uint8)
            image_list.append(image_path)
            images.append(img)
    return image_list, images

def create_4cs(v_scaling, v_rot, v_shearing, grid_size, unitcell, miller, t):

    # Combine transformation matrices
    # M = T.dot(S).dot(SH).dot(R)
    # scaling
    M1 = create_affine_matrix(angles = (0, 0, 0), translation = (0, 0, 0), scaling = v_scaling, shearing = (0, 0, 0, 0, 0, 0))
    # rotation
    M2 = create_affine_matrix(angles = v_rot, translation = (0, 0, 0), scaling = (1, 1, 1), shearing = (0, 0, 0, 0, 0, 0))
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
    B = level_set_equation_matrix(A)
    
    level_set_transformed0 = B[0].reshape(X.shape)
    level_set_transformed1 = B[1].reshape(X.shape)
    level_set_transformed2 = B[2].reshape(X.shape)
    level_set_transformed3 = B[3].reshape(X.shape)
    
    bin_im = (make_binary(level_set_transformed0, t), make_binary(level_set_transformed1, t), make_binary(level_set_transformed2, t), make_binary(level_set_transformed3, t))
    
    ##############################################################################
    # levelset F090
    if corner == 'L':
        label1 = 'F090'
    elif corner == 'R':
        label1 = 'R090'
    # Apply transformation to grid points
    norm1 = miller_indices_to_normal(miller[0], miller[1], miller[2])
    norm1 = transform_normal(norm1, M2)
    norm1 = miller_indices_to_normal(norm1[0], norm1[1], norm1[2])
    
    ##############################################################################
    # levelset F024
    if corner == 'L':
        label2 = 'F024'
    elif corner == 'R':
        label2 = 'R024'
    norm2 = transform_normal(norm1, R2)
    norm2 = miller_indices_to_normal(norm2[0], norm2[1], norm2[2])

    
    ##############################################################################
    # levelset L090
    if corner == 'L':
        label3 = 'L090'
    elif corner == 'R':
        label3 = 'F090'
    norm3 = transform_normal(norm1, R3)
    norm3 = miller_indices_to_normal(norm3[0], norm3[1], norm3[2])
    
    ##############################################################################
    # levelset L024
    if corner == 'L':
        label4 = 'L024'
    elif corner == 'R':
        label4 = 'F024'
    norm4 = transform_normal(norm3, R4)
    norm4 = miller_indices_to_normal(norm4[0], norm4[1], norm4[2])
    
    Rtsn = create_affine_matrix(angles = (0, 90, 0), translation = (0, 0, 0), scaling = (1, 1, 1), shearing = (0, 0, 0, 0, 0, 0))
    topsurf_norm = transform_normal(norm1, Rtsn)
    topsurf_norm = miller_indices_to_normal(topsurf_norm[0], topsurf_norm[1], topsurf_norm[2])
    
    norm_images = (norm1, norm2, norm3, norm4)
    label_images = (label1, label2, label3, label4)
    
    return (bin_im, norm_images, label_images, M, A, topsurf_norm)

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

def translation_matrix(tx, ty, tz):
    return np.array([[1, 0, 0, tx],
                     [0, 1, 0, ty],
                     [0, 0, 1, tz],
                     [0, 0, 0, 1]])

def scaling_matrix(sx, sy, sz):
    return np.diag([sx, sy, sz, 1])

def shearing_matrix(sxy, sxz, syx, syz, szx, szy):
    return np.array([[1, sxy, sxz, 0],
                     [syx, 1, syz, 0],
                     [szx, szy, 1, 0],
                     [0, 0, 0, 1]])

def level_set_equation_matrix(M):
    level_set_matrix0 = np.sin(M[0]) * np.cos(M[1]) + np.sin(M[1]) * np.cos(M[2]) + np.sin(M[2]) * np.cos(M[0])
    level_set_matrix1 = (np.sin(M[4]) * np.cos(M[5]) + np.sin(M[5]) * np.cos(M[6]) + np.sin(M[6]) * np.cos(M[4]))
    level_set_matrix2 = (np.sin(M[8]) * np.cos(M[9]) + np.sin(M[9]) * np.cos(M[10]) + np.sin(M[10]) * np.cos(M[8]))
    level_set_matrix3 = (np.sin(M[12]) * np.cos(M[13]) + np.sin(M[13]) * np.cos(M[14]) + np.sin(M[14]) * np.cos(M[12]))
    level_set_matrix = (level_set_matrix0, level_set_matrix1, level_set_matrix2, level_set_matrix3)
    return level_set_matrix

def make_binary(matrix, t):
    binary = np.where(matrix < t, 1, 0) * 255
    return binary

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

def miller_indices_to_normal(h, k, l):
    # Find the normal vector to the plane using Miller indices
    output_vector = [0, 0, 0]
    normal_vector = np.array([h, k, l])
    normal_vector = normal_vector / np.linalg.norm(normal_vector)
    output_vector[0] = normal_vector[0]
    output_vector[1] = normal_vector[1]
    output_vector[2] = normal_vector[2]
    return output_vector

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

def matrix_to_image(matrix):
    # Scale the float64 image to the range [0, 255]
    image = Image.fromarray(matrix)
    scaled_image = ((image - np.min(image)) / (np.max(image) - np.min(image)) * 255).astype(np.uint8)
    return scaled_image

def save_data_csv(tops, topr, topsh, topthr, surface_on_top, singlec, topc, runs, time_sf, save, c, best4):
    # Specify the file path
    gmt = time.gmtime() 
    file_path = save + f'/{gmt[0]}{gmt[1]:02}{gmt[2]:02}_crosscorr_arrays_data_{c}_{k:02}.csv'
    
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
        {'Description': 'time needed', 'Array': time_sf},
        {'Description': 'threshold', 'Array': topthr}
    ]
    
    # Write data to CSV file
    with open(file_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=columns)
        writer.writeheader()
        for row in data:
            writer.writerow(row)

def plot_4cs(bin_im, label_images, directory, j):

    bin_im_np_orig = [0, 0, 0, 0]

    bin_im_np_orig[0] = bin_im[0][1]
    # plt.imshow(bin_im_np_orig[0])
    cv2.imwrite(directory + f'/{label_images[0]}_crosscorr_match_{j}_{k}_{corner}.png', bin_im_np_orig[0])
    
    bin_im_np_orig[1] = bin_im[1][1]
    # plt.imshow(bin_im_np_orig[1])
    cv2.imwrite(directory + f'/{label_images[1]}_crosscorr_match_{j}_{k}_{corner}.png', bin_im_np_orig[1])
    
    bin_im_np_orig[2] = bin_im[2][1]
    # plt.imshow(bin_im_np_orig[2])
    cv2.imwrite(directory + f'/{label_images[2]}_crosscorr_match_{j}_{k}_{corner}.png', bin_im_np_orig[2])

    bin_im_np_orig[3] = bin_im[3][1]
    # plt.imshow(bin_im_np_orig[3])
    cv2.imwrite(directory + f'/{label_images[3]}_crosscorr_match_{j}_{k}_{corner}.png', bin_im_np_orig[3])

        
            

def print_top_correlations_left(max_color, min_color, color_palette, top_score, topcorr_overall, bin_im, final_max_location, cropped_samples, top_v_sc, top_v_ro, top_v_sh, top_v_thr, ts, transparency, transparency_sample):
    
    color_scaling = int(round(100*(max_color-min_color), 2))
    color_max = round(max_color, 2)
    color_min = 0.0
    # color_min = round(min_color, 2)
    # textbox color according to correlation value
    text_values = np.linspace(color_min, color_max, color_scaling)
    # Create a seaborn color palette
    palette = sns.color_palette(color_palette, n_colors=len(text_values))
    ###########################################################################################

    fig_corr, axs_corr = plt.subplots(1, cs_per_volume, figsize=(30, 10)) 
    fig_corr.suptitle(f'Correlation of {samples_dir}', fontsize=16) 
        
    # Display the original image
    axs_corr = axs_corr.flatten()
    if corner == 'L':
        axs_corr[0].set_title('L90')
        axs_corr[1].set_title('L24')
        axs_corr[2].set_title('F90')
        axs_corr[3].set_title('F24') 
    elif corner == 'R':
        axs_corr[0].set_title('F90')
        axs_corr[1].set_title('F24')
        axs_corr[2].set_title('R90')
        axs_corr[3].set_title('R24')    
    
    # Display the "best_combos" 

    h, w = np.array(bin_im[0][1]).shape
    
    iL90 = 0
    iL24 = 1
    iF90 = 2
    iF24 = 3
    
    
    overlay_L90 = cropped_samples[2].copy()  
    overlay_L90 = np.uint8((transparency_sample)*overlay_L90[:,:])
    overlay_L90[final_max_location[2][1][1]:final_max_location[2][1][1]+h, final_max_location[2][1][0]:final_max_location[2][1][0]+w] = np.uint8(transparency*bin_im[2][1][:,:] + (1-transparency)*overlay_L90[final_max_location[2][1][1]:final_max_location[2][1][1]+h, final_max_location[2][1][0]:final_max_location[2][1][0]+w])
    axs_corr[iL90].imshow(overlay_L90, cmap='gray')
                
    if top_score[2][1] <= color_min:
        axs_corr[iL90].text(overlay_L90.shape[1] + round(overlay_L90.shape[1]/20), overlay_L90.shape[0] - round(overlay_L90.shape[0]/12), "%.2f" %top_score[2][1], bbox=dict(facecolor=palette[0], alpha=0.5))
    elif top_score[2][1] >= color_max:
        axs_corr[iL90].text(overlay_L90.shape[1] + round(overlay_L90.shape[1]/20), overlay_L90.shape[0] - round(overlay_L90.shape[0]/12), "%.2f" %top_score[2][1], bbox=dict(facecolor=palette[int(color_scaling - 1)], alpha=0.5))
    else:
        axs_corr[iL90].text(overlay_L90.shape[1] + round(overlay_L90.shape[1]/20), overlay_L90.shape[0] - round(overlay_L90.shape[0]/12), "%.2f" %top_score[2][1], bbox=dict(facecolor=palette[int(round(float(top_score[2][1]) - color_min, 2)*100)], alpha=0.5))

    overlay_L24 = cropped_samples[3].copy()
    overlay_L24[:, :] = np.uint8((transparency_sample)*overlay_L24[:,:])
    overlay_L24[final_max_location[3][1][1]:final_max_location[3][1][1]+h, final_max_location[3][1][0]:final_max_location[3][1][0]+w] = np.uint8(transparency*bin_im[3][1][:,:] + (1-transparency)*overlay_L24[final_max_location[3][1][1]:final_max_location[3][1][1]+h, final_max_location[3][1][0]:final_max_location[3][1][0]+w])
    axs_corr[iL24].imshow(overlay_L24, cmap='gray')
    
    if top_score[3][1] <= color_min:
        axs_corr[iL24].text(overlay_L24.shape[1] + round(overlay_L24.shape[1]/20), overlay_L24.shape[0] - round(overlay_L24.shape[0]/12), "%.2f" %top_score[3][1], bbox=dict(facecolor=palette[0], alpha=0.5))
    elif top_score[3][1] >= color_max:
        axs_corr[iL24].text(overlay_L24.shape[1] + round(overlay_L24.shape[1]/20), overlay_L24.shape[0] - round(overlay_L24.shape[0]/12), "%.2f" %top_score[3][1], bbox=dict(facecolor=palette[int(color_scaling - 1)], alpha=0.5))
    else:
        axs_corr[iL24].text(overlay_L24.shape[1] + round(overlay_L24.shape[1]/20), overlay_L24.shape[0] - round(overlay_L24.shape[0]/12), "%.2f" %top_score[3][1], bbox=dict(facecolor=palette[int(round(float(top_score[3][1]) - color_min, 2)*100)], alpha=0.5))


    overlay_F90 = cropped_samples[0].copy()
    overlay_F90[:, :] = np.uint8((transparency_sample)*overlay_F90[:,:])
    overlay_F90[final_max_location[0][1][1]:final_max_location[0][1][1]+h, final_max_location[0][1][0]:final_max_location[0][1][0]+w] = np.uint8(transparency*bin_im[0][1][:,:] + (1-transparency)*overlay_F90[final_max_location[0][1][1]:final_max_location[0][1][1]+h, final_max_location[0][1][0]:final_max_location[0][1][0]+w])
    axs_corr[iF90].imshow(overlay_F90, cmap='gray')
 
    if top_score[0][1] <= color_min:
        axs_corr[iF90].text(overlay_F90.shape[1] + round(overlay_F90.shape[1]/20), overlay_F90.shape[0] - round(overlay_F90.shape[0]/12), "%.2f" %top_score[0][1], bbox=dict(facecolor=palette[0], alpha=0.5))
    elif top_score[0][1] >= color_max:
        axs_corr[iF90].text(overlay_F90.shape[1] + round(overlay_F90.shape[1]/20), overlay_F90.shape[0] - round(overlay_F90.shape[0]/12), "%.2f" %top_score[0][1], bbox=dict(facecolor=palette[int(color_scaling - 1)], alpha=0.5))
    else:
        axs_corr[iF90].text(overlay_F90.shape[1] + round(overlay_F90.shape[1]/20), overlay_F90.shape[0] - round(overlay_F90.shape[0]/12), "%.2f" %top_score[0][1], bbox=dict(facecolor=palette[int(round(float(top_score[0][1]) - color_min, 2)*100)], alpha=0.5))

    
    overlay_F24 = cropped_samples[1].copy()
    overlay_F24[:, :] = np.uint8((transparency_sample)*overlay_F24[:,:])
    overlay_F24[final_max_location[1][1][1]:final_max_location[1][1][1]+h, final_max_location[1][1][0]:final_max_location[1][1][0]+w] = np.uint8(transparency*bin_im[1][1][:,:] + (1-transparency)*overlay_F24[final_max_location[1][1][1]:final_max_location[1][1][1]+h, final_max_location[1][1][0]:final_max_location[1][1][0]+w])
    axs_corr[iF24].imshow(overlay_F24, cmap='gray')
    
    if top_score[1][1] <= color_min:
        axs_corr[iF24].text(overlay_F24.shape[1] + round(overlay_F24.shape[1]/20), overlay_F24.shape[0] - round(overlay_F24.shape[0]/12), "%.2f" %top_score[1][1], bbox=dict(facecolor=palette[0], alpha=0.5))
    elif top_score[1][1] >= color_max:
        axs_corr[iF24].text(overlay_F24.shape[1] + round(overlay_F24.shape[1]/20), overlay_F24.shape[0] - round(overlay_F24.shape[0]/12), "%.2f" %top_score[1][1], bbox=dict(facecolor=palette[int(color_scaling - 1)], alpha=0.5))
    else:
        axs_corr[iF24].text(overlay_F24.shape[1] + round(overlay_F24.shape[1]/20), overlay_F24.shape[0] - round(overlay_F24.shape[0]/12), "%.2f" %top_score[1][1], bbox=dict(facecolor=palette[int(round(float(top_score[1][1]) - color_min, 2)*100)], alpha=0.5))
    
    top_corr = 1/2*np.sqrt(pow(top_score[2][1], 2) + pow(top_score[3][1], 2) + pow(top_score[0][1], 2) + pow(top_score[1][1], 2))
    
    axs_corr[iF24].text(overlay_F24.shape[1] + round(overlay_F24.shape[0]/20), overlay_F24.shape[0] - round(overlay_F24.shape[0]/3.5), f'threshold \n {top_v_thr} \nscaling \n {top_v_sc[1][0]} \n {top_v_sc[1][1]} \n {top_v_sc[1][2]}' + 
    f'\nrotations \n {top_v_ro[1][0]} \n {top_v_ro[1][1]} \n {top_v_ro[1][2]}'+
    f'\nshearing \n {top_v_sh[1][0]} \n {top_v_sh[1][1]} \n {top_v_sh[1][2]}'+
    f'\n {top_v_sh[1][3]} \n {top_v_sh[1][4]} \n {top_v_sh[1][5]}'+
    f'\ntopsurface \nh {round(float(ts[1][0]), 2)} \nk {round(float(ts[1][1]), 2)} \nl {round(float(ts[1][2]), 2)}'+
    f'\ntotal corr {topcorr_overall}')
    
    if top_corr <= color_min:
        axs_corr[iF24].text(overlay_F24.shape[1] + round(overlay_F24.shape[0]/20), overlay_F24.shape[0] - round(overlay_F24.shape[0]/6), "%.2f" %top_corr, bbox=dict(facecolor=palette[0], alpha=0.5))
    elif top_corr >= color_max:
        axs_corr[iF24].text(overlay_F24.shape[1] + round(overlay_F24.shape[0]/20), overlay_F24.shape[0] - round(overlay_F24.shape[0]/6), "%.2f" %top_corr, bbox=dict(facecolor=palette[int(color_scaling - 1)], alpha=0.5))
    else:
        axs_corr[iF24].text(overlay_F24.shape[1] + round(overlay_F24.shape[0]/20), overlay_F24.shape[0] - round(overlay_F24.shape[0]/6), "%.2f" %top_corr, bbox=dict(facecolor=palette[int(round(top_score[0][1] - color_min, 2)*100)-1], alpha=0.5))

    # record current timestamp
    gmt = time.gmtime()          
    plt.savefig(save_folder + f'/{gmt.tm_year:04}{gmt.tm_mon:02}{gmt.tm_mday:02}_crosscorr_plot_{corner}_{k:02}.png')
 

def print_sample(max_color, min_color, color_palette, top_score, topcorr_overall, bin_im, final_max_location, cropped_samples, top_v_sc, top_v_ro, top_v_sh, top_v_thr, ts, transparency, transparency_s):
    
    color_scaling = int(round(100*(max_color-min_color), 2))
    color_max = round(max_color, 2)
    color_min = 0.0
    # color_min = round(min_color, 2)
    # textbox color according to correlation value
    text_values = np.linspace(color_min, color_max, color_scaling)
    # Create a seaborn color palette
    palette = sns.color_palette(color_palette, n_colors=len(text_values))
    
    transparency, transparency_s = 0, 1
    ###########################################################################################

    fig_corr, axs_corr = plt.subplots(1, cs_per_volume, figsize=(30, 10)) 
    fig_corr.suptitle(f'Correlation of {samples_dir}', fontsize=16) 
        
    # Display the original image
    axs_corr = axs_corr.flatten()
    if corner == 'L':
        axs_corr[0].set_title('L90')
        axs_corr[1].set_title('L24')
        axs_corr[2].set_title('F90')
        axs_corr[3].set_title('F24') 
    elif corner == 'R':
        axs_corr[0].set_title('F90')
        axs_corr[1].set_title('F24')
        axs_corr[2].set_title('R90')
        axs_corr[3].set_title('R24')    
    
    # Display the "best_combos" 

    h, w = np.array(bin_im[0][1]).shape
    
    iL90 = 0
    iL24 = 1
    iF90 = 2
    iF24 = 3
    
    
    overlay_L90 = cropped_samples[2].copy()  
    overlay_L90 = np.uint8((transparency_s)*overlay_L90[:,:])
    overlay_L90[final_max_location[2][1][1]:final_max_location[2][1][1]+h, final_max_location[2][1][0]:final_max_location[2][1][0]+w] = np.uint8(transparency*bin_im[2][1][:,:] + (1-transparency)*overlay_L90[final_max_location[2][1][1]:final_max_location[2][1][1]+h, final_max_location[2][1][0]:final_max_location[2][1][0]+w])
    axs_corr[iL90].imshow(overlay_L90, cmap='gray')
                
    # if top_score[2][1] <= color_min:
    #     axs_corr[iL90].text(overlay_L90.shape[1] + round(overlay_L90.shape[1]/20), overlay_L90.shape[0] - round(overlay_L90.shape[0]/12), "%.2f" %top_score[2][1], bbox=dict(facecolor=palette[0], alpha=0.5))
    # elif top_score[2][1] >= color_max:
    #     axs_corr[iL90].text(overlay_L90.shape[1] + round(overlay_L90.shape[1]/20), overlay_L90.shape[0] - round(overlay_L90.shape[0]/12), "%.2f" %top_score[2][1], bbox=dict(facecolor=palette[int(color_scaling - 1)], alpha=0.5))
    # else:
    #     axs_corr[iL90].text(overlay_L90.shape[1] + round(overlay_L90.shape[1]/20), overlay_L90.shape[0] - round(overlay_L90.shape[0]/12), "%.2f" %top_score[2][1], bbox=dict(facecolor=palette[int(round(float(top_score[2][1]) - color_min, 2)*100)], alpha=0.5))

    overlay_L24 = cropped_samples[3].copy()
    overlay_L24[:, :] = np.uint8((transparency_s)*overlay_L24[:,:])
    overlay_L24[final_max_location[3][1][1]:final_max_location[3][1][1]+h, final_max_location[3][1][0]:final_max_location[3][1][0]+w] = np.uint8(transparency*bin_im[3][1][:,:] + (1-transparency)*overlay_L24[final_max_location[3][1][1]:final_max_location[3][1][1]+h, final_max_location[3][1][0]:final_max_location[3][1][0]+w])
    axs_corr[iL24].imshow(overlay_L24, cmap='gray')
    
    # if top_score[3][1] <= color_min:
    #     axs_corr[iL24].text(overlay_L24.shape[1] + round(overlay_L24.shape[1]/20), overlay_L24.shape[0] - round(overlay_L24.shape[0]/12), "%.2f" %top_score[3][1], bbox=dict(facecolor=palette[0], alpha=0.5))
    # elif top_score[3][1] >= color_max:
    #     axs_corr[iL24].text(overlay_L24.shape[1] + round(overlay_L24.shape[1]/20), overlay_L24.shape[0] - round(overlay_L24.shape[0]/12), "%.2f" %top_score[3][1], bbox=dict(facecolor=palette[int(color_scaling - 1)], alpha=0.5))
    # else:
    #     axs_corr[iL24].text(overlay_L24.shape[1] + round(overlay_L24.shape[1]/20), overlay_L24.shape[0] - round(overlay_L24.shape[0]/12), "%.2f" %top_score[3][1], bbox=dict(facecolor=palette[int(round(float(top_score[3][1]) - color_min, 2)*100)], alpha=0.5))


    overlay_F90 = cropped_samples[0].copy()
    overlay_F90[:, :] = np.uint8((transparency_s)*overlay_F90[:,:])
    overlay_F90[final_max_location[0][1][1]:final_max_location[0][1][1]+h, final_max_location[0][1][0]:final_max_location[0][1][0]+w] = np.uint8(transparency*bin_im[0][1][:,:] + (1-transparency)*overlay_F90[final_max_location[0][1][1]:final_max_location[0][1][1]+h, final_max_location[0][1][0]:final_max_location[0][1][0]+w])
    axs_corr[iF90].imshow(overlay_F90, cmap='gray')
 
    # if top_score[0][1] <= color_min:
    #     axs_corr[iF90].text(overlay_F90.shape[1] + round(overlay_F90.shape[1]/20), overlay_F90.shape[0] - round(overlay_F90.shape[0]/12), "%.2f" %top_score[0][1], bbox=dict(facecolor=palette[0], alpha=0.5))
    # elif top_score[0][1] >= color_max:
    #     axs_corr[iF90].text(overlay_F90.shape[1] + round(overlay_F90.shape[1]/20), overlay_F90.shape[0] - round(overlay_F90.shape[0]/12), "%.2f" %top_score[0][1], bbox=dict(facecolor=palette[int(color_scaling - 1)], alpha=0.5))
    # else:
    #     axs_corr[iF90].text(overlay_F90.shape[1] + round(overlay_F90.shape[1]/20), overlay_F90.shape[0] - round(overlay_F90.shape[0]/12), "%.2f" %top_score[0][1], bbox=dict(facecolor=palette[int(round(float(top_score[0][1]) - color_min, 2)*100)], alpha=0.5))

    
    overlay_F24 = cropped_samples[1].copy()
    overlay_F24[:, :] = np.uint8((transparency_s)*overlay_F24[:,:])
    overlay_F24[final_max_location[1][1][1]:final_max_location[1][1][1]+h, final_max_location[1][1][0]:final_max_location[1][1][0]+w] = np.uint8(transparency*bin_im[1][1][:,:] + (1-transparency)*overlay_F24[final_max_location[1][1][1]:final_max_location[1][1][1]+h, final_max_location[1][1][0]:final_max_location[1][1][0]+w])
    axs_corr[iF24].imshow(overlay_F24, cmap='gray')
    
    # if top_score[1][1] <= color_min:
    #     axs_corr[iF24].text(overlay_F24.shape[1] + round(overlay_F24.shape[1]/20), overlay_F24.shape[0] - round(overlay_F24.shape[0]/12), "%.2f" %top_score[1][1], bbox=dict(facecolor=palette[0], alpha=0.5))
    # elif top_score[1][1] >= color_max:
    #     axs_corr[iF24].text(overlay_F24.shape[1] + round(overlay_F24.shape[1]/20), overlay_F24.shape[0] - round(overlay_F24.shape[0]/12), "%.2f" %top_score[1][1], bbox=dict(facecolor=palette[int(color_scaling - 1)], alpha=0.5))
    # else:
    #     axs_corr[iF24].text(overlay_F24.shape[1] + round(overlay_F24.shape[1]/20), overlay_F24.shape[0] - round(overlay_F24.shape[0]/12), "%.2f" %top_score[1][1], bbox=dict(facecolor=palette[int(round(float(top_score[1][1]) - color_min, 2)*100)], alpha=0.5))
    
    top_corr = 1/2*np.sqrt(pow(top_score[2][1], 2) + pow(top_score[3][1], 2) + pow(top_score[0][1], 2) + pow(top_score[1][1], 2))
    
    # axs_corr[iF24].text(overlay_F24.shape[1] + round(overlay_F24.shape[0]/20), overlay_F24.shape[0] - round(overlay_F24.shape[0]/3.5), f'threshold \n {top_v_thr} \nscaling \n {top_v_sc[1][0]} \n {top_v_sc[1][1]} \n {top_v_sc[1][2]}' + 
    # f'\nrotations \n {top_v_ro[1][0]} \n {top_v_ro[1][1]} \n {top_v_ro[1][2]}'+
    # f'\nshearing \n {top_v_sh[1][0]} \n {top_v_sh[1][1]} \n {top_v_sh[1][2]}'+
    # f'\n {top_v_sh[1][3]} \n {top_v_sh[1][4]} \n {top_v_sh[1][5]}'+
    # f'\ntopsurface \nh {round(float(ts[1][0]), 2)} \nk {round(float(ts[1][1]), 2)} \nl {round(float(ts[1][2]), 2)}'+
    # f'\ntotal corr {topcorr_overall}')
    
    # if top_corr <= color_min:
    #     axs_corr[iF24].text(overlay_F24.shape[1] + round(overlay_F24.shape[0]/20), overlay_F24.shape[0] - round(overlay_F24.shape[0]/6), "%.2f" %top_corr, bbox=dict(facecolor=palette[0], alpha=0.5))
    # elif top_corr >= color_max:
    #     axs_corr[iF24].text(overlay_F24.shape[1] + round(overlay_F24.shape[0]/20), overlay_F24.shape[0] - round(overlay_F24.shape[0]/6), "%.2f" %top_corr, bbox=dict(facecolor=palette[int(color_scaling - 1)], alpha=0.5))
    # else:
    #     axs_corr[iF24].text(overlay_F24.shape[1] + round(overlay_F24.shape[0]/20), overlay_F24.shape[0] - round(overlay_F24.shape[0]/6), "%.2f" %top_corr, bbox=dict(facecolor=palette[int(round(top_score[0][1] - color_min, 2)*100)-1], alpha=0.5))

    # record current timestamp
    gmt = time.gmtime()          
    plt.savefig(save_folder + f'/{gmt.tm_year:04}{gmt.tm_mon:02}{gmt.tm_mday:02}_sample_plot_{corner}.png')
 
    
 
def big_loop(rf, sf, sd, vsca, vrot, vshe, thr, gs, uc, scsq, miller, co, square_shift, mi):    
    
    # Defining the original basis vectors (example values)
    original_basis_vectors = np.array([[1, 0, 0],
                                       [0, 1, 0],
                                       [0, 0, 1]])
    
    # crosscorrelation
    top_corr_arr = [0]
    top_parameters = [0]
    top_parameters_scores = [0]
    top_translation = [0]
    top_translation_scores = [0]
    top_matrix_F90 = [0]
    top_F90_scores = [0]
    top_matrix_F24 = [0]
    top_F24_scores = [0]
    top_matrix_L90 = [0]
    top_L90_scores = [0]
    top_matrix_L24 = [0]
    top_L24_scores = [0]
    
    model_matrices_F90 = [0]
    model_matrices_F24 = [0]
    model_matrices_L90 = [0]
    model_matrices_L24 = [0]
    model_matrices_R90 = [0]
    model_matrices_R24 = [0]
    
    max_location_F90 = [0]
    max_location_F24 = [0]
    max_location_L90 = [0]
    max_location_L24 = [0]
    max_location_R90 = [0]
    max_location_R24 = [0]
    
    top_v_rotations = [0]
    top_v_translations = [0]
    top_v_scaling = [0]
    top_v_shearing = [0]
    top_v_threshold = [0]
    top_v_surface = [0]
    top_v_singlecorr = [0]
    
    top_label = [0]
    
    # Read in sample and model image
    samples = read_images_from_folder(sd)
    
    time_start = time.time()       

    t = thr
    
    v_translations = (0, 0, 0)
    
    v_scaling = vsca
    v_rotations = vrot
    v_shearing = vshe
    
    # performing crosscorrelation of the model with the 4 FIB SEM cuts
    binary_images, norm_images, labels, matrices, A, topsurface = create_4cs(v_scaling, v_rotations, v_shearing, gs, uc, miller, t)
    top_transf_matrices, top_score, max_location, model_matrices, top_v_sca, top_v_rot, top_v_she, topsurfaces, cropped_samples, top_corr_a = perform_crosscorrelation_left(sd, scsq, square_shift, binary_images, norm_images, topsurface, matrices, 
                                                                                                                                                                                                       top_F90_scores, top_matrix_F90, max_location_F90, model_matrices_F90, 
                                                                                                                                                                                                       top_F24_scores, top_matrix_F24, max_location_F24, model_matrices_F24, 
                                                                                                                                                                                                       top_L90_scores, top_matrix_L90, max_location_L90, model_matrices_L90, 
                                                                                                                                                                                                       top_L24_scores, top_matrix_L24, max_location_L24, model_matrices_L24, 
                                                                                                                                                                                                       top_v_rotations, top_v_translations, top_v_scaling, top_v_shearing, top_label, top_corr_arr,
                                                                                                                                                                                                       v_rotations, v_translations, v_scaling, v_shearing)
    #######            

    top_corr_arr = top_corr_a
    top_v_rotations = top_v_rot
    top_v_scaling = top_v_sca
    top_v_shearing = top_v_she
    top_v_threshold = t
    top_v_surface = topsurfaces
    top_v_singlecorr = top_score
    
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

    save_data_csv(top_v_scaling, top_v_rotations, top_v_shearing, top_v_threshold, top_v_surface, top_v_singlecorr, top_corr_arr, iterations, time_sofar_used, sf, co, best4)

    # color coding
    max_color = 1
    min_color = 0
    color_palette = (colors_plot)
    transparency_overlay = 0.1
    transparency_sample = 0.05
    
    for i in range(0, print_top):
        plot_4cs(model_matrices, labels, sf, i)

    # print_top_correlations_left(max_color, min_color, color_palette, top_score, model_matrices, cropped_samples, top_v_scaling, top_v_rotations, top_v_shearing, topsurfaces, transparency_overlay, transparency_sample, max_location, sf, co)
    print_top_correlations_left(max_color, min_color, color_palette, top_score, top_corr_a, model_matrices, max_location, cropped_samples, top_v_sca, top_v_rot, top_v_she, top_v_threshold, topsurfaces, transparency_overlay, transparency_sample)
    print_sample(max_color, min_color, color_palette, top_score, top_corr_a, model_matrices, max_location, cropped_samples, top_v_sca, top_v_rot, top_v_she, top_v_threshold, topsurfaces, transparency_overlay, transparency_sample)



########################################################

# Example data
np.random.seed(42)  # For reproducibility
x = np.random.rand(100)
y = np.random.rand(100)
z = np.random.rand(100)
X = []
Y = []
Z = []
X_start = []
Y_start = []
Z_start = []
corr = []
single_corr_1 = []
single_corr_2 = []
single_corr_3 = []
single_corr_4 = []
rotation = []
scale = []
shear = []
threshold = []
int_corr = []
plt_corr = []
values = np.random.rand(100)  # Related values for color mapping

#################################################################################
# Sources either right side or left side of the FIB session

# corner = 'L'
# root_folder = 'C:/Users/iselire/OneDrive/studium/unifr/solvent_evaporation_annealing/evaluation_crosssections/tests/20260531_tests/20260531_tests_L_rot'
# samples_dir = 'C:/Users/iselire/OneDrive/studium/unifr/solvent_evaporation_annealing/evaluation_crosssections/tests/20260531_tests/20260531_tests_L_rot'
# save_folder = 'C:/Users/iselire/OneDrive/studium/unifr/solvent_evaporation_annealing/evaluation_crosssections/tests/20260531_tests/20260531_tests_L_rot'


# corner = 'R'
# root_folder = 'C:/Users/iselire/OneDrive/studium/unifr/solvent_evaporation_annealing/evaluation_crosssections/tests/20260531_tests/20260531_tests_R_rot'
# samples_dir = 'C:/Users/iselire/OneDrive/studium/unifr/solvent_evaporation_annealing/evaluation_crosssections/tests/20260531_tests/20260531_tests_R_rot'
# save_folder = 'C:/Users/iselire/OneDrive/studium/unifr/solvent_evaporation_annealing/evaluation_crosssections/tests/20260531_tests/20260531_tests_R_rot'

corner = 'L'
root_folder = 'C:/Users/iselire/OneDrive/studium/unifr/solvent_evaporation_annealing/evaluation_crosssections/tests_torch/test_torch_L'
samples_dir = 'C:/Users/iselire/OneDrive/studium/unifr/solvent_evaporation_annealing/evaluation_crosssections/tests_torch/test_torch_L'
save_folder = 'C:/Users/iselire/OneDrive/studium/unifr/solvent_evaporation_annealing/evaluation_crosssections/tests_torch/test_torch_L'

dir_exists(save_folder)

# root_folder = 'C:/Users/IseliRe/Downloads/test'
csv_files_data_1, csv_files_data_2, csv_files_data_3, csv_files_data_4, csv_files_data_5, csv_files_data_6, csv_files_data_7, csv_files_data_8 = read_all_csv_files(root_folder)

max_threshold = 1.0
min_threshold = 0.0 
number_of_prints = 1
print_top_number = 'False' 
# print_top_number = 'True'
# create_rot_gif = 'False' 
create_rot_gif = 'False'

first_miller = (1, 0, 0)
unitcell_pixel = 60
grid = (400, 400, 400)
sample_size = 1000
translations_per_cut = 70
best4 = 1
iterations = 1

colors_plot = 'binary' # viridis, RdYlBu, gnuplot2, binary, afmhot, gist_heat, hot, jet, CMRmap, copper

# x, y - shift per image
# L24, L90, F24, F90 or F24, F90, R24, R90
square_shift_pixels = ([(0, 0), (0, 0), (0, 0), (0, 0)],
                        [(0, 0), (0, 0), (0, 0), (0, 0)])

best_combos = 5
print_top = 1
cs_per_volume = 4

#################################################################################

for directory in range(1, len(csv_files_data_2), 1):
    topsurface_arrays = overallcorr_to_array(csv_files_data_1[directory][1][0], 'no')
    overallcorr_arrays = overallcorr_to_array(csv_files_data_2[directory][1][0], 'no')
    # starting_topsurface = overallcorr_to_array(csv_files_data_3[directory][1][0], 'no')
    single_correlation = overallcorr_to_array(csv_files_data_4[directory][1][0], 'no')
    scaling_arrays = vectorstring_to_array(csv_files_data_5[directory][1][0])
    rotation_arrays = vectorstring_to_array(csv_files_data_6[directory][1][0])
    shearing_arrays = vectorstring_to_array(csv_files_data_7[directory][1][0])
    threshold_arrays = csv_files_data_8[directory][1][0]
    # threshold_arrays = overallcorr_to_array(csv_files_data_8[directory][1][0])
    for i in range(1, len(overallcorr_arrays)):
        X.append(topsurface_arrays[i][0])
        Y.append(topsurface_arrays[i][1])
        Z.append(topsurface_arrays[i][2])
        corr.append(overallcorr_arrays[i])
        single_corr_1.append(single_correlation[i][0])
        single_corr_2.append(single_correlation[i][1])
        single_corr_3.append(single_correlation[i][2])
        single_corr_4.append(single_correlation[i][3])
        int_corr.append(overallcorr_arrays[i])
        plt_corr.append(overallcorr_arrays[i])
        scale.append(scaling_arrays[i])
        rotation.append(rotation_arrays[i])
        shear.append(shearing_arrays[i])
        threshold.append(threshold_arrays)
        
    # for i in range(0, len(csv_files_data_3[directory][1])):
    #     # print('here')
    #     X_start.append(float(starting_topsurface[0]))
    #     Y_start.append(float(starting_topsurface[1]))
    #     Z_start.append(float(starting_topsurface[2]))
        
    


    
for u in range(0, len(corr)): 
    if corr[u] > 0 and corr[u] < max_threshold and single_corr_1[u] < max_threshold and single_corr_2[u] < max_threshold and single_corr_3[u] < max_threshold and single_corr_4[u] < max_threshold: 
        int_corr[u] = int((np.round(pow(corr[u]*100/30, 2) + 1)))
    if corr[u] > 0 and corr[u] > min_threshold : 
        int_corr[u] = int((np.round(pow(corr[u]*100/30, 2) + 1)))
    else:
        int_corr[u] = 0
        plt_corr[u] = 0
# Create a 3D scatter plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# getting the original colormap using cm.get_cmap() function 
orig_map=plt.cm.get_cmap(colors_plot) 
  
# reversing the original colormap using reversed() function 
reversed_map = orig_map.reversed()

max_ind = sorted(range(len(corr)), key=lambda sub: corr[sub])[-number_of_prints:]
max_scale = 0
for i in range(0, len(max_ind)):
    if corr[max_ind[i]] > max_scale and corr[max_ind[i]] < max_threshold and single_corr_1[u] < max_threshold and single_corr_2[u] < max_threshold and single_corr_3[u] < max_threshold and single_corr_4[u] < max_threshold:
        max_scale = corr[max_ind[i]]

# Scatter plot with colors based on the 'values'
sc = ax.scatter(X, Y, Z, c=plt_corr, cmap=orig_map, s = int_corr, vmin = min_threshold, vmax = max_threshold)
# sc_start = ax.scatter(X_start, Y_start, Z_start, c='yellow', s = 20, alpha = 0.1)

ax.scatter(1.1, 0, 0, c='magenta', s = 30)
ax.scatter(0, 1.1, 0, c='magenta', s = 30)
ax.scatter(0, 0, 1.1, c='magenta', s = 30)
ax.scatter(0, 0, -1, c='orange', s = 30)
#############################################
# Define the parameters for the ring
num_points = 60
theta = np.linspace(0, 2 * np.pi, num_points)
radius = 1
height = 0.1

# Generate the points for the ring
x = radius * np.cos(theta)
y = radius * np.sin(theta)
z = np.zeros(num_points) * height

# Plot the ring
# ax.scatter(x, y, z, color='blue', alpha=0.2)
#############################################



# Add color bar to show the color scale
colorbar = plt.colorbar(sc)
colorbar.set_label('Correlation')

# Set labels
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

ax.set_box_aspect([1, 1, 1])

# Show plot
plt.show()

fig.savefig(save_folder + f'/sample_scatterplot_topsurface_{corner}' + '.png', dpi=1200)
fig.savefig(save_folder + f'/sample_scatterplot_topsurface_{corner}' + '.svg', dpi=900)


if create_rot_gif == 'True':
    # Directory to save frames
    frames_dir = save_folder + '/scattergif'
    os.makedirs(frames_dir, exist_ok=True)
    
    # Create and save frames
    num_frames = 120
    for i in range(num_frames):
        ax.view_init(elev=10, azim=i*(360/num_frames))  # Rotate the view
        frame_path = f"{frames_dir}/frame_{i:02d}.png"
        plt.savefig(frame_path)
        print(f'creating frame {i} of {num_frames} for gif')

    # Create a GIF from the frames
    frames = []
    for i in range(num_frames):
        frame_path = f"{frames_dir}/frame_{i:02d}.png"
        frames.append(imageio.imread(frame_path))

    # Save the GIF
    gif_path = save_folder + '/3d_scatter_rotation.gif'
    imageio.mimsave(gif_path, frames, fps=10)
    


if print_top_number == 'True':
    for k in range(0, len(max_ind)):
        print(f'printing {k} of number_of_prints, max_ind is {max_ind[k]}')
        big_loop(root_folder, save_folder, samples_dir, scale[max_ind[k]], rotation[max_ind[k]], shear[max_ind[k]], threshold[max_ind[k]], grid, unitcell_pixel, sample_size, first_miller, corner, square_shift_pixels[1], k)
