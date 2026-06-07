# -*- coding: utf-8 -*-
"""
Created on Tue May 21 15:13:56 2024

@author: iselire
"""

import multiprocessing as mp
import time
import numpy as np
import math
from numpy import random

import matplotlib.pyplot as plt
import os
import math
import csv
import ast

# scaling the image matrix
import cv2

# to stop script if conditions are not met
import sys

# array to image
from PIL import Image
#text color
import seaborn as sns
# import imageio
import imageio.v2 as imageio


####################################################

def sleep_one_sec():
    print('Sleeping 1 second')
    time.sleep(1)
    print('Waking up')
    
def make_calculation_one(numbers):
    for number in numbers: 
        result_a.append(math.sqrt(number * 3))

def make_calculation_two(numbers):
    for number in numbers: 
        result_b.append(math.sqrt(number * 4))

def make_calculation_three(numbers):
    for number in numbers: 
        result_c.append(math.sqrt(number * 5))
     
# Function to parse the string representation of the array
def parse_nested_array_string(array_string):
    if isinstance(array_string, str):
        # Clean the string to make it evaluable
        clean_string = array_string.replace("[array([", "[[").replace("]), array([", "],[").replace("])]", "]]")
        
        # Use ast.literal_eval to evaluate the cleaned string as a Python expression
        parsed_list = ast.literal_eval(clean_string)
        
        # Convert the parsed list of lists to a list of NumPy arrays
        array_list = [np.array(arr) for arr in parsed_list]
        
        return array_list
    elif isinstance(array_string, np.ndarray):
        return array_string
    else:
        raise ValueError("Input must be a string or a numpy array")
        
        
# Function to read data from CSV file and store in arrays
def read_csv(filename):
    # Initialize empty lists to store data
    column_values = []
    row_des = []
    row_arr = []
    # Open the CSV file for reading
    with open(filename, 'r') as file:
        # Create a CSV reader object
        csv_reader = csv.reader(file)
        # Iterate over each row in the CSV file
        skip_line = 12
        for row in csv_reader:
            # Assuming the CSV file has two columns
            # Append values from each row to respective lists
            if skip_line == 10 or skip_line == 9 or skip_line == 8 or skip_line == 7 or skip_line == 4:
                row_des = row[0]
                row_arr = eval(row[1])
                column_values.append(row_arr)  # Convert to appropriate data type if needed
                skip_line = skip_line - 1
                # print(skip_line)
            elif skip_line == 6:
                # print(f'here {(row[1])}')
                # print(f'here {parse_nested_array_string(row[1])}')
                row_des = row[0]
                row_arr = parse_nested_array_string(row[1])
                column_values.append(row_arr)  # Convert to appropriate data type if needed
                skip_line = skip_line - 1
            else:
                skip_line = skip_line - 1
                print(skip_line)
    return column_values


def save_data_csv(tops, topr, topsh, surface_on_top, singlec, topc, runs, time_sf, save, c, best4, sl, thr):
    # Specify the file path
    gmt = time.gmtime() 
    file_path = save + f'/{gmt[0]}{gmt[1]:02}{gmt[2]:02}_crosscorr_arrays_data_{c}_{iterations}_test_{best4}.csv'
    
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
        {'Description': 'start of iteration', 'Array': sl},
        {'Description': 'threshold', 'Array': thr}
    ]
    
    # Write data to CSV file
    with open(file_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=columns)
        writer.writeheader()
        for row in data:
            writer.writerow(row)

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
    image = Image.fromarray(matrix.astype(np.uint8))
    scaled_image = ((image - np.min(image)) / (np.max(image) - np.min(image)) * 255).astype(np.uint8)
    #plt.imshow(scaled_image)
    #plt.show()
    return scaled_image

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

def perform_crosscorrelation_left(samples_dir, scaling_square, square_shift, bin_im, norm_images, tsn, M, thr,
                                  tF90s, tmF90, mlF90, mmF90,
                                  tF24s, tmF24, mlF24, mmF24,
                                  tL90s, tmL90, mlL90, mmL90,
                                  tL24s, tmL24, mlL24, mmL24, 
                                  tvro, tvtr, tvsc, tvsh, tl, tcarr,thresh,
                                  vro, vtr, vsc, vsh): 
    # Read in sample and model image
    samples = read_images_from_folder(samples_dir)
    sample_F90 = imageio.imread(samples[0][1], pilmode='L').astype(np.uint8)
    sample_F90_crop = sample_F90[square_shift[2][0]:(scaling_square + square_shift[2][0]), square_shift[2][1]:(scaling_square + square_shift[2][1])]
    sample_F24 = imageio.imread(samples[0][0], pilmode='L').astype(np.uint8)
    sample_F24_crop = sample_F24[square_shift[3][0]:(scaling_square + square_shift[3][0]), square_shift[3][1]:(scaling_square + square_shift[3][1])]
    sample_L90 = imageio.imread(samples[0][3], pilmode='L').astype(np.uint8)
    sample_L90_crop = sample_L90[square_shift[0][0]:(scaling_square + square_shift[0][0]), square_shift[0][1]:(scaling_square + square_shift[0][1])]
    sample_L24 = imageio.imread(samples[0][2], pilmode='L').astype(np.uint8)
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

    if (len(tF90s) <= best_combos and a > max(tcarr)):
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
        thresh.append(thr)
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
        del thresh[min_index]
        

    top_transf_matrices = (tmF90, tmF24, tmL90, tmL24)
    top_score = (tF90s, tF24s, tL90s, tL24s)
    max_location = (mlF90, mlF24, mlL90, mlL24)
    bin_im = (mmF90, mmF24, mmL90, mmL24)
    cropped_samples = (sample_F90_crop, sample_F24_crop, sample_L90_crop, sample_L24_crop)
    
    return    top_transf_matrices, top_score, max_location, bin_im, tvsc, tvro, tvsh, tl, thresh, cropped_samples, tcarr



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
            img = imageio.imread(image_path, pilmode='L').astype(np.uint8)
            image_list.append(image_path)
            images.append(img)
    return image_list, images

def plot_4cs(bin_im, label_images, directory, b4, tcorrarr, k):

    bin_im_np_orig = [0, 0, 0, 0]

    if len(tcorrarr) == 2:
        max_ind = sorted(range(len(tcorrarr)), key=lambda sub: tcorrarr[sub])[-1:]

        bin_im_np_orig[0] = bin_im[k][max_ind[0]]
        # plt.imshow(bin_im_np_orig[0])
        cv2.imwrite(directory + f'/{label_images[k]}_crosscorr_match_{b4}_0_{iterations}.png', bin_im_np_orig[0])

    elif len(tcorrarr) == 3:
        max_ind = sorted(range(len(tcorrarr)), key=lambda sub: tcorrarr[sub])[-2:]

        bin_im_np_orig[0] = bin_im[k][max_ind[0]]
        # plt.imshow(bin_im_np_orig[0])
        cv2.imwrite(directory + f'/{label_images[k]}_crosscorr_match_{b4}_0_{iterations}.png', bin_im_np_orig[0])

        bin_im_np_orig[1] = bin_im[k][max_ind[1]]
        # plt.imshow(bin_im_np_orig[1])
        cv2.imwrite(directory + f'/{label_images[k]}_crosscorr_match_{b4}_1_{iterations}.png', bin_im_np_orig[1])

    elif len(tcorrarr) == 4:
        max_ind = sorted(range(len(tcorrarr)), key=lambda sub: tcorrarr[sub])[-3:]

        bin_im_np_orig[0] = bin_im[k][max_ind[0]]
        # plt.imshow(bin_im_np_orig[0])
        cv2.imwrite(directory + f'/{label_images[k]}_crosscorr_match_{b4}_0_{iterations}.png', bin_im_np_orig[0])

        bin_im_np_orig[1] = bin_im[k][max_ind[1]]
        # plt.imshow(bin_im_np_orig[1])
        cv2.imwrite(directory + f'/{label_images[k]}_crosscorr_match_{b4}_1_{iterations}.png', bin_im_np_orig[1])

        bin_im_np_orig[2] = bin_im[k][max_ind[2]]
        # plt.imshow(bin_im_np_orig[2])
        cv2.imwrite(directory + f'/{label_images[k]}_crosscorr_match_{b4}_2_{iterations}.png', bin_im_np_orig[2])

    elif len(tcorrarr) > 4:
        max_ind = sorted(range(len(tcorrarr)), key=lambda sub: tcorrarr[sub])[-4:]

        bin_im_np_orig[0] = bin_im[k][max_ind[0]]
        # plt.imshow(bin_im_np_orig[0])
        cv2.imwrite(directory + f'/{label_images[k]}_crosscorr_match_{b4}_0_{iterations}.png', bin_im_np_orig[0])

        bin_im_np_orig[1] = bin_im[k][max_ind[1]]
        # plt.imshow(bin_im_np_orig[1])
        cv2.imwrite(directory + f'/{label_images[k]}_crosscorr_match_{b4}_1_{iterations}.png', bin_im_np_orig[1])

        bin_im_np_orig[2] = bin_im[k][max_ind[2]]
        # plt.imshow(bin_im_np_orig[2])
        cv2.imwrite(directory + f'/{label_images[k]}_crosscorr_match_{b4}_2_{iterations}.png', bin_im_np_orig[2])

        bin_im_np_orig[3] = bin_im[k][max_ind[3]]
        # plt.imshow(bin_im_np_orig[3])
        cv2.imwrite(directory + f'/{label_images[k]}_crosscorr_match_{b4}_3_{iterations}.png', bin_im_np_orig[3])
    
def print_top_correlations_left(max_color, min_color, color_palette, top_score, tcorrarr, bin_im, cropped_samples, top_v_scaling,
                                top_v_rotations, top_v_shearing, ts, thresh, transparency, transparency_sample, ml, directory, c, b4):
    
    color_scaling = int(round(100*(max_color-min_color), 2))
    color_max = round(max_color, 2)
    color_min = round(min_color, 2)
    # textbox color according to correlation value
    text_values = np.linspace(color_min, color_max, color_scaling)
    # Create a seaborn color palette
    palette = sns.color_palette(color_palette, n_colors=len(text_values))
    ###########################################################################################
    if len(tcorrarr) == 2:
        max_ind = sorted(range(len(tcorrarr)), key=lambda sub: tcorrarr[sub])[-1:]
        plot_n = 1
    elif len(tcorrarr) == 3:
        max_ind = sorted(range(len(tcorrarr)), key=lambda sub: tcorrarr[sub])[-2:]
        plot_n = 2
    elif len(tcorrarr) == 4:
        max_ind = sorted(range(len(tcorrarr)), key=lambda sub: tcorrarr[sub])[-3:]
        plot_n = 3
    elif len(tcorrarr) > 4:
        max_ind = sorted(range(len(tcorrarr)), key=lambda sub: tcorrarr[sub])[-4:]
        plot_n = 4


    fig_corr, axs_corr = plt.subplots(plot_n, cs_per_volume, figsize=(30, 30))
    fig_corr.suptitle(f'Correlation of {samples_dir[:]}', fontsize=16)
        
    # Display the original image
    axs_corr = axs_corr.flatten()
    axs_corr[0].set_title('L90')
    axs_corr[1].set_title('L24')
    axs_corr[2].set_title('F90')
    axs_corr[3].set_title('F24')    
    
    # Display the "best_combos" 
    for i in range(0, plot_n):

        print(f'len is {len(bin_im[0])}')
        #h, w = bin_im[0][i][0].shape
        h, w, z = grid_size_model
        
        iL90 = i*4 + 0
        iL24 = i*4 + 1
        iF90 = i*4 + 2
        iF24 = i*4 + 3
        
        print(f'i is {i}')
        print(color_scaling)

        overlay_L90 = cropped_samples[2].copy()  
        overlay_L90[:, :] = np.uint8((transparency_sample)*overlay_L90[:,:])
        overlay_L90[ml[2][max_ind[i]][1]:ml[2][max_ind[i]][1]+h, ml[2][max_ind[i]][0]:ml[2][max_ind[i]][0]+w] = np.uint8(transparency*bin_im[2][max_ind[i]] + (1-transparency)*overlay_L90[ml[2][max_ind[i]][1]:ml[2][max_ind[i]][1]+h, ml[2][max_ind[i]][0]:ml[2][max_ind[i]][0]+w])
        axs_corr[iL90].imshow(overlay_L90, cmap='gray')
                    
        if top_score[2][max_ind[i]] <= color_min:
            axs_corr[iL90].text(overlay_L90.shape[1] + round(overlay_L90.shape[1]/20), overlay_L90.shape[0] - round(overlay_L90.shape[0]/12), "%.2f" %top_score[2][max_ind[i]], bbox=dict(facecolor=palette[0], alpha=0.5))
        elif top_score[2][max_ind[i]] >= color_max:
            axs_corr[iL90].text(overlay_L90.shape[1] + round(overlay_L90.shape[1]/20), overlay_L90.shape[0] - round(overlay_L90.shape[0]/12), "%.2f" %top_score[2][max_ind[i]], bbox=dict(facecolor=palette[int(color_scaling - 1)], alpha=0.5))
        else:
            axs_corr[iL90].text(overlay_L90.shape[1] + round(overlay_L90.shape[1]/20), overlay_L90.shape[0] - round(overlay_L90.shape[0]/12), "%.2f" %top_score[2][max_ind[i]], bbox=dict(facecolor=palette[int(round(top_score[2][max_ind[i]] - color_min, 2)*100)], alpha=0.5))
    
        overlay_L24 = cropped_samples[3].copy()
        overlay_L24[:, :] = np.uint8((transparency_sample)*overlay_L24[:,:])
        overlay_L24[ml[3][max_ind[i]][1]:ml[3][max_ind[i]][1]+h, ml[3][max_ind[i]][0]:ml[3][max_ind[i]][0]+w] = np.uint8(transparency*bin_im[3][max_ind[i]][:,:] + (1-transparency)*overlay_L24[ml[3][max_ind[i]][1]:ml[3][max_ind[i]][1]+h, ml[3][max_ind[i]][0]:ml[3][max_ind[i]][0]+w])
        axs_corr[iL24].imshow(overlay_L24, cmap='gray')
        
        if top_score[3][max_ind[i]] <= color_min:
            axs_corr[iL24].text(overlay_L24.shape[1] + round(overlay_L24.shape[1]/20), overlay_L24.shape[0] - round(overlay_L24.shape[0]/12), "%.2f" %top_score[3][max_ind[i]], bbox=dict(facecolor=palette[0], alpha=0.5))
        elif top_score[3][max_ind[i]] >= color_max:
            axs_corr[iL24].text(overlay_L24.shape[1] + round(overlay_L24.shape[1]/20), overlay_L24.shape[0] - round(overlay_L24.shape[0]/12), "%.2f" %top_score[3][max_ind[i]], bbox=dict(facecolor=palette[int(color_scaling - 1)], alpha=0.5))
        else:
            axs_corr[iL24].text(overlay_L24.shape[1] + round(overlay_L24.shape[1]/20), overlay_L24.shape[0] - round(overlay_L24.shape[0]/12), "%.2f" %top_score[3][max_ind[i]], bbox=dict(facecolor=palette[int(round(top_score[3][max_ind[i]] - color_min, 2)*100)], alpha=0.5))
    
    
        overlay_F90 = cropped_samples[0].copy()
        overlay_F90[:, :] = np.uint8((transparency_sample)*overlay_F90[:,:])
        overlay_F90[ml[0][max_ind[i]][1]:ml[0][max_ind[i]][1]+h, ml[0][max_ind[i]][0]:ml[0][max_ind[i]][0]+w] = np.uint8(transparency*bin_im[0][max_ind[i]][:,:] + (1-transparency)*overlay_F90[ml[0][max_ind[i]][1]:ml[0][max_ind[i]][1]+h, ml[0][max_ind[i]][0]:ml[0][max_ind[i]][0]+w])
        axs_corr[iF90].imshow(overlay_F90, cmap='gray')
 
        if top_score[0][max_ind[i]] <= color_min:
            axs_corr[iF90].text(overlay_F90.shape[1] + round(overlay_F90.shape[1]/20), overlay_F90.shape[0] - round(overlay_F90.shape[0]/12), "%.2f" %top_score[0][max_ind[i]], bbox=dict(facecolor=palette[0], alpha=0.5))
        elif top_score[0][max_ind[i]] >= color_max:
            axs_corr[iF90].text(overlay_F90.shape[1] + round(overlay_F90.shape[1]/20), overlay_F90.shape[0] - round(overlay_F90.shape[0]/12), "%.2f" %top_score[0][max_ind[i]], bbox=dict(facecolor=palette[int(color_scaling - 1)], alpha=0.5))
        else:
            axs_corr[iF90].text(overlay_F90.shape[1] + round(overlay_F90.shape[1]/20), overlay_F90.shape[0] - round(overlay_F90.shape[0]/12), "%.2f" %top_score[0][max_ind[i]], bbox=dict(facecolor=palette[int(round(top_score[0][max_ind[i]] - color_min, 2)*100)], alpha=0.5))
    
        
        overlay_F24 = cropped_samples[1].copy()
        overlay_F24[:, :] = np.uint8((transparency_sample)*overlay_F24[:,:])
        overlay_F24[ml[1][max_ind[i]][1]:ml[1][max_ind[i]][1]+h, ml[1][max_ind[i]][0]:ml[1][max_ind[i]][0]+w] = np.uint8(transparency*bin_im[1][max_ind[i]][:,:] + (1-transparency)*overlay_F24[ml[1][max_ind[i]][1]:ml[1][max_ind[i]][1]+h, ml[1][max_ind[i]][0]:ml[1][max_ind[i]][0]+w])
        axs_corr[iF24].imshow(overlay_F24, cmap='gray')
        
        if top_score[1][max_ind[i]] <= color_min:
            axs_corr[iF24].text(overlay_F24.shape[1] + round(overlay_F24.shape[1]/20), overlay_F24.shape[0] - round(overlay_F24.shape[0]/12), "%.2f" %top_score[1][max_ind[i]], bbox=dict(facecolor=palette[0], alpha=0.5))
        elif top_score[1][max_ind[i]] >= color_max:
            axs_corr[iF24].text(overlay_F24.shape[1] + round(overlay_F24.shape[1]/20), overlay_F24.shape[0] - round(overlay_F24.shape[0]/12), "%.2f" %top_score[1][max_ind[i]], bbox=dict(facecolor=palette[int(color_scaling - 1)], alpha=0.5))
        else:
            axs_corr[iF24].text(overlay_F24.shape[1] + round(overlay_F24.shape[1]/20), overlay_F24.shape[0] - round(overlay_F24.shape[0]/12), "%.2f" %top_score[1][max_ind[i]], bbox=dict(facecolor=palette[int(round(top_score[1][max_ind[i]] - color_min, 2)*100)], alpha=0.5))


        top_corr = 1/2*np.sqrt(pow(top_score[2][max_ind[i]], 2) + pow(top_score[3][max_ind[i]], 2) + pow(top_score[0][max_ind[i]], 2) + pow(top_score[1][max_ind[i]], 2))
        axs_corr[iF24].text(overlay_F24.shape[1] + round(overlay_F24.shape[0]/20), overlay_F24.shape[0] - round(overlay_F24.shape[0]/3.5),
                            f'threshold \n {thresh[max_ind[i]]} \nscaling \n {top_v_scaling[max_ind[i]][0]} \n {top_v_scaling[max_ind[i]][1]} \n {top_v_scaling[max_ind[i]][2]}' +
                            f'\nrotations \n {top_v_rotations[max_ind[i]][0]} \n {top_v_rotations[max_ind[i]][1]} \n {top_v_rotations[max_ind[i]][2]}'+
                            f'\nshearing \n {top_v_shearing[max_ind[i]][0]} \n {top_v_shearing[max_ind[i]][1]} \n {top_v_shearing[max_ind[i]][2]}'+
                            f'\n {top_v_shearing[max_ind[i]][3]} \n {top_v_shearing[max_ind[i]][4]} \n {top_v_shearing[max_ind[i]][5]}'+
                            f'\ntopsurface \nh {round(ts[max_ind[i]][0], 2)} \nk {round(ts[max_ind[i]][1], 2)} \nl {round(ts[max_ind[i]][2], 2)}'+
                            f'\ntotal corr {top_corr}')
        
        if top_corr <= color_min:
            axs_corr[iF24].text(overlay_F24.shape[1] + round(overlay_F24.shape[0]/20), overlay_F24.shape[0] - round(overlay_F24.shape[0]/6), "%.2f" %top_corr, bbox=dict(facecolor=palette[0], alpha=0.5))
        elif top_corr >= color_max:
            axs_corr[iF24].text(overlay_F24.shape[1] + round(overlay_F24.shape[0]/20), overlay_F24.shape[0] - round(overlay_F24.shape[0]/6), "%.2f" %top_corr, bbox=dict(facecolor=palette[int(color_scaling - 1)], alpha=0.5))
        else:
            axs_corr[iF24].text(overlay_F24.shape[1] + round(overlay_F24.shape[0]/20), overlay_F24.shape[0] - round(overlay_F24.shape[0]/6), "%.2f" %top_corr, bbox=dict(facecolor=palette[int(round(top_score[1][max_ind[i]] - color_min, 2)*100)-1], alpha=0.5))

    # record current timestamp
    gmt = time.gmtime()        
    plt.savefig(directory + f'/{gmt[0]}{gmt[1]:02}{gmt[2]:02}_crosscorr_plot_{c}_{iterations}_test_{b4}.png')
    plt.savefig(directory + f'/{gmt[0]}{gmt[1]:02}{gmt[2]:02}_crosscorr_plot_{c}_{iterations}_test_{b4}.svg')


def print_top_correlations_right(max_color, min_color, color_palette, top_score, tcorrarr, bin_im, cropped_samples, top_v_scaling,
                                top_v_rotations, top_v_shearing, ts, thresh, transparency, transparency_sample, ml, directory, c, b4):
    color_scaling = int(round(100 * (max_color - min_color), 2))
    color_max = round(max_color, 2)
    color_min = round(min_color, 2)
    # textbox color according to correlation value
    text_values = np.linspace(color_min, color_max, color_scaling)
    # Create a seaborn color palette
    palette = sns.color_palette(color_palette, n_colors=len(text_values))
    ###########################################################################################
    if len(tcorrarr) == 2:
        max_ind = sorted(range(len(tcorrarr)), key=lambda sub: tcorrarr[sub])[-1:]
        plot_n = 1
    elif len(tcorrarr) == 3:
        max_ind = sorted(range(len(tcorrarr)), key=lambda sub: tcorrarr[sub])[-2:]
        plot_n = 2
    elif len(tcorrarr) == 4:
        max_ind = sorted(range(len(tcorrarr)), key=lambda sub: tcorrarr[sub])[-3:]
        plot_n = 3
    elif len(tcorrarr) > 4:
        max_ind = sorted(range(len(tcorrarr)), key=lambda sub: tcorrarr[sub])[-4:]
        plot_n = 4

    fig_corr, axs_corr = plt.subplots(plot_n, cs_per_volume, figsize=(30, 30))
    fig_corr.suptitle(f'Correlation of {samples_dir[:]}', fontsize=16)

    # Display the original image
    axs_corr = axs_corr.flatten()
    axs_corr[0].set_title('F90')
    axs_corr[1].set_title('F24')
    axs_corr[2].set_title('R90')
    axs_corr[3].set_title('R24')


    # Display the "best_combos"
    for i in range(0, plot_n):
        #h, w = bin_im[0][i].shape
        h, w, z = grid_size_model

        iF90 = i * 4 + 0
        iF24 = i * 4 + 1
        iR90 = i * 4 + 2
        iR24 = i * 4 + 3

        # print(f'i is {i}')
        # print(color_scaling)
        overlay_R90 = cropped_samples[2].copy()
        overlay_R90[:, :] = np.uint8((transparency_sample) * overlay_R90[:, :])
        overlay_R90[ml[2][max_ind[i]][1]:ml[2][max_ind[i]][1] + h,
        ml[2][max_ind[i]][0]:ml[2][max_ind[i]][0] + w] = np.uint8(
            transparency * bin_im[2][max_ind[i]][:, :] + (1 - transparency) * overlay_R90[
                                                                              ml[2][max_ind[i]][1]:ml[2][max_ind[i]][
                                                                                                       1] + h,
                                                                              ml[2][max_ind[i]][0]:ml[2][max_ind[i]][
                                                                                                       0] + w])
        axs_corr[iR90].imshow(overlay_R90, cmap='gray')

        if top_score[2][max_ind[i]] <= color_min:
            axs_corr[iR90].text(overlay_R90.shape[1] + round(overlay_R90.shape[1] / 20),
                                overlay_R90.shape[0] - round(overlay_R90.shape[0] / 12),
                                "%.2f" % top_score[2][max_ind[i]], bbox=dict(facecolor=palette[0], alpha=0.5))
        elif top_score[2][max_ind[i]] >= color_max:
            axs_corr[iR90].text(overlay_R90.shape[1] + round(overlay_R90.shape[1] / 20),
                                overlay_R90.shape[0] - round(overlay_R90.shape[0] / 12),
                                "%.2f" % top_score[2][max_ind[i]],
                                bbox=dict(facecolor=palette[int(color_scaling - 1)], alpha=0.5))
        else:
            axs_corr[iR90].text(overlay_R90.shape[1] + round(overlay_R90.shape[1] / 20),
                                overlay_R90.shape[0] - round(overlay_R90.shape[0] / 12),
                                "%.2f" % top_score[2][max_ind[i]],
                                bbox=dict(facecolor=palette[int(round(top_score[2][max_ind[i]] - color_min, 2) * 100)],
                                          alpha=0.5))

        overlay_R24 = cropped_samples[3].copy()
        overlay_R24[:, :] = np.uint8((transparency_sample) * overlay_R24[:, :])
        overlay_R24[ml[3][max_ind[i]][1]:ml[3][max_ind[i]][1] + h,
        ml[3][max_ind[i]][0]:ml[3][max_ind[i]][0] + w] = np.uint8(
            transparency * bin_im[3][max_ind[i]][:, :] + (1 - transparency) * overlay_R24[
                                                                              ml[3][max_ind[i]][1]:ml[3][max_ind[i]][
                                                                                                       1] + h,
                                                                              ml[3][max_ind[i]][0]:ml[3][max_ind[i]][
                                                                                                       0] + w])
        axs_corr[iR24].imshow(overlay_R24, cmap='gray')

        if top_score[3][max_ind[i]] <= color_min:
            axs_corr[iR24].text(overlay_R24.shape[1] + round(overlay_R24.shape[1] / 20),
                                overlay_R24.shape[0] - round(overlay_R24.shape[0] / 12),
                                "%.2f" % top_score[3][max_ind[i]], bbox=dict(facecolor=palette[0], alpha=0.5))
        elif top_score[3][max_ind[i]] >= color_max:
            axs_corr[iR24].text(overlay_R24.shape[1] + round(overlay_R24.shape[1] / 20),
                                overlay_R24.shape[0] - round(overlay_R24.shape[0] / 12),
                                "%.2f" % top_score[3][max_ind[i]],
                                bbox=dict(facecolor=palette[int(color_scaling - 1)], alpha=0.5))
        else:
            axs_corr[iR24].text(overlay_R24.shape[1] + round(overlay_R24.shape[1] / 20),
                                overlay_R24.shape[0] - round(overlay_R24.shape[0] / 12),
                                "%.2f" % top_score[3][max_ind[i]],
                                bbox=dict(facecolor=palette[int(round(top_score[3][max_ind[i]] - color_min, 2) * 100)],
                                          alpha=0.5))

        overlay_F90 = cropped_samples[0].copy()
        overlay_F90[:, :] = np.uint8((transparency_sample) * overlay_F90[:, :])
        overlay_F90[ml[0][max_ind[i]][1]:ml[0][max_ind[i]][1] + h,
        ml[0][max_ind[i]][0]:ml[0][max_ind[i]][0] + w] = np.uint8(
            transparency * bin_im[0][max_ind[i]][:, :] + (1 - transparency) * overlay_F90[
                                                                              ml[0][max_ind[i]][1]:ml[0][max_ind[i]][
                                                                                                       1] + h,
                                                                              ml[0][max_ind[i]][0]:ml[0][max_ind[i]][
                                                                                                       0] + w])
        axs_corr[iF90].imshow(overlay_F90, cmap='gray')

        if top_score[0][max_ind[i]] <= color_min:
            axs_corr[iF90].text(overlay_F90.shape[1] + round(overlay_F90.shape[1] / 20),
                                overlay_F90.shape[0] - round(overlay_F90.shape[0] / 12),
                                "%.2f" % top_score[0][max_ind[i]], bbox=dict(facecolor=palette[0], alpha=0.5))
        elif top_score[0][max_ind[i]] >= color_max:
            axs_corr[iF90].text(overlay_F90.shape[1] + round(overlay_F90.shape[1] / 20),
                                overlay_F90.shape[0] - round(overlay_F90.shape[0] / 12),
                                "%.2f" % top_score[0][max_ind[i]],
                                bbox=dict(facecolor=palette[int(color_scaling - 1)], alpha=0.5))
        else:
            axs_corr[iF90].text(overlay_F90.shape[1] + round(overlay_F90.shape[1] / 20),
                                overlay_F90.shape[0] - round(overlay_F90.shape[0] / 12),
                                "%.2f" % top_score[0][max_ind[i]],
                                bbox=dict(facecolor=palette[int(round(top_score[0][max_ind[i]] - color_min, 2) * 100)],
                                          alpha=0.5))

        overlay_F24 = cropped_samples[1].copy()
        overlay_F24[:, :] = np.uint8((transparency_sample) * overlay_F24[:, :])
        overlay_F24[ml[1][max_ind[i]][1]:ml[1][max_ind[i]][1] + h,
        ml[1][max_ind[i]][0]:ml[1][max_ind[i]][0] + w] = np.uint8(
            transparency * bin_im[1][max_ind[i]][:, :] + (1 - transparency) * overlay_F24[
                                                                              ml[1][max_ind[i]][1]:ml[1][max_ind[i]][
                                                                                                       1] + h,
                                                                              ml[1][max_ind[i]][0]:ml[1][max_ind[i]][
                                                                                                       0] + w])
        axs_corr[iF24].imshow(overlay_F24, cmap='gray')

        if top_score[1][max_ind[i]] <= color_min:
            axs_corr[iF24].text(overlay_F24.shape[1] + round(overlay_F24.shape[1] / 20),
                                overlay_F24.shape[0] - round(overlay_F24.shape[0] / 12),
                                "%.2f" % top_score[1][max_ind[i]], bbox=dict(facecolor=palette[0], alpha=0.5))
        elif top_score[1][max_ind[i]] >= color_max:
            axs_corr[iF24].text(overlay_F24.shape[1] + round(overlay_F24.shape[1] / 20),
                                overlay_F24.shape[0] - round(overlay_F24.shape[0] / 12),
                                "%.2f" % top_score[1][max_ind[i]],
                                bbox=dict(facecolor=palette[int(color_scaling - 1)], alpha=0.5))
        else:
            axs_corr[iF24].text(overlay_F24.shape[1] + round(overlay_F24.shape[1] / 20),
                                overlay_F24.shape[0] - round(overlay_F24.shape[0] / 12),
                                "%.2f" % top_score[1][max_ind[i]],
                                bbox=dict(facecolor=palette[int(round(top_score[1][max_ind[i]] - color_min, 2) * 100)],
                                          alpha=0.5))

        top_corr = 1 / 2 * np.sqrt(
            pow(top_score[2][max_ind[i]], 2) + pow(top_score[3][max_ind[i]], 2) + pow(top_score[0][max_ind[i]],
                                                                                      2) + pow(top_score[1][max_ind[i]],
                                                                                               2))


        axs_corr[iR24].text(overlay_F24.shape[1] + round(overlay_F24.shape[0] / 20),
                            overlay_F24.shape[0] - round(overlay_F24.shape[0] / 3.5),
                            f'threshold \n {thresh[max_ind[i]]} \nscaling \n {top_v_scaling[max_ind[i]][0]} \n {top_v_scaling[max_ind[i]][1]} \n {top_v_scaling[max_ind[i]][2]}' +
                            f'\nrotations \n {top_v_rotations[max_ind[i]][0]} \n {top_v_rotations[max_ind[i]][1]} \n {top_v_rotations[max_ind[i]][2]}' +
                            f'\nshearing \n {top_v_shearing[max_ind[i]][0]} \n {top_v_shearing[max_ind[i]][1]} \n {top_v_shearing[max_ind[i]][2]}' +
                            f'\n {top_v_shearing[max_ind[i]][3]} \n {top_v_shearing[max_ind[i]][4]} \n {top_v_shearing[max_ind[i]][5]}' +
                            f'\ntopsurface \nh {round(ts[max_ind[i]][0], 2)} \nk {round(ts[max_ind[i]][1], 2)} \nl {round(ts[max_ind[i]][2], 2)}' +
                            f'\ntotal corr {top_corr}')
        # print(ts)
        if top_corr <= color_min:
            axs_corr[iR24].text(overlay_F24.shape[1] + round(overlay_F24.shape[0] / 20),
                                overlay_F24.shape[0] - round(overlay_F24.shape[0] / 6), "%.2f" % top_corr,
                                bbox=dict(facecolor=palette[0], alpha=0.5))
        elif top_corr >= color_max:
            axs_corr[iR24].text(overlay_F24.shape[1] + round(overlay_F24.shape[0] / 20),
                                overlay_F24.shape[0] - round(overlay_F24.shape[0] / 6), "%.2f" % top_corr,
                                bbox=dict(facecolor=palette[int(color_scaling - 1)], alpha=0.5))
        else:
            axs_corr[iR24].text(overlay_F24.shape[1] + round(overlay_F24.shape[0] / 20),
                                overlay_F24.shape[0] - round(overlay_F24.shape[0] / 6), "%.2f" % top_corr, bbox=dict(
                    facecolor=palette[int(round(top_score[1][max_ind[i]] - color_min, 2) * 100) - 1], alpha=0.5))

    # record current timestamp
    gmt = time.gmtime()
    plt.savefig(directory + f'/{gmt[0]}{gmt[1]:02}{gmt[2]:02}_crosscorr_plot_{c}_{iterations}_test_{b4}.png')
    plt.savefig(directory + f'/{gmt[0]}{gmt[1]:02}{gmt[2]:02}_crosscorr_plot_{c}_{iterations}_test_{b4}.svg')



def big_loop(sf, sd, gs, uc, miller, co, square_shift, iv_inp):
    
    # Defining the original basis vectors (example values)
    original_basis_vectors = np.array([[1, 0, 0],
                                       [0, 1, 0],
                                       [0, 0, 1]])
    
    # crosscorrelation
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
    
    max_location_F90 = [0]
    max_location_F24 = [0]
    max_location_L90 = [0]
    max_location_L24 = [0]

    top_v_rotations = []
    top_v_translations = []
    top_v_scaling = []
    top_v_shearing = []
    top_v_threshold = []
    top_v_surface = []
    top_v_singlecorr = []
    top_corr_arr = []
    top_v_max_location = []
    top_v_model_matrices = []
    top_v_cropped_samples = []
    top_label = []
    start_label = []
    for i in range(0, starting_points):
        top_v_rotations.append([0])
        top_v_translations.append([0])
        top_v_scaling.append([0])
        top_v_shearing.append([0])
        top_v_threshold.append([0])
        top_v_surface.append([0])
        top_v_singlecorr.append([0])
        top_corr_arr.append([0])
        top_v_max_location.append([0])
        top_v_model_matrices.append([0])
        top_v_cropped_samples.append([0])
        top_label.append([0])
        start_label.append([0])
    
    # Read in sample and model image
    time_start = time.time()
    for starting_nr in range(0, starting_points):
        w = 0
        t = iv_inp[0][starting_nr]
        v_translations = (0, 0, 0)
        v_scaling = [iv_inp[1][starting_nr][0],
                     iv_inp[1][starting_nr][1],
                     iv_inp[1][starting_nr][2]]
        v_rotations = [iv_inp[2][starting_nr][0],
                       iv_inp[2][starting_nr][1],
                       iv_inp[2][starting_nr][2]]
        v_shearing = [iv_inp[3][starting_nr][0],
                      iv_inp[3][starting_nr][1],
                      iv_inp[3][starting_nr][2],
                      iv_inp[3][starting_nr][3],
                      iv_inp[3][starting_nr][4],
                      iv_inp[3][starting_nr][5]]

        # performing crosscorrelation of the model with the 4 FIB SEM cuts
        binary_images, norm_images, labels, matrices, A, topsurface = create_4cs(v_scaling, v_rotations,
                                                                                 v_shearing, gs, uc,
                                                                                 miller, t)

        top_transf_matrices, top_score, max_location, model_matrices, top_v_sca, top_v_rot, top_v_she, topsurfaces, t, cropped_samples, top_corr_a = perform_crosscorrelation_left(
            sd, sample_pixel_size, square_shift, binary_images, norm_images, topsurface, matrices, t,
            top_F90_scores, top_matrix_F90, max_location_F90, model_matrices_F90,
            top_F24_scores, top_matrix_F24, max_location_F24, model_matrices_F24,
            top_L90_scores, top_matrix_L90, max_location_L90, model_matrices_L90,
            top_L24_scores, top_matrix_L24, max_location_L24, model_matrices_L24,
            top_v_rotations[starting_nr], top_v_translations[starting_nr], top_v_scaling[starting_nr], top_v_shearing[starting_nr], top_label[starting_nr], top_corr_arr[starting_nr],
            top_v_threshold[starting_nr],
            v_rotations, v_translations, v_scaling, v_shearing)

        start_label[starting_nr] = [float(x) for x in topsurfaces[1]]

        if len(top_corr_a) > 1:
            v_corr = top_corr_a[1]
        elif len(top_corr_a) == 1:
            v_corr = top_corr_a[0]

        while (w < iterations):
        # for x in range(0, 360, 10):
        #     for y in range(0, 360, 10):
        #         for z in range(0, 360, 10):

                    if v_corr > max(top_corr_arr[starting_nr]):
                        t = iv_inp[0][starting_nr]
                        v_translations = (0, 0, 0)
                        v_scaling = [iv_inp[1][starting_nr][0],
                                     iv_inp[1][starting_nr][1],
                                     iv_inp[1][starting_nr][2]]
                        v_rotations = [iv_inp[2][starting_nr][0],
                                       iv_inp[2][starting_nr][1],
                                       iv_inp[2][starting_nr][2]]
                        v_shearing = [iv_inp[3][starting_nr][0],
                                      iv_inp[3][starting_nr][1],
                                      iv_inp[3][starting_nr][2],
                                      iv_inp[3][starting_nr][3],
                                      iv_inp[3][starting_nr][4],
                                      iv_inp[3][starting_nr][5]]

                        # performing crosscorrelation of the model with the 4 FIB SEM cuts
                        binary_images, norm_images, labels, matrices, A, topsurface = create_4cs(v_scaling, v_rotations,
                                                                                                 v_shearing, gs, uc,
                                                                                                 miller, t)
                        top_transf_matrices, top_score, max_location, model_matrices, top_v_sca, top_v_rot, top_v_she, topsurfaces, t, cropped_samples, top_corr_a = perform_crosscorrelation_left(
                            sd, sample_pixel_size, square_shift, binary_images, norm_images, topsurface, matrices, t,
                            top_F90_scores, top_matrix_F90, max_location_F90, model_matrices_F90,
                            top_F24_scores, top_matrix_F24, max_location_F24, model_matrices_F24,
                            top_L90_scores, top_matrix_L90, max_location_L90, model_matrices_L90,
                            top_L24_scores, top_matrix_L24, max_location_L24, model_matrices_L24,
                            top_v_rotations[starting_nr], top_v_translations[starting_nr], top_v_scaling[starting_nr], top_v_shearing[starting_nr], top_label[starting_nr], top_corr_arr[starting_nr], top_v_threshold[starting_nr],
                            v_rotations, v_translations, v_scaling, v_shearing)

                    if v_corr == max(top_corr_arr[starting_nr]):
                        #t = iv_inp[0][starting_nr]
                        t = iv_inp[0][starting_nr] + pow(-1, random.randint(10)) * wobble_threshold * float(random.randint(100)) / 100
                        v_translations = (0, 0, 0)
                        vsc_x = iv_inp[1][starting_nr][0] + pow(-1, random.randint(10)) * np.random.randint(0, 100)/100 * wobble_sca
                        vsc_y = iv_inp[1][starting_nr][1] + pow(-1, random.randint(10)) * np.random.randint(0, 100)/100 * wobble_sca
                        vsc_z = 1/(vsc_x * vsc_y)
                        v_scaling = [vsc_x, vsc_y, vsc_z]
                        v_scaling = [1, 1, 1]
                        v_rotations = [iv_inp[2][starting_nr][0] + pow(-1, random.randint(10)) * np.random.randint(0, 100)/100 * wobble_rot,
                                       iv_inp[2][starting_nr][1] + pow(-1, random.randint(10)) * np.random.randint(0, 100)/100 * wobble_rot,
                                       iv_inp[2][starting_nr][2] + pow(-1, random.randint(10)) * np.random.randint(0, 100)/100 * wobble_rot]
                        v_shearing = [iv_inp[3][starting_nr][0] + pow(-1, random.randint(10)) * np.random.randint(0, 100)/100 * wobble_she,
                                      iv_inp[3][starting_nr][1] + pow(-1, random.randint(10)) * np.random.randint(0, 100)/100 * wobble_she,
                                      iv_inp[3][starting_nr][2] + pow(-1, random.randint(10)) * np.random.randint(0, 100)/100 * wobble_she,
                                      iv_inp[3][starting_nr][3] + pow(-1, random.randint(10)) * np.random.randint(0, 100)/100 * wobble_she,
                                      iv_inp[3][starting_nr][4] + pow(-1, random.randint(10)) * np.random.randint(0, 100)/100 * wobble_she,
                                      iv_inp[3][starting_nr][5] + pow(-1, random.randint(10)) * np.random.randint(0, 100)/100 * wobble_she]
                        v_shearing = [0, 0, 0, 0, 0, 0]
                        # performing crosscorrelation of the model with the 4 FIB SEM cuts
                        binary_images, norm_images, labels, matrices, A, topsurface = create_4cs(v_scaling, v_rotations,
                                                                                                 v_shearing, gs, uc,
                                                                                                 miller, t)
                        top_transf_matrices, top_score, max_location, model_matrices, top_v_sca, top_v_rot, top_v_she, topsurfaces, t, cropped_samples, top_corr_a = perform_crosscorrelation_left(
                            sd, sample_pixel_size, square_shift, binary_images, norm_images, topsurface, matrices, t,
                            top_F90_scores, top_matrix_F90, max_location_F90, model_matrices_F90,
                            top_F24_scores, top_matrix_F24, max_location_F24, model_matrices_F24,
                            top_L90_scores, top_matrix_L90, max_location_L90, model_matrices_L90,
                            top_L24_scores, top_matrix_L24, max_location_L24, model_matrices_L24,
                            top_v_rotations[starting_nr], top_v_translations[starting_nr], top_v_scaling[starting_nr], top_v_shearing[starting_nr], top_label[starting_nr], top_corr_arr[starting_nr], top_v_threshold[starting_nr],
                            v_rotations, v_translations, v_scaling, v_shearing)

                    elif v_corr < max(top_corr_arr[starting_nr]):
                        max_ind = top_corr_arr[starting_nr].index(max(top_corr_arr[starting_nr]))
                        #t = iv_inp[0][starting_nr]
                        t = top_v_threshold[starting_nr][max_ind] + pow(-1, random.randint(10)) * wobble_threshold * float(random.randint(100)) / 100

                        # v_rotations = (float(random.randint(3600))/10, float(random.randint(3600))/10, float(random.randint(3600))/10)
                        v_translations = (0, 0, 0)
                        # v_scaling = (1 + pow(-1, random.randint(10))*float(random.randint(30))/100, 1 + pow(-1, random.randint(10))*float(random.randint(30))/100, 1 + pow(-1, random.randint(10))*float(random.randint(30))/100)
                        # v_shearing = (pow(-1, random.randint(10))*float(random.randint(10))/100, pow(-1, random.randint(10))*float(random.randint(10))/100, pow(-1, random.randint(10))*float(random.randint(10))/100, pow(-1, random.randint(10))*float(random.randint(10))/100, pow(-1, random.randint(10))*float(random.randint(10))/100, pow(-1, random.randint(10))*float(random.randint(10))/100)

                        # v_rotations = (0, 0, 0)
                        # v_rotations = (0+x, 0+y, 0+z)
                        # v_translations = (0, 0, 0)
                        #v_scaling = (1, 1, 1)
                        #v_shearing = (0, 0, 0, 0, 0, 0)

                        v_scaling = [top_v_scaling[starting_nr][max_ind][0] + pow(-1, random.randint(10)) * np.random.randint(0, 100)/100 * wobble_sca,
                                     top_v_scaling[starting_nr][max_ind][1] + pow(-1, random.randint(10)) * np.random.randint(0, 100)/100 * wobble_sca,
                                     top_v_scaling[starting_nr][max_ind][2] + pow(-1, random.randint(10)) * np.random.randint(0, 100)/100 * wobble_sca]
                        v_rotations = [top_v_rotations[starting_nr][max_ind][0] + pow(-1, random.randint(10)) * np.random.randint(0, 100)/100 * wobble_rot,
                                       top_v_rotations[starting_nr][max_ind][1] + pow(-1, random.randint(10)) * np.random.randint(0, 100)/100 * wobble_rot,
                                       top_v_rotations[starting_nr][max_ind][2] + pow(-1, random.randint(10)) * np.random.randint(0, 100)/100 * wobble_rot]
                        v_shearing = [top_v_shearing[starting_nr][max_ind][0] + pow(-1, random.randint(10)) * np.random.randint(0, 100)/100 * wobble_she,
                                      top_v_shearing[starting_nr][max_ind][1] + pow(-1, random.randint(10)) * np.random.randint(0, 100)/100 * wobble_she,
                                      top_v_shearing[starting_nr][max_ind][2] + pow(-1, random.randint(10)) * np.random.randint(0, 100)/100 * wobble_she,
                                      top_v_shearing[starting_nr][max_ind][3] + pow(-1, random.randint(10)) * np.random.randint(0, 100)/100 * wobble_she,
                                      top_v_shearing[starting_nr][max_ind][4] + pow(-1, random.randint(10)) * np.random.randint(0, 100)/100 * wobble_she,
                                      top_v_shearing[starting_nr][max_ind][5] + pow(-1, random.randint(10)) * np.random.randint(0, 100)/100 * wobble_she]

                        # performing crosscorrelation of the model with the 4 FIB SEM cuts
                        binary_images, norm_images, labels, matrices, A, topsurface = create_4cs(v_scaling, v_rotations, v_shearing, gs, uc, miller, t)
                        top_transf_matrices, top_score, max_location, model_matrices, top_v_sca, top_v_rot, top_v_she, topsurfaces, t, cropped_samples, top_corr_a = perform_crosscorrelation_left(sd, sample_pixel_size, square_shift, binary_images, norm_images, topsurface, matrices, t,
                                                                                                                                                                                                               top_F90_scores, top_matrix_F90, max_location_F90, model_matrices_F90,
                                                                                                                                                                                                           top_F24_scores, top_matrix_F24, max_location_F24, model_matrices_F24, 
                                                                                                                                                                                                           top_L90_scores, top_matrix_L90, max_location_L90, model_matrices_L90, 
                                                                                                                                                                                                           top_L24_scores, top_matrix_L24, max_location_L24, model_matrices_L24, 
                                                                                                                                                                                                           top_v_rotations[starting_nr], top_v_translations[starting_nr], top_v_scaling[starting_nr], top_v_shearing[starting_nr], top_label[starting_nr], top_corr_arr[starting_nr], top_v_threshold[starting_nr],
                                                                                                                                                                                                           v_rotations, v_translations, v_scaling, v_shearing)
                    ####
                    

                    #print('here')
                    #print(top_v_rot)
                    #print(topsurfaces)
                    #print(top_corr_a)
                    #print(max(top_corr_arr)[0])
                    top_corr_arr[starting_nr] = [float(x) for x in top_corr_a[:]]
                    top_corr_arr[starting_nr][0] = 0
                    #print(top_corr_arr[starting_nr])
                    top_v_rotations[starting_nr] = top_v_rot
                    top_v_scaling[starting_nr] = top_v_sca
                    top_v_shearing[starting_nr] = top_v_she
                    top_v_threshold[starting_nr] = t
                    top_v_surface[starting_nr] = topsurfaces
                    arr = np.array(topsurfaces[1:]).astype(float)  # use astype here
                    topsurfaces = arr.tolist()  # back to plain list
                    #print(lst_float)
                    top_v_surface_adapted = [0]
                    [top_v_surface_adapted.append(x) for x in topsurfaces]
                    #print(topsurfaces[:])
                    #print(top_v_surface_adapted)
                    top_v_singlecorr[starting_nr] = top_score
                    top_v_max_location[starting_nr] = max_location
                    top_v_model_matrices[starting_nr] = model_matrices
                    top_v_cropped_samples[starting_nr] = cropped_samples
                    
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
            
                    save_data_csv(top_v_scaling[starting_nr][:], top_v_rotations[starting_nr][:], top_v_shearing[starting_nr][:], top_v_surface_adapted, top_v_singlecorr[starting_nr][:], top_corr_arr[starting_nr][:], np.array(iterations), time_sofar_used, sf, co, starting_nr, start_label[starting_nr][:], top_v_threshold[starting_nr][:])
                    # print(f'corr is: {top_corr_arr}')
                    w = w + 1
                    # print(f'xyz is {v_rotations}')

                    percentage = (float(starting_nr)*float(iterations) + float(round(w))) / (float(starting_points)*float(iterations)) * 100
                    print(f'{round(percentage, 2)}% with {len(top_corr_arr[starting_nr])} of {best_combos} saved iterations')
                    print(f'loop {starting_nr+1} of {starting_points}')

                    # Convert elapsed time to days, hours, minutes, and seconds
                    days = int(elapsed_time_seconds // (24 * 3600))
                    hours = int((elapsed_time_seconds % (24 * 3600)) // 3600)
                    minutes = int((elapsed_time_seconds % 3600) // 60)
                    seconds = int(elapsed_time_seconds % 60)

                    time_sofar_used = f'elapsed time: {days}d, {hours}h, {minutes}min, {seconds}s'
                    print(time_sofar_used)
                    estimated_seconds = float(elapsed_time_seconds) / (float(percentage)) * 100
                    # Convert elapsed time to days, hours, minutes, and seconds
                    days = int(estimated_seconds // (24 * 3600))
                    hours = int((estimated_seconds % (24 * 3600)) // 3600)
                    minutes = int((estimated_seconds % 3600) // 60)
                    seconds = int(estimated_seconds % 60)

                    estimated_time = f'estimated time: {days}d, {hours}h, {minutes}min, {seconds}s'
                    print(estimated_time)
           
        # color coding
        max_color = 1
        min_color = 0
        color_palette = ("viridis")
        transparency_overlay = 0.1
        transparency_sample = 0.05

        for i in range(0, cs_per_volume):

            plot_4cs(top_v_model_matrices[starting_nr], labels, sf, starting_nr, top_corr_arr[starting_nr], i)


        if co == 'L':
            print_top_correlations_left(max_color, min_color, color_palette, top_v_singlecorr[starting_nr], top_corr_arr[starting_nr],
                                        top_v_model_matrices[starting_nr], top_v_cropped_samples[starting_nr],
                                        top_v_scaling[starting_nr], top_v_rotations[starting_nr], top_v_shearing[starting_nr], top_v_surface[starting_nr], top_v_threshold[starting_nr],
                                        transparency_overlay, transparency_sample, top_v_max_location[starting_nr], sf, co, starting_nr)
        if co == 'R':
            print_top_correlations_right(max_color, min_color, color_palette, top_v_singlecorr[starting_nr], top_corr_arr[starting_nr],
                                         top_v_model_matrices[starting_nr], top_v_cropped_samples[starting_nr],
                                        top_v_scaling[starting_nr], top_v_rotations[starting_nr], top_v_shearing[starting_nr], top_v_surface[starting_nr], top_v_threshold[starting_nr],
                                        transparency_overlay, transparency_sample, top_v_max_location[starting_nr], sf, co, starting_nr)

def dir_exists(directory):
    # Check if the directory already exists
    if not os.path.exists(directory):
        # Create the directory
        os.makedirs(directory)
        print("Directory created successfully!")
    else:
        print("Directory already exists!")
        
        
####################################################

result_a = []
result_b = []
result_c = []

# directory for saving the files
saving_files = ["C:/Users/iselire/OneDrive/studium/unifr/solvent_evaporation_annealing/evaluation_crosssections/tests/01",
                "C:/Users/iselire/OneDrive/studium/unifr/solvent_evaporation_annealing/evaluation_crosssections/tests/02",
                "C:/Users/iselire/OneDrive/studium/unifr/solvent_evaporation_annealing/evaluation_crosssections/tests/03",
                "C:/Users/iselire/OneDrive/studium/unifr/solvent_evaporation_annealing/evaluation_crosssections/tests/04",
                "C:/Users/iselire/OneDrive/studium/unifr/solvent_evaporation_annealing/evaluation_crosssections/tests/05",
                "C:/Users/iselire/OneDrive/studium/unifr/solvent_evaporation_annealing/evaluation_crosssections/tests/06"]

# samples should be named sample_F24 etc. or F24 etc.
samples_dir = ["C:/Users/iselire/OneDrive/studium/unifr/solvent_evaporation_annealing/evaluation_crosssections/data/distorted_test_L",
               "C:/Users/iselire/OneDrive/studium/unifr/solvent_evaporation_annealing/evaluation_crosssections/data/distorted_test_R"]

#initial_values = ["C:/Users/IseliRe/PycharmProjects/pythonProject/20240509_multiprocessing/tests/the_match_test/20240601_crosscorr_arrays_data_L_10_test.csv"]
#iv_arrays = read_csv(initial_values[0])
#iv_input = iv_arrays
#print(iv_input)


# Fill fraction, max value of levelset is 1.4990768364186535
fill_fraction = 0.40
threshold = -0.8

wobble_rot = 30
wobble_sca = 0
wobble_she = 0
wobble_threshold = 0

starting_points = 5
threshold_start = []
scaling_start = []
rotation_start = []
shearing_start = []
topsurface_start = []
correlation_start = []

scaling_start_value = [1, 1, 1]
shearing_start_value = [0, 0, 0, 0, 0, 0]

for i in range(0, starting_points):
    threshold_start.append(threshold)
    #sc_x = scaling_start_value[0] + pow(-1, random.randint(10))*float(random.randint(100))/100 * wobble_sca
    #sc_y = scaling_start_value[1] + pow(-1, random.randint(10))*float(random.randint(100))/100 * wobble_sca
    #sc_z = 1/(sc_x * sc_y)
    #sc_all = (sc_x, sc_y, sc_z)
    sc_all = [1, 1, 1]
    scaling_start.append(sc_all)
    rotation_start.append((float(random.randint(3600))/10, float(random.randint(3600))/10, float(random.randint(3600))/10))
    #shearing_start.append((shearing_start_value[0] + pow(-1, random.randint(10))*float(random.randint(100))/100 * wobble_she,
    #                       shearing_start_value[0] + pow(-1, random.randint(10))*float(random.randint(100))/100 * wobble_she,
    #                       shearing_start_value[0] + pow(-1, random.randint(10))*float(random.randint(100))/100 * wobble_she,
    #                       shearing_start_value[0] + pow(-1, random.randint(10))*float(random.randint(100))/100 * wobble_she,
    #                       shearing_start_value[0] + pow(-1, random.randint(10))*float(random.randint(100))/100 * wobble_she,
    #                       shearing_start_value[0] + pow(-1, random.randint(10))*float(random.randint(100))/100 * wobble_she))
    shearing_start.append([0, 0, 0, 0, 0, 0])
    topsurface_start.append([0, 0, 0])
    correlation_start.append(0)

iv_input = []
iv_input.append(threshold_start)
iv_input.append(scaling_start)
iv_input.append(rotation_start)
iv_input.append(shearing_start)
iv_input.append(topsurface_start)
iv_input.append(correlation_start)

for f in saving_files:
    dir_exists(f)

corner = ['L', 'R']

first_miller = (1, 0, 0)
unitcell_in_pixel = 60
grid_size_model = (80, 80, 80)
translations_per_cut = 70

    
best_combos = 300
print_top = 4
cs_per_volume = 4


sample_pixel_size = 200
# y, x - shift per image
# L90, L24, F90, F24 or R90, R24, F90, F24
sample_square_shift = ([(0, 0), (0, 0), (0, 0), (0, 0)],
                        [(0, 0), (0, 0), (0, 0), (0, 0)])

iterations = 500

####################################################

 
if __name__ == '__main__':

####################################################    
    # number_list = list(range(100000))
    
    # start = time.time()
    # make_calculation_one(number_list,)
    # make_calculation_two(number_list,)
    # make_calculation_three(number_list,)
    # end = time.time()
    # print(f'1st execution: {end-start}')
    
    # start = time.time()
    # p1 = mp.Process(target = make_calculation_one, args = (number_list,))
    # p2 = mp.Process(target = make_calculation_two, args = (number_list,))
    # p3 = mp.Process(target = make_calculation_three, args = (number_list,))
    
    # p1.start()
    # p2.start()
    # p3.start()
    
    # p1.join()
    # p2.join()
    # p3.join()
    # end = time.time()
    # print(f'2nd execution: {end-start}')
    
    # temp_a = result_a
    # temp_b = result_b
    # temp_c = result_c
####################################################
    '''
    start = time.perf_counter()
    sleep_one_sec()
    sleep_one_sec()
    sleep_one_sec()
    sleep_one_sec()
    sleep_one_sec()
    end = time.perf_counter()
    print(f'1st execution: {end-start}')
    
    start = time.perf_counter()
    p1 = mp.Process(target = sleep_one_sec)
    p2 = mp.Process(target = sleep_one_sec)
    p3 = mp.Process(target = sleep_one_sec)
    p4 = mp.Process(target = sleep_one_sec)
    p5 = mp.Process(target = sleep_one_sec)
    
    p1.start()
    p2.start()
    p3.start()
    p4.start()
    p5.start()
    
    p1.join()
    p2.join()
    p3.join()
    p4.join()
    p5.join()
    end = time.perf_counter()
    print(f'2nd execution: {end-start}')
    '''
#################################################### 


    start = time.time()
    p1 = mp.Process(target=big_loop, args=(
        saving_files[0], samples_dir[0], grid_size_model, unitcell_in_pixel, first_miller, corner[0],
        sample_square_shift[0], iv_input))
    p2 = mp.Process(target=big_loop, args=(
        saving_files[1], samples_dir[0], grid_size_model, unitcell_in_pixel, first_miller, corner[0],
        sample_square_shift[0], iv_input))
    p3 = mp.Process(target=big_loop, args=(
        saving_files[2], samples_dir[0], grid_size_model, unitcell_in_pixel, first_miller, corner[0],
        sample_square_shift[0], iv_input))
    p4 = mp.Process(target=big_loop, args=(
        saving_files[3], samples_dir[1], grid_size_model, unitcell_in_pixel, first_miller, corner[1],
        sample_square_shift[1], iv_input))
    p5 = mp.Process(target=big_loop, args=(
        saving_files[4], samples_dir[1], grid_size_model, unitcell_in_pixel, first_miller, corner[1],
        sample_square_shift[1], iv_input))
    p6 = mp.Process(target=big_loop, args=(
        saving_files[5], samples_dir[1], grid_size_model, unitcell_in_pixel, first_miller, corner[1],
        sample_square_shift[1], iv_input))
#    p7 = mp.Process(target=big_loop, args=(
#        saving_files[6], samples_dir[0], grid_size_model, unitcell_in_pixel, first_miller, corner[0],
#        sample_square_shift[0], iv_input))
#    p8 = mp.Process(target=big_loop, args=(
#        saving_files[7], samples_dir[0], grid_size_model, unitcell_in_pixel, first_miller, corner[0],
#        sample_square_shift[0], iv_input))
#    p9 = mp.Process(target=big_loop, args=(
#        saving_files[8], samples_dir[0], grid_size_model, unitcell_in_pixel, first_miller, corner[0],
#        sample_square_shift[0], iv_input))
#    p10 = mp.Process(target=big_loop, args=(
#        saving_files[9], samples_dir[1], grid_size_model, unitcell_in_pixel, first_miller, corner[1],
#        sample_square_shift[1], iv_input))
#    p11 = mp.Process(target=big_loop, args=(
#        saving_files[10], samples_dir[1], grid_size_model, unitcell_in_pixel, first_miller, corner[1],
#        sample_square_shift[1], iv_input))
#    p12 = mp.Process(target=big_loop, args=(
#        saving_files[11], samples_dir[1], grid_size_model, unitcell_in_pixel, first_miller, corner[1],
#        sample_square_shift[1], iv_input))

    p1.start()
    p2.start()
    p3.start()
    p4.start()
    p5.start()
    p6.start()
#    p7.start()
#    p8.start()
#    p9.start()
#    p10.start()
#    p11.start()
#    p12.start()

    p1.join()
    p2.join()
    p3.join()
    p4.join()
    p5.join()
    p6.join()
#    p7.join()
#    p8.join()
#    p9.join()
#    p10.join()
#    p11.join()
#    p12.join()
    end = time.time()

    print(f'1st execution: {end-start}')

    

