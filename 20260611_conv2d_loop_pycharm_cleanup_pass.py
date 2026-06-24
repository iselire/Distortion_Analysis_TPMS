# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 16:12:33 2024

@author: IseliRe
"""

import numpy as np
from numpy import random
import matplotlib.pyplot as plt
import os
import math
import csv
import scipy.optimize as spo

import multiprocessing as mp

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

import torch
import torch.nn.functional as F
from torchvision import transforms
import torchvision.transforms.functional as TF


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
            img = imageio.imread(image_path, pilmode='L').astype(np.uint8)
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
        skip_line = 12
        for row in csv_reader:
            # Assuming the CSV file has two columns
            # Append values from each row to respective lists
            if skip_line == 11 or skip_line == 10 or skip_line == 9:
                #print(row)
                row_des = row[0]
                row_arr = eval(row[1])
                column_values.append(row_arr)  # Convert to appropriate data type if needed
                skip_line = skip_line - 1
                #print(skip_line)
            else:
                skip_line = skip_line - 1
                #print(skip_line)
    return column_values

def wobble(n_it, wob_sc, wob_ro, wob_sh, ig, corner_label):
    # Initialize the best solution and best objective value
    # takes in a parameter set of the initial values
    # and saves the cross correlation as best_objective_score
    # then wobbles the parameter set for n_it iterations

    best_param_set = ig
    # best_objective_score = cov2d_opti(ig, corner_label)
    candidate_param_set = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    #print(candidate_param_set)

    # Generate a random
    wobbling_array = (pow(-1, random.randint(10)) * np.random.rand(10) * wob_sc,
    pow(-1, random.randint(10)) * np.random.rand(10) * wob_sc,
    pow(-1, random.randint(10)) * np.random.rand(10) * wob_sc,
    pow(-1, random.randint(10)) * np.random.rand(10) * wob_ro,
    pow(-1, random.randint(10)) * np.random.rand(10) * wob_ro,
    pow(-1, random.randint(10)) * np.random.rand(10) * wob_ro,
    pow(-1, random.randint(10)) * np.random.rand(10) * wob_sh,
    pow(-1, random.randint(10)) * np.random.rand(10) * wob_sh,
    pow(-1, random.randint(10)) * np.random.rand(10) * wob_sh,
    pow(-1, random.randint(10)) * np.random.rand(10) * wob_sh,
    pow(-1, random.randint(10)) * np.random.rand(10) * wob_sh,
    pow(-1, random.randint(10)) * np.random.rand(10) * wob_sh)

    # Apply wobbling to the current solution
    for i in range(0, len(best_param_set)):
        candidate_param_set[i] = best_param_set[i] + wobbling_array[i][0]
    # Evaluate the objective function at the wobbled solution
    candidate_objective_score = cov2d_opti(candidate_param_set, corner_label)
    # Update the best solution if the wobbled solution is better
    best_param_set = candidate_param_set
    best_objective_score = candidate_objective_score
    #print(best_objective_score)

    return   best_param_set, best_objective_score

def cov2d_opti(x, corner_label):
    # takes in a parameter set, creates cross-sections and
    # performs the cross correlation.
    # output is the overall correlation of the 4 cross-sections combined

    scale1 = x[0]
    scale2 = x[1]
    scale3 = x[2]
    rot1 = x[3]
    rot2 = x[4]
    rot3 = x[5]
    shear1 = x[6]
    shear2 = x[7]
    shear3 = x[8]
    shear4 = x[9]
    shear5 = x[10]
    shear6 = x[11]

    v_translations = (0, 0, 0)
    scaling_values_opti = (scale1, scale2, scale3)
    rotation_values_opti = (rot1, rot2, rot3)
    shear_values_opti = (shear1, shear2, shear3, shear4, shear5, shear6)


    # performing crosscorrelation of the model with the 4 FIB SEM cuts
    model_binary_images_opti, model_levelset_images_opti, labels_opti, M_opti, A_opti, topsurface_opti = create_4cs(scaling_values_opti, rotation_values_opti, shear_values_opti, grid_size,
                                                                        unitcell, miller, t, corner_label)

    #plt.imshow(model_binary_images[0][0])
    #plt.show()
    #plt.imshow(model_binary_images[0][1])
    #plt.show()
    #plt.imshow(model_binary_images[0][2])
    #plt.show()
    if corner_label == 'L':
        overall_corr_values_raw = only_crosscorrelation_left(model_binary_images_opti, sample_L24_crop,
                                  sample_L90_crop, sample_F24_crop, sample_F90_crop, M_opti)
    elif corner_label == 'R':
        top_transf_matrices, per_slice_corr_scores, max_location, model_cross_sections, top_v_scaling, top_v_rotations, top_v_shearing, cropped_samples, top_surface_vectors, overall_corr_values_raw, top_model_slices_by_view = perform_crosscorrelation_right(
        samples_dir, scaling_square, sample_square_shift, model_binary_images, model_levelset_images, topsurface, sample_L24_crop, sample_L90_crop,
        sample_F24_crop, sample_F90_crop, M, scaling_values, rotation_values, shear_values)

    overall_corr_values_raw = overall_corr_values_raw.squeeze()
    overall_corr_values = overall_corr_values_raw

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

    #save_data_csv(top_v_scaling, top_v_rotations, top_v_shearing, top_surface_vectors, per_slice_corr_scores, top_corr_arr, time_sofar_used, num_iterations)
    #print(f'corr is: {top_corr_arr}')
    #print(f'{time_sofar_used}')

    return overall_corr_values


def create_4cs(scaling_values, rotation_values, shear_values, grid_size, unitcell, miller, thr, corner_label):
    device = 'cpu'  # cpu, cuda

    # Combine transformation matrices
    # M = T.dot(S).dot(SH).dot(R)
    # scaling
    M1 = create_affine_matrix(angles=(0, 0, 0), translation=(0, 0, 0), scaling=scaling_values, shearing=(0, 0, 0, 0, 0, 0))
    # rotation
    M2 = create_affine_matrix(angles=rotation_values, translation=(0, 0, 0), scaling=(1, 1, 1), shearing=(0, 0, 0, 0, 0, 0))
    # shearing
    M3 = create_affine_matrix(angles=(0, 0, 0), translation=(0, 0, 0), scaling=(1, 1, 1), shearing=shear_values)

    M = (M3).dot(M2).dot(M1)

    # Define grid points
    x = np.linspace(-math.pi * grid_size[0] / (unitcell), math.pi * grid_size[0] / (unitcell), grid_size[0])
    y = np.linspace(-math.pi * grid_size[1] / (unitcell), math.pi * grid_size[1] / (unitcell), grid_size[1])
    z = np.linspace(-math.pi * grid_size[2] / (unitcell), math.pi * grid_size[2] / (unitcell), grid_size[2])
    X, Y, Z = np.meshgrid(x, y, z)

    ##############################################################################
    # Rotation matrix
    ##############################################################################

    R2 = create_affine_matrix(angles=(0, 66, 0), translation=(0, 0, 0), scaling=(1, 1, 1), shearing=(0, 0, 0, 0, 0, 0))
    R3 = create_affine_matrix(angles=(0, 0, 90), translation=(0, 0, 0), scaling=(1, 1, 1), shearing=(0, 0, 0, 0, 0, 0))
    R4 = create_affine_matrix(angles=(0, 66, 0), translation=(0, 0, 0), scaling=(1, 1, 1), shearing=(0, 0, 0, 0, 0, 0))
    M4 = np.dot(R3, M)

    rot_matrix = np.block([
        [M, np.zeros((4, 12))],
        [np.zeros((4, 4)), np.dot(R2, M), np.zeros((4, 8))],
        [np.zeros((4, 8)), np.dot(R3, M), np.zeros((4, 4))],
        [np.zeros((4, 12)), np.dot(R4, M4)]
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

    bin_im = (make_binary(level_set_transformed0), make_binary(level_set_transformed1),
              make_binary(level_set_transformed2), make_binary(level_set_transformed3))

    ##############################################################################
    # levelset F090
    if corner_label == 'L':
        label1 = 'F090'
    elif corner_label == 'R':
        label1 = 'R090'
    # Apply transformation to grid points
    norm1 = miller_indices_to_normal(miller[0], miller[1], miller[2])
    norm1 = transform_normal(norm1, M2)
    norm1 = miller_indices_to_normal(norm1[0], norm1[1], norm1[2])

    ##############################################################################
    # levelset F024
    if corner_label == 'L':
        label2 = 'F024'
    elif corner_label == 'R':
        label2 = 'R024'
    norm2 = transform_normal(norm1, R2)
    norm2 = miller_indices_to_normal(norm2[0], norm2[1], norm2[2])

    ##############################################################################
    # levelset L090
    if corner_label == 'L':
        label3 = 'L090'
    elif corner_label == 'R':
        label3 = 'F090'
    norm3 = transform_normal(norm1, R3)
    norm3 = miller_indices_to_normal(norm3[0], norm3[1], norm3[2])

    ##############################################################################
    # levelset L024
    if corner_label == 'L':
        label4 = 'L024'
    elif corner_label == 'R':
        label4 = 'F024'
    norm4 = transform_normal(norm3, R4)
    norm4 = miller_indices_to_normal(norm4[0], norm4[1], norm4[2])

    Rtsn = create_affine_matrix(angles=(0, 90, 0), translation=(0, 0, 0), scaling=(1, 1, 1),
                                shearing=(0, 0, 0, 0, 0, 0))
    topsurf_norm = transform_normal(norm1, Rtsn)
    topsurf_norm = miller_indices_to_normal(topsurf_norm[0], topsurf_norm[1], topsurf_norm[2])

    model_levelset_images = (norm1, norm2, norm3, norm4)
    label_images = (label1, label2, label3, label4)

    return (bin_im, model_levelset_images, label_images, M, A, topsurf_norm)


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

# Define translation matrix function
def translation_matrix(tx, ty, tz):
    return np.array([[1, 0, 0, tx],
                     [0, 1, 0, ty],
                     [0, 0, 1, tz],
                     [0, 0, 0, 1]])

# Define scaling matrix function
def scaling_matrix(sx, sy, sz):
    return np.diag([sx, sy, sz, 1])


# Define rotation matrix function
def rotation_matrix(angle_x, angle_y, angle_z):
    angle_x = float(angle_x) / 360 * 2 * math.pi
    angle_y = float(angle_y) / 360 * 2 * math.pi
    angle_z = float(angle_z) / 360 * 2 * math.pi
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

# Define shearing matrix function
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

def make_binary(matrix):
    binary = np.where(matrix < t, 1, 0) * 255
    return binary

def make_binary_testfile(matrix, t, alpha):
    binary = np.where(matrix > t, 1, 0) * 255 * alpha
    return binary

# Function to find the normal vector to a plane given Miller indices
def miller_indices_to_normal(h, k, l):
    # Find the normal vector to the plane using Miller indices
    normal_vector = np.array([h, k, l])
    return normal_vector / np.linalg.norm(normal_vector)


def transform_normal(vector, matrix):
    # Extend the 3x1 vector to homogeneous coordinates by adding a fourth dimension with a value of 1
    extended_vector = np.hstack([vector, [1]])
    extended_vector = np.transpose(extended_vector)

    # Apply the affine transformation matrix to the extended vector
    m_want = np.linalg.inv(matrix).T
    transformed_vector = np.dot(m_want, extended_vector)

    # Extract the rotated 3x1 vector from the transformed vector
    deformed_vector = transformed_vector[:3]
    deformed_vector = deformed_vector / np.linalg.norm(deformed_vector)

    return deformed_vector


def perform_crosscorrelation_left(samples_dir, scaling_square, sample_square_shift, bin_im_cc, model_levelset_images, top_surface_vector, sample_L24_crop,
                                  sample_L90_crop, sample_F24_crop, sample_F90_crop, M, scaling_values, v_rotations,
                                  shear_values):
    device = 'cpu'  # cpu, cuda

    # Read in sample and model image

    top_F90_temp = [0, 0, 0, 0]
    top_F90_scores_temp = [0, 0, 0, 0]
    top_F24_temp = [0, 0, 0, 0]
    top_F24_scores_temp = [0, 0, 0, 0]
    top_L90_temp = [0, 0, 0, 0]
    top_L90_scores_temp = [0, 0, 0, 0]
    top_L24_temp = [0, 0, 0, 0]
    top_L24_scores_temp = [0, 0, 0, 0]

    overall_corr_values = [0, 0, 0, 0]

    model_matrices_F90_temp = [0, 0, 0, 0]
    model_matrices_F24_temp = [0, 0, 0, 0]
    model_matrices_L90_temp = [0, 0, 0, 0]
    model_matrices_L24_temp = [0, 0, 0, 0]

    model_slice_F90_temp = [0, 0, 0, 0]
    model_slice_F24_temp = [0, 0, 0, 0]
    model_slice_L90_temp = [0, 0, 0, 0]
    model_slice_L24_temp = [0, 0, 0, 0]

    max_location_F90x_temp = [0, 0, 0, 0]
    max_location_F90y_temp = [0, 0, 0, 0]
    max_location_F24x_temp = [0, 0, 0, 0]
    max_location_F24y_temp = [0, 0, 0, 0]
    max_location_L90x_temp = [0, 0, 0, 0]
    max_location_L90y_temp = [0, 0, 0, 0]
    max_location_L24x_temp = [0, 0, 0, 0]
    max_location_L24y_temp = [0, 0, 0, 0]

    # turn sample images to float tensors
    sample_F90_crop = sample_F90_crop.float()
    sample_F90_crop = sample_F90_crop.unsqueeze(0)

    sample_F24_crop = sample_F24_crop.float()
    sample_F24_crop = sample_F24_crop.unsqueeze(0)

    sample_L90_crop = sample_L90_crop.float()
    sample_L90_crop = sample_L90_crop.unsqueeze(0)

    sample_L24_crop = sample_L24_crop.float()
    sample_L24_crop = sample_L24_crop.unsqueeze(0)

    # Iterate through the translation of the binary matrices left and front
    for k in range(0, norm_translation):

        # Ensure both matrices are float tensors
        binary_images_F90 = torch.from_numpy(bin_im_cc[0][k]).float().to(device)
        binary_images_F90 = binary_images_F90.unsqueeze(0).unsqueeze(0)

        # normalise the float tensors
        img = (sample_F90_crop - sample_F90_crop.mean())/ (sample_F90_crop.std() + 1e-8)
        bin_im_norm = (binary_images_F90 - binary_images_F90.mean())/ (binary_images_F90.std() + 1e-8)

        result_F90 = F.conv2d(img, bin_im_norm, padding=0)/(grid_size[1] * grid_size[2])

        binary_images_F24 = torch.from_numpy(bin_im_cc[1][k]).float().to(device)
        binary_images_F24 = binary_images_F24.unsqueeze(0).unsqueeze(0)

        # normalise the float tensors
        img = (sample_F24_crop - sample_F24_crop.mean()) / (sample_F24_crop.std() + 1e-8)
        bin_im_norm = (binary_images_F24 - binary_images_F24.mean()) / (binary_images_F24.std() + 1e-8)

        result_F24 = F.conv2d(img, bin_im_norm, padding=0)/(grid_size[1] * grid_size[2])

        binary_images_L90 = torch.from_numpy(bin_im_cc[2][k]).float().to(device)
        binary_images_L90 = binary_images_L90.unsqueeze(0).unsqueeze(0)

        # normalise the float tensors
        img = (sample_L90_crop - sample_L90_crop.mean())/ (sample_L90_crop.std() + 1e-8)
        bin_im_norm = (binary_images_L90 - binary_images_L90.mean())/ (binary_images_L90.std() + 1e-8)

        result_L90 = F.conv2d(img, bin_im_norm, padding=0)/(grid_size[1] * grid_size[2])

        binary_images_L24 = torch.from_numpy(bin_im_cc[3][k]).float().to(device)
        binary_images_L24 = binary_images_L24.unsqueeze(0).unsqueeze(0)

        # normalise the float tensors
        img = (sample_L24_crop - sample_L24_crop.mean())/ (sample_L24_crop.std() + 1e-8)
        bin_im_norm = (binary_images_L24 - binary_images_L24.mean())/ (binary_images_L24.std() + 1e-8)

        result_L24 = F.conv2d(img, bin_im_norm, padding=0)/(grid_size[1] * grid_size[2])

        flat_idx = torch.argmax(result_F90)
        max_loc_F90y = flat_idx // result_F90.shape[-1]
        max_loc_F90x = flat_idx % result_F90.shape[-1]
        max_val_F90 = result_F90[0, 0, max_loc_F90y, max_loc_F90x]

        flat_idx = torch.argmax(result_F24)
        max_loc_F24y = flat_idx // result_F24.shape[-1]
        max_loc_F24x = flat_idx % result_F24.shape[-1]
        max_val_F24 = result_F24[0, 0, max_loc_F24y, max_loc_F24x]

        flat_idx = torch.argmax(result_L90)
        max_loc_L90y = flat_idx // result_L90.shape[-1]
        max_loc_L90x = flat_idx % result_L90.shape[-1]
        max_val_L90 = result_L90[0, 0, max_loc_L90y, max_loc_L90x]

        flat_idx = torch.argmax(result_L24)
        max_loc_L24y = flat_idx // result_L24.shape[-1]
        max_loc_L24x = flat_idx % result_L24.shape[-1]
        max_val_L24 = result_L24[0, 0, max_loc_L24y, max_loc_L24x]

        # Check if this translation provides a better match than the top translations
        if max_val_F90 > min(top_F90_scores_temp):
            top_F90_temp.append([M, k])
            top_F90_scores_temp.append(max_val_F90)
            max_location_F90x_temp.append(max_loc_F90x)
            max_location_F90y_temp.append(max_loc_F90y)
            model_matrices_F90_temp.append(binary_images_F90)
            model_slice_F90_temp.append(k)

            # Keep only the top rotations
            if len(top_F90_scores_temp) > best_combos_single_run:
                min_index = top_F90_scores_temp.index(min(top_F90_scores_temp))
                del top_F90_temp[min_index]
                del top_F90_scores_temp[min_index]
                del max_location_F90x_temp[min_index]
                del max_location_F90y_temp[min_index]
                del model_matrices_F90_temp[min_index]
                del model_slice_F90_temp[min_index]

        if max_val_F24 > min(top_F24_scores_temp):
            top_F24_temp.append([M, k])
            top_F24_scores_temp.append(max_val_F24)
            max_location_F24x_temp.append(max_loc_F24x)
            max_location_F24y_temp.append(max_loc_F24y)
            model_matrices_F24_temp.append(binary_images_F24)
            model_slice_F24_temp.append(k)

            # Keep only the top rotations
            if len(top_F24_scores_temp) > best_combos_single_run:
                min_index = top_F24_scores_temp.index(min(top_F24_scores_temp))
                del top_F24_temp[min_index]
                del top_F24_scores_temp[min_index]
                del max_location_F24x_temp[min_index]
                del max_location_F24y_temp[min_index]
                del model_matrices_F24_temp[min_index]
                del model_slice_F24_temp[min_index]

        if max_val_L90 > min(top_L90_scores_temp):
            top_L90_temp.append([M, k])
            top_L90_scores_temp.append(max_val_L90)
            max_location_L90x_temp.append(max_loc_L90x)
            max_location_L90y_temp.append(max_loc_L90y)
            model_matrices_L90_temp.append(binary_images_L90)
            model_slice_L90_temp.append(k)

            # Keep only the top three rotations
            if len(top_L90_scores_temp) > best_combos_single_run:
                min_index = top_L90_scores_temp.index(min(top_L90_scores_temp))
                del top_L90_temp[min_index]
                del top_L90_scores_temp[min_index]
                del max_location_L90x_temp[min_index]
                del max_location_L90y_temp[min_index]
                del model_matrices_L90_temp[min_index]
                del model_slice_L90_temp[min_index]

        if max_val_L24 > min(top_L24_scores_temp):
            top_L24_temp.append([M, k])
            top_L24_scores_temp.append(max_val_L24)
            max_location_L24x_temp.append(max_loc_L24x)
            max_location_L24y_temp.append(max_loc_L24y)
            model_matrices_L24_temp.append(binary_images_L24)
            model_slice_L24_temp.append(k)

            # Keep only the top three rotations
            if len(top_L24_scores_temp) > best_combos_single_run:
                min_index = top_L24_scores_temp.index(min(top_L24_scores_temp))
                del top_L24_temp[min_index]
                del top_L24_scores_temp[min_index]
                del max_location_L24x_temp[min_index]
                del max_location_L24y_temp[min_index]
                del model_matrices_L24_temp[min_index]
                del model_slice_L24_temp[min_index]

    if len(top_F90_scores) <= best_combos:
        max_F90 = top_F90_scores_temp.index(max(top_F90_scores_temp))
        top_F90.append(top_F90_temp[max_F90])
        top_F90_scores.append(top_F90_scores_temp[max_F90])
        max_location_F90x.append(max_location_F90x_temp[max_F90])
        max_location_F90y.append(max_location_F90y_temp[max_F90])
        model_matrices_F90.append(model_matrices_F90_temp[max_F90])
        model_slice_F90.append(model_slice_F90_temp[max_F90])

        max_F24 = top_F24_scores_temp.index(max(top_F24_scores_temp))
        top_F24.append(top_F24_temp[max_F24])
        top_F24_scores.append(top_F24_scores_temp[max_F24])
        max_location_F24x.append(max_location_F24x_temp[max_F24])
        max_location_F24y.append(max_location_F24y_temp[max_F24])
        model_matrices_F24.append(model_matrices_F24_temp[max_F24])
        model_slice_F24.append(model_slice_F24_temp[max_F24])

        max_L24 = top_L24_scores_temp.index(max(top_L24_scores_temp))
        top_L24.append(top_L24_temp[max_L24])
        top_L24_scores.append(top_L24_scores_temp[max_L24])
        max_location_L24x.append(max_location_L24x_temp[max_L24])
        max_location_L24y.append(max_location_L24y_temp[max_L24])
        model_matrices_L24.append(model_matrices_L24_temp[max_L24])
        model_slice_L24.append(model_slice_L24_temp[max_L24])

        max_L90 = top_L90_scores_temp.index(max(top_L90_scores_temp))
        top_L90.append(top_L90_temp[max_L90])
        top_L90_scores.append(top_L90_scores_temp[max_L90])
        max_location_L90x.append(max_location_L90x_temp[max_L90])
        max_location_L90y.append(max_location_L90y_temp[max_L90])
        model_matrices_L90.append(model_matrices_L90_temp[max_L90])
        model_slice_L90.append(model_slice_L90_temp[max_L90])

        top_v_rotations.append(v_rotations)
        # top_v_translations.append(v_translations)
        top_v_scaling.append(scaling_values)
        top_v_shearing.append(shear_values)

        top_surface_vectors.append(top_surface_vector)

        # compute overall correlation
        a = (1 / np.sqrt(float(cs_per_volume))) * np.sqrt(
            pow(float(top_F90_scores_temp[max_F90]), 2) + pow(float(top_F24_scores_temp[max_F24]), 2) +
            pow(float(top_L90_scores_temp[max_L90]), 2) + pow(float(top_L24_scores_temp[max_L24]), 2))
        top_corr_arr.append(a)
        overall_corr_values.append(a)

    if len(top_F90_scores) > best_combos:
        # continue here

        # Keep only the top combos

        min_index = overall_corr_values.index(min(overall_corr_values))
        del top_F90[min_index]
        del top_F90_scores[min_index]
        del max_location_F90x[min_index]
        del max_location_F90y[min_index]
        del model_matrices_F90[min_index]
        del model_slice_F90[min_index]

        del top_F24[min_index]
        del top_F24_scores[min_index]
        del max_location_F24x[min_index]
        del max_location_F24y[min_index]
        del model_matrices_F24[min_index]
        del model_slice_F24[min_index]

        del top_L90[min_index]
        del top_L90_scores[min_index]
        del max_location_L90x[min_index]
        del max_location_L90y[min_index]
        del model_matrices_L90[min_index]
        del model_slice_L90[min_index]

        del top_L24[min_index]
        del top_L24_scores[min_index]
        del max_location_L24x[min_index]
        del max_location_L24y[min_index]
        del model_matrices_L24[min_index]
        del model_slice_L24[min_index]

        del top_v_rotations[min_index]
        del top_v_scaling[min_index]
        del top_v_shearing[min_index]
        del top_corr_arr[min_index]

        del top_surface_vectors[min_index]

    # Move tensor to CPU
    # top_F90_scores_temp_cpu = [0, 0, 0, 0]
    # top_F24_scores_temp_cpu = [0, 0, 0, 0]
    # top_L90_scores_temp_cpu = [0, 0, 0, 0]
    # top_L24_scores_temp_cpu = [0, 0, 0, 0]

    # top_F90_scores_temp_cpu[0], top_F24_scores_temp_cpu[0], top_L90_scores_temp_cpu[0], top_L24_scores_temp_cpu[0] = top_F90_scores_temp[0].cpu(), top_F24_scores_temp[0].cpu(), top_L90_scores_temp[0].cpu(), top_L24_scores_temp[0].cpu()
    # top_F90_scores_temp_cpu[1], top_F24_scores_temp_cpu[1], top_L90_scores_temp_cpu[1], top_L24_scores_temp_cpu[1] = top_F90_scores_temp[1].cpu(), top_F24_scores_temp[1].cpu(), top_L90_scores_temp[1].cpu(), top_L24_scores_temp[1].cpu()
    # top_F90_scores_temp_cpu[2], top_F24_scores_temp_cpu[2], top_L90_scores_temp_cpu[2], top_L24_scores_temp_cpu[2] = top_F90_scores_temp[2].cpu(), top_F24_scores_temp[2].cpu(), top_L90_scores_temp[2].cpu(), top_L24_scores_temp[2].cpu()
    # top_F90_scores_temp_cpu[3], top_F24_scores_temp_cpu[3], top_L90_scores_temp_cpu[3], top_L24_scores_temp_cpu[3] = top_F90_scores_temp[3].cpu(), top_F24_scores_temp[3].cpu(), top_L90_scores_temp[3].cpu(), top_L24_scores_temp[3].cpu()

    # Convert tensor to NumPy array
    # top_F90_scores_temp_np, top_F24_scores_temp_np, top_L90_scores_temp_np, top_L24_scores_temp_np = top_F90_scores_temp_cpu.numpy(), top_F24_scores_temp_cpu.numpy(), top_L90_scores_temp_cpu.numpy(), top_L24_scores_temp_cpu.numpy()

    tc0 = (1 / np.sqrt(cs_per_volume)) * np.sqrt(
        pow(float(top_F90_scores_temp[0]), 2) + pow(float(top_F24_scores_temp[0]), 2) + pow(
            float(top_L90_scores_temp[0]), 2) + pow(float(top_L24_scores_temp[0]), 2))
    tc1 = (1 / np.sqrt(cs_per_volume)) * np.sqrt(
        pow(float(top_F90_scores_temp[1]), 2) + pow(float(top_F24_scores_temp[1]), 2) + pow(
            float(top_L90_scores_temp[1]), 2) + pow(float(top_L24_scores_temp[1]), 2))
    tc2 = (1 / np.sqrt(cs_per_volume)) * np.sqrt(
        pow(float(top_F90_scores_temp[2]), 2) + pow(float(top_F24_scores_temp[2]), 2) + pow(
            float(top_L90_scores_temp[2]), 2) + pow(float(top_L24_scores_temp[2]), 2))
    tc3 = (1 / np.sqrt(cs_per_volume)) * np.sqrt(
        pow(float(top_F90_scores_temp[3]), 2) + pow(float(top_F24_scores_temp[3]), 2) + pow(
            float(top_L90_scores_temp[3]), 2) + pow(float(top_L24_scores_temp[3]), 2))
    tcs = (tc0, tc1, tc2, tc3)
    overall_corr_values = max(tcs)

    # overall_corr_values = (1/np.sqrt(cs_per_volume))*np.sqrt(pow(max(top_F90_scores), 2) + pow(max(top_F24_scores), 2) + pow(max(top_L90_scores), 2) + pow(max(top_L24_scores), 2))
    top_transf_matrices = (top_F90, top_F24, top_L90, top_L24)
    per_slice_corr_scores = (top_F90_scores, top_F24_scores, top_L90_scores, top_L24_scores)
    max_location = (max_location_F90x, max_location_F90y, max_location_F24x, max_location_F24y, max_location_L90x,
                    max_location_L90y, max_location_L24x, max_location_L24y)
    bin_im_cc = (model_matrices_F90, model_matrices_F24, model_matrices_L90, model_matrices_L24)
    top_slices = (model_slice_F90, model_slice_F24, model_slice_L90, model_slice_L24)
    cropped_samples = (sample_F90_crop, sample_F24_crop, sample_L90_crop, sample_L24_crop)
    # print(mean_tensor_F90.shape, mean_tensor_F24.shape, mean_tensor_L90.shape, mean_tensor_L24.shape)
    # print(f'overall_corr_values is {overall_corr_values}')

    return top_transf_matrices, per_slice_corr_scores, max_location, bin_im_cc, top_v_scaling, top_v_rotations, top_v_shearing, cropped_samples, top_surface_vectors, overall_corr_values, top_slices

def perform_crosscorrelation_right(samples_dir, scaling_square, sample_square_shift, bin_im_cc, model_levelset_images, top_surface_vector, sample_F24_crop,
                                  sample_F90_crop, sample_R24_crop, sample_R90_crop, M,
                                   scaling_values, v_rotations, shear_values):
    device = 'cpu'  # cpu, cuda

    # Read in sample and model image

    top_R90_temp = [0, 0, 0, 0]
    top_R90_scores_temp = [0, 0, 0, 0]
    top_R24_temp = [0, 0, 0, 0]
    top_R24_scores_temp = [0, 0, 0, 0]
    top_F90_temp = [0, 0, 0, 0]
    top_F90_scores_temp = [0, 0, 0, 0]
    top_F24_temp = [0, 0, 0, 0]
    top_F24_scores_temp = [0, 0, 0, 0]

    overall_corr_values = [0, 0, 0, 0]

    model_matrices_R90_temp = [0, 0, 0, 0]
    model_matrices_R24_temp = [0, 0, 0, 0]
    model_matrices_F90_temp = [0, 0, 0, 0]
    model_matrices_F24_temp = [0, 0, 0, 0]

    model_slice_R90_temp = [0, 0, 0, 0]
    model_slice_R24_temp = [0, 0, 0, 0]
    model_slice_F90_temp = [0, 0, 0, 0]
    model_slice_F24_temp = [0, 0, 0, 0]

    max_location_R90x_temp = [0, 0, 0, 0]
    max_location_R90y_temp = [0, 0, 0, 0]
    max_location_R24x_temp = [0, 0, 0, 0]
    max_location_R24y_temp = [0, 0, 0, 0]
    max_location_F90x_temp = [0, 0, 0, 0]
    max_location_F90y_temp = [0, 0, 0, 0]
    max_location_F24x_temp = [0, 0, 0, 0]
    max_location_F24y_temp = [0, 0, 0, 0]

    # turn sample images to float tensors
    sample_R90_crop = sample_R90_crop.float()
    sample_R90_crop = sample_R90_crop.unsqueeze(0)

    sample_R24_crop = sample_R24_crop.float()
    sample_R24_crop = sample_R24_crop.unsqueeze(0)

    sample_F90_crop = sample_F90_crop.float()
    sample_F90_crop = sample_F90_crop.unsqueeze(0)

    sample_F24_crop = sample_F24_crop.float()
    sample_F24_crop = sample_F24_crop.unsqueeze(0)

    # Iterate through the translation of the binary matrices left and front
    for k in range(0, norm_translation):

        # Ensure both matrices are float tensors
        binary_images_R90 = torch.from_numpy(bin_im_cc[0][k]).float().to(device)
        binary_images_R90 = binary_images_R90.unsqueeze(0).unsqueeze(0)

        # normalise the float tensors
        img = (sample_R90_crop - sample_R90_crop.mean())/ (sample_R90_crop.std() + 1e-8)
        bin_im_norm = (binary_images_R90 - binary_images_R90.mean())/ (binary_images_R90.std() + 1e-8)

        result_R90 = F.conv2d(img, bin_im_norm, padding=0)/(grid_size[1] * grid_size[2])

        binary_images_R24 = torch.from_numpy(bin_im_cc[1][k]).float().to(device)
        binary_images_R24 = binary_images_R24.unsqueeze(0).unsqueeze(0)

        # normalise the float tensors
        img = (sample_R24_crop - sample_R24_crop.mean()) / (sample_R24_crop.std() + 1e-8)
        bin_im_norm = (binary_images_R24 - binary_images_R24.mean()) / (binary_images_R24.std() + 1e-8)

        result_R24 = F.conv2d(img, bin_im_norm, padding=0)/(grid_size[1] * grid_size[2])

        binary_images_F90 = torch.from_numpy(bin_im_cc[2][k]).float().to(device)
        binary_images_F90 = binary_images_F90.unsqueeze(0).unsqueeze(0)

        # normalise the float tensors
        img = (sample_F90_crop - sample_F90_crop.mean())/ (sample_F90_crop.std() + 1e-8)
        bin_im_norm = (binary_images_F90 - binary_images_F90.mean())/ (binary_images_F90.std() + 1e-8)

        result_F90 = F.conv2d(img, bin_im_norm, padding=0)/(grid_size[1] * grid_size[2])

        binary_images_F24 = torch.from_numpy(bin_im_cc[3][k]).float().to(device)
        binary_images_F24 = binary_images_F24.unsqueeze(0).unsqueeze(0)

        # normalise the float tensors
        img = (sample_F24_crop - sample_F24_crop.mean())/ (sample_F24_crop.std() + 1e-8)
        bin_im_norm = (binary_images_F24 - binary_images_F24.mean())/ (binary_images_F24.std() + 1e-8)

        result_F24 = F.conv2d(img, bin_im_norm, padding=0)/(grid_size[1] * grid_size[2])

        flat_idx = torch.argmax(result_R90)
        max_loc_R90y = flat_idx // result_R90.shape[-1]
        max_loc_R90x = flat_idx % result_R90.shape[-1]
        max_val_R90 = result_R90[0, 0, max_loc_R90y, max_loc_R90x]

        flat_idx = torch.argmax(result_R24)
        max_loc_R24y = flat_idx // result_R24.shape[-1]
        max_loc_R24x = flat_idx % result_R24.shape[-1]
        max_val_R24 = result_R24[0, 0, max_loc_R24y, max_loc_R24x]

        flat_idx = torch.argmax(result_F90)
        max_loc_F90y = flat_idx // result_F90.shape[-1]
        max_loc_F90x = flat_idx % result_F90.shape[-1]
        max_val_F90 = result_F90[0, 0, max_loc_F90y, max_loc_F90x]

        flat_idx = torch.argmax(result_F24)
        max_loc_F24y = flat_idx // result_F24.shape[-1]
        max_loc_F24x = flat_idx % result_F24.shape[-1]
        max_val_F24 = result_F24[0, 0, max_loc_F24y, max_loc_F24x]

        # Check if this translation provides a better match than the top translations
        if max_val_R90 > min(top_R90_scores_temp):
            top_R90_temp.append([M, k])
            top_R90_scores_temp.append(max_val_R90)
            max_location_R90x_temp.append(max_loc_R90x)
            max_location_R90y_temp.append(max_loc_R90y)
            model_matrices_R90_temp.append(binary_images_R90)
            model_slice_R90_temp.append(k)

            # Keep only the top rotations
            if len(top_R90_scores_temp) > best_combos_single_run:
                min_index = top_R90_scores_temp.index(min(top_R90_scores_temp))
                del top_R90_temp[min_index]
                del top_R90_scores_temp[min_index]
                del max_location_R90x_temp[min_index]
                del max_location_R90y_temp[min_index]
                del model_matrices_R90_temp[min_index]
                del model_slice_R90_temp[min_index]

        if max_val_R24 > min(top_R24_scores_temp):
            top_R24_temp.append([M, k])
            top_R24_scores_temp.append(max_val_R24)
            max_location_R24x_temp.append(max_loc_R24x)
            max_location_R24y_temp.append(max_loc_R24y)
            model_matrices_R24_temp.append(binary_images_R24)
            model_slice_R24_temp.append(k)

            # Keep only the top rotations
            if len(top_R24_scores_temp) > best_combos_single_run:
                min_index = top_R24_scores_temp.index(min(top_R24_scores_temp))
                del top_R24_temp[min_index]
                del top_R24_scores_temp[min_index]
                del max_location_R24x_temp[min_index]
                del max_location_R24y_temp[min_index]
                del model_matrices_R24_temp[min_index]
                del model_slice_R24_temp[min_index]

        if max_val_F90 > min(top_F90_scores_temp):
            top_F90_temp.append([M, k])
            top_F90_scores_temp.append(max_val_F90)
            max_location_F90x_temp.append(max_loc_F90x)
            max_location_F90y_temp.append(max_loc_F90y)
            model_matrices_F90_temp.append(binary_images_F90)
            model_slice_F90_temp.append(k)

            # Keep only the top three rotations
            if len(top_F90_scores_temp) > best_combos_single_run:
                min_index = top_F90_scores_temp.index(min(top_F90_scores_temp))
                del top_F90_temp[min_index]
                del top_F90_scores_temp[min_index]
                del max_location_F90x_temp[min_index]
                del max_location_F90y_temp[min_index]
                del model_matrices_F90_temp[min_index]
                del model_slice_F90_temp[min_index]

        if max_val_F24 > min(top_F24_scores_temp):
            top_F24_temp.append([M, k])
            top_F24_scores_temp.append(max_val_F24)
            max_location_F24x_temp.append(max_loc_F24x)
            max_location_F24y_temp.append(max_loc_F24y)
            model_matrices_F24_temp.append(binary_images_F24)
            model_slice_F24_temp.append(k)

            # Keep only the top three rotations
            if len(top_F24_scores_temp) > best_combos_single_run:
                min_index = top_F24_scores_temp.index(min(top_F24_scores_temp))
                del top_F24_temp[min_index]
                del top_F24_scores_temp[min_index]
                del max_location_F24x_temp[min_index]
                del max_location_F24y_temp[min_index]
                del model_matrices_F24_temp[min_index]
                del model_slice_F24_temp[min_index]

    if len(top_R90_scores) <= best_combos:
        max_R90 = top_R90_scores_temp.index(max(top_R90_scores_temp))
        top_R90.append(top_R90_temp[max_R90])
        top_R90_scores.append(top_R90_scores_temp[max_R90])
        max_location_R90x.append(max_location_R90x_temp[max_R90])
        max_location_R90y.append(max_location_R90y_temp[max_R90])
        model_matrices_R90.append(model_matrices_R90_temp[max_R90])
        model_slice_R90.append(model_slice_R90_temp[max_R90])

        max_R24 = top_R24_scores_temp.index(max(top_R24_scores_temp))
        top_R24.append(top_R24_temp[max_R24])
        top_R24_scores.append(top_R24_scores_temp[max_R24])
        max_location_R24x.append(max_location_R24x_temp[max_R24])
        max_location_R24y.append(max_location_R24y_temp[max_R24])
        model_matrices_R24.append(model_matrices_R24_temp[max_R24])
        model_slice_R24.append(model_slice_R24_temp[max_R24])

        max_F24 = top_F24_scores_temp.index(max(top_F24_scores_temp))
        top_F24.append(top_F24_temp[max_F24])
        top_F24_scores.append(top_F24_scores_temp[max_F24])
        max_location_F24x.append(max_location_F24x_temp[max_F24])
        max_location_F24y.append(max_location_F24y_temp[max_F24])
        model_matrices_F24.append(model_matrices_F24_temp[max_F24])
        model_slice_F24.append(model_slice_F24_temp[max_F24])

        max_F90 = top_F90_scores_temp.index(max(top_F90_scores_temp))
        top_F90.append(top_F90_temp[max_F90])
        top_F90_scores.append(top_F90_scores_temp[max_F90])
        max_location_F90x.append(max_location_F90x_temp[max_F90])
        max_location_F90y.append(max_location_F90y_temp[max_F90])
        model_matrices_F90.append(model_matrices_F90_temp[max_F90])
        model_slice_F90.append(model_slice_F90_temp[max_F90])

        top_v_rotations.append(v_rotations)
        # top_v_translations.append(v_translations)
        top_v_scaling.append(scaling_values)
        top_v_shearing.append(shear_values)

        top_surface_vectors.append(top_surface_vector)

        # compute overall correlation
        a = (1 / np.sqrt(float(cs_per_volume))) * np.sqrt(
            pow(float(top_R90_scores_temp[max_R90]), 2) + pow(float(top_R24_scores_temp[max_R24]), 2) +
            pow(float(top_F90_scores_temp[max_F90]), 2) + pow(float(top_F24_scores_temp[max_F24]), 2))
        top_corr_arr.append(a)
        overall_corr_values.append(a)

    if len(top_R90_scores) > best_combos:
        # continue here

        # Keep only the top combos

        min_index = overall_corr_values.index(min(overall_corr_values))
        del top_R90[min_index]
        del top_R90_scores[min_index]
        del max_location_R90x[min_index]
        del max_location_R90y[min_index]
        del model_matrices_R90[min_index]
        del model_slice_R90[min_index]

        del top_R24[min_index]
        del top_R24_scores[min_index]
        del max_location_R24x[min_index]
        del max_location_R24y[min_index]
        del model_matrices_R24[min_index]
        del model_slice_R24[min_index]

        del top_F90[min_index]
        del top_F90_scores[min_index]
        del max_location_F90x[min_index]
        del max_location_F90y[min_index]
        del model_matrices_F90[min_index]
        del model_slice_F90[min_index]

        del top_F24[min_index]
        del top_F24_scores[min_index]
        del max_location_F24x[min_index]
        del max_location_F24y[min_index]
        del model_matrices_F24[min_index]
        del model_slice_F24[min_index]

        del top_v_rotations[min_index]
        del top_v_scaling[min_index]
        del top_v_shearing[min_index]
        del top_corr_arr[min_index]

        del top_surface_vectors[min_index]

    # Move tensor to CPU
    # top_R90_scores_temp_cpu = [0, 0, 0, 0]
    # top_R24_scores_temp_cpu = [0, 0, 0, 0]
    # top_F90_scores_temp_cpu = [0, 0, 0, 0]
    # top_F24_scores_temp_cpu = [0, 0, 0, 0]

    # top_R90_scores_temp_cpu[0], top_R24_scores_temp_cpu[0], top_F90_scores_temp_cpu[0], top_F24_scores_temp_cpu[0] = top_R90_scores_temp[0].cpu(), top_R24_scores_temp[0].cpu(), top_F90_scores_temp[0].cpu(), top_F24_scores_temp[0].cpu()
    # top_R90_scores_temp_cpu[1], top_R24_scores_temp_cpu[1], top_F90_scores_temp_cpu[1], top_F24_scores_temp_cpu[1] = top_R90_scores_temp[1].cpu(), top_R24_scores_temp[1].cpu(), top_F90_scores_temp[1].cpu(), top_F24_scores_temp[1].cpu()
    # top_R90_scores_temp_cpu[2], top_R24_scores_temp_cpu[2], top_F90_scores_temp_cpu[2], top_F24_scores_temp_cpu[2] = top_R90_scores_temp[2].cpu(), top_R24_scores_temp[2].cpu(), top_F90_scores_temp[2].cpu(), top_F24_scores_temp[2].cpu()
    # top_R90_scores_temp_cpu[3], top_R24_scores_temp_cpu[3], top_F90_scores_temp_cpu[3], top_F24_scores_temp_cpu[3] = top_R90_scores_temp[3].cpu(), top_R24_scores_temp[3].cpu(), top_F90_scores_temp[3].cpu(), top_F24_scores_temp[3].cpu()

    # Convert tensor to NumPy array
    # top_R90_scores_temp_np, top_R24_scores_temp_np, top_F90_scores_temp_np, top_F24_scores_temp_np = top_R90_scores_temp_cpu.numpy(), top_R24_scores_temp_cpu.numpy(), top_F90_scores_temp_cpu.numpy(), top_F24_scores_temp_cpu.numpy()

    tc0 = (1 / np.sqrt(cs_per_volume)) * np.sqrt(
        pow(float(top_R90_scores_temp[0]), 2) + pow(float(top_R24_scores_temp[0]), 2) + pow(
            float(top_F90_scores_temp[0]), 2) + pow(float(top_F24_scores_temp[0]), 2))
    tc1 = (1 / np.sqrt(cs_per_volume)) * np.sqrt(
        pow(float(top_R90_scores_temp[1]), 2) + pow(float(top_R24_scores_temp[1]), 2) + pow(
            float(top_F90_scores_temp[1]), 2) + pow(float(top_F24_scores_temp[1]), 2))
    tc2 = (1 / np.sqrt(cs_per_volume)) * np.sqrt(
        pow(float(top_R90_scores_temp[2]), 2) + pow(float(top_R24_scores_temp[2]), 2) + pow(
            float(top_F90_scores_temp[2]), 2) + pow(float(top_F24_scores_temp[2]), 2))
    tc3 = (1 / np.sqrt(cs_per_volume)) * np.sqrt(
        pow(float(top_R90_scores_temp[3]), 2) + pow(float(top_R24_scores_temp[3]), 2) + pow(
            float(top_F90_scores_temp[3]), 2) + pow(float(top_F24_scores_temp[3]), 2))
    tcs = (tc0, tc1, tc2, tc3)
    overall_corr_values = max(tcs)

    # overall_corr_values = (1/np.sqrt(cs_per_volume))*np.sqrt(pow(max(top_R90_scores), 2) + pow(max(top_R24_scores), 2) + pow(max(top_F90_scores), 2) + pow(max(top_F24_scores), 2))
    top_transf_matrices = (top_R90, top_R24, top_F90, top_F24)
    per_slice_corr_scores = (top_R90_scores, top_R24_scores, top_F90_scores, top_F24_scores)
    max_location = (max_location_R90x, max_location_R90y, max_location_R24x, max_location_R24y, max_location_F90x,
                    max_location_F90y, max_location_F24x, max_location_F24y)
    bin_im_cc = (model_matrices_R90, model_matrices_R24, model_matrices_F90, model_matrices_F24)
    top_slices = (model_slice_R90, model_slice_R24, model_slice_F90, model_slice_F24)
    cropped_samples = (sample_R90_crop, sample_R24_crop, sample_F90_crop, sample_F24_crop)
    # print(mean_tensor_R90.shape, mean_tensor_R24.shape, mean_tensor_F90.shape, mean_tensor_F24.shape)
    # print(f'overall_corr_values is {overall_corr_values}')

    return top_transf_matrices, per_slice_corr_scores, max_location, bin_im_cc, top_v_scaling, top_v_rotations, top_v_shearing, cropped_samples, top_surface_vectors, overall_corr_values, top_slices

def only_crosscorrelation_left(bin_im_cc, sample_L24_crop,
                                  sample_L90_crop, sample_F24_crop, sample_F90_crop, M):
    device = 'cpu'  # cpu, cuda

    # Read in sample and model image

    top_F90_temp = [0, 0, 0, 0]
    top_F90_scores_temp = [0, 0, 0, 0]
    top_F24_temp = [0, 0, 0, 0]
    top_F24_scores_temp = [0, 0, 0, 0]
    top_L90_temp = [0, 0, 0, 0]
    top_L90_scores_temp = [0, 0, 0, 0]
    top_L24_temp = [0, 0, 0, 0]
    top_L24_scores_temp = [0, 0, 0, 0]

    model_matrices_F90_temp = [0, 0, 0, 0]
    model_matrices_F24_temp = [0, 0, 0, 0]
    model_matrices_L90_temp = [0, 0, 0, 0]
    model_matrices_L24_temp = [0, 0, 0, 0]

    model_slice_F90_temp = [0, 0, 0, 0]
    model_slice_F24_temp = [0, 0, 0, 0]
    model_slice_L90_temp = [0, 0, 0, 0]
    model_slice_L24_temp = [0, 0, 0, 0]

    max_location_F90x_temp = [0, 0, 0, 0]
    max_location_F90y_temp = [0, 0, 0, 0]
    max_location_F24x_temp = [0, 0, 0, 0]
    max_location_F24y_temp = [0, 0, 0, 0]
    max_location_L90x_temp = [0, 0, 0, 0]
    max_location_L90y_temp = [0, 0, 0, 0]
    max_location_L24x_temp = [0, 0, 0, 0]
    max_location_L24y_temp = [0, 0, 0, 0]

    # turn sample images to float tensors
    sample_F90_crop = sample_F90_crop.float()
    sample_F90_crop = sample_F90_crop.unsqueeze(0)

    sample_F24_crop = sample_F24_crop.float()
    sample_F24_crop = sample_F24_crop.unsqueeze(0)

    sample_L90_crop = sample_L90_crop.float()
    sample_L90_crop = sample_L90_crop.unsqueeze(0)

    sample_L24_crop = sample_L24_crop.float()
    sample_L24_crop = sample_L24_crop.unsqueeze(0)

    # Iterate through the translation of the binary matrices left and front
    for k in range(0, norm_translation):

        # Ensure both matrices are float tensors
        binary_images_F90 = torch.from_numpy(bin_im_cc[0][k]).float().to(device)
        binary_images_F90 = binary_images_F90.unsqueeze(0).unsqueeze(0)

        # normalise the float tensors
        img = (sample_F90_crop - sample_F90_crop.mean())/ (sample_F90_crop.std() + 1e-8)
        bin_im_norm = (binary_images_F90 - binary_images_F90.mean())/ (binary_images_F90.std() + 1e-8)

        result_F90 = F.conv2d(img, bin_im_norm, padding=0)/(grid_size[1] * grid_size[2])

        binary_images_F24 = torch.from_numpy(bin_im_cc[1][k]).float().to(device)
        binary_images_F24 = binary_images_F24.unsqueeze(0).unsqueeze(0)

        # normalise the float tensors
        img = (sample_F24_crop - sample_F24_crop.mean()) / (sample_F24_crop.std() + 1e-8)
        bin_im_norm = (binary_images_F24 - binary_images_F24.mean()) / (binary_images_F24.std() + 1e-8)

        result_F24 = F.conv2d(img, bin_im_norm, padding=0)/(grid_size[1] * grid_size[2])

        binary_images_L90 = torch.from_numpy(bin_im_cc[2][k]).float().to(device)
        binary_images_L90 = binary_images_L90.unsqueeze(0).unsqueeze(0)

        # normalise the float tensors
        img = (sample_L90_crop - sample_L90_crop.mean())/ (sample_L90_crop.std() + 1e-8)
        bin_im_norm = (binary_images_L90 - binary_images_L90.mean())/ (binary_images_L90.std() + 1e-8)

        result_L90 = F.conv2d(img, bin_im_norm, padding=0)/(grid_size[1] * grid_size[2])

        binary_images_L24 = torch.from_numpy(bin_im_cc[3][k]).float().to(device)
        binary_images_L24 = binary_images_L24.unsqueeze(0).unsqueeze(0)

        # normalise the float tensors
        img = (sample_L24_crop - sample_L24_crop.mean())/ (sample_L24_crop.std() + 1e-8)
        bin_im_norm = (binary_images_L24 - binary_images_L24.mean())/ (binary_images_L24.std() + 1e-8)

        result_L24 = F.conv2d(img, bin_im_norm, padding=0)/(grid_size[1] * grid_size[2])

        flat_idx = torch.argmax(result_F90)
        max_loc_F90y = flat_idx // result_F90.shape[-1]
        max_loc_F90x = flat_idx % result_F90.shape[-1]
        max_val_F90 = result_F90[0, 0, max_loc_F90x, max_loc_F90y]

        flat_idx = torch.argmax(result_F24)
        max_loc_F24y = flat_idx // result_F24.shape[-1]
        max_loc_F24x = flat_idx % result_F24.shape[-1]
        max_val_F24 = result_F24[0, 0, max_loc_F24x, max_loc_F24y]

        flat_idx = torch.argmax(result_L90)
        max_loc_L90y = flat_idx // result_L90.shape[-1]
        max_loc_L90x = flat_idx % result_L90.shape[-1]
        max_val_L90 = result_L90[0, 0, max_loc_L90x, max_loc_L90y]

        flat_idx = torch.argmax(result_L24)
        max_loc_L24y = flat_idx // result_L24.shape[-1]
        max_loc_L24x = flat_idx % result_L24.shape[-1]
        max_val_L24 = result_L24[0, 0, max_loc_L24x, max_loc_L24y]

        # Check if this translation provides a better match than the top translations
        if max_val_F90 > max(top_F90_scores_temp):
            top_F90_temp.append([M, k])
            top_F90_scores_temp.append(max_val_F90)
            max_location_F90x_temp.append(max_loc_F90x)
            max_location_F90y_temp.append(max_loc_F90y)
            model_matrices_F90_temp.append(binary_images_F90)
            model_slice_F90_temp.append(k)

            # Keep only the top rotations
            if len(top_F90_scores_temp) > best_combos_single_run:
                min_index = top_F90_scores_temp.index(min(top_F90_scores_temp))
                del top_F90_temp[min_index]
                del top_F90_scores_temp[min_index]
                del max_location_F90x_temp[min_index]
                del max_location_F90y_temp[min_index]
                del model_matrices_F90_temp[min_index]
                del model_slice_F90_temp[min_index]

        if max_val_F24 > min(top_F24_scores_temp):
            top_F24_temp.append([M, k])
            top_F24_scores_temp.append(max_val_F24)
            max_location_F24x_temp.append(max_loc_F24x)
            max_location_F24y_temp.append(max_loc_F24y)
            model_matrices_F24_temp.append(binary_images_F24)
            model_slice_F24_temp.append(k)

            # Keep only the top rotations
            if len(top_F24_scores_temp) > best_combos_single_run:
                min_index = top_F24_scores_temp.index(min(top_F24_scores_temp))
                del top_F24_temp[min_index]
                del top_F24_scores_temp[min_index]
                del max_location_F24x_temp[min_index]
                del max_location_F24y_temp[min_index]
                del model_matrices_F24_temp[min_index]
                del model_slice_F24_temp[min_index]

        if max_val_L90 > min(top_L90_scores_temp):
            top_L90_temp.append([M, k])
            top_L90_scores_temp.append(max_val_L90)
            max_location_L90x_temp.append(max_loc_L90x)
            max_location_L90y_temp.append(max_loc_L90y)
            model_matrices_L90_temp.append(binary_images_L90)
            model_slice_L90_temp.append(k)

            # Keep only the top three rotations
            if len(top_L90_scores_temp) > best_combos_single_run:
                min_index = top_L90_scores_temp.index(min(top_L90_scores_temp))
                del top_L90_temp[min_index]
                del top_L90_scores_temp[min_index]
                del max_location_L90x_temp[min_index]
                del max_location_L90y_temp[min_index]
                del model_matrices_L90_temp[min_index]
                del model_slice_L90_temp[min_index]

        if max_val_L24 > min(top_L24_scores_temp):
            top_L24_temp.append([M, k])
            top_L24_scores_temp.append(max_val_L24)
            max_location_L24x_temp.append(max_loc_L24x)
            max_location_L24y_temp.append(max_loc_L24y)
            model_matrices_L24_temp.append(binary_images_L24)
            model_slice_L24_temp.append(k)

            # Keep only the top three rotations
            if len(top_L24_scores_temp) > best_combos_single_run:
                min_index = top_L24_scores_temp.index(min(top_L24_scores_temp))
                del top_L24_temp[min_index]
                del top_L24_scores_temp[min_index]
                del max_location_L24x_temp[min_index]
                del max_location_L24y_temp[min_index]
                del model_matrices_L24_temp[min_index]
                del model_slice_L24_temp[min_index]

    max_F90 = top_F90_scores_temp.index(max(top_F90_scores_temp))
    max_F24 = top_F24_scores_temp.index(max(top_F24_scores_temp))
    max_L90 = top_L90_scores_temp.index(max(top_L90_scores_temp))
    max_L24 = top_L24_scores_temp.index(max(top_L24_scores_temp))

    tc0 = (1 / np.sqrt(cs_per_volume)) * np.sqrt(
        pow(float(top_F90_scores_temp[max_F90]), 2) + pow(float(top_F24_scores_temp[max_F24]), 2) + pow(
            float(top_L90_scores_temp[max_L90]), 2) + pow(float(top_L24_scores_temp[max_L24]), 2))

    overall_corr_values = tc0

    return overall_corr_values


def perform_crosscorrelation_multi_left(samples_dir, scaling_square, sample_square_shift, bin_im_cc, model_levelset_images, top_surface_vector, sample_L24_crop,
                                        sample_L90_crop, sample_F24_crop, sample_F90_crop, M, thr,
                                        tF90s, tF24s, tL90s, tL24s,
                                        tF90, tF24, tL90, tL24,
                                        mlF90x, mlF24x, mlL90x, mlL24x,
                                        mlF90y, mlF24y, mlL90y, mlL24y,
                                        mmF90, mmF24, mmL90, mmL24,
                                        msF90, msF24, msL90, msL24,
                                        tvscaling, tvrotation, tvshearing,
                                        max_location, tlabel, tcarr,
                                        tscorr, thresh, tmm,
                                        scaling_values, v_rotations, shear_values):
    device = 'cpu'  # cpu, cuda

    # Read in sample and model image

    top_F90_temp = [0]
    top_F90_scores_temp = [0]
    top_F24_temp = [0]
    top_F24_scores_temp = [0]
    top_L90_temp = [0]
    top_L90_scores_temp = [0]
    top_L24_temp = [0]
    top_L24_scores_temp = [0]

    overall_corr_values = [0]

    model_matrices_F90_temp = [0]
    model_matrices_F24_temp = [0]
    model_matrices_L90_temp = [0]
    model_matrices_L24_temp = [0]

    model_slice_F90_temp = [0]
    model_slice_F24_temp = [0]
    model_slice_L90_temp = [0]
    model_slice_L24_temp = [0]

    max_location_F90x_temp = [0]
    max_location_F90y_temp = [0]
    max_location_F24x_temp = [0]
    max_location_F24y_temp = [0]
    max_location_L90x_temp = [0]
    max_location_L90y_temp = [0]
    max_location_L24x_temp = [0]
    max_location_L24y_temp = [0]


    # turn sample images to float tensors
    sample_F90_crop = sample_F90_crop.float()
    sample_F90_crop = sample_F90_crop.unsqueeze(0)

    sample_F24_crop = sample_F24_crop.float()
    sample_F24_crop = sample_F24_crop.unsqueeze(0)

    sample_L90_crop = sample_L90_crop.float()
    sample_L90_crop = sample_L90_crop.unsqueeze(0)

    sample_L24_crop = sample_L24_crop.float()
    sample_L24_crop = sample_L24_crop.unsqueeze(0)

    # Iterate through the translation of the binary matrices left and front
    for k in range(0, norm_translation):

        # Ensure both matrices are float tensors
        binary_images_F90 = torch.from_numpy(bin_im_cc[0][k]).float().to(device)
        binary_images_F90 = binary_images_F90.unsqueeze(0).unsqueeze(0)

        # normalise the float tensors
        img = (sample_F90_crop - sample_F90_crop.mean())/ (sample_F90_crop.std() + 1e-8)
        bin_im_norm = (binary_images_F90 - binary_images_F90.mean())/ (binary_images_F90.std() + 1e-8)

        result_F90 = F.conv2d(img, bin_im_norm, padding=0)/(grid_size[1] * grid_size[2])

        binary_images_F24 = torch.from_numpy(bin_im_cc[1][k]).float().to(device)
        binary_images_F24 = binary_images_F24.unsqueeze(0).unsqueeze(0)

        # normalise the float tensors
        img = (sample_F24_crop - sample_F24_crop.mean()) / (sample_F24_crop.std() + 1e-8)
        bin_im_norm = (binary_images_F24 - binary_images_F24.mean()) / (binary_images_F24.std() + 1e-8)

        result_F24 = F.conv2d(img, bin_im_norm, padding=0)/(grid_size[1] * grid_size[2])

        binary_images_L90 = torch.from_numpy(bin_im_cc[2][k]).float().to(device)
        binary_images_L90 = binary_images_L90.unsqueeze(0).unsqueeze(0)

        # normalise the float tensors
        img = (sample_L90_crop - sample_L90_crop.mean())/ (sample_L90_crop.std() + 1e-8)
        bin_im_norm = (binary_images_L90 - binary_images_L90.mean())/ (binary_images_L90.std() + 1e-8)

        result_L90 = F.conv2d(img, bin_im_norm, padding=0)/(grid_size[1] * grid_size[2])

        binary_images_L24 = torch.from_numpy(bin_im_cc[3][k]).float().to(device)
        binary_images_L24 = binary_images_L24.unsqueeze(0).unsqueeze(0)

        # normalise the float tensors
        img = (sample_L24_crop - sample_L24_crop.mean())/ (sample_L24_crop.std() + 1e-8)
        bin_im_norm = (binary_images_L24 - binary_images_L24.mean())/ (binary_images_L24.std() + 1e-8)

        result_L24 = F.conv2d(img, bin_im_norm, padding=0)/(grid_size[1] * grid_size[2])

        flat_idx = torch.argmax(result_F90)
        max_loc_F90y = flat_idx // result_F90.shape[-1]
        max_loc_F90x = flat_idx % result_F90.shape[-1]
        max_val_F90 = result_F90[0, 0, max_loc_F90x, max_loc_F90y]

        flat_idx = torch.argmax(result_F24)
        max_loc_F24y = flat_idx // result_F24.shape[-1]
        max_loc_F24x = flat_idx % result_F24.shape[-1]
        max_val_F24 = result_F24[0, 0, max_loc_F24x, max_loc_F24y]

        flat_idx = torch.argmax(result_L90)
        max_loc_L90y = flat_idx // result_L90.shape[-1]
        max_loc_L90x = flat_idx % result_L90.shape[-1]
        max_val_L90 = result_L90[0, 0, max_loc_L90x, max_loc_L90y]

        flat_idx = torch.argmax(result_L24)
        max_loc_L24y = flat_idx // result_L24.shape[-1]
        max_loc_L24x = flat_idx % result_L24.shape[-1]
        max_val_L24 = result_L24[0, 0, max_loc_L24x, max_loc_L24y]

        # Check if this translation provides a better match than the top translations
        if max_val_F90 > min(top_F90_scores_temp):
            top_F90_temp.append([M, k])
            top_F90_scores_temp.append(max_val_F90)
            max_location_F90x_temp.append(max_loc_F90x)
            max_location_F90y_temp.append(max_loc_F90y)
            model_matrices_F90_temp.append(binary_images_F90)
            model_slice_F90_temp.append(k)

            # Keep only the top rotations
            if len(top_F90_scores_temp) > best_combos_single_run:
                min_index = top_F90_scores_temp.index(min(top_F90_scores_temp))
                del top_F90_temp[min_index]
                del top_F90_scores_temp[min_index]
                del max_location_F90x_temp[min_index]
                del max_location_F90y_temp[min_index]
                del model_matrices_F90_temp[min_index]
                del model_slice_F90_temp[min_index]

        if max_val_F24 > min(top_F24_scores_temp):
            top_F24_temp.append([M, k])
            top_F24_scores_temp.append(max_val_F24)
            max_location_F24x_temp.append(max_loc_F24x)
            max_location_F24y_temp.append(max_loc_F24y)
            model_matrices_F24_temp.append(binary_images_F24)
            model_slice_F24_temp.append(k)

            # Keep only the top rotations
            if len(top_F24_scores_temp) > best_combos_single_run:
                min_index = top_F24_scores_temp.index(min(top_F24_scores_temp))
                del top_F24_temp[min_index]
                del top_F24_scores_temp[min_index]
                del max_location_F24x_temp[min_index]
                del max_location_F24y_temp[min_index]
                del model_matrices_F24_temp[min_index]
                del model_slice_F24_temp[min_index]

        if max_val_L90 > min(top_L90_scores_temp):
            top_L90_temp.append([M, k])
            top_L90_scores_temp.append(max_val_L90)
            max_location_L90x_temp.append(max_loc_L90x)
            max_location_L90y_temp.append(max_loc_L90y)
            model_matrices_L90_temp.append(binary_images_L90)
            model_slice_L90_temp.append(k)

            # Keep only the top three rotations
            if len(top_L90_scores_temp) > best_combos_single_run:
                min_index = top_L90_scores_temp.index(min(top_L90_scores_temp))
                del top_L90_temp[min_index]
                del top_L90_scores_temp[min_index]
                del max_location_L90x_temp[min_index]
                del max_location_L90y_temp[min_index]
                del model_matrices_L90_temp[min_index]
                del model_slice_L90_temp[min_index]

        if max_val_L24 > min(top_L24_scores_temp):
            top_L24_temp.append([M, k])
            top_L24_scores_temp.append(max_val_L24)
            max_location_L24x_temp.append(max_loc_L24x)
            max_location_L24y_temp.append(max_loc_L24y)
            model_matrices_L24_temp.append(binary_images_L24)
            model_slice_L24_temp.append(k)

            # Keep only the top three rotations
            if len(top_L24_scores_temp) > best_combos_single_run:
                min_index = top_L24_scores_temp.index(min(top_L24_scores_temp))
                del top_L24_temp[min_index]
                del top_L24_scores_temp[min_index]
                del max_location_L24x_temp[min_index]
                del max_location_L24y_temp[min_index]
                del model_matrices_L24_temp[min_index]
                del model_slice_L24_temp[min_index]


    max_F90 = top_F90_scores_temp.index(max(top_F90_scores_temp))
    max_F24 = top_F24_scores_temp.index(max(top_F24_scores_temp))
    max_L90 = top_L90_scores_temp.index(max(top_L90_scores_temp))
    max_L24 = top_L24_scores_temp.index(max(top_L24_scores_temp))

    a = (1 / np.sqrt(float(cs_per_volume))) * np.sqrt(
        pow(float(top_F90_scores_temp[max_F90]), 2) + pow(float(top_F24_scores_temp[max_F24]), 2) +
        pow(float(top_L90_scores_temp[max_L90]), 2) + pow(float(top_L24_scores_temp[max_L24]), 2))

    print(f'corr {a} vs {max(tcarr)}')
    print(f'{len(tcarr)}')
    if max(tcarr) <= a:
        max_F90 = top_F90_scores_temp.index(max(top_F90_scores_temp))

        max_F24 = top_F24_scores_temp.index(max(top_F24_scores_temp))

        max_L90 = top_L90_scores_temp.index(max(top_L90_scores_temp))

        max_L24 = top_L24_scores_temp.index(max(top_L24_scores_temp))

        tmm_append = (model_matrices_F90_temp[max_F90], model_matrices_F24_temp[max_F24],
                      model_matrices_L90_temp[max_L90], model_matrices_L24_temp[max_L24])
        tmm.append(tmm_append)

        location_append = (max_location_F90x_temp[max_F90], max_location_F90y_temp[max_F90],
         max_location_F24x_temp[max_F24], max_location_F24y_temp[max_F24],
         max_location_L90x_temp[max_L90], max_location_L90y_temp[max_L90],
         max_location_L24x_temp[max_L24], max_location_L24y_temp[max_L24])
        max_location.append(location_append)

        tscorr_append = (float(top_F90_scores_temp[max_F90]), float(top_F24_scores_temp[max_F24]), float(top_L90_scores_temp[max_L90]), float(top_L24_scores_temp[max_L24]))
        tscorr.append(tscorr_append)
        # top_v_translations.append(v_translations)
        tvscaling.append(scaling_values)
        tvrotation.append(v_rotations)
        tvshearing.append(shear_values)

        thresh.append(thr)

        tlabel_append = [float(top_surface_vector[0]), float(top_surface_vector[1]), float(top_surface_vector[2])]
        tlabel.append(tlabel_append)

        # compute overall correlation
        a = (1 / np.sqrt(float(cs_per_volume))) * np.sqrt(
            pow(float(top_F90_scores_temp[max_F90]), 2) + pow(float(top_F24_scores_temp[max_F24]), 2) +
            pow(float(top_L90_scores_temp[max_L90]), 2) + pow(float(top_L24_scores_temp[max_L24]), 2))
        tcarr.append(float(a))
        overall_corr_values.append(a)

    if len(tcarr) > best_combos:
        # continue here

        # Keep only the top combos

        min_index = tcarr.index(min(tcarr))
        print(f'latest corr {tcarr[-1]}')

        del tmm[min_index]
        del max_location[min_index]
        del tscorr[min_index]

        del tvscaling[min_index]
        del tvrotation[min_index]
        del tvshearing[min_index]
        del tcarr[min_index]
        del thresh[min_index]

        del tlabel[min_index]

    # Convert tensor to NumPy array
    # top_F90_scores_temp_np, top_F24_scores_temp_np, top_L90_scores_temp_np, top_L24_scores_temp_np = top_F90_scores_temp_cpu.numpy(), top_F24_scores_temp_cpu.numpy(), top_L90_scores_temp_cpu.numpy(), top_L24_scores_temp_cpu.numpy()

    overall_corr_values = (1 / np.sqrt(cs_per_volume)) * np.sqrt(
        pow(float(max(top_F90_scores_temp)), 2) + pow(float(max(top_F24_scores_temp)), 2) + pow(
            float(max(top_L90_scores_temp)), 2) + pow(float(max(top_L24_scores_temp)), 2))

    # overall_corr_values = (1/np.sqrt(cs_per_volume))*np.sqrt(pow(max(top_F90_scores), 2) + pow(max(top_F24_scores), 2) + pow(max(top_L90_scores), 2) + pow(max(top_L24_scores), 2))
    top_transf_matrices = (tF90, tF24, tL90, tL24)
    top_slices = (msF90, msF24, msL90, msL24)
    cropped_samples = (sample_F90_crop, sample_F24_crop, sample_L90_crop, sample_L24_crop)
    # print(mean_tensor_F90.shape, mean_tensor_F24.shape, mean_tensor_L90.shape, mean_tensor_L24.shape)
    # print(f'overall_corr_values is {overall_corr_values}')

    return top_transf_matrices, tscorr, max_location, tmm, tvscaling, tvrotation, tvshearing, cropped_samples, tlabel, overall_corr_values, tcarr, top_slices, thresh

def save_data_csv(scaling_rows, rotation_rows, shearing_rows, surface_on_top, single_corr_values, topc, elapsed_time_text, iteration_count, corner_label):
    # Specify the file path
    gmt = time.gmtime()
    file_path = saving_files[0] + f'/{gmt[0]}{gmt[1]:02}{gmt[2]:02}_conv2d_arrays_data_{corner_label}.csv'

    # Define column headers
    columns = ['Description', 'Array']

    # Define the data to be written
    data = [
        {'Description': 'scaling', 'Array': scaling_rows},
        {'Description': 'rotation', 'Array': rotation_rows},
        {'Description': 'shearing', 'Array': shearing_rows},
        {'Description': 'topsurface', 'Array': surface_on_top},
        {'Description': 'single_correlation', 'Array': single_corr_values},
        {'Description': 'overall_correlation', 'Array': topc},
        {'Description': 'iterations', 'Array': iteration_count},
        {'Description': 'time needed', 'Array': elapsed_time_text}
    ]

    # Write data to CSV file
    with open(file_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=columns)
        writer.writeheader()
        for row in data:
            writer.writerow(row)

def save_data_csv_multiple(scaling_rows, rotation_rows, shearing_rows, surface_on_top, single_corr_values, topc, elapsed_time_text, iteration_count, corner_label, directory_save, starting_nr_save):
    # Specify the file path
    gmt = time.gmtime()
    file_path = directory_save + f'/{gmt[0]}{gmt[1]:02}{gmt[2]:02}_conv2d_arrays_data_{corner_label}_{starting_nr_save}.csv'

    # Define column headers
    columns = ['Description', 'Array']

    # Define the data to be written
    data = [
        {'Description': 'scaling', 'Array': scaling_rows},
        {'Description': 'rotation', 'Array': rotation_rows},
        {'Description': 'shearing', 'Array': shearing_rows},
        {'Description': 'topsurface', 'Array': surface_on_top},
        {'Description': 'single_correlation', 'Array': single_corr_values},
        {'Description': 'overall_correlation', 'Array': topc},
        {'Description': 'iterations', 'Array': iteration_count},
        {'Description': 'time needed', 'Array': elapsed_time_text}
    ]

    # Write data to CSV file
    with open(file_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=columns)
        writer.writeheader()
        for row in data:
            writer.writerow(row)

def plot_4cs(bin_im, slice_number, label_images, directory_print, k, starting_nr_print):
    bin_im_np_orig = [0, 0, 0, 0]

    fig_bin = plt.figure()
    fig_bin.subplots_adjust(hspace=0.4, top=0.85)
    fig_bin.suptitle("Crossections", fontsize=15)
    ax1_bin = fig_bin.add_subplot(1, 4, 1)
    ax1_bin.title.set_text(f'{label_images[0]}')
    bin_im_np = bin_im[0].squeeze()
    bin_im_np = bin_im_np.cpu()
    bin_im_np = bin_im_np.numpy()
    bin_im_np_orig[0] = bin_im_np.reshape(grid_size[0], grid_size[1])
    img = np.clip(bin_im_np_orig[0], 0, 255).astype(np.uint8)
    plt.imshow(img)
    cv2.imwrite(directory_print + f'/{label_images[0]}_conv2d_match_{starting_nr_print}_{k}_{num_iterations}_optimize.png', img)

    ax2_bin = fig_bin.add_subplot(2, 3, 2)
    ax2_bin.title.set_text(f'{label_images[1]}')
    bin_im_np = bin_im[1].squeeze()
    bin_im_np = bin_im_np.cpu()
    bin_im_np = bin_im_np.numpy()
    bin_im_np_orig[1] = bin_im_np.reshape(grid_size[0], grid_size[1])
    img = np.clip(bin_im_np_orig[1], 0, 255).astype(np.uint8)
    plt.imshow(img)
    cv2.imwrite(directory_print + f'/{label_images[1]}_conv2d_match_{starting_nr_print}_{k}_{num_iterations}_optimize.png', img)

    ax3_bin = fig_bin.add_subplot(2, 3, 3)
    ax3_bin.title.set_text(f'{label_images[2]}')
    bin_im_np = bin_im[2].squeeze()
    bin_im_np = bin_im_np.cpu()
    bin_im_np = bin_im_np.numpy()
    bin_im_np_orig[2] = bin_im_np.reshape(grid_size[0], grid_size[1])
    img = np.clip(bin_im_np_orig[2], 0, 255).astype(np.uint8)
    plt.imshow(img)
    cv2.imwrite(directory_print + f'/{label_images[2]}_conv2d_match_{starting_nr_print}_{k}_{num_iterations}_optimize.png', img)

    ax4_bin = fig_bin.add_subplot(2, 3, 4)
    ax4_bin.title.set_text(f'{label_images[3]}')
    bin_im_np = bin_im[3].squeeze()
    bin_im_np = bin_im_np.cpu()
    bin_im_np = bin_im_np.numpy()
    bin_im_np_orig[3] = bin_im_np.reshape(grid_size[0], grid_size[1])
    img = np.clip(bin_im_np_orig[3], 0, 255).astype(np.uint8)
    plt.imshow(img)
    cv2.imwrite(directory_print + f'/{label_images[3]}_conv2d_match_{starting_nr_print}_{k}_{num_iterations}_optimize.png', img)

def plot_4cs_multi(bin_im, label_images, directory_print, k, starting_nr_print):
    bin_im_np_orig = [0, 0, 0, 0]

    fig_bin = plt.figure()
    fig_bin.subplots_adjust(hspace=0.4, top=0.85)
    fig_bin.suptitle("Crossections", fontsize=15)
    ax1_bin = fig_bin.add_subplot(1, 4, 1)
    ax1_bin.title.set_text(f'{label_images[0]}')
    print(k)
    print(len(bin_im))
    bin_im_np = bin_im[k][0].squeeze()
    bin_im_np = bin_im_np.cpu()
    bin_im_np = bin_im_np.numpy()
    bin_im_np_orig[0] = bin_im_np.reshape(grid_size[0], grid_size[1])
    img = np.clip(bin_im_np_orig[0], 0, 255).astype(np.uint8)
    plt.imshow(img)
    cv2.imwrite(directory_print + f'/{label_images[0]}_conv2d_match_{starting_nr_print}_{k}_{num_iterations}_optimize.png', img)

    ax2_bin = fig_bin.add_subplot(2, 3, 2)
    ax2_bin.title.set_text(f'{label_images[1]}')
    bin_im_np = bin_im[k][1].squeeze()
    bin_im_np = bin_im_np.cpu()
    bin_im_np = bin_im_np.numpy()
    bin_im_np_orig[1] = bin_im_np.reshape(grid_size[0], grid_size[1])
    img = np.clip(bin_im_np_orig[1], 0, 255).astype(np.uint8)
    plt.imshow(img)
    cv2.imwrite(directory_print + f'/{label_images[1]}_conv2d_match_{starting_nr_print}_{k}_{num_iterations}_optimize.png', img)

    ax3_bin = fig_bin.add_subplot(2, 3, 3)
    ax3_bin.title.set_text(f'{label_images[2]}')
    bin_im_np = bin_im[k][2].squeeze()
    bin_im_np = bin_im_np.cpu()
    bin_im_np = bin_im_np.numpy()
    bin_im_np_orig[2] = bin_im_np.reshape(grid_size[0], grid_size[1])
    img = np.clip(bin_im_np_orig[2], 0, 255).astype(np.uint8)
    plt.imshow(img)
    cv2.imwrite(directory_print + f'/{label_images[2]}_conv2d_match_{starting_nr_print}_{k}_{num_iterations}_optimize.png', img)

    ax4_bin = fig_bin.add_subplot(2, 3, 4)
    ax4_bin.title.set_text(f'{label_images[3]}')
    bin_im_np = bin_im[k][3].squeeze()
    bin_im_np = bin_im_np.cpu()
    bin_im_np = bin_im_np.numpy()
    bin_im_np_orig[3] = bin_im_np.reshape(grid_size[0], grid_size[1])
    img = np.clip(bin_im_np_orig[3], 0, 255).astype(np.uint8)
    plt.imshow(img)
    cv2.imwrite(directory_print + f'/{label_images[3]}_conv2d_match_{starting_nr_print}_{k}_{num_iterations}_optimize.png', img)

def print_top_correlations_left(max_color, min_color, color_palette, per_slice_corr_scores, bin_im, cropped_sample_images, top_v_scaling,
                                top_v_rotations, top_v_shearing, top_surface_vector, transparency):
    color_scaling = int(round(100 * (max_color - min_color), 2))
    color_max = round(rounddown_0_01(max_color), 2)
    color_min = round(rounddown_0_01(min_color), 2)
    # textbox color according to correlation value
    text_values = np.linspace(color_min, color_max, color_scaling)
    # Create a seaborn color palette
    palette = sns.color_palette(color_palette, n_colors=len(text_values))

    print(len(overall_corr_values))
    ###########################################################################################
    if len(overall_corr_values) == 1:
        max_ind = sorted(range(len(overall_corr_values)), key=lambda sub: top_corr_arr[sub])[-1:]
        plot_n = 1
    elif len(overall_corr_values) == 2:
        max_ind = sorted(range(len(overall_corr_values)), key=lambda sub: top_corr_arr[sub])[-2:]
        plot_n = 2
    elif len(overall_corr_values) == 3:
        max_ind = sorted(range(len(overall_corr_values)), key=lambda sub: top_corr_arr[sub])[-3:]
        plot_n = 3
    elif len(overall_corr_values) >= 4:
        max_ind = sorted(range(len(overall_corr_values)), key=lambda sub: top_corr_arr[sub])[-4:]
        plot_n = 4
    ###########################################################################################

    print('plot n is ' + str(plot_n))
    print(per_slice_corr_scores)
    print(overall_corr_values)
    print(max_location)
    fig_corr, axs_corr = plt.subplots(cs_per_volume, plot_n, figsize=(30, 30))
    fig_corr.suptitle(f'Correlation of {sample_directory[63:]}', fontsize=16)

    # Display the original image
    axs_corr = axs_corr.flatten()
    axs_corr[0].set_title('L90')
    axs_corr[1].set_title('L24')
    axs_corr[2].set_title('F90')
    axs_corr[3].set_title('F24')

    cropped_samples_np = [0, 0, 0, 0]
    bin_im_np_orig = [0, 0, 0, 0]
    bin_im_np = [0, 0, 0, 0]

    for i in range(0, cs_per_volume):
        cropped_samples_np[i] = cropped_sample_images[i].squeeze()
        cropped_samples_np[i] = cropped_samples_np[i].cpu()
        cropped_samples_np[i] = cropped_samples_np[i].numpy()
        plt.imshow(cropped_samples_np[i])
        cropped_samples_np[i] = cropped_samples_np[i].reshape(scaling_square, scaling_square)

    # Display the "best_combos"
    for i in range(0, plot_n):

        bin_im_np[0] = bin_im[0][i].squeeze()
        bin_im_np[1] = bin_im[1][i].squeeze()
        bin_im_np[2] = bin_im[2][i].squeeze()
        bin_im_np[3] = bin_im[3][i].squeeze()

        for r in range(0, cs_per_volume):
            bin_im_np[r] = bin_im_np[r].cpu()
            bin_im_np[r] = bin_im_np[r].numpy()
            bin_im_np_orig[r] = bin_im_np[r].reshape(grid_size[0], grid_size[1])

        h = bin_im_np_orig[i].shape[0]
        w = bin_im_np_orig[i].shape[1]

        iL90 = i * 4 + 0
        iL24 = i * 4 + 1
        iF90 = i * 4 + 2
        iF24 = i * 4 + 3

        max_loc_L90x = max_location[4][i].squeeze()
        max_loc_L90y = max_location[5][i].squeeze()
        overlay_L90 = cropped_samples_np[2].copy()

        max_loc_L90x = max_loc_L90x.cpu()
        max_loc_L90y = max_loc_L90y.cpu()

        overlay_L90[max_loc_L90x.numpy():max_loc_L90x.numpy() + w,
        max_loc_L90y.numpy():max_loc_L90y.numpy() + h] = np.uint8(
            transparency * bin_im_np_orig[2][:, :] + (1 - transparency) * overlay_L90[
                max_loc_L90x.numpy():max_loc_L90x.numpy() + w,
                max_loc_L90y.numpy():max_loc_L90y.numpy() + h])

        axs_corr[iL90].imshow(transparency * cropped_samples_np[2] + (1 - transparency) * overlay_L90, cmap='gray')

        top_score_iL90 = per_slice_corr_scores[3][i].squeeze()
        top_score_iL90 = top_score_iL90.cpu()
        top_score_iL90 = float(top_score_iL90.numpy())
        if per_slice_corr_scores[2][i] <= color_min:
            axs_corr[iL90].text(overlay_L90.shape[1] + round(overlay_L90.shape[1] / 20),
                                overlay_L90.shape[0] - round(overlay_L90.shape[0] / 5), "%.2f" % per_slice_corr_scores[2][i],
                                bbox=dict(facecolor=palette[0], alpha=0.5))
        elif per_slice_corr_scores[2][i] >= color_max:
            axs_corr[iL90].text(overlay_L90.shape[1] + round(overlay_L90.shape[1] / 20),
                                overlay_L90.shape[0] - round(overlay_L90.shape[0] / 5), "%.2f" % per_slice_corr_scores[2][i],
                                bbox=dict(facecolor=palette[int(color_scaling - 1)], alpha=0.5))
        else:
            axs_corr[iL90].text(overlay_L90.shape[1] + round(overlay_L90.shape[1] / 20),
                                overlay_L90.shape[0] - round(overlay_L90.shape[0] / 5), "%.2f" % per_slice_corr_scores[2][i],
                                bbox=dict(facecolor=palette[int(rounddown_0_01(per_slice_corr_scores[2][i] - color_min) * 100)],
                                          alpha=0.5))
        ###

        # here the coordinates are somehow weirdly shifted, used to be [6], then [7]
        max_loc_L24x = max_location[7][i].squeeze()
        max_loc_L24y = max_location[6][i].squeeze()
        overlay_L24 = cropped_samples_np[3].copy()

        max_loc_L24x = max_loc_L24x.cpu()
        max_loc_L24y = max_loc_L24y.cpu()

        # print((grid_size[1] - max_loc_L24y.numpy()), (grid_size[0]-max_loc_L24x.numpy()))
        overlay_L24[max_loc_L24x.numpy():max_loc_L24x.numpy() + w,
        max_loc_L24y.numpy():max_loc_L24y.numpy() + h] = np.uint8(
            transparency * bin_im_np_orig[3][:, :] + (1 - transparency) * overlay_L24[
                max_loc_L24x.numpy():max_loc_L24x.numpy() + w,
                max_loc_L24y.numpy():max_loc_L24y.numpy() + h])

        axs_corr[iL24].imshow(transparency * cropped_samples_np[3] + (1 - transparency) * overlay_L24, cmap='gray')

        if per_slice_corr_scores[3][i] <= color_min:
            axs_corr[iL24].text(overlay_L24.shape[1] + round(overlay_L24.shape[1] / 20),
                                overlay_L24.shape[0] - round(overlay_L24.shape[0] / 5), "%.2f" % per_slice_corr_scores[3][i],
                                bbox=dict(facecolor=palette[0], alpha=0.5))
        elif per_slice_corr_scores[3][i] >= color_max:
            axs_corr[iL24].text(overlay_L24.shape[1] + round(overlay_L24.shape[1] / 20),
                                overlay_L24.shape[0] - round(overlay_L24.shape[0] / 5), "%.2f" % per_slice_corr_scores[3][i],
                                bbox=dict(facecolor=palette[int(color_scaling - 1)], alpha=0.5))
        else:
            axs_corr[iL24].text(overlay_L24.shape[1] + round(overlay_L24.shape[1] / 20),
                                overlay_L24.shape[0] - round(overlay_L24.shape[0] / 5), "%.2f" % per_slice_corr_scores[3][i],
                                bbox=dict(facecolor=palette[int(rounddown_0_01(per_slice_corr_scores[3][i] - color_min) * 100)],
                                          alpha=0.5))
        ###

        # print(max_location)
        max_loc_F90x = max_location[0][i].squeeze()
        max_loc_F90y = max_location[1][i].squeeze()
        overlay_F90 = cropped_samples_np[0].copy()

        max_loc_F90x = max_loc_F90x.cpu()
        max_loc_F90y = max_loc_F90y.cpu()

        overlay_F90[
            max_loc_F90x.numpy():max_loc_F90x.numpy() + w, max_loc_F90y.numpy():max_loc_F90y.numpy() + h] = np.uint8(
            transparency * bin_im_np_orig[0][:, :] + (1 - transparency) * overlay_F90[
                max_loc_F90x.numpy():max_loc_F90x.numpy() + w, max_loc_F90y.numpy():max_loc_F90y.numpy() + h])

        axs_corr[iF90].imshow(transparency * cropped_samples_np[0] + (1 - transparency) * overlay_F90, cmap='gray')

        if per_slice_corr_scores[0][i] <= color_min:
            axs_corr[iF90].text(overlay_F90.shape[1] + round(overlay_F90.shape[1] / 20),
                                overlay_F90.shape[0] - round(overlay_F90.shape[0] / 5), "%.2f" % per_slice_corr_scores[0][i],
                                bbox=dict(facecolor=palette[0], alpha=0.5))
        elif per_slice_corr_scores[0][i] >= color_max:
            axs_corr[iF90].text(overlay_F90.shape[1] + round(overlay_F90.shape[1] / 20),
                                overlay_F90.shape[0] - round(overlay_F90.shape[0] / 5), "%.2f" % per_slice_corr_scores[0][i],
                                bbox=dict(facecolor=palette[int(color_scaling - 1)], alpha=0.5))
        else:
            axs_corr[iF90].text(overlay_F90.shape[1] + round(overlay_F90.shape[1] / 20),
                                overlay_F90.shape[0] - round(overlay_F90.shape[0] / 5), "%.2f" % per_slice_corr_scores[0][i],
                                bbox=dict(facecolor=palette[int(rounddown_0_01(per_slice_corr_scores[0][i] - color_min) * 100)],
                                          alpha=0.5))
        ###

        # print(max_location)
        max_loc_F24x = max_location[2][i].squeeze()
        max_loc_F24y = max_location[3][i].squeeze()
        overlay_F24 = cropped_samples_np[1].copy()

        max_loc_F24x = max_loc_F24x.cpu()
        max_loc_F24y = max_loc_F24y.cpu()

        # print((grid_size[1] - max_loc_F24y.numpy()), (grid_size[0]-max_loc_F24x.numpy()))
        overlay_F24[
            max_loc_F24x.numpy():max_loc_F24x.numpy() + w, max_loc_F24y.numpy():max_loc_F24y.numpy() + h] = np.uint8(
            transparency * bin_im_np_orig[1][:, :] + (1 - transparency) * overlay_F24[
                max_loc_F24x.numpy():max_loc_F24x.numpy() + w, max_loc_F24y.numpy():max_loc_F24y.numpy() + h])

        axs_corr[iF24].imshow(transparency * cropped_samples_np[1] + (1 - transparency) * overlay_F24, cmap='gray')

        if per_slice_corr_scores[1][i] <= color_min:
            axs_corr[iF24].text(overlay_F24.shape[1] + round(overlay_F24.shape[1] / 20),
                                overlay_F24.shape[0] - round(overlay_F24.shape[0] / 5), "%.2f" % per_slice_corr_scores[1][i],
                                bbox=dict(facecolor=palette[0], alpha=0.5))
        elif per_slice_corr_scores[1][i] >= color_max:
            axs_corr[iF24].text(overlay_F24.shape[1] + round(overlay_F24.shape[1] / 20),
                                overlay_F24.shape[0] - round(overlay_F24.shape[0] / 5), "%.2f" % per_slice_corr_scores[1][i],
                                bbox=dict(facecolor=palette[int(color_scaling - 1)], alpha=0.5))
        else:
            axs_corr[iF24].text(overlay_F24.shape[1] + round(overlay_F24.shape[1] / 20),
                                overlay_F24.shape[0] - round(overlay_F24.shape[0] / 5), "%.2f" % per_slice_corr_scores[1][i],
                                bbox=dict(facecolor=palette[int(rounddown_0_01(per_slice_corr_scores[1][i] - color_min) * 100)],
                                          alpha=0.5))

        per_slice_corr_scores[0][i] = per_slice_corr_scores[0][i].cpu()
        per_slice_corr_scores[1][i] = per_slice_corr_scores[1][i].cpu()
        per_slice_corr_scores[2][i] = per_slice_corr_scores[2][i].cpu()
        per_slice_corr_scores[3][i] = per_slice_corr_scores[3][i].cpu()

        top_corr = 1 / 2 * np.sqrt(pow(float(per_slice_corr_scores[2][i]), 2) + pow(float(per_slice_corr_scores[3][i]), 2) +
                                   pow(float(per_slice_corr_scores[0][i]), 2) + pow(float(per_slice_corr_scores[1][i]), 2))

        axs_corr[iF24].text(overlay_F24.shape[1] + round(overlay_F24.shape[0] / 20),
                            overlay_F24.shape[0] - round(overlay_F24.shape[0] / 2.8),
                            f'scaling \n {top_v_scaling[1][0]} \n {top_v_scaling[1][1]} \n {top_v_scaling[1][2]}' +
                            f'\nrotations \n {top_v_rotations[i][0]} \n {top_v_rotations[i][1]} \n {top_v_rotations[i][2]}' +
                            f'\nshearing \n {top_v_shearing[i][0]} \n {top_v_shearing[i][1]} \n {top_v_shearing[i][2]}' +
                            f'\n {top_v_shearing[i][3]} \n {top_v_shearing[i][4]} \n {top_v_shearing[i][5]}' +
                            f'\ntopsurface \nh {rounddown_0_01(top_surface_vector[i][0])} \nk {rounddown_0_01(top_surface_vector[i][1])} \nl {rounddown_0_01(top_surface_vector[i][2])}' +
                            f'\ntotal corr {top_corr}')

        top_corr = top_corr.squeeze()
        top_corr = float(top_corr)
        if top_corr <= color_min:
            axs_corr[iF24].text(overlay_F24.shape[1] + round(overlay_F24.shape[0] / 20),
                                overlay_F24.shape[0] - round(overlay_F24.shape[0] / 3.5), "%.2f" % top_corr,
                                bbox=dict(facecolor=palette[0], alpha=0.5))
        elif top_corr >= color_max:
            axs_corr[iF24].text(overlay_F24.shape[1] + round(overlay_F24.shape[0] / 20),
                                overlay_F24.shape[0] - round(overlay_F24.shape[0] / 3.5), "%.2f" % top_corr,
                                bbox=dict(facecolor=palette[int(color_scaling - 1)], alpha=0.5))
        else:
            axs_corr[iF24].text(overlay_F24.shape[1] + round(overlay_F24.shape[0] / 20),
                                overlay_F24.shape[0] - round(overlay_F24.shape[0] / 3.5), "%.2f" % top_corr,
                                bbox=dict(facecolor=palette[int(round(top_corr - color_min, 2) * 100) - 1], alpha=0.5))

    # record current timestamp
    gmt = time.gmtime()
    plt.savefig(directory + f'/{gmt[0]}{gmt[1]:02}{gmt[2]:02}_conv2d_plot_{corner[0]}_{num_iterations}_optimize.png')

def print_top_correlations_left_multi(max_color, min_color, color_palette, per_slice_corr_scores_print, bin_im_print, cropped_sample_images, top_v_scaling_print,
                                top_v_rotations_print, top_v_shearing_print, top_surface_vector_print,
                                      top_corr_print, top_v_singlecorr_print, max_loc_print, starting_nr_print, transparency_o, transparency_s, directory_print, sample_directory):
    color_scaling = int(round(100 * (max_color - min_color), 2))
    color_max = round(rounddown_0_01(max_color), 2)
    color_min = round(rounddown_0_01(min_color), 2)
    # textbox color according to correlation value
    text_values = np.linspace(color_min, color_max, color_scaling)
    # Create a seaborn color palette
    palette = sns.color_palette(color_palette, n_colors=len(text_values))

    print(len(top_corr_print))
    ###########################################################################################
    if len(top_corr_print) == 1:
        max_ind = sorted(range(len(top_corr_print)), key=lambda sub: top_corr_print[sub])[-1:]
        plot_n = 1
        top_idx_print = (np.argsort(top_corr_print)[-1])
    elif len(top_corr_print) == 2:
        max_ind = sorted(range(len(top_corr_print)), key=lambda sub: top_corr_print[sub])[-2:]
        plot_n = 2
        top_idx_print = (np.argsort(top_corr_print)[-1], np.argsort(top_corr_print)[-2])
    elif len(top_corr_print) == 3:
        max_ind = sorted(range(len(top_corr_print)), key=lambda sub: top_corr_print[sub])[-3:]
        plot_n = 3
        top_idx_print = (np.argsort(top_corr_print)[-1], np.argsort(top_corr_print)[-2],
                   np.argsort(top_corr_print)[-3])
    elif len(top_corr_print) >= 4:
        max_ind = sorted(range(len(top_corr_print)), key=lambda sub: top_corr_print[sub])[-4:]
        plot_n = 4
        top_idx_print = (np.argsort(top_corr_print)[-1], np.argsort(top_corr_print)[-2],
                   np.argsort(top_corr_print)[-3], np.argsort(top_corr_print)[-4])

        ###########################################################################################

    print('plot n is ' + str(plot_n))
    print(per_slice_corr_scores_print)
    print(top_corr_print)
    print(top_surface_vector_print)
    fig_corr, axs_corr = plt.subplots(cs_per_volume, plot_n, figsize=(30, 30))
    fig_corr.suptitle(f'Correlation of {sample_directory[63:]}', fontsize=16)

    # Display the original image
    axs_corr = axs_corr.flatten()
    axs_corr[0].set_title('L90')
    axs_corr[1].set_title('L24')
    axs_corr[2].set_title('F90')
    axs_corr[3].set_title('F24')

    cropped_samples_np = [0, 0, 0, 0]
    bin_im_print_np_orig = [0, 0, 0, 0]
    bin_im_print_np = [0, 0, 0, 0]

    for i in range(0, cs_per_volume):
        cropped_samples_np[i] = cropped_sample_images[i].squeeze()
        cropped_samples_np[i] = cropped_samples_np[i].cpu()
        cropped_samples_np[i] = cropped_samples_np[i].numpy()
        # plt.imshow(cropped_samples_np[i])
        # plt.show()
        cropped_samples_np[i] = cropped_samples_np[i].reshape(scaling_square, scaling_square)

    # Display the "best_combos"
    for i in range(0, plot_n):
        # print(i)
        # print(len(bin_im_print))
        # print(len(bin_im_print[0]))
        # print(bin_im_print[0])
        # print(len(bin_im_print[0][1]))
        # print(bin_im_print[0][1].squeeze())
        # print(bin_im_print[0][1].squeeze().squeeze())
        # plt.imshow(bin_im_print[0][i].squeeze())
        # plt.show()
        bin_im_print_np[0] = bin_im_print[top_idx_print[i]][0].squeeze()
        bin_im_print_np[1] = bin_im_print[top_idx_print[i]][1].squeeze()
        bin_im_print_np[2] = bin_im_print[top_idx_print[i]][2].squeeze()
        bin_im_print_np[3] = bin_im_print[top_idx_print[i]][3].squeeze()

        for r in range(0, cs_per_volume):
            bin_im_print_np[r] = bin_im_print_np[r].cpu()
            bin_im_print_np[r] = bin_im_print_np[r].numpy()
            bin_im_print_np_orig[r] = bin_im_print_np[r].reshape(grid_size[0], grid_size[1])

        h = bin_im_print_np_orig[i].shape[0]
        w = bin_im_print_np_orig[i].shape[1]

        iL90 = i * 4 + 0
        iL24 = i * 4 + 1
        iF90 = i * 4 + 2
        iF24 = i * 4 + 3

        print('here in the plot')
        # print(top_v_scaling_print)
        # print(top_v_rotations_print)
        # print(top_v_shearing_print)
        # print(top_surface_vector_print)
        # print(max_loc_print)
        # print(per_slice_corr_scores)

        max_loc_L90x = max_loc_print[top_idx_print[i]][4].squeeze()
        max_loc_L90y = max_loc_print[top_idx_print[i]][5].squeeze()
        overlay_L90 = make_binary_testfile(cropped_samples_np[2].copy(), 0.5, transparency_s)

        overlay_L90[max_loc_L90x.numpy():max_loc_L90x.numpy() + w,
        max_loc_L90y.numpy():max_loc_L90y.numpy() + h] = np.uint8(
            transparency_s * bin_im_print_np_orig[2][:, :] + transparency_o * overlay_L90[
                max_loc_L90x.numpy():max_loc_L90x.numpy() + w,
                max_loc_L90y.numpy():max_loc_L90y.numpy() + h])

        axs_corr[iL90].imshow(transparency_s * cropped_samples_np[2] + overlay_L90, cmap='gray')

        top_score_iL90 = float(top_v_singlecorr_print[i][2])
        if top_v_singlecorr_print[top_idx_print[i]][2] <= color_min:
            axs_corr[iL90].text(overlay_L90.shape[1] + round(overlay_L90.shape[1] / 20),
                                overlay_L90.shape[0] - round(overlay_L90.shape[0] / 5), "%.2f" % top_v_singlecorr_print[top_idx_print[i]][2],
                                bbox=dict(facecolor=palette[0], alpha=0.5))
        elif top_v_singlecorr_print[top_idx_print[i]][2] >= color_max:
            axs_corr[iL90].text(overlay_L90.shape[1] + round(overlay_L90.shape[1] / 20),
                                overlay_L90.shape[0] - round(overlay_L90.shape[0] / 5), "%.2f" % top_v_singlecorr_print[top_idx_print[i]][2],
                                bbox=dict(facecolor=palette[int(color_scaling - 1)], alpha=0.5))
        else:
            axs_corr[iL90].text(overlay_L90.shape[1] + round(overlay_L90.shape[1] / 20),
                                overlay_L90.shape[0] - round(overlay_L90.shape[0] / 5), "%.2f" % top_v_singlecorr_print[top_idx_print[i]][2],
                                bbox=dict(facecolor=palette[int(rounddown_0_01(top_v_singlecorr_print[top_idx_print[i]][2] - color_min) * 100)],
                                          alpha=0.5))
        ###

        # here the coordinates are somehow weirdly shifted, used to be [6], then [7]
        max_loc_L24x = max_loc_print[top_idx_print[i]][7].squeeze()
        max_loc_L24y = max_loc_print[top_idx_print[i]][6].squeeze()
        overlay_L24 = make_binary_testfile(cropped_samples_np[3].copy(), 0.5, transparency_s)

        max_loc_L24x = max_loc_L24x.cpu()
        max_loc_L24y = max_loc_L24y.cpu()

        # print((grid_size[1] - max_loc_L24y.numpy()), (grid_size[0]-max_loc_L24x.numpy()))
        overlay_L24[max_loc_L24x.numpy():max_loc_L24x.numpy() + w,
        max_loc_L24y.numpy():max_loc_L24y.numpy() + h] = np.uint8(
            transparency_s * bin_im_print_np_orig[3][:, :] + transparency_o * overlay_L24[
                max_loc_L24x.numpy():max_loc_L24x.numpy() + w,
                max_loc_L24y.numpy():max_loc_L24y.numpy() + h])

        axs_corr[iL24].imshow(transparency_s * cropped_samples_np[3] + overlay_L24, cmap='gray')

        if top_v_singlecorr_print[top_idx_print[i]][3] <= color_min:
            axs_corr[iL24].text(overlay_L24.shape[1] + round(overlay_L24.shape[1] / 20),
                                overlay_L24.shape[0] - round(overlay_L24.shape[0] / 5), "%.2f" % top_v_singlecorr_print[top_idx_print[i]][3],
                                bbox=dict(facecolor=palette[0], alpha=0.5))
        elif top_v_singlecorr_print[top_idx_print[i]][3] >= color_max:
            axs_corr[iL24].text(overlay_L24.shape[1] + round(overlay_L24.shape[1] / 20),
                                overlay_L24.shape[0] - round(overlay_L24.shape[0] / 5), "%.2f" % top_v_singlecorr_print[top_idx_print[i]][3],
                                bbox=dict(facecolor=palette[int(color_scaling - 1)], alpha=0.5))
        else:
            axs_corr[iL24].text(overlay_L24.shape[1] + round(overlay_L24.shape[1] / 20),
                                overlay_L24.shape[0] - round(overlay_L24.shape[0] / 5), "%.2f" % top_v_singlecorr_print[top_idx_print[i]][3],
                                bbox=dict(facecolor=palette[int(rounddown_0_01(top_v_singlecorr_print[top_idx_print[i]][3] - color_min) * 100)],
                                          alpha=0.5))
        ###

        # print(max_location)
        max_loc_F90x = max_loc_print[top_idx_print[i]][0].squeeze()
        max_loc_F90y = max_loc_print[top_idx_print[i]][1].squeeze()
        overlay_F90 = make_binary_testfile(cropped_samples_np[0].copy(), 0.5, transparency_s)

        max_loc_F90x = max_loc_F90x.cpu()
        max_loc_F90y = max_loc_F90y.cpu()

        overlay_F90[
            max_loc_F90x.numpy():max_loc_F90x.numpy() + w, max_loc_F90y.numpy():max_loc_F90y.numpy() + h] = np.uint8(
            transparency_s * bin_im_print_np_orig[0][:, :] + transparency_o * overlay_F90[
                max_loc_F90x.numpy():max_loc_F90x.numpy() + w, max_loc_F90y.numpy():max_loc_F90y.numpy() + h])

        axs_corr[iF90].imshow(transparency_s * cropped_samples_np[0] + overlay_F90, cmap='gray')

        if top_v_singlecorr_print[top_idx_print[i]][0] <= color_min:
            axs_corr[iF90].text(overlay_F90.shape[1] + round(overlay_F90.shape[1] / 20),
                                overlay_F90.shape[0] - round(overlay_F90.shape[0] / 5), "%.2f" % top_v_singlecorr_print[top_idx_print[i]][0],
                                bbox=dict(facecolor=palette[0], alpha=0.5))
        elif top_v_singlecorr_print[top_idx_print[i]][0] >= color_max:
            axs_corr[iF90].text(overlay_F90.shape[1] + round(overlay_F90.shape[1] / 20),
                                overlay_F90.shape[0] - round(overlay_F90.shape[0] / 5), "%.2f" % top_v_singlecorr_print[top_idx_print[i]][0],
                                bbox=dict(facecolor=palette[int(color_scaling - 1)], alpha=0.5))
        else:
            axs_corr[iF90].text(overlay_F90.shape[1] + round(overlay_F90.shape[1] / 20),
                                overlay_F90.shape[0] - round(overlay_F90.shape[0] / 5), "%.2f" % top_v_singlecorr_print[top_idx_print[i]][0],
                                bbox=dict(facecolor=palette[int(rounddown_0_01(top_v_singlecorr_print[top_idx_print[i]][0] - color_min) * 100)],
                                          alpha=0.5))
        ###

        # print(max_location)
        max_loc_F24x = max_loc_print[top_idx_print[i]][2].squeeze()
        max_loc_F24y = max_loc_print[top_idx_print[i]][3].squeeze()
        overlay_F24 = make_binary_testfile(cropped_samples_np[1].copy(), 0.5, transparency_s)

        max_loc_F24x = max_loc_F24x.cpu()
        max_loc_F24y = max_loc_F24y.cpu()

        # print((grid_size[1] - max_loc_F24y.numpy()), (grid_size[0]-max_loc_F24x.numpy()))
        overlay_F24[
            max_loc_F24x.numpy():max_loc_F24x.numpy() + w, max_loc_F24y.numpy():max_loc_F24y.numpy() + h] = np.uint8(
            transparency_s * bin_im_print_np_orig[1][:, :] + transparency_o * overlay_F24[
                max_loc_F24x.numpy():max_loc_F24x.numpy() + w, max_loc_F24y.numpy():max_loc_F24y.numpy() + h])

        axs_corr[iF24].imshow(transparency_s * cropped_samples_np[1] + overlay_F24, cmap='gray')

        if top_v_singlecorr_print[top_idx_print[i]][1] <= color_min:
            axs_corr[iF24].text(overlay_F24.shape[1] + round(overlay_F24.shape[1] / 20),
                                overlay_F24.shape[0] - round(overlay_F24.shape[0] / 5), "%.2f" % top_v_singlecorr_print[top_idx_print[i]][1],
                                bbox=dict(facecolor=palette[0], alpha=0.5))
        elif top_v_singlecorr_print[top_idx_print[i]][1] >= color_max:
            axs_corr[iF24].text(overlay_F24.shape[1] + round(overlay_F24.shape[1] / 20),
                                overlay_F24.shape[0] - round(overlay_F24.shape[0] / 5), "%.2f" % top_v_singlecorr_print[top_idx_print[i]][1],
                                bbox=dict(facecolor=palette[int(color_scaling - 1)], alpha=0.5))
        else:
            axs_corr[iF24].text(overlay_F24.shape[1] + round(overlay_F24.shape[1] / 20),
                                overlay_F24.shape[0] - round(overlay_F24.shape[0] / 5), "%.2f" % top_v_singlecorr_print[top_idx_print[i]][1],
                                bbox=dict(facecolor=palette[int(rounddown_0_01(top_v_singlecorr_print[top_idx_print[i]][1] - color_min) * 100)],
                                          alpha=0.5))

        singlecorr_0 = float(top_v_singlecorr_print[top_idx_print[i]][0])
        singlecorr_1 = float(top_v_singlecorr_print[top_idx_print[i]][1])
        singlecorr_2 = float(top_v_singlecorr_print[top_idx_print[i]][2])
        singlecorr_3 = float(top_v_singlecorr_print[top_idx_print[i]][3])

        top_corr = 1 / 2 * np.sqrt(pow(float(singlecorr_2), 2) + pow(float(singlecorr_3), 2) +
                                   pow(float(singlecorr_0), 2) + pow(float(singlecorr_1), 2))


        print(top_corr)
        axs_corr[iF24].text(overlay_F24.shape[1] + round(overlay_F24.shape[0] / 20),
                            overlay_F24.shape[0] - round(overlay_F24.shape[0] / 2.8),
                            f'scaling \n {top_v_scaling_print[top_idx_print[i]][0]} \n {top_v_scaling_print[top_idx_print[i]][1]} \n {top_v_scaling_print[top_idx_print[i]][2]}' +
                            f'\nrotations \n {top_v_rotations_print[top_idx_print[i]][0]} \n {top_v_rotations_print[top_idx_print[i]][1]} \n {top_v_rotations_print[top_idx_print[i]][2]}' +
                            f'\nshearing \n {top_v_shearing_print[top_idx_print[i]][0]} \n {top_v_shearing_print[top_idx_print[i]][1]} \n {top_v_shearing_print[top_idx_print[i]][2]}' +
                            f'\n {top_v_shearing_print[top_idx_print[i]][3]} \n {top_v_shearing_print[top_idx_print[i]][4]} \n {top_v_shearing_print[top_idx_print[i]][5]}' +
                            f'\ntopsurface \nh {rounddown_0_01(top_surface_vector_print[top_idx_print[i]][0])} \nk {rounddown_0_01(top_surface_vector_print[top_idx_print[i]][1])} \nl {rounddown_0_01(top_surface_vector_print[top_idx_print[i]][2])}' +
                            f'\ntotal corr {top_corr}')

        top_corr = top_corr.squeeze()
        top_corr = float(top_corr)
        if top_corr <= color_min:
            axs_corr[iF24].text(overlay_F24.shape[1] + round(overlay_F24.shape[0] / 20),
                                overlay_F24.shape[0] - round(overlay_F24.shape[0] / 3.5), "%.2f" % top_corr,
                                bbox=dict(facecolor=palette[0], alpha=0.5))
        elif top_corr >= color_max:
            axs_corr[iF24].text(overlay_F24.shape[1] + round(overlay_F24.shape[0] / 20),
                                overlay_F24.shape[0] - round(overlay_F24.shape[0] / 3.5), "%.2f" % top_corr,
                                bbox=dict(facecolor=palette[int(color_scaling - 1)], alpha=0.5))
        else:
            axs_corr[iF24].text(overlay_F24.shape[1] + round(overlay_F24.shape[0] / 20),
                                overlay_F24.shape[0] - round(overlay_F24.shape[0] / 3.5), "%.2f" % top_corr,
                                bbox=dict(facecolor=palette[int(round(top_corr - color_min, 2) * 100) - 1], alpha=0.5))

    # record current timestamp
    gmt = time.gmtime()
    plt.savefig(directory_print + f'/{gmt[0]}{gmt[1]:02}{gmt[2]:02}_conv2d_plot_{corner[0]}_{starting_nr_print}_{num_iterations}_optimize.png')


def rounddown_0_01(x):
    return int(math.floor(x / 0.01)) * 0.01

def big_loop(sf, sd, gs, uc, miller, co, square_shift, iv_inp):

    ##############################################################################
    # empty arrays for the crosscorrelation
    top_parameters = [0]
    top_parameters_scores = [0]
    top_translation = [0]
    top_translation_scores = [0]
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

    model_matrices_F90 = [0]
    model_matrices_F24 = [0]
    model_matrices_L90 = [0]
    model_matrices_L24 = [0]
    model_matrices_R90 = [0]
    model_matrices_R24 = [0]

    model_slice_F90 = [0]
    model_slice_F24 = [0]
    model_slice_L90 = [0]
    model_slice_L24 = [0]
    model_slice_R90 = [0]
    model_slice_R24 = [0]

    max_location_F90x = [0]
    max_location_F90y = [0]
    max_location_F24x = [0]
    max_location_F24y = [0]
    max_location_L90x = [0]
    max_location_L90y = [0]
    max_location_L24x = [0]
    max_location_L24y = [0]
    max_location_R90 = [0]
    max_location_R24 = [0]

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
    top_model_slices_by_view = []
    top_v_cropped_samples = []
    top_surface_vectors = []
    start_label = []

    selected_scaling_values = []
    selected_rotation_values = []
    selected_shear_values = []
    selected_overall_corr_values = []
    temp_scaling = [1, 1, 1]
    temp_rotations = [0, 0, 0]
    temp_shear = [0, 0, 0, 0, 0, 0]



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
        top_model_slices_by_view.append([0])
        top_v_cropped_samples.append([0])
        top_surface_vectors.append([0])
        start_label.append([0])

        # Read in sample and model image
    time_start = time.time()
    for starting_nr in range(0, starting_points):
        w = 0
        t = iv_inp[0][starting_nr]
        v_translations = (0, 0, 0)
        scaling_values = [iv_inp[1][starting_nr][0],
                     iv_inp[1][starting_nr][1],
                     iv_inp[1][starting_nr][2]]
        v_rotations = [iv_inp[2][starting_nr][0],
                       iv_inp[2][starting_nr][1],
                       iv_inp[2][starting_nr][2]]
        shear_values = [iv_inp[3][starting_nr][0],
                      iv_inp[3][starting_nr][1],
                      iv_inp[3][starting_nr][2],
                      iv_inp[3][starting_nr][3],
                      iv_inp[3][starting_nr][4],
                      iv_inp[3][starting_nr][5]]

        # performing crosscorrelation of the model with the 4 FIB SEM cuts
        model_binary_images, model_levelset_images, labels, M, A, topsurface = create_4cs(scaling_values, v_rotations,
                                                                                 shear_values, gs, uc,
                                                                                 miller, t, co)

        if co == 'L':
            (top_transf_matrices, top_v_singlecorr[starting_nr], top_v_max_location[starting_nr], top_v_model_matrices[starting_nr],
             top_v_scaling[starting_nr], top_v_rotations[starting_nr], top_v_shearing[starting_nr],
             cropped_samples, top_surface_vectors[starting_nr], overall_corr_values_raw, top_corr_arr[starting_nr],
             top_model_slices_by_view[starting_nr], top_v_threshold[starting_nr]) = perform_crosscorrelation_multi_left(
                samples_dir, scaling_square, sample_square_shift_single, model_binary_images, model_levelset_images, topsurface,
                sample_L24_crop, sample_L90_crop, sample_F24_crop, sample_F90_crop, M, t,
                top_F90_scores, top_F24_scores, top_L90_scores, top_L24_scores,
                top_F90, top_F24, top_L90, top_L24,
                max_location_F90x, max_location_F24x, max_location_L90x, max_location_L24x,
                max_location_F90y, max_location_F24y, max_location_L90y, max_location_L24y,
                model_matrices_F90, model_matrices_F24, model_matrices_L90, model_matrices_L24,
                model_slice_F90, model_slice_F24, model_slice_L90, model_slice_L24,
                top_v_scaling[starting_nr], top_v_rotations[starting_nr], top_v_shearing[starting_nr],
                top_v_max_location[starting_nr], top_surface_vectors[starting_nr], top_corr_arr[starting_nr],
                top_v_singlecorr[starting_nr], top_v_threshold[starting_nr], top_v_model_matrices[starting_nr],
                scaling_values, v_rotations, shear_values)
        elif co == 'R':
            (top_transf_matrices, per_slice_corr_scores, max_location, model_cross_sections, top_v_scaling, top_v_rotations,
             top_v_shearing,
             cropped_samples, top_surface_vectors, overall_corr_values_raw, top_model_slices_by_view) = perform_crosscorrelation_right(
                samples_dir, scaling_square, sample_square_shift_single, model_binary_images, model_levelset_images, topsurface,
                sample_F24_crop,
                sample_F90_crop, sample_R24_crop, sample_R90_crop, M, temp_scaling, temp_rotations, temp_shear)



        while (w < num_iterations):

            if w == 0:
                max_ind = 0

                del top_v_rotations[starting_nr][0]
                # top_v_translations[starting_nr].append(top_v_translations[starting_nr][1])
                del top_v_scaling[starting_nr][0]
                del top_v_shearing[starting_nr][0]
                del top_v_threshold[starting_nr][0]
                # top_v_surface[starting_nr].append(top_v_surface[starting_nr][1])
                del top_v_singlecorr[starting_nr][0]
                del top_corr_arr[starting_nr][0]
                del top_v_max_location[starting_nr][0]
                del top_v_model_matrices[starting_nr][0]
                del top_model_slices_by_view[starting_nr][0][0]
                del top_model_slices_by_view[starting_nr][1][0]
                del top_model_slices_by_view[starting_nr][2][0]
                del top_model_slices_by_view[starting_nr][3][0]
                del top_surface_vectors[starting_nr][0]

                temp_t = iv_inp[0][starting_nr]
                temp_translations = (0, 0, 0)
                temp_scaling = top_v_scaling[starting_nr]
                temp_rotations = top_v_rotations[starting_nr]
                temp_shear = top_v_shearing[starting_nr]

                temp_input_loop = [top_v_scaling[starting_nr][0][0], top_v_scaling[starting_nr][0][1],
                                   top_v_scaling[starting_nr][0][2],
                                   top_v_rotations[starting_nr][0][0], top_v_rotations[starting_nr][0][1],
                                   top_v_rotations[starting_nr][0][2],
                                   top_v_shearing[starting_nr][0][0], top_v_shearing[starting_nr][0][1],
                                   top_v_shearing[starting_nr][0][2],
                                   top_v_shearing[starting_nr][0][3], top_v_shearing[starting_nr][0][4],
                                   top_v_shearing[starting_nr][0][5]]
                candidate_param_sets, candidate_objective_values = wobble(num_iterations, wob_scale, wob_rot, wob_shear,
                                                                          temp_input_loop,
                                                                          co)

                selected_scaling_values = [float(candidate_param_sets[0]), float(candidate_param_sets[1]),
                                           float(candidate_param_sets[2])]
                selected_rotation_values = [float(candidate_param_sets[3]), float(candidate_param_sets[4]),
                                            float(candidate_param_sets[5])]
                selected_shear_values = [float(candidate_param_sets[6]), float(candidate_param_sets[7]),
                                         float(candidate_param_sets[8]), float(candidate_param_sets[9]),
                                         float(candidate_param_sets[10]), float(candidate_param_sets[11])]
                # print(f'new {candidate_objective_values, candidate_param_sets}')

                print(f'new {candidate_objective_values} vs {max(top_corr_arr[starting_nr])}')
                # print(f'array: {top_corr_arr[starting_nr]}')

                if candidate_objective_values >= max(top_corr_arr[starting_nr]):
                    # performing crosscorrelation of the model with the 4 FIB SEM cuts
                    model_binary_images, model_levelset_images, labels, M, A, topsurface = create_4cs(
                        selected_scaling_values, selected_rotation_values,
                        selected_shear_values, gs, uc,
                        miller, t, co)
                    # top_corr_arr[starting_nr].append(candidate_objective_values)
                    # print(f'new {candidate_objective_values, candidate_param_sets}')
                    # print(f'new {top_corr_arr[starting_nr]}')

                    if co == 'L':
                        (top_transf_matrices, top_v_singlecorr[starting_nr], top_v_max_location[starting_nr],
                         top_v_model_matrices[starting_nr],
                         top_v_scaling[starting_nr], top_v_rotations[starting_nr], top_v_shearing[starting_nr],
                         cropped_samples, top_surface_vectors[starting_nr], overall_corr_values_raw,
                         top_corr_arr[starting_nr],
                         top_model_slices_by_view[starting_nr],
                         top_v_threshold[starting_nr]) = perform_crosscorrelation_multi_left(
                            samples_dir, scaling_square, sample_square_shift_single, model_binary_images,
                            model_levelset_images, topsurface,
                            sample_L24_crop, sample_L90_crop, sample_F24_crop, sample_F90_crop, M, t,
                            top_F90_scores, top_F24_scores, top_L90_scores, top_L24_scores,
                            top_F90, top_F24, top_L90, top_L24,
                            max_location_F90x, max_location_F24x, max_location_L90x, max_location_L24x,
                            max_location_F90y, max_location_F24y, max_location_L90y, max_location_L24y,
                            model_matrices_F90, model_matrices_F24, model_matrices_L90, model_matrices_L24,
                            model_slice_F90, model_slice_F24, model_slice_L90, model_slice_L24,
                            top_v_scaling[starting_nr], top_v_rotations[starting_nr], top_v_shearing[starting_nr],
                            top_v_max_location[starting_nr], top_surface_vectors[starting_nr],
                            top_corr_arr[starting_nr],
                            top_v_singlecorr[starting_nr], top_v_threshold[starting_nr],
                            top_v_model_matrices[starting_nr],
                            selected_scaling_values, selected_rotation_values, selected_shear_values)
                    elif co == 'R':
                        (top_transf_matrices, per_slice_corr_scores, max_location, model_cross_sections, top_v_scaling,
                         top_v_rotations,
                         top_v_shearing,
                         cropped_samples, top_surface_vectors, overall_corr_values_raw,
                         top_model_slices_by_view) = perform_crosscorrelation_right(
                            samples_dir, scaling_square, sample_square_shift_single, model_binary_images,
                            model_levelset_images,
                            topsurface,
                            sample_L24_crop,
                            sample_L90_crop, sample_F24_crop, sample_F90_crop, M, temp_scaling, temp_rotations,
                            temp_shear)

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

                save_data_csv_multiple(top_v_scaling[starting_nr], top_v_rotations[starting_nr], top_v_shearing[starting_nr], top_surface_vectors[starting_nr],
                                       top_v_singlecorr[starting_nr], top_corr_arr[starting_nr],
                                       time_sofar_used,
                                       num_iterations, single_corner, sf, starting_nr)

                percentage = (float(starting_nr) * float(num_iterations) + float(round(w))) / (
                        float(starting_points) * float(num_iterations)) * 100
                # print(f'{round(percentage, 3)}% with {len(top_corr_arr[starting_nr])} of {best_combos} saved iterations')
                # print(f'loop {starting_nr + 1} of {starting_points}')

                # Convert elapsed time to days, hours, minutes, and seconds
                days = int(elapsed_time_seconds // (24 * 3600))
                hours = int((elapsed_time_seconds % (24 * 3600)) // 3600)
                minutes = int((elapsed_time_seconds % 3600) // 60)
                seconds = int(elapsed_time_seconds % 60)

                time_sofar_used = f'elapsed time: {days}d, {hours}h, {minutes}min, {seconds}s'
                print(time_sofar_used)
                estimated_seconds = float(elapsed_time_seconds) / (float(percentage) + 0.001) * 100
                # Convert elapsed time to days, hours, minutes, and seconds
                days = int(estimated_seconds // (24 * 3600))
                hours = int((estimated_seconds % (24 * 3600)) // 3600)
                minutes = int((estimated_seconds % 3600) // 60)
                seconds = int(estimated_seconds % 60)

                estimated_time = f'estimated time: {days}d, {hours}h, {minutes}min, {seconds}s'
                print(estimated_time)

                for k in range(0, best_combos-1):

                    top_v_rotations[starting_nr].append(top_v_rotations[starting_nr][0])
                    # top_v_translations[starting_nr].append(top_v_translations[starting_nr][1])
                    top_v_scaling[starting_nr].append(top_v_scaling[starting_nr][0])
                    top_v_shearing[starting_nr].append(top_v_shearing[starting_nr][0])
                    top_v_threshold[starting_nr].append(top_v_threshold[starting_nr][0])
                    # top_v_surface[starting_nr].append(top_v_surface[starting_nr][1])
                    top_v_singlecorr[starting_nr].append(top_v_singlecorr[starting_nr][0])
                    top_corr_arr[starting_nr].append(top_corr_arr[starting_nr][0])
                    top_v_max_location[starting_nr].append(top_v_max_location[starting_nr][0])
                    top_v_model_matrices[starting_nr].append(top_v_model_matrices[starting_nr][0])
                    top_surface_vectors[starting_nr].append(top_surface_vectors[starting_nr][0])

                    top_parameters = [0]
                    top_parameters_scores = [0]
                    top_translation = [0]
                    top_translation_scores = [0]
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

                    model_matrices_F90 = [0]
                    model_matrices_F24 = [0]
                    model_matrices_L90 = [0]
                    model_matrices_L24 = [0]
                    model_matrices_R90 = [0]
                    model_matrices_R24 = [0]

                    model_slice_F90 = [0]
                    model_slice_F24 = [0]
                    model_slice_L90 = [0]
                    model_slice_L24 = [0]
                    model_slice_R90 = [0]
                    model_slice_R24 = [0]

                    max_location_F90x = [0]
                    max_location_F90y = [0]
                    max_location_F24x = [0]
                    max_location_F24y = [0]
                    max_location_L90x = [0]
                    max_location_L90y = [0]
                    max_location_L24x = [0]
                    max_location_L24y = [0]
                    max_location_R90 = [0]
                    max_location_R24 = [0]

                w = w + 1

            else:
                max_ind = top_corr_arr[starting_nr].index(max(top_corr_arr[starting_nr]))



                temp_t = iv_inp[0][starting_nr]
                temp_translations = (0, 0, 0)
                temp_scaling = top_v_scaling[starting_nr][max_ind]
                temp_rotations = top_v_rotations[starting_nr][max_ind]
                temp_shear = top_v_shearing[starting_nr][max_ind]

                temp_input_loop = [top_v_scaling[starting_nr][max_ind][0], top_v_scaling[starting_nr][max_ind][1], top_v_scaling[starting_nr][max_ind][2],
                                   top_v_rotations[starting_nr][max_ind][0], top_v_rotations[starting_nr][max_ind][1], top_v_rotations[starting_nr][max_ind][2],
                                   top_v_shearing[starting_nr][max_ind][0], top_v_shearing[starting_nr][max_ind][1], top_v_shearing[starting_nr][max_ind][2],
                                   top_v_shearing[starting_nr][max_ind][3], top_v_shearing[starting_nr][max_ind][4], top_v_shearing[starting_nr][max_ind][5]]


                candidate_param_sets, candidate_objective_values = wobble(num_iterations, wob_scale, wob_rot, wob_shear, temp_input_loop,
                                              co)

                selected_scaling_values = [float(candidate_param_sets[0]), float(candidate_param_sets[1]), float(candidate_param_sets[2])]
                selected_rotation_values = [float(candidate_param_sets[3]), float(candidate_param_sets[4]), float(candidate_param_sets[5])]
                selected_shear_values = [float(candidate_param_sets[6]), float(candidate_param_sets[7]), float(candidate_param_sets[8]), float(candidate_param_sets[9]),
                     float(candidate_param_sets[10]), float(candidate_param_sets[11])]
                # print(f'new {candidate_objective_values, candidate_param_sets}')

                print(f'new {candidate_objective_values} vs {max(top_corr_arr[starting_nr])}')
                # print(f'array: {top_corr_arr[starting_nr]}')

                if candidate_objective_values >= max(top_corr_arr[starting_nr]):
                    # performing crosscorrelation of the model with the 4 FIB SEM cuts
                    model_binary_images, model_levelset_images, labels, M, A, topsurface = create_4cs(selected_scaling_values, selected_rotation_values,
                                                                                      selected_shear_values, gs, uc,
                                                                                      miller, t, co)
                    # top_corr_arr[starting_nr].append(candidate_objective_values)
                    # print(f'new {candidate_objective_values, candidate_param_sets}')
                    # print(f'new {top_corr_arr[starting_nr]}')

                    if co == 'L':
                        (top_transf_matrices, top_v_singlecorr[starting_nr], top_v_max_location[starting_nr],
                         top_v_model_matrices[starting_nr],
                         top_v_scaling[starting_nr], top_v_rotations[starting_nr], top_v_shearing[starting_nr],
                         cropped_samples, top_surface_vectors[starting_nr], overall_corr_values_raw,
                         top_corr_arr[starting_nr],
                         top_model_slices_by_view[starting_nr],
                         top_v_threshold[starting_nr]) = perform_crosscorrelation_multi_left(
                            samples_dir, scaling_square, sample_square_shift_single, model_binary_images,
                            model_levelset_images, topsurface,
                            sample_L24_crop, sample_L90_crop, sample_F24_crop, sample_F90_crop, M, t,
                            top_F90_scores, top_F24_scores, top_L90_scores, top_L24_scores,
                            top_F90, top_F24, top_L90, top_L24,
                            max_location_F90x, max_location_F24x, max_location_L90x, max_location_L24x,
                            max_location_F90y, max_location_F24y, max_location_L90y, max_location_L24y,
                            model_matrices_F90, model_matrices_F24, model_matrices_L90, model_matrices_L24,
                            model_slice_F90, model_slice_F24, model_slice_L90, model_slice_L24,
                            top_v_scaling[starting_nr], top_v_rotations[starting_nr], top_v_shearing[starting_nr],
                            top_v_max_location[starting_nr], top_surface_vectors[starting_nr], top_corr_arr[starting_nr],
                            top_v_singlecorr[starting_nr], top_v_threshold[starting_nr], top_v_model_matrices[starting_nr],
                            selected_scaling_values, selected_rotation_values, selected_shear_values)
                    elif co == 'R':
                        (top_transf_matrices, per_slice_corr_scores, max_location, model_cross_sections, top_v_scaling, top_v_rotations,
                         top_v_shearing,
                         cropped_samples, top_surface_vectors, overall_corr_values_raw, top_model_slices_by_view) = perform_crosscorrelation_right(
                            samples_dir, scaling_square, sample_square_shift_single, model_binary_images, model_levelset_images,
                            topsurface,
                            sample_L24_crop,
                            sample_L90_crop, sample_F24_crop, sample_F90_crop, M, temp_scaling, temp_rotations,
                            temp_shear)


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

                save_data_csv_multiple(top_v_scaling[starting_nr], top_v_rotations[starting_nr], top_v_shearing[starting_nr], top_surface_vectors[starting_nr],
                                       top_v_singlecorr[starting_nr], top_corr_arr[starting_nr],
                                       time_sofar_used,
                                       num_iterations, single_corner, sf, starting_nr)


                percentage = (float(starting_nr) * float(num_iterations) + float(round(w))) / (
                        float(starting_points) * float(num_iterations)) * 100
                print(
                    f'{round(percentage, 3)}% with {len(top_corr_arr[starting_nr])} of {best_combos} saved iterations')
                print(f'loop {starting_nr + 1} of {starting_points}')

                # Convert elapsed time to days, hours, minutes, and seconds
                days = int(elapsed_time_seconds // (24 * 3600))
                hours = int((elapsed_time_seconds % (24 * 3600)) // 3600)
                minutes = int((elapsed_time_seconds % 3600) // 60)
                seconds = int(elapsed_time_seconds % 60)

                time_sofar_used = f'elapsed time: {days}d, {hours}h, {minutes}min, {seconds}s'
                print(time_sofar_used)
                estimated_seconds = float(elapsed_time_seconds) / (float(percentage) + 0.001) * 100
                # Convert elapsed time to days, hours, minutes, and seconds
                days = int(estimated_seconds // (24 * 3600))
                hours = int((estimated_seconds % (24 * 3600)) // 3600)
                minutes = int((estimated_seconds % 3600) // 60)
                seconds = int(estimated_seconds % 60)

                estimated_time = f'estimated time: {days}d, {hours}h, {minutes}min, {seconds}s'
                print(estimated_time)

                w = w + 1

        # color coding
        max_color = 1
        min_color = 0
        color_palette = ("viridis")
        transparency_overlay = 0.4
        transparency_sample = 0.1

        top_idx = (np.argsort(top_corr_arr[starting_nr])[-1], np.argsort(top_corr_arr[starting_nr])[-2],
                   np.argsort(top_corr_arr[starting_nr])[-3], np.argsort(top_corr_arr[starting_nr])[-4])
        for i in range(0, print_top):
            # print('here')
            # print(top_v_model_matrices[starting_nr])
            # print(len(top_v_model_matrices[starting_nr]))
            # print(len(top_corr_arr[starting_nr]))
            # print(top_v_model_matrices[starting_nr][0])
            # print(labels)
            # print(i)
            plot_4cs_multi(top_v_model_matrices[starting_nr], labels, sf, top_idx[i], starting_nr)

        if co == 'L':

            print(top_v_model_matrices[starting_nr][0])
            print(top_v_model_matrices[starting_nr][0][0])
            print_top_correlations_left_multi(max_color, min_color, color_palette, top_model_slices_by_view[starting_nr], top_v_model_matrices[starting_nr], cropped_samples,
                                        top_v_scaling[starting_nr], top_v_rotations[starting_nr], top_v_shearing[starting_nr], top_surface_vectors[starting_nr],
                                              top_corr_arr[starting_nr], top_v_singlecorr[starting_nr], top_v_max_location[starting_nr],
                                              starting_nr, transparency_overlay, transparency_sample, sf, sd)
        if co == 'R':
            print_top_correlations_right(max_color, min_color, color_palette, top_v_singlecorr[starting_nr],
                                         top_corr_arr[starting_nr],
                                         top_v_model_matrices[starting_nr], top_v_cropped_samples[starting_nr],
                                         top_v_scaling[starting_nr], top_v_rotations[starting_nr],
                                         top_v_shearing[starting_nr], top_v_surface[starting_nr],
                                         top_v_threshold[starting_nr],
                                         transparency_overlay, transparency_sample, top_v_max_location[starting_nr],
                                         sf, co, starting_nr)

    ##############################################################################

##############################################################################
##############################################################################
##############################################################################
##############################################################################

# directory for saving the files
saving_files = ["C:/Users/iselire/OneDrive/studium/unifr/solvent_evaporation_annealing/evaluation_crosssections/tests_torch/01",
                "C:/Users/iselire/OneDrive/studium/unifr/solvent_evaporation_annealing/evaluation_crosssections/tests_torch/02",
                "C:/Users/iselire/OneDrive/studium/unifr/solvent_evaporation_annealing/evaluation_crosssections/tests_torch/03"]
directory = saving_files[0]
# samples should be named sample_F24 etc. or F24 etc.
samples_dir = "C:/Users/iselire/OneDrive/studium/unifr/solvent_evaporation_annealing/evaluation_crosssections/data/rot_test_L"
for i in range(0, len(saving_files)):
    dir_exists(saving_files[i])
# where initial values are stored
initial_values = "C:/Users/IseliRe/OneDrive/studium/unifr/solvent_evaporation_annealing/evaluation_crosssections/tests_torch/initial_values/20260531_conv2d_initial_values_rotation.csv"

##############################################################################

# Fill fraction, max value of levelset is 1.4990768364186535
ff = 0.40
if ff >= 0.5:
    t = 0 + (math.sin((ff - 0.5) * math.pi / 2)) * 1.4990768364186535
elif ff <= 0.5:
    t = 0 - (math.sin((ff) * math.pi / 2)) * 1.4990768364186535

elif ff > 1 or ff < 0:
    print('The fill fraction needs to be between 0 and 1')
    sys.exit()

t = -0.05  # threshold

# Defining the original basis vectors (example values)
original_basis_vectors = np.array([[1, 0, 0],
                                   [0, 1, 0],
                                   [0, 0, 1]])

miller = (1, 0, 0)
unitcell = 40
grid_size = (100, 100, 100)
norm_translation = 100

iv_arrays = read_csv(initial_values)
time_start = time.time()
iv_input = iv_arrays

best_combos = 8 # minimum 4
best_combos_single_run = 4
cs_per_volume = 4 # should be 4
print_top = 4 # are the top number of binary images generated
iteration_loops = 'multiple' # 'single', 'multiple'
multiple_initials = 'initial_values' #initial_values, random
# if multiple, this is the number of randomly generated starting points
if multiple_initials == 'random':
    starting_points = 10
elif multiple_initials == 'initial_values':
    starting_points = len(iv_input[0])
# x, y - shift per image taken from the sample
# L24, L90, F24, F90 or F24, F90, R24, R90
sample_square_shift = ([(0, 0), (0, 0), (0, 0), (0, 0)],
                        [(0, 0), (0, 0), (0, 0), (0, 0)])
sample_square_shift_single = [(0, 0), (0, 0), (0, 0), (0, 0)]
# size of the square taken from the sample
scaling_square = 160

##############################################################################

# creating the settings for the cross correlation

print(f'the path is ...{samples_dir[:]}')
#val = input("Should we perform a crosscorrelation with translation? (1 = yes, 0 = no) : ")
val = '1'
print(val + ' --> 1 means we perform a cross correlation')

#corner = input("left (L) or right (R) corner of the sample? ")
corner = ['L', 'R']
single_corner = corner[0]
print(single_corner + ' is the corner we observe')

# Define the number of iterations and the wobbling factor
num_iterations = 10
wob_scale = 0
wob_rot = 1
wob_shear = 0

print('These are the initial values imported for the evalutation: ')
print('scaling: ' + str(iv_input[0]))
print('rotation: ' + str(iv_input[1]))
print('shearing: ' + str(iv_input[2]))
print('together with the wobble parameters: ')
print('scaling = ' + str(wob_scale) + ', rotation = ' + str(wob_rot) + ', shear = ' + str(wob_shear))


##############################################################################
# loop for the cross correlation

if iteration_loops == 'single':

    ##############################################################################

    # Read in sample image

    samples = read_images_from_folder(samples_dir)

    transform = transforms.ToTensor()
    image_F90 = Image.open(samples[0][1])
    sample_F90_c = image_F90.crop((sample_square_shift_single[2][0], sample_square_shift_single[2][1], (scaling_square + sample_square_shift_single[2][0]),
                                   (scaling_square + sample_square_shift_single[2][1])))
    sample_F90_crop = transform(sample_F90_c)

    image_F24 = Image.open(samples[0][0])
    sample_F24_c = image_F24.crop((sample_square_shift_single[3][0], sample_square_shift_single[3][1], (scaling_square + sample_square_shift_single[3][0]),
                                   (scaling_square + sample_square_shift_single[3][1])))
    sample_F24_crop = transform(sample_F24_c)

    image_L90 = Image.open(samples[0][3])
    sample_L90_c = image_L90.crop((sample_square_shift_single[0][0], sample_square_shift_single[0][1], (scaling_square + sample_square_shift_single[0][0]),
                                   (scaling_square + sample_square_shift_single[0][1])))
    sample_L90_crop = transform(sample_L90_c)

    image_L24 = Image.open(samples[0][2])
    sample_L24_c = image_L24.crop((sample_square_shift_single[1][0], sample_square_shift_single[1][1], (scaling_square + sample_square_shift_single[1][0]),
                                   (scaling_square + sample_square_shift_single[1][1])))
    sample_L24_crop = transform(sample_L24_c)

    ##############################################################################
    # empty arrays for the crosscorrelation
    top_corr_arr = [0]
    top_parameters = [0]
    top_parameters_scores = [0]
    top_translation = [0]
    top_translation_scores = [0]
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

    model_matrices_F90 = [0]
    model_matrices_F24 = [0]
    model_matrices_L90 = [0]
    model_matrices_L24 = [0]
    model_matrices_R90 = [0]
    model_matrices_R24 = [0]

    model_slice_F90 = [0]
    model_slice_F24 = [0]
    model_slice_L90 = [0]
    model_slice_L24 = [0]
    model_slice_R90 = [0]
    model_slice_R24 = [0]

    max_location_F90x = [0]
    max_location_F90y = [0]
    max_location_F24x = [0]
    max_location_F24y = [0]
    max_location_L90x = [0]
    max_location_L90y = [0]
    max_location_L24x = [0]
    max_location_L24y = [0]
    max_location_R90 = [0]
    max_location_R24 = [0]

    top_v_rotations = [0]
    top_v_translations = [0]
    top_v_scaling = [0]
    top_v_shearing = [0]

    selected_scaling_values = []
    selected_rotation_values = []
    selected_shear_values = []
    selected_overall_corr_values = []
    temp_scaling = [0, 0, 0]
    temp_rotations = [0, 0, 0]
    temp_shear = [0, 0, 0, 0, 0, 0]

    top_surface_vectors = [0]

    ##############################################################################

    for i in range(0, len(iv_input[0])):
        iv_input_loop = (iv_input[0][0 + i][0], iv_input[0][0 + i][1], iv_input[0][0 + i][2], iv_input[1][0 + i][0], iv_input[1][0 + i][1], iv_input[1][0 + i][2],
        iv_input[2][0 + i][0], iv_input[2][0 + i][1], iv_input[2][0 + i][2], iv_input[2][0 + i][3], iv_input[2][0 + i][4], iv_input[2][0 + i][5])

        candidate_param_sets, candidate_objective_values = wobble(num_iterations, wob_scale, wob_rot, wob_shear, iv_input_loop, single_corner)

        selected_scaling_values.append([float(candidate_param_sets[0]), float(candidate_param_sets[1]), float(candidate_param_sets[2])])
        selected_rotation_values.append([float(candidate_param_sets[3]), float(candidate_param_sets[4]), float(candidate_param_sets[5])])
        selected_shear_values.append([float(candidate_param_sets[6]), float(candidate_param_sets[7]), float(candidate_param_sets[8]), float(candidate_param_sets[9]), float(candidate_param_sets[10]), float(candidate_param_sets[11])])
        selected_overall_corr_values.append(candidate_objective_values)
        # print(f'new {candidate_objective_values, candidate_param_sets}')
        candidate_param_sets, candidate_objective_values = [], []
        # print(f'new {selected_overall_corr_values}')

    ###########################################################################################

    # plotting


    # color coding
    max_color = 1
    min_color = 0
    color_palette = ("viridis")
    transparency_overlay = 0.995

    for i in range(0, 4):
        # performing crosscorrelation of the model with the 4 FIB SEM cuts
        temp_scaling = selected_scaling_values[i]
        temp_rotations = selected_rotation_values[i]
        temp_shear = selected_shear_values[i]
        model_binary_images, model_levelset_images, labels, M, A, topsurface = create_4cs(temp_scaling, temp_rotations, temp_shear,
                                                                            grid_size,
                                                                            unitcell, miller, t, single_corner)
        (top_transf_matrices, per_slice_corr_scores, max_location, model_cross_sections, top_v_scaling, top_v_rotations, top_v_shearing,
         cropped_samples, top_surface_vectors, overall_corr_values_raw, top_model_slices_by_view) = perform_crosscorrelation_left(
            samples_dir, scaling_square, sample_square_shift_single, model_binary_images, model_levelset_images, topsurface, sample_L24_crop,
            sample_L90_crop, sample_F24_crop, sample_F90_crop, M, temp_scaling, temp_rotations, temp_shear)

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

    save_data_csv(selected_scaling_values, selected_rotation_values, selected_shear_values, top_surface_vectors, per_slice_corr_scores, selected_overall_corr_values, time_sofar_used,
                  num_iterations, single_corner)
    print(f'corr is: {top_corr_arr}')
    print(f'{time_sofar_used}')

    for i in range(0, print_top):
        # print('here')
        # print(len(model_cross_sections[i]))
        # print(model_cross_sections[i])
        # print(model_cross_sections[i][0])
        # print(len(model_cross_sections[i][1]))
        # print(model_cross_sections[i][1])
        # print(len(model_cross_sections[i][1][0]))
        # print(model_cross_sections[i][1][0])
        plot_4cs(model_cross_sections[i], top_model_slices_by_view[i], labels, directory, i, 0)

    if single_corner == 'L':
        print_top_correlations_left(max_color, min_color, color_palette, per_slice_corr_scores, model_cross_sections, cropped_samples, selected_scaling_values, selected_rotation_values, selected_shear_values, top_surface_vectors, transparency_overlay)

##############################################################################

elif iteration_loops == 'multiple':

    ##############################################################################

    # Read in sample image

    samples = read_images_from_folder(samples_dir)

    transform = transforms.ToTensor()
    image_F90 = Image.open(samples[0][1])
    sample_F90_c = image_F90.crop((sample_square_shift[0][2][0], sample_square_shift[0][2][1],
                                   (scaling_square + sample_square_shift[0][2][0]),
                                   (scaling_square + sample_square_shift[0][2][1])))
    sample_F90_crop = transform(sample_F90_c)

    image_F24 = Image.open(samples[0][0])
    sample_F24_c = image_F24.crop((sample_square_shift[0][3][0], sample_square_shift[0][3][1],
                                   (scaling_square + sample_square_shift[0][3][0]),
                                   (scaling_square + sample_square_shift[0][3][1])))
    sample_F24_crop = transform(sample_F24_c)

    image_L90 = Image.open(samples[0][3])
    sample_L90_c = image_L90.crop((sample_square_shift[0][0][0], sample_square_shift[0][0][1],
                                   (scaling_square + sample_square_shift[0][0][0]),
                                   (scaling_square + sample_square_shift[0][0][1])))
    sample_L90_crop = transform(sample_L90_c)

    image_L24 = Image.open(samples[0][2])
    sample_L24_c = image_L24.crop((sample_square_shift[0][1][0], sample_square_shift[0][1][1],
                                   (scaling_square + sample_square_shift[0][1][0]),
                                   (scaling_square + sample_square_shift[0][1][1])))
    sample_L24_crop = transform(sample_L24_c)

    ##############################################################################

    threshold_start = []
    scaling_start = []
    rotation_start = []
    shearing_start = []
    topsurface_start = []
    correlation_start = []

    if multiple_initials == 'random':
        for i in range(0, starting_points):
            threshold_start.append(t)
            # sc_x = scaling_start_value[0] + pow(-1, random.randint(10))*float(random.randint(100))/100 * wobble_sca
            # sc_y = scaling_start_value[1] + pow(-1, random.randint(10))*float(random.randint(100))/100 * wobble_sca
            # sc_z = 1/(sc_x * sc_y)
            # sc_all = (sc_x, sc_y, sc_z)
            sc_all = [1, 1, 1]
            scaling_start.append(sc_all)
            rotation_start.append(
                [float(random.randint(3600)) / 10, float(random.randint(3600)) / 10, float(random.randint(3600)) / 10])
            # shearing_start.append([shearing_start_value[0] + pow(-1, random.randint(10))*float(random.randint(100))/100 * wobble_she,
            #                       shearing_start_value[0] + pow(-1, random.randint(10))*float(random.randint(100))/100 * wobble_she,
            #                       shearing_start_value[0] + pow(-1, random.randint(10))*float(random.randint(100))/100 * wobble_she,
            #                       shearing_start_value[0] + pow(-1, random.randint(10))*float(random.randint(100))/100 * wobble_she,
            #                       shearing_start_value[0] + pow(-1, random.randint(10))*float(random.randint(100))/100 * wobble_she,
            #                       shearing_start_value[0] + pow(-1, random.randint(10))*float(random.randint(100))/100 * wobble_she])
            shearing_start.append([0, 0, 0, 0, 0, 0])
            topsurface_start.append([0, 0, 0])
            correlation_start.append(0)

    elif multiple_initials == 'initial_values':
        for i in range(0, len(iv_input[0])):
            threshold_start.append(t)
            scaling_start.append([iv_input[0][0 + i][0], iv_input[0][0 + i][1], iv_input[0][0 + i][2]])
            rotation_start.append([iv_input[1][0 + i][0], iv_input[1][0 + i][1], iv_input[1][0 + i][2]])
            shearing_start.append([iv_input[2][0 + i][0], iv_input[2][0 + i][1], iv_input[2][0 + i][2],
                                   iv_input[2][0 + i][3], iv_input[2][0 + i][4], iv_input[2][0 + i][5]])
            topsurface_start.append([0, 0, 0])
            correlation_start.append(0)

    iv_input = []
    iv_input.append(threshold_start)
    iv_input.append(scaling_start)
    iv_input.append(rotation_start)
    iv_input.append(shearing_start)
    iv_input.append(topsurface_start)
    iv_input.append(correlation_start)
    ##############################################################################

    if __name__ == '__main__':

        start = time.time()
        p1 = mp.Process(target=big_loop, args=(
            saving_files[0], samples_dir[0], grid_size, unitcell, miller, corner[0],
            sample_square_shift[0], iv_input))
        p2 = mp.Process(target=big_loop, args=(
            saving_files[1], samples_dir[0], grid_size, unitcell, miller, corner[0],
            sample_square_shift[0], iv_input))
        # p3 = mp.Process(target=big_loop, args=(
        #     saving_files[3], samples_dir[0], grid_size, unitcell, miller, corner[0],
        #     sample_square_shift[0], iv_input))


        p1.start()
        p2.start()
        # p3.start()

        p1.join()
        p2.join()
        # p3.join()

        end = time.time()

        print(f'1st execution: {end-start}')