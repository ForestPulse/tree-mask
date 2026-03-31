#!/usr/bin/env python

import numpy as np
import os
import argparse
from scipy.interpolate import BSpline
from tqdm import tqdm



parser = argparse.ArgumentParser()
parser.add_argument("--working_directory", 
                    help="path to the working directory", 
                    default= "/data/ahsoka/eocp/forestpulse/01_data/02_processed_data/tree_mask_noFORCE")
parser.add_argument("--year", help="year of the training points", default= '2021')
args = parser.parse_args()

def prepare_data():
    #===========================================
    #======= Step 1: Convert to 2D array =======
    #===========================================
    arrays_x = []
    arrays_y = []
    expected_nrows = 220
    for file in sorted(os.listdir(os.path.join(args.working_directory, '1_samples', f'samples_{args.year}'))):
        if file.startswith('x_') and file.endswith('.csv'):
            try:
                arr_x = np.loadtxt(os.path.join(args.working_directory, '1_samples',f'samples_{args.year}',file))
                print(f"Loaded {file} with shape {arr_x.shape}")
                if arr_x.shape[1] != expected_nrows:
                    print(f"Warning: File {file} has {arr_x.shape[1]} rows, expected {expected_nrows}. Skipping this file.")
                    continue
                else:
                    arrays_x.append(arr_x)
            except Exception as e:
                print(f"Skip {file}: Fehler beim Laden ({e})")
        if file.startswith('y_') and file.endswith('.csv'):
            try:
                arr_y = np.loadtxt(os.path.join(args.working_directory, '1_samples',f'samples_{args.year}',file))
                print(f"Loaded {file} with shape {arr_y.shape}")
                if arr_y.shape[0] ==0:
                    print(f"Warning: File {file} has {arr_y.shape[0]} rows. Skipping this file.")
                    continue
                else:
                    arrays_y.append(arr_y)
            except Exception as e:
                print(f"Skip {file}: Fehler beim Laden ({e})")
    
    # concatenate the arrays
    samples = np.concatenate(arrays_x, axis=0)
    response = np.concatenate(arrays_y, axis=0)
    print('sample data shape: ', samples.shape)

    # remove nodata samples
    valid_indices = samples[:, 0] != -9999
    samples = samples[valid_indices]
    response = response[valid_indices]
    valid_indices = samples[:, 0] != 0
    samples = samples[valid_indices]
    response = response[valid_indices]
    print('sample data shape after removing nodata: ', samples.shape)
    print('response data shape after removing nodata: ', response.shape)

    # Reshape der Daten in (num_rows, 10, 220)
    #reshaped_2D_array = samples.reshape(samples.shape[0], 10, 22).swapaxes(1, 2)  
    #print('2D array shape: ', reshaped_2D_array.shape)  

    #===========================================
    #====== Step 2: Train/Test data split ======
    #===========================================
    # Leere Listen initialisieren
    train_x_list = []
    train_y_list = []
    train_coords_list = []

    test_x_list = []
    test_y_list = []
    test_coords_list = []

    val_x_list = []
    val_y_list = []
    val_coords_list = []
    # mix every class individually
    for LC_class in np.unique(response):
        print('Processing LC class: ', LC_class, ' length: ', len(response[response == LC_class]))

        #LC_array_samples = samples[response == LC_class, :, :]
        LC_array_samples = samples[response == LC_class, :]
        LC_array_response = response[response == LC_class]
        num_samples = LC_array_samples.shape[0]
        #train_ratio = 0.6
        #test_ratio = 0.8
        train_ratio = 0.7
        if LC_class == 4 or LC_class == 7 or LC_class == 8:
            #train_ratio = 0.33
            #test_ratio = 0.67
            train_ratio = 0.5
        # make a random index array
        random_indices = np.random.permutation(num_samples)
        #calculate training/test array size
        train_size = int(num_samples * train_ratio)
        #test_size = int(num_samples * test_ratio)
        # allocation of train and test indices
        train_indices = random_indices[:train_size]
        test_indices = random_indices[train_size:]
        #test_indices = random_indices[train_size:test_size]
        #val_indices = random_indices[test_size:]

        train_class_x = LC_array_samples[train_indices]
        train_class_y = LC_array_response[train_indices]

        test_class_x = LC_array_samples[test_indices]
        test_class_y = LC_array_response[test_indices]

        #val_class_x = LC_array_samples[val_indices]
       # val_class_y = LC_array_response[val_indices]

        # append to lists
        train_x_list.append(train_class_x)
        train_y_list.append(train_class_y)

        test_x_list.append(test_class_x)
        test_y_list.append(test_class_y)

        #val_x_list.append(val_class_x)
        #val_y_list.append(val_class_y)

    # concatenate everything
    train_x = np.concatenate(train_x_list, axis=0)
    train_y = np.concatenate(train_y_list, axis=0)

    test_x = np.concatenate(test_x_list, axis=0)
    test_y = np.concatenate(test_y_list, axis=0)

    #val_x = np.concatenate(val_x_list, axis=0)
    #val_y = np.concatenate(val_y_list, axis=0)

    #===========================================
    #==== Step 3: Store the joined Arrays  =====
    #===========================================
    if not os.path.exists(os.path.join(args.working_directory, '2_train_test')):
        os.makedirs(os.path.join(args.working_directory, '2_train_test'))
    # train arrays
    np.save( os.path.join(args.working_directory, '2_train_test', 'train_x.npy') , arr=train_x.astype('uint16'))
    print('train data samples: ', train_x.shape)
    np.save( os.path.join(args.working_directory, '2_train_test', 'train_y.npy') , arr=train_y.astype('uint8'))
    print('train data response: ', train_y.shape)
    # test arrays
    np.save( os.path.join(args.working_directory, '2_train_test', 'test_x.npy') , arr=test_x.astype('uint16'))
    print('test data samples: ', test_x.shape)
    np.save( os.path.join(args.working_directory, '2_train_test', 'test_y.npy') , arr=test_y.astype('uint8'))
    print('test data response: ', test_y.shape)
    # val arrays
    #np.save( os.path.join(args.working_directory, '2_train_test_val', 'val_x.npy') , arr=val_x.astype('uint16'))
    #print('val data samples: ', val_x.shape)
    #np.save( os.path.join(args.working_directory, '2_train_test_val', 'val_y.npy') , arr=val_y.astype('uint8'))
    #print('val data response: ', val_y.shape)

if __name__ == '__main__':
    prepare_data()