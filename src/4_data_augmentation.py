#!/usr/bin/env python

import numpy as np
import random
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--working_directory", help="path to the working directory", default= "/data/ahsoka/eocp/forestpulse/01_data/02_processed_data/tree_mask_LCC")
args = parser.parse_args()

# 1. Random Observation Selection (ROS)
def ROS(x_array, y_array, augmentation_num ,sparsity_prop):
    ROS_array_x = []
    ROS_array_y = []
    for rep in range(augmentation_num):
        for sample_num in range(x_array.shape[0]):
            sample = x_array[sample_num,:,:].copy()
            blue_band = sample[:,0] #!
            sparsity_percentage = random.randint(0, sparsity_prop)
            valid_indices = np.where(blue_band > 0)[0]

            # Zufällig Indizes auswählen, die maskiert werden
            indices_to_mask = np.random.choice(valid_indices, 
                                            size= int(len(valid_indices) * (sparsity_percentage / 100)), 
                                            replace=False)
            # Setze die ausgewählten Werte auf 0
            sample[indices_to_mask,:] = 0

            #add modified sample to dataset
            ROS_array_x.append(sample)
            ROS_array_y.append(y_array[sample_num])
            #ROS_array_y.append(y_array[sample_num,:])

    ROS_array_x = np.stack(ROS_array_x)
    ROS_array_y = np.stack(ROS_array_y)
    return ROS_array_x, ROS_array_y

# 2. Random Day Shifting (RDS)
def RDS(x_array, y_array):
    RDS_array_x = []
    RDS_array_y = []
    for sample_num in range(x_array.shape[0]):
    #for sample_num in range(5):
        sample = x_array[sample_num,:,:].copy()
        #print(sample[0,:])
        blue_band = sample[:,0]
        valid_indices = np.where(blue_band > 0)[0]
        valid_data = sample[valid_indices,:]
        sample[valid_indices,:] = 0
        #print(valid_indices)
        #random shifting
        for i in range(len(valid_indices)):
            random_shift = random.randint(-7, 7)
            # avoid time series edge cases
            if (valid_indices[i] + random_shift < 0) or (valid_indices[i] + random_shift) > 365-1 :
                valid_indices[i] = valid_indices[i]
            else:
                valid_indices[i] = valid_indices[i] + random_shift 
        # set data to new shifted days
        sample[valid_indices,:] = valid_data
        # add sample and correspondung response array to resulting array
        RDS_array_x.append(sample)
        RDS_array_y.append(y_array[sample_num])
        #RDS_array_y.append(y_array[sample_num,:])

    # stack x and y array
    RDS_array_x = np.stack(RDS_array_x)
    RDS_array_y = np.stack(RDS_array_y)
    return RDS_array_x, RDS_array_y

if __name__ == '__main__':
    # load training data (x and y)
    x_array = np.load(os.path.join(args.working_directory, '3_train_test_data', 'train_x.npy'))
    y_array = np.load(os.path.join(args.working_directory, '3_train_test_data', 'train_y.npy'))
    print(x_array.shape, y_array.shape)

    # augemtation settings
    sparsity = 90 # Pham et al. 2024 states that a random augmentation of up to 90 % is good for training
    augmentation_num = 3

    # 1st step: Random observation selection  
    x_ROS, y_ROS = ROS(x_array,y_array,augmentation_num,sparsity)
    print(x_ROS.shape, y_ROS.shape)

    # 2nd step: Random Day Shifting (of the Random selected samples)
    x_augmented, y_augmented = RDS(x_ROS, y_ROS)
    print(x_augmented.shape, y_augmented.shape)

    # 3rd: Merge of training and augmented array
    x_TE_augmented = np.concatenate([x_array, x_augmented], axis=0)
    y_TE_augmented = np.concatenate([y_array, y_augmented], axis=0)
    print(x_TE_augmented.shape)
    print(y_TE_augmented.shape)

    if not os.path.exists(os.path.join(args.working_directory, '4_augmented_data')):
        os.makedirs(os.path.join(args.working_directory, '4_augmented_data'))
    np.save(os.path.join(args.working_directory, '4_augmented_data', 'TE_augmented_x.npy'), arr=x_TE_augmented.astype('uint16'))
    np.save(os.path.join(args.working_directory, '4_augmented_data', 'TE_augmented_y.npy'), arr=y_TE_augmented.astype('uint8'))