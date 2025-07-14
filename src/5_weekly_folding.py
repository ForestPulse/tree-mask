import numpy as np
import random
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--working_directory", help="path to the working directory", default= "/data/ahsoka/eocp/forestpulse/01_data/02_processed_data/tree_mask_LCC")
args = parser.parse_args()

def fold_array(x_array):
    # remove last day, for possible reshaping
    data = x_array[:, :364, :] 
    # Reshape to (:, 52, 7, 10)
    data_weeks = data.reshape(data.shape[0], 52, 7, data.shape[2])
    # Mask values != 0
    mask_nonzero = data_weeks != 0
    # Sum for every week
    sum_values = np.where(mask_nonzero, data_weeks, 0).sum(axis=2)
    # count values ≠ 0 per week
    count_nonzero = mask_nonzero.sum(axis=2)
    # calculate mean for each week
    # where all values are 0, result will be zero
    # => prevent Division with 0 in np.where
    mean_weeks = np.where(count_nonzero > 0, sum_values / count_nonzero, 0)
    #print(mean_weeks.shape)
    print(mean_weeks[0, :, 0]) 
    #print(data_weeks[0, :, :, 0]) 
    print(mean_weeks[1000, :, 0])
    #print(data_weeks[1, :, :, 0]) 
    return np.round(mean_weeks).astype(np.uint16)


if __name__ == '__main__':
    train_x = np.load(os.path.join(args.working_directory, '4_augmented_data', 'TE_augmented_x.npy'))
    test_x = np.load(os.path.join(args.working_directory, '3_train_test_data', 'test_x.npy'))
    folded_train_x = fold_array(train_x)
    folded_test_x = fold_array(test_x)
    print('Folded train data shape: ', folded_train_x.shape)
    print('Folded test data shape: ', folded_test_x.shape)

    if not os.path.exists(os.path.join(args.working_directory, '5_folded_data')):
        os.makedirs(os.path.join(args.working_directory, '5_folded_data'))
    np.save(os.path.join(args.working_directory, '5_folded_data', 'train_augmented_folded_x.npy'), arr=folded_train_x)
    np.save(os.path.join(args.working_directory, '5_folded_data', 'test_augmented_folded_x.npy'), arr=folded_test_x)