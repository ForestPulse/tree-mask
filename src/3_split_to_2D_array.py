import numpy as np
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--working_directory", help="path to the working directory", default= "/data/ahsoka/eocp/forestpulse/INTERNAL/tree_mask")
args = parser.parse_args()

def prepare_data():
    #===========================================
    #======= Step 1: Convert to 2D array =======
    #===========================================

    FORCE_sample = np.loadtxt(os.path.join(args.working_directory, '2_FORCE_samples','sample.txt'))
    # NoData to zero (in Pham et al. 2024)
    FORCE_sample[FORCE_sample < 0] = 0
    # Reshape der Daten in (num_rows, 10, 365)
    reshaped_2D_array = FORCE_sample.reshape(FORCE_sample.shape[0], 10, 365).swapaxes(1, 2)
    # check if the reshape worked
    '''
    print(reshaped_2D_array[4,0:20,0])
    print(reshaped_2D_array[4,0:20,1])
    print(reshaped_2D_array[4,0:20,2])
    print(reshaped_2D_array[4,0:20,3])
    print('--------------------------------------------------------')
    print(reshaped_2D_array[3,20:30,0])
    print(reshaped_2D_array[3,20:30,1])
    print(reshaped_2D_array[3,20:30,2])
    print(reshaped_2D_array[3,20:30,3])
    '''
    # 2nd: response array
    FORCE_response = np.loadtxt(os.path.join(args.working_directory, '2_FORCE_samples','response.txt'))
    FORCE_coords = np.loadtxt(os.path.join(args.working_directory, '2_FORCE_samples','coord.txt'))

    #===========================================
    #====== Step 2: Train/Test data split ======
    #===========================================

    num_samples = FORCE_sample.shape[0]
    train_ratio = 0.7
    # make a random index array
    random_indices = np.random.permutation(num_samples)
    #calculate training/test array size
    train_size = int(num_samples * train_ratio)
    # allocation of train and test indices
    train_indices = random_indices[:train_size]
    test_indices = random_indices[train_size:]

    train_x = reshaped_2D_array[train_indices]
    train_y = FORCE_response[train_indices]
    train_coords = FORCE_coords[train_indices]
    print(train_x.shape, train_y.shape ,train_coords.shape)

    test_x = reshaped_2D_array[test_indices]
    test_y = FORCE_response[test_indices]
    test_coords = FORCE_coords[test_indices]
    print(test_x.shape, test_y.shape ,test_coords.shape)

    #===========================================
    #==== Step 3: Store the joined Arrays  =====
    #===========================================
    if not os.path.exists(os.path.join(args.working_directory, '3_train_test_data')):
        os.makedirs(os.path.join(args.working_directory, '3_train_test_data'))
    # train arrays
    np.save( os.path.join(args.working_directory, '3_train_test_data', 'train_x.npy') , arr=train_x.astype('uint16'))
    print('train data samples: ', train_x.shape)
    np.save( os.path.join(args.working_directory, '3_train_test_data', 'train_y.npy') , arr=train_y.astype('uint8'))
    print('train data response: ', train_y.shape)
    np.save( os.path.join(args.working_directory, '3_train_test_data', 'train_coords.npy') , arr=train_coords)
    print('train data coords: ', train_coords.shape)
    # test arrays
    np.save( os.path.join(args.working_directory, '3_train_test_data', 'test_x.npy') , arr=test_x.astype('uint16'))
    print('test data samples: ', test_x.shape)
    np.save( os.path.join(args.working_directory, '3_train_test_data', 'test_y.npy') , arr=test_y.astype('uint8'))
    print('test data response: ', test_y.shape)
    np.save( os.path.join(args.working_directory, '3_train_test_data', 'test_coords.npy') , arr=test_coords)
    print('test data coords: ', test_coords.shape)

if __name__ == '__main__':
    prepare_data()