import numpy as np
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--working_directory", help="path to the working directory", default= "/data/ahsoka/eocp/forestpulse/01_data/02_processed_data/tree_mask_LCC")
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
    # 2nd: response array
    FORCE_response = np.loadtxt(os.path.join(args.working_directory, '2_FORCE_samples','response.txt'))
    FORCE_coords = np.loadtxt(os.path.join(args.working_directory, '2_FORCE_samples','coord.txt'))

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
    # mix every class individually
    for LC_class in np.unique(FORCE_response):

        LC_array_samples = reshaped_2D_array[FORCE_response == LC_class, :, :]
        LC_array_response = FORCE_response[FORCE_response == LC_class]
        LC_array_coords = FORCE_coords[FORCE_response == LC_class,:]
        num_samples = LC_array_samples.shape[0]
        train_ratio = 0.7
        # make a random index array
        random_indices = np.random.permutation(num_samples)
        #calculate training/test array size
        train_size = int(num_samples * train_ratio)
        # allocation of train and test indices
        train_indices = random_indices[:train_size]
        test_indices = random_indices[train_size:]

        train_class_x = LC_array_samples[train_indices]
        train_class_y = LC_array_response[train_indices]
        train_class_coords = LC_array_coords[train_indices]

        test_class_x = LC_array_samples[test_indices]
        test_class_y = LC_array_response[test_indices]
        test_class_coords = LC_array_coords[test_indices]

        # append to lists
        train_x_list.append(train_class_x)
        train_y_list.append(train_class_y)
        train_coords_list.append(train_class_coords)

        test_x_list.append(test_class_x)
        test_y_list.append(test_class_y)
        test_coords_list.append(test_class_coords)

    # concatenate everything
    train_x = np.concatenate(train_x_list, axis=0)
    train_y = np.concatenate(train_y_list, axis=0)
    train_coords = np.concatenate(train_coords_list, axis=0)

    test_x = np.concatenate(test_x_list, axis=0)
    test_y = np.concatenate(test_y_list, axis=0)
    test_coords = np.concatenate(test_coords_list, axis=0)

    #===========================================
    #==== Step 3: Store the joined Arrays  =====
    #===========================================
    if not os.path.exists(os.path.join(args.working_directory, '3_train_test_data')):
        os.makedirs(os.path.join(args.working_directory, '3_train_test_data'))
    # train arrays
    np.save( os.path.join(args.working_directory, '3_train_test_data', 'train_x.npy') , arr=train_x.astype('uint16'))
    #print('train data samples: ', train_x.shape)
    np.save( os.path.join(args.working_directory, '3_train_test_data', 'train_y.npy') , arr=train_y.astype('uint8'))
    #print('train data response: ', train_y.shape)
    np.save( os.path.join(args.working_directory, '3_train_test_data', 'train_coords.npy') , arr=train_coords)
    #print('train data coords: ', train_coords.shape)
    # test arrays
    np.save( os.path.join(args.working_directory, '3_train_test_data', 'test_x.npy') , arr=test_x.astype('uint16'))
    #print('test data samples: ', test_x.shape)
    np.save( os.path.join(args.working_directory, '3_train_test_data', 'test_y.npy') , arr=test_y.astype('uint8'))
    #print('test data response: ', test_y.shape)
    np.save( os.path.join(args.working_directory, '3_train_test_data', 'test_coords.npy') , arr=test_coords)
   # print('test data coords: ', test_coords.shape)

if __name__ == '__main__':
    prepare_data()