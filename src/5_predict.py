#!/usr/bin/env python

import numpy as np
from osgeo import gdal
import rasterio
from tqdm import tqdm
from joblib import Parallel, delayed
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from datetime import datetime
import argparse
import ast
from scipy.interpolate import BSpline

parser = argparse.ArgumentParser()
parser.add_argument("--working_directory", 
                    help="path to the working directory", 
                    default= "/data/ahsoka/eocp/forestpulse/01_data/02_processed_data/tree_mask_noFORCE")
parser.add_argument("--dc_folder", 
                    help="path to the folder of spline coefficents", 
                    default= "/data/ahsoka/eocp/forestpulse/01_data/02_processed_data/ThermSpline_DC")
parser.add_argument("--year", 
                    help="year of the mask", 
                    default= "2021")
parser.add_argument("--tile", 
                    help="FORCEtile which should be predicted", 
                    default= 'X0055_Y0053')
parser.add_argument("--mask_dir", 
                    help="directory of the germany mask", 
                    default= '/data/ahsoka/dc/deu/mask')
parser.add_argument("--mask_name", 
                    help="name of the germany mask file", 
                    default= 'DEU.tif')
parser.add_argument("--name_list", 
                    help="names of the landcover classes", 
                    default= "['Artificial Land', 'Cropland', 'Woodland', 'Shrubland', 'Grassland', 'Bare Land', 'Water Areas', 'Wetlands']" )
parser.add_argument("--version", 
                    help="version of the model to be used",
                      default = '5')
#parser.add_argument("--output_directory", help="FORCEtile which should be predicted", default= "/data/ahsoka/eocp/forestpulse/01_data/02_processed_data/tree_mask")
args = parser.parse_args()

def get_stack(tile, year):
    file_path = os.path.join(args.dc_folder, tile, f'ThermSpline_coefs_{args.year}.tif')
    with rasterio.Env(GDAL_NUM_THREADS="1"):
        with rasterio.open(file_path) as src:
            stack = src.read()
            stack = np.moveaxis(stack, 0, -1)
    return stack

def predict(tile, year, model):
    def pred(model, x):
        y_pred = model(x, training=False)
        return y_pred.numpy()
    def norm(a):
        a_out = a/10000.
        a_out[a_out<0] = 0
        return a_out
    
    # =============================================
    #              loading data 
    # =============================================
    
    file = os.path.join(args.dc_folder, tile, f'ThermSpline_coefs_{args.year}.tif')
    if not os.path.isfile(file):
        print('Not tile, skipping!')
        return
    
    out_dir = os.path.join(args.working_directory, "5_prediction", tile)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # start processing here
    print('Start: ', tile, ' ', year, ' [...]')
    x_in = get_stack(tile, year)   
    y_out = np.zeros([x_in.shape[0], x_in.shape[1],8])

    print('Predicting [...] ', datetime.now().strftime('%H:%M:%S')),
    if not os.path.isfile(os.path.join(args.mask_dir, args.tile ,args.mask_name)):
        print('Not in Germany, skipping!')
        return

    with rasterio.open(os.path.join(args.mask_dir, args.tile ,args.mask_name)) as src:
        meta = src.meta.copy()
        germany_mask = src.read(1)

    # =============================================
    #              prediction 
    # =============================================
    for i in tqdm(range(y_out.shape[1])): # 3000 iterations = about 2 minutes (much smaller model)
        x_batch = x_in[i, ...]
        x_batch = norm(x_batch.astype(np.float32))
        #print(x_batch.shape)
        y_out[i, ...] = pred(model, x_batch)

    # classification of dominant species
    y_out_clf = np.argmax(y_out, axis= -1)
    y_out_clf += 1
    y_out_clf = y_out_clf.astype(np.uint8)
    y_out_clf[germany_mask == 0] = 255

    # forest binary mask
    forest_binary = y_out_clf
    forest_binary[forest_binary != 3] = 0 # all other
    forest_binary[forest_binary == 3] = 1 # Woodland
    forest_binary[germany_mask == 0] = 255 # outside germany

    # =============================================
    #              writing outputs 
    # =============================================
    print("Exporting [...] ", datetime.now().strftime('%H:%M:%S'))
    # 1. propability of classification
    meta.update(
        count=len(ast.literal_eval(args.name_list)),  # Anzahl der Layer entspricht der Anzahl der TIFFs
        dtype='uint8',          # 8-Bit Integer
        compress='ZSTD'          # LZW-Komprimierung
    )
    with rasterio.open(os.path.join(out_dir, f'LC_prop_{args.year}.tif'), 
                       'w', **meta) as dst:
        dst.descriptions = ast.literal_eval(args.name_list)
        for i in range(y_out.shape[-1]):
            dst.write(y_out[..., i].astype(np.uint8), i+1)
            dst.set_band_description(i+1, ast.literal_eval(args.name_list)[i])

    # 2. classification of dominant species
    meta.update(
        count=1,  # Anzahl der Layer entspricht der Anzahl der TIFFs
        dtype='uint8',          # 8-Bit Integer
        compress='ZSTD'          # LZW-Komprimierung
    )
    with rasterio.open(os.path.join(out_dir, f'LC_class_{args.year}.tif'), 
                       'w', **meta) as dst:
        dst.write(y_out_clf.astype(np.uint8), 1)
        dst.set_band_description(1, 'LC_Class')

    # 3. binary forest mask
    #Export_forest_binary(forest_binary)
    with rasterio.open(os.path.join(out_dir, f'Forest_{args.year}.tif'), 
                       'w', **meta) as dst:
        dst.write(forest_binary.astype(np.uint8), 1)
        dst.set_band_description(1, 'Forest')
    print('Finished: ', tile, ' ', year , ' ', datetime.now().strftime('%H:%M:%S'))

if __name__ == '__main__':
    # load model
    version = args.version
    model_path = os.path.join(args.working_directory, '3_trained_model','version' +
                              str(version), 'saved_model'+ str(version)+ '.keras')
    model = tf.keras.models.load_model(model_path)
    # start prediction
    predict(args.tile, int(args.year) , model)
