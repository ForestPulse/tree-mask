
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

parser = argparse.ArgumentParser()
parser.add_argument("--working_directory", help="path to the working directory", default= "/data/ahsoka/eocp/forestpulse/01_data/02_processed_data/tree_mask_LCC")
parser.add_argument("--year", help="year of the mask", default= "2021")
parser.add_argument("--tile", help="FORCEtile which should be predicted", default= 'X0055_Y0053')
parser.add_argument("--name_list", help="names of the landcover classes", default= "['Artificial Land', 'Cropland', 'Woodland', 'Shrubland', 'Grassland', 'Bare Land', 'Water Areas', 'Wetlands']" )
#parser.add_argument("--output_directory", help="FORCEtile which should be predicted", default= "/data/ahsoka/eocp/forestpulse/01_data/02_processed_data/tree_mask")
args = parser.parse_args()

def get_stack(tile, year):
    def get_band(band_name):
        print('Loading band: ', band_name, ' - ' , datetime.now().strftime('%H:%M:%S')) 
        DC_path =  os.path.join(args.working_directory, '1_DC_FBW')
        band_path = os.path.join(DC_path, tile, '{y1}0101-{y2}1231_001-365_HL_TSA_SEN2L_{band_name}_FBW.tif'.format(y1=year, y2=year,band_name=band_name))
        with rasterio.open(band_path) as src:
            band = src.read()
        band = np.moveaxis(band, 0, -1) 
        return band
    print('Loading Data [...] - ', datetime.now().strftime('%H:%M:%S'))
    band_list = ['BLU', 'GRN', 'RED', 'RE1', 'RE2', 'RE3', 'BNR', 'NIR', 'SW1', 'SW2']# 10 seconds per band (40 seconds if not in cache)
    stack = np.array([get_band(b) for b in band_list])
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
    def Export(arr_in):
        path = os.path.join(args.working_directory, '1_DC_FBW', tile, '{y1}0101-{y2}1231_001-365_HL_TSA_SEN2L_BLU_FBW.tif'.format(y1=year, y2=year))
        ds = gdal.Open(path)
        band = ds.GetRasterBand(1)
        arr = band.ReadAsArray()
        [cols, rows] = arr.shape
        bands = arr_in.shape[-1]
        driver = gdal.GetDriverByName("GTiff")
        path_out = os.path.join(args.working_directory, '8_prediction', tile, 'LC_prop_' + str(year) + '.tif')
        if os.path.exists(path_out):
            os.remove(path_out)
        outdata = driver.Create(path_out, rows, cols, bands, gdal.GDT_Float32)
        #outdata = driver.Create(path_out, rows, cols, 1, gdal.GDT_Byte)
        # set geotransform and projection
        outdata.SetGeoTransform(ds.GetGeoTransform())
        outdata.SetProjection(ds.GetProjection())
        for i in range(bands):
            outdata.GetRasterBand(i + 1).WriteArray(arr_in[..., i])
            outdata.GetRasterBand(i + 1).SetNoDataValue(-9999)
            outdata.GetRasterBand(i + 1).SetDescription(ast.literal_eval(args.name_list)[i])
        outdata.FlushCache()
        outdata = None
        band=None
        ds=None

    def Export_classification(arr_in):
        path = os.path.join(args.working_directory, '1_DC_FBW', tile, '{y1}0101-{y2}1231_001-365_HL_TSA_SEN2L_BLU_FBW.tif'.format(y1=year, y2=year))
        ds = gdal.Open(path)
        band = ds.GetRasterBand(1)
        arr = band.ReadAsArray()
        [cols, rows] = arr.shape
        driver = gdal.GetDriverByName("GTiff")
        path_out = os.path.join(args.working_directory, '8_prediction', tile, 'LC_class_' + str(year) + '.tif')
        if os.path.exists(path_out):
            os.remove(path_out)
        outdata = driver.Create(path_out, rows, cols, 1, gdal.GDT_Byte)
        outdata.SetGeoTransform(ds.GetGeoTransform())##sets same geotransform as input
        outdata.SetProjection(ds.GetProjection())##sets same projection as input
        outdata.GetRasterBand(1).WriteArray(arr_in)
        outdata.GetRasterBand(1).SetNoDataValue(-9999)
        outdata.FlushCache() ##saves to disk!!
        outdata = None
        band=None
        ds=None
    
    blue_band = os.path.join(args.working_directory, '1_DC_FBW', tile, '{y1}0101-{y2}1231_001-365_HL_TSA_SEN2L_BLU_FBW.tif'.format(y1=year, y2=year))
    if not os.path.isfile(blue_band):
        print('Not tile, skipping!')
        return
    
    out_dir = os.path.join(args.working_directory, "8_prediction", tile)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # start processing here
    print('Start: ', tile, ' ', year, ' [...]')
    x_in = get_stack(tile, year)   
    y_out = np.zeros([x_in.shape[0], x_in.shape[1],8])
    print('Predicting [...] ', datetime.now().strftime('%H:%M:%S'))
    print(x_in.shape)
    print(y_out.shape)

    for i in tqdm(range(y_out.shape[1])): # duration about 15 minutes
        x_batch = x_in[i, ...]
        x_batch = norm(x_batch.astype(np.float32))
        y_out[i, ...] = pred(model, x_batch)
    
    # classification of dominant species
    y_out_clf = np.argmax(y_out, axis= -1)
    y_out_clf += 1
    y_out_clf = y_out_clf.astype(np.int8)

    print("Exporting [...] ", datetime.now().strftime('%H:%M:%S'))
    Export(y_out)
    Export_classification(y_out_clf)
    print('Finished: ', tile, ' ', year , ' ', datetime.now().strftime('%H:%M:%S'))

if __name__ == '__main__':
    # load model
    model_path = os.path.join(args.working_directory, '6_trained_model','version' +str(1), 'saved_model'+ str(1)+ '.keras')
    model = tf.keras.models.load_model(model_path)
    # start prediction
    predict(args.tile, int(args.year) , model)
    #print("Wohoooooo")