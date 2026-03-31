#! /home/ahsoka/klehr/anaconda3/envs/Synth_Mix/bin python3
from tracemalloc import start

import geopandas as gpd
import rasterio
from rasterio.windows import Window
import numpy as np
from tqdm import tqdm
import os
import argparse
from joblib import Parallel, delayed
from shapely.geometry import box

parser = argparse.ArgumentParser()
parser.add_argument("--dc_folder", help="path to the spline coefficents data-cube", 
                    default= "/data/ahsoka/eocp/forestpulse/01_data/02_processed_data/ThermSpline_DC" )
parser.add_argument("--training_points", help="path to the file of the training points geopackage", 
                    default= "/data/ahsoka/eocp/forestpulse/INTERNAL/BWI4/forest_mask/forest_mask_samples_LCC.gpkg") # for 2021
parser.add_argument("--tile", help="the tile to process", default= 'X0055_Y0053')
parser.add_argument("--year", help="year of the training points", default= '2021')
parser.add_argument("--working_directory", help="path to the file of the training points geopackage", 
                    default= "/data/ahsoka/eocp/forestpulse/01_data/02_processed_data/tree_mask_LCC_SplineCoefs")
args = parser.parse_args()

#-----------------------------------------------------------------------------------------------------
# This script is the first step for the tree species fraction mapping using the c_thermal spline coefs
# It extracts the pure pixel values for the training points and stores them in a folder. 
# The output is used in the next step to create synthetic mixtures for training the model (step 3).
#-----------------------------------------------------------------------------------------------------
def to_point(geom):
    if geom.geom_type == "MultiPoint":
        return list(geom.geoms)[0]
    return geom

def extract_points(tile):
    dc_path = os.path.join(args.dc_folder, tile)
    file_path = os.path.join(dc_path, f'ThermSpline_coefs_{args.year}.tif')
    if not os.path.exists(file_path):
        return
    
    with rasterio.open(file_path) as src:
        band_1 = src.read(1)  # get raster information (position, height, width etc.)
        bounds = src.bounds 

        raster_geom = box(bounds.left, bounds.bottom, bounds.right, bounds.top)
            
        # 1. load point data
        gdf = gpd.read_file(args.training_points)
        gdf = gdf.to_crs("EPSG:3035")

        # 2. clip the point data to the raster bounds
        gdf_clipped = gdf[gdf.intersects(raster_geom)].copy()
        gdf_clipped["geometry"] = gdf_clipped.geometry.apply(to_point)

        # 3. extract raster values at the point locations
        coords = [(geom.x, geom.y) for geom in gdf_clipped.geometry]
        classes = gdf_clipped["mask_class"].values
        samples = list(src.sample(coords))
        
        # 4. save the extracted samples
        np.savetxt(os.path.join(args.working_directory, '1_samples', f'samples_{str(args.year)}',f'y_{tile}.csv'),
                    classes, fmt="%d")
        np.savetxt(os.path.join(args.working_directory, '1_samples', f'samples_{str(args.year)}',f'x_{tile}.csv'),
                    samples, fmt="%d")
        return
        
if __name__ == '__main__':
    if not os.path.exists(os.path.join(args.working_directory, '1_samples', f'samples_{str(args.year)}')):
        os.makedirs(os.path.join(args.working_directory, '1_samples', f'samples_{str(args.year)}'))
    # -------------- single tile for testing --------------
    #tile = 'X0055_Y0053'
    if os.path.exists(os.path.join(args.working_directory, '1_samples', 
                                   f'samples_{str(args.year)}',
                                   f'y_{args.tile}.csv')) and (
       os.path.exists(os.path.join(args.working_directory, '1_samples', 
                                   f'samples_{str(args.year)}',f'x_{args.tile}.csv'))):
        print(f"Samples for tile {args.tile} already exist. Skipping extraction.")
    else:
        extract_points(args.tile)
    #------------------------------------------------------

    # --------- parallel processing for all tiles ---------
    #tiles = [tile for tile in os.listdir(args.dc_folder) if tile.startswith("X")]
    #Parallel(n_jobs=20)(delayed(extract_points)(tile) for tile in tqdm(tiles, desc="Processing tiles"))
    #------------------------------------------------------