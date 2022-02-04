import requests
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
import rasterio
import rasterio.mask
from rasterio.features import shapes
from rasterio.transform import from_origin
from rasterio.warp import calculate_default_transform, reproject, Resampling
from pyproj import CRS
from pyproj import Transformer
import json
import geojson
from geojson_rewind import rewind
from geojson import Feature, FeatureCollection, dump
import math
import time
from rasterio.merge import merge
import glob
from glob import glob
import os
import fiona
from osgeo import gdal

#API Documentation: https://services1.arcgisonline.co.nz/arcgis/sdk/rest/index.html#//02ss00000062000000
#Export Map URL: https://services1.arcgisonline.co.nz/arcgis/rest/services/Imagery/newzealand/MapServer/export
#The image pulled from ArcGIS is 4096 x 4096 pixels, and equivalent to 1000m x 1000m area. 

def main():
    tic = time.perf_counter()
    url = 'https://services1.arcgisonline.co.nz/arcgis/rest/services/Imagery/newzealand/MapServer/export'

    imageDirectory = 'C:/Users/skwe9/Desktop/folder/image.png'
    modelDirectory = "C:/Users/skwe9/Desktop/folder/SavedModel.h5"
    mainDirectory = 'C:/Users/skwe9/Desktop/folder/'
    jsonDirectory = 'C:/Users/skwe9/Desktop/folder/sampleProps.geojson'

    # Get a list of top left corners
    topLeftList = get_topLeftList(jsonDirectory) 
    print("1000x1000 blocks found: "+ str(len(topLeftList)))

    # Run our ML model
    for count in range(len(topLeftList)):
        #input top left XY coordinates 
        topLeftX = topLeftList[count][0]
        topLeftY = topLeftList[count][1]

        bbox = str(topLeftX) + ", " + str(topLeftY) + ", " + str(topLeftX + 1000) + ", " + str(topLeftY - 1000) #get bottom right coordinates
        data = { #specify image properties pulled from ArcGIS
            'bbox': bbox, 
            'size': '4096,4096', 
            'format': 'png',
            'transparent': 'false',
            'f': 'image'
        }

        # Prepare the image
        get_image(url, imageDirectory, data)
        print("got image no: " + str(count))
        image = Image.open(imageDirectory).convert("RGB")
        image = np.asarray(image) #(4096, 4096, 3) - RGB image sized 4096x4096 pixels
        tiles = split_image(image, (256,256)) #(256,256,256,3) - 256 RGB images sized 256x256 pixels

        # Normalize the image
        images_list = []
        for i in range(256):
            images_list.append(tiles[i]/255) #normalize the image pixel values from (0~255) to (0~1)
            x = np.asarray(images_list)

        # Make Predictions
        model = load_model(modelDirectory, custom_objects={"focal_tversky":focal_tversky,"tversky":tversky,"tversky_loss":tversky_loss})
        result = model.predict(x) #(256,256,256,1) - 256 2D arrays sized 256x256
        result = (result > 0.5).astype(int) #set a threshold - anything above 0.5 becomes 1

        # Combine the result arrays into one giant array
        combinedArray = np.hstack(result) #(256, 65536, 1)
        newArray = np.hsplit(combinedArray, 16) #65536/16 = 4096
        newArray = np.vstack(newArray) #(4096, 4096, 1)
        result = np.squeeze(newArray) #(4096, 4096)

        # Write the result to a GeoTiff file
        #Source: https://gis.stackexchange.com/questions/279953/numpy-array-to-gtiff-using-rasterio-without-source-raster
        tifDirectory = mainDirectory + 'L' + str(count) + ".tif" #L0.tif, L1.tif etc.
        print(tifDirectory)
        # transformer = Transformer.from_crs('epsg:2193', 'epsg:4326', always_xy=True)
        # topLeftX, topLeftY = transformer.transform(topLeftX, topLeftY)
        transform = from_origin(topLeftX, topLeftY, 0.244140625, 0.244140625) #(top left corner x, top left corner y, 1000/4096, 1000/4096)
        new_dataset = rasterio.open(tifDirectory, 'w', driver='GTiff', height=result.shape[0], width=result.shape[1], count=1, dtype=str(result.dtype),
                                crs=CRS.from_epsg(2193),
                                transform=transform)
        new_dataset.write(result, indexes=1)
        new_dataset.close()

    # Merge the GeoTIFF files together into one
    #Source: https://automating-gis-processes.github.io/CSC18/lessons/L6/raster-mosaic.html
    dirpath = mainDirectory
    out_fp = 'C:/Users/skwe9/Desktop/folder/mosaic.tif'
    search_criteria = "L*.tif"
    q = os.path.join(dirpath, search_criteria)
    dem_fps = glob.glob(q)
    src_files_to_mosaic = []
    for fp in dem_fps:
        src = rasterio.open(fp)
        src_files_to_mosaic.append(src)
    mosaic, out_trans = merge(src_files_to_mosaic)
    out_meta = src.meta.copy()
    out_meta.update({"driver": "GTiff",
               "height": mosaic.shape[1],
                "width": mosaic.shape[2],
                "transform": out_trans,
                "crs": CRS.from_epsg(2193)
                }
                )
    with rasterio.open(out_fp, "w", **out_meta) as dest:
        dest.write(mosaic)

    dst_crs = 'EPSG:4326'
    # Convert Raster EPSG from NZ to WGS84
    with rasterio.open("C:/Users/skwe9/Desktop/folder/mosaic.tif") as src:
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds)
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': dst_crs,
            'transform': transform,
            'width': width,
            'height': height
        })

        with rasterio.open("C:/Users/skwe9/Desktop/folder/new_mosaic.tif", 'w', **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.nearest)

    # Clip raster
    # Source: https://rasterio.readthedocs.io/en/latest/topics/masking-by-shapefile.html
    srcDS = gdal.OpenEx(jsonDirectory)
    gdal.VectorTranslate('C:/Users/skwe9/Desktop/folder/test.shp', srcDS, format='ESRI Shapefile')

    with fiona.open('C:/Users/skwe9/Desktop/folder/test.shp', "r") as shapefile:
        polygonShape = [feature["geometry"] for feature in shapefile]

    with rasterio.open("C:/Users/skwe9/Desktop/folder/new_mosaic.tif") as src:
        out_image, out_transform = rasterio.mask.mask(src, polygonShape, crop=True)
        out_meta = src.meta

    out_meta.update({"driver": "GTiff",
                    "height": out_image.shape[1],
                    "width": out_image.shape[2],
                    "transform": out_transform})

    with rasterio.open("C:/Users/skwe9/Desktop/folder/new_mosaic.tif", "w", **out_meta) as dest:
        dest.write(out_image)

    # Convert GeoTIFF to GeoJSON
    #Source:https://gis.stackexchange.com/questions/187877/how-to-polygonize-raster-to-shapely-polygons/187883
    mask = None
    results = []
    with rasterio.Env():
        with rasterio.open('C:/Users/skwe9/Desktop/folder/new_mosaic.tif') as src:
            image = src.read(1) # first band
            for i, (s, v) in enumerate(shapes(image, mask=mask, transform=src.transform)):
                if v == 1: #save the polygons that have value of 1 (erosion)
                    results.append(Feature(geometry=s, properties={'raster_val': v}))
                else:
                    continue

    #https://gis.stackexchange.com/questions/130963/write-geojson-into-a-geojson-file-with-python
    #https://gis.stackexchange.com/questions/259944/polygons-and-multipolygons-should-follow-the-right-hand-rule
    feature_collection = FeatureCollection(results)
    rewoundResults = rewind(feature_collection)
    with open('C:/Users/skwe9/Desktop/folder/myfile.geojson', 'w') as f:
        dump(rewoundResults, f)

    # # Polygonize a raster band
    # #source: https://pcjericks.github.io/py-gdalogr-cookbook/raster_layers.html#polygonize-a-raster-band
    # tifDirectory = 'C:/Users/skwe9/Desktop/mosaic.tif'
    # src_ds = gdal.Open(tifDirectory)
    # if src_ds is None:
    #     print ('Unable to open file')
    #     sys.exit(1)
    # else:
    #     srcband = src_ds.GetRasterBand(1)

    # srs = osr.SpatialReference()
    # srs.ImportFromEPSG(2193)

    # # create output datasource
    # #Source: https://gis.stackexchange.com/questions/279032/output-field-name-when-using-gdal-polygonize-in-python
    # drv = ogr.GetDriverByName("ESRI Shapefile")
    # dst_ds = drv.CreateDataSource( polygonDirectory + ".shp" )
    # dst_layer = dst_ds.CreateLayer( polygonDirectory, srs = srs )
    # fd = ogr.FieldDefn("DN", ogr.OFTInteger)
    # dst_layer.CreateField(fd)
    # dst_field = dst_layer.GetLayerDefn().GetFieldIndex("DN")
    # gdal.Polygonize( srcband, srcband, dst_layer, dst_field, [], callback=None )

    toc = time.perf_counter()
    print(f"Finished in {toc - tic:0.4f} seconds")



#Functions used for main():
def get_topLeftList(jsonDirectory):
    with open(jsonDirectory) as f:
        data = json.load(f)
    
    for feature in data['features']:
        cood = feature['geometry']['coordinates']
        poly = geojson.Polygon(cood)

    line = get_bbox(list(geojson.utils.coords(poly)))
    maxX = line[1][0]
    maxY = line[0][1]
    minX = line[0][0]
    minY = line[1][1]

    width = maxX - minX
    height = maxY - minY

    if width % 1000 != 0:
        widthNo = math.ceil(width / 1000)

    if height % 1000 != 0:
        heightNo = math.ceil(height / 1000)

    topLeftList = []

    for i in range(heightNo):
        for j in range(widthNo):
            newX = minX + j * 1000
            newY = maxY - i * 1000
            topLeftList.append((newX, newY))
    return topLeftList

def get_bbox(coord_list): 
    #Source: https://gis.stackexchange.com/questions/313011/convert-geojson-object-to-bounding-box-using-python-3-6
    box = []
    for i in (0,1):
        res = sorted(coord_list, key=lambda x:x[i])
        box.append((res[0][i],res[-1][i]))
    x1,y1 = box[0][0],box[1][1] #top left XY
    x2,y2 = box[0][1],box[1][0] #bottom right XY
    transformer = Transformer.from_crs('epsg:4326', 'epsg:2193', always_xy=True)
    topLeft = transformer.transform(x1,y1)
    botRight = transformer.transform(x2,y2)
    return topLeft, botRight

def get_image(url: str, directory: str, data: dict):
    recieve = requests.get(url, params=data)
    if recieve.status_code == 200: 
        with open(f'{directory}','wb') as f:
            f.write(recieve.content) 

def split_image(image: np.ndarray, kernel_size: tuple):
    #Source: https://towardsdatascience.com/efficiently-splitting-an-image-into-tiles-in-python-using-numpy-d1bf0dd7b6f7
    img_height, img_width, channels = image.shape
    tile_height, tile_width = kernel_size
    tiled_array = image.reshape(img_height // tile_height, tile_height, img_width // tile_width, tile_width, channels)
    tiled_array = tiled_array.swapaxes(1,2)
    tiled_array = tiled_array.reshape(-1, 256, 256, 3)
    return tiled_array



#Custom Objects for ML model - needs these functions to load the model
def tversky(y_true, y_pred):
    smooth=1.
    y_true_pos = K.flatten(y_true)
    y_pred_pos = K.flatten(y_pred)
    true_pos = K.sum(y_true_pos * y_pred_pos)
    false_neg = K.sum(y_true_pos * (1-y_pred_pos))
    false_pos = K.sum((1-y_true_pos)*y_pred_pos)
    alpha = 0.7
    return (true_pos + smooth)/(true_pos + alpha*false_neg + (1-alpha)*false_pos + smooth)

def focal_tversky(y_true,y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    pt_1 = tversky(y_true, y_pred)
    gamma = 0.75
    return K.pow((1-pt_1), gamma)

def tversky_loss(y_true, y_pred):
    return 1 - tversky(y_true,y_pred)


if __name__ == "__main__":
    main()
