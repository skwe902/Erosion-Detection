import requests
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
import rasterio
from rasterio.transform import from_origin
from pyproj import CRS
from osgeo import gdal, ogr, osr
import sys

#API Documentation: https://services1.arcgisonline.co.nz/arcgis/sdk/rest/index.html#//02ss00000062000000
#Export Map URL: https://services1.arcgisonline.co.nz/arcgis/rest/services/Imagery/newzealand/MapServer/export
#The image pulled from ArcGIS is 4096 x 4096 pixels, and equivalent to 1000m x 1000m area. 

def main():
    imageDirectory = 'C:/Users/skwe9/Desktop/getImages/image.png'
    modelDirectory = "C:/Users/skwe9/Desktop/SavedModel.h5"
    tifDirectory = 'C:/Users/skwe9/Desktop/test1.tif'
    polygonDirectory = "C:/Users/skwe9/Desktop/POLYGONIZED_STUFF"
    url = 'https://services1.arcgisonline.co.nz/arcgis/rest/services/Imagery/newzealand/MapServer/export'
    tilesize = (256,256) #the size we want to chunk the images

    data = { #specify image properties pulled from ArcGIS
        'bbox': "1905111.347, 5800115.650, 1906111.347, 5799115.650", 
        'size': '4096,4096', 
        'format': 'png',
        'transparent': 'false',
        'f': 'image'
    }

    # Prepare the image
    get_image(url, imageDirectory, data)
    image = Image.open(imageDirectory).convert("RGB")
    image = np.asarray(image) #(4096, 4096, 3) - RGB image sized 4096x4096 pixels
    tiles = split_image(image, tilesize) #(256,256,256,3) - 256 RGB images sized 256x256 pixels

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
    transform = from_origin(1905111.347, 5800115.650, 0.244140625, 0.244140625) #(top left corner x, top left corner y, 1000/4096, 1000/4096)
    new_dataset = rasterio.open(tifDirectory, 'w', driver='GTiff', height=result.shape[0], width=result.shape[1], count=1, dtype=str(result.dtype),
                            crs=CRS.from_epsg(2193),
                            transform=transform)
    new_dataset.write(result, indexes=1)
    new_dataset.close()

    # Polygonize a raster band
    #source: https://pcjericks.github.io/py-gdalogr-cookbook/raster_layers.html#polygonize-a-raster-band
    src_ds = gdal.Open(tifDirectory)
    if src_ds is None:
        print ('Unable to open file')
        sys.exit(1)
    else:
        srcband = src_ds.GetRasterBand(1)

    srs = osr.SpatialReference()
    srs.ImportFromEPSG(2193)

    # create output datasource
    #Source: https://gis.stackexchange.com/questions/279032/output-field-name-when-using-gdal-polygonize-in-python
    drv = ogr.GetDriverByName("ESRI Shapefile")
    dst_ds = drv.CreateDataSource( polygonDirectory + ".shp" )
    dst_layer = dst_ds.CreateLayer( polygonDirectory, srs = srs )
    fd = ogr.FieldDefn("DN", ogr.OFTInteger)
    dst_layer.CreateField(fd)
    dst_field = dst_layer.GetLayerDefn().GetFieldIndex("DN")
    gdal.Polygonize( srcband, srcband, dst_layer, dst_field, [], callback=None )



#Functions used for main():
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