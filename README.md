# Erosion Detection

Written by **Andy Kweon** <br />

## Aim of the project:
To detect sediment loss from aerial images of hills around the Bay of Plenty region through ML model.

## High-level overview of the code:
The python script first makes a GET request to the ArcGIS online services to get a 4096 x 4096 sized image. 
<p align="center">
<img src = "https://user-images.githubusercontent.com/63220455/151877600-d179500d-a3e3-4ce8-ad5b-7de2d968743c.png"> <br />
Fig 1. Pulled image from ArcGIS server
</p>

We then run our ML model, and return a numpy array of size 4096 x 4096 with ones and zeros, with 1 being erosion and 0 being background.
<p align="center">
<img src = "https://user-images.githubusercontent.com/63220455/151877933-38564374-a980-4fcb-acad-0856cc5bdbc5.png"> <br />
Fig 2. Output from ML model
</p>

The numpy array is then written as a GeoTiff file, which is georeferenced.
<p align="center">
<img src = "https://user-images.githubusercontent.com/63220455/151878190-c55f7c6f-5ccb-4ed3-a473-164be15dec8a.png"> <br />
Fig 3. GeoTiff file opened in QGIS
</p>

Finally, the GeoTiff file is converted into shapely polygons.
<p align="center">
<img src = "https://user-images.githubusercontent.com/63220455/151878541-4245ed99-4935-4d6f-98b0-936c7804dddb.png"> <br />
Fig 3. Polygon overlayed on top of google map in QGIS
</p>

## Note:
The trained model that I made (.h5 file) is not included due to size limitations of Github and my internship contract. <br />
The model I used was U-Net, and there are plenty of resources online on binary segmantaion which you can use to recreate the model. <br />
To see the results / performance of the model, I have included [Results.pdf](https://github.com/skwe902/Erosion-Detection/blob/main/Results.pdf) file which contains the test results of the model. 
