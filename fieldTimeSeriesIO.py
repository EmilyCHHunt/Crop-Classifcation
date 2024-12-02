#####
# Creation Date: 06/11/2023
# Purpose: Inputting Planet .tif files and outputting saved .npy files of each field and it's layers
#####

import numpy as np
import matplotlib.pyplot as plt
import rasterio as rio
from rasterio import features
import geopandas as gpd
import os

##################################################

# documentation https://assets.planet.com/docs/Fusion-Tech-Spec_v1.0.0.pdf

INPUTIMAGEDIR =r"C:\Users\emily\OneDrive - Swansea University\Documents\Proj/18E-242N/PF-SR"
INPUTQADIR =  r"C:\Users\emily\OneDrive - Swansea University\Documents\Proj/18E-242N/PF-QA"
INPUTLABELS = r"C:\Users\emily\OneDrive - Swansea University\Documents\Proj/br-18E-242N-crop-labels-train-2018.geojson"
OUTPUTFILEPATH = r"C:\Users\emily\OneDrive - Swansea University\Documents\Proj/OutputFields"

# INPUTIMAGEDIR = "A:/Files/Work/Denethor/33N/18E-242N/PF-SR/"
# INPUTQADIR = "A:/Files/Work/Denethor/33N/18E-242N/PF-QA/"
# INPUTLABELS = "A:/Files/Work/Denethor/br-18E-242N-crop-labels-train-2018.geojson"
# OUTPUTFILEPATH = "A:/Files/Work/Denethor/OutputFields"

def writeFieldMetadata(fieldLabels, outputPath):
    # make sure output directory exists
    if os.path.isdir(outputPath):
        print("Output Directory Exists")
    else:
        os.mkdir(outputPath)
        print("Output Directory Created")

    # write the data for each field, naming the file after the field id number
    for index, row in fieldLabels.iterrows():
        outFile = open(os.path.join(outputPath,str(row[0]) + ".txt"), "w")
        # writing as fid cropid cropname minx miny maxx maxy
        left, bottom, right, top = row.geometry.bounds
        outFile.write(f"{row[0]} {row[3]} {row[4]} {left} {bottom} {right} {top}")

    print("Field Metadata Files Written")

def writeDatesMetadata(dates, outputPath):
    # make sure output directory exists
    if os.path.isdir(outputPath):
        print("Output Directory Exists")
    else:
        os.mkdir(outputPath)
        print("Output Directory Created")

    # open the file
    outFile = open(os.path.join(outputPath,"usedDates.txt"), "w")

    # write the data for each field, naming the file after the field id number
    for date in dates:
        outFile.write(date + "\n")
        
    print("Date Metadata File Written")

def main():
    """The general function that will take the input .tif data and output products"""

    # get geodataframe of field labels
    fieldLabels = gpd.read_file(INPUTLABELS)

    # for later progress tracking
    numFields = fieldLabels.shape[0]

    # write up metadata files for all fields
    writeFieldMetadata(fieldLabels, OUTPUTFILEPATH)

    # get a list of imagery dates to consider - held as file name strings
    imageryDates = os.listdir(INPUTIMAGEDIR)

    # reduce these dates down to every 5 days - should be 73 dates per year
    reducedImageryDates = [imageryDate for imageryDate in imageryDates[::5]]

    inputImages = [os.path.join(INPUTIMAGEDIR, reducedImageryDate) for reducedImageryDate in reducedImageryDates]
    inputQA = [os.path.join(INPUTQADIR, reducedImageryDate) for reducedImageryDate in reducedImageryDates]

    # write up list of used file dates to make life easier
    writeDatesMetadata(reducedImageryDates, OUTPUTFILEPATH)

    # get transform and crs of images
    with rio.open(inputImages[0]) as image:
        crs = image.crs
        transform = image.transform

    # for each field, add the relevant data from this date to the numpy array
    for index, field in fieldLabels.iterrows():
        # get window the field exists within in the image
        left, bottom, right, top = field.geometry.bounds
        fieldWindow = rio.windows.from_bounds(left, bottom, right, top, transform)

        # get transform of the individual field
        with rio.open(inputImages[0]) as image:
            win_transform = image.window_transform(fieldWindow)

        # read in data from that window
        fieldImageData = np.stack([rio.open(temporalImage).read(window=fieldWindow) for temporalImage in inputImages])
        fieldQAData = np.stack([rio.open(temporalImage).read(window=fieldWindow) for temporalImage in inputQA])

        # rasterise multipolygon to create a mask of pixels within the field
        out_shape = fieldImageData[0].shape
        # check the shape is valid
        if out_shape[0] == 0 or out_shape[1] == 0:
            print(f"Field with id {field[0]} is too small in one dimension to be valid, skipping.")
            continue
        mask = features.rasterize([field.geometry], all_touched=True,transform=win_transform, out_shape=fieldImageData[0, 0].shape)

        # increase the mask to as many layers as origin image
        mask3d = np.broadcast_to(mask, fieldImageData.shape)

        # mask the image data layers - can't mask QA products as for some a 0 value is good
        fieldImageData[mask3d == 0] = 0

        # concatenate layers of the QA product onto the image data
        # documentation https://assets.planet.com/docs/Fusion-Tech-Spec_v1.0.0.pdf
        fieldImageData = np.concatenate((fieldImageData, fieldQAData), axis = 1)
        # indexing is now blue, green, red, nir, synth data %, pixel observation day offset, cloud+shadow mask, pixel tracability, reference scenes, blue uncertainty, green uncertainty, red uncertainty, nir uncertainty

        # reduce the data down to just layers we want
        fieldImageData = fieldImageData[:,[0,1,2,3,4,6],:,:]
        # indexing is now blue, green, red, nir, synth data %, cloud+shadow mask

        # write the data for this field
        # get file name/path to save to
        fid = field[0]
        savepath = os.path.join(OUTPUTFILEPATH, str(fid) + ".npy")
        # save the array to a numpy binary file
        if fieldImageData is not None:
            np.save(savepath, fieldImageData)
            print(f"Field {fid} saved ({index+1}/{numFields})")
        else:
            print(f"Field {fid} could not be saved as it was too small in a given dimension for pixel representation")

    # # display the image
    # plt.imshow(fieldImageData[0], cmap = "gray", interpolation="nearest")
    # plt.axis('off')
    # plt.show()

##################################################

if __name__ == "__main__":
    main()
