#####
# Creation Date: 13/11/2023
# Purpose: Finding the largest square in all fields in the dataset
#####

import numpy as np
import matplotlib.pyplot as plt
import rasterio as rio
from rasterio import features
import geopandas as gpd
import os
import math
import multiprocessing

##################################################

# documentation https://assets.planet.com/docs/Fusion-Tech-Spec_v1.0.0.pdf

INPUTIMAGEDIR =r"C:\Users\emily\OneDrive - Swansea University\Documents\Proj/18E-242N/PF-SR"
INPUTQADIR =  r"C:\Users\emily\OneDrive - Swansea University\Documents\Proj/18E-242N/PF-QA"
INPUTLABELS = r"C:\Users\emily\OneDrive - Swansea University\Documents\Proj/br-18E-242N-crop-labels-train-2018.geojson"
OUTPUTFILEPATH = r"C:\Users\emily\OneDrive - Swansea University\Documents\Proj/OutputFields"

# INPUTIMAGEDIR = "A:/Files/Work/Denethor/33N/18E-242N/PF-SR"
# INPUTQADIR = "A:/Files/Work/Denethor/33N/18E-242N/PF-QA"
# INPUTLABELS = "A:/Files/Work/Denethor/br-18E-242N-crop-labels-train-2018.geojson"
# OUTPUTFILEPATH = "A:/Files/Work/Denethor/OutputFields"

##################################################

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

def getFieldArrays(numFields, dates, fieldLabels):
    # create a list to hold field np arrays (to be Time x Band x H x W)
    fieldArrays = [None]*numFields

    # loop through every date
    for date in dates:
        # load imagery and QA file
        tileImage = rio.open(os.path.join(INPUTIMAGEDIR, date))
        tileQA = rio.open(os.path.join(INPUTQADIR, date))

        # get transform and crs of image
        crs = tileImage.crs
        transform = tileImage.transform

        # for each field, add the relevant data from this date to the numpy array
        for index, field in fieldLabels.iterrows():
            # get window the field exists within in the image
            left, bottom, right, top = field.geometry.bounds
            fieldWindow = rio.windows.from_bounds(left, bottom, right, top, transform)

            # get transform of the individual field
            win_transform = tileImage.window_transform(fieldWindow)

            # read in data from that window
            fieldImageData = tileImage.read(window=fieldWindow)
            fieldQAData = tileQA.read(window=fieldWindow)

            # rasterise multipolygon to create a mask of pixels within the field
            out_shape = fieldImageData[0].shape
            # check the shape is valid
            if out_shape[0] == 0 or out_shape[1] == 0:
                print(f"Field with id {field[0]} is too small in one dimension to be valid, skipping.")
                continue
            mask = features.rasterize([field.geometry], all_touched=True, transform=win_transform, out_shape=out_shape)

            # increase the mask to as many layers as origin image
            mask3d = np.broadcast_to(mask, fieldImageData.shape)

            # mask the image data layers
            fieldImageData[mask3d == 0] = 0

            # concatenate certain layers of the QA product onto the data for this time band
            # documentation https://assets.planet.com/docs/Fusion-Tech-Spec_v1.0.0.pdf
            fieldImageData = np.concatenate((fieldImageData, [fieldQAData[0]]), axis = 0) # synthetic pixel map
            fieldImageData = np.concatenate((fieldImageData, [fieldQAData[2]]), axis = 0) # cloud mask pixel map

            # concatenate this time band onto the relevant entry in the list of field arrays
            if fieldArrays[index] is None:
                fieldArrays[index] = fieldImageData
            else:
                fieldArrays[index] = np.concatenate((fieldArrays[index], fieldImageData), axis = 0)

        print(f"Field data from {date} read")

        # delete the variables holding the images from this date band - they're big and I only have so much RAM
        del tileImage
        del tileQA

    return fieldArrays

def getBestSizedROI(fieldData):
    """Poolable function to get list of best square ROI sizes and locations in a set of fields"""
    fieldLabels = fieldData[0]
    fieldArrays = fieldData[1]

    # iterate through arrays looking for largest possible square of valid data
    bestsList = []
    stdIndex = 0
    for index, field in fieldLabels.iterrows():
        # bad data check
        if fieldArrays[stdIndex] is None:
            # skip data if too small to make an image
            bestsList.append((0, (0,0)))
            stdIndex += 1
            continue

        # reduce image to 1s and 0s for simplicity
        workingFieldImage = np.clip(fieldArrays[stdIndex][0], a_min = 0, a_max=1)

        # create array to hold size of maximum square of 1s from that location
        bestSizeArray = np.zeros((len(workingFieldImage), len(workingFieldImage[0])))

        # work through every location in the field array
        for x in range(0,len(workingFieldImage)):
            for y in range(0,len(workingFieldImage[0])):
                # check if origin pixel is 0 to save time:
                if workingFieldImage[x][y] == 0:
                    bestSizeArray[x][y] = 0
                else:
                    # grow a square outwards from origin pixel
                    currentROIGrowth = 0
                    nullPixelUnmet = True
                    while nullPixelUnmet:
                        lowX = x-currentROIGrowth
                        highX = x+currentROIGrowth+1
                        lowY = y-currentROIGrowth
                        highY = y+currentROIGrowth+1
                        # check if created square would be within array bounds
                        if lowX >= 0 and lowY >= 0 and highX <= len(workingFieldImage) and highY <= len(workingFieldImage[0]):
                            checkedSquare = workingFieldImage[lowX:highX, lowY:highY]
                        else:
                            bestSizeArray[x][y] = ((currentROIGrowth-1) * 2) + 1
                            nullPixelUnmet = False
                            continue
                        # check if the created square to check is all 1s
                        if np.any(np.isin(checkedSquare, 0)):
                            bestSizeArray[x][y] = ((currentROIGrowth-1) * 2) + 1
                            nullPixelUnmet = False
                            continue
                        else:
                            currentROIGrowth += 1

        # now to process into highest value and it's location
        maxSize = np.max(bestSizeArray)
        # can choose any location with the maximum value, as long as it is deterministic 
        # looking for the first location in a set pattern is pretty deterministic
        maxFound = False
        for x in range(0,len(bestSizeArray)):
            for y in range(0,len(bestSizeArray[0])):
                if bestSizeArray[x][y] == maxSize:
                    maxLocation = (x, y)
                    maxFound == True
                    break
            if maxFound:
                break

        # want to write this value pair to a list
        bestsList.append((maxSize, maxLocation))

        print(f"Field {field[0]} processed ({len(bestsList)}/{len(fieldLabels)})")

        stdIndex += 1

        # # visualiser - only turn on when demoing
        # plt.imshow(bestSizeArray, cmap = "gray", interpolation="nearest")
        # plt.axis('off')
        # plt.show()

    print("Batch complete")

    return bestsList

##################################################
        
def main():
    """The general function that will take the input .tif data and output products"""

    # get geodataframe of field labels
    fieldLabels = gpd.read_file(INPUTLABELS)

    # need to set up the empty data structures for each field
    numFields = fieldLabels.shape[0]

    # write up metadata files for all fields
    writeFieldMetadata(fieldLabels, OUTPUTFILEPATH)

    # get the name of the an image date, doesn't really matter which date, to analyse field shapes
    testImageryDate = os.listdir(INPUTIMAGEDIR)[0]

    # get an array of the field shape for each field in one date
    fieldArrays = getFieldArrays(numFields, [testImageryDate], fieldLabels)
    # data limiter, for testing
    # fieldLabels = fieldLabels[0:50]
    # fieldArrays = fieldArrays[0:50]

    # multithreading setup
    cores = multiprocessing.cpu_count()
    rowsPerThread = math.ceil(len(fieldLabels)/(cores-1))
    splitLabels = [fieldLabels[i:i+rowsPerThread] for i in range(0,len(fieldLabels),rowsPerThread)]
    splitArrays = [fieldArrays[i:i+rowsPerThread] for i in range(0,len(fieldArrays),rowsPerThread)]
    splitData = zip(splitLabels, splitArrays)

    # send the split labels and arrays through the ROI get function:
    with multiprocessing.Pool(cores-1) as pool:
        returnedLists = pool.map(getBestSizedROI, splitData)

    # build returned lists into one list
    bestsList = [item for sublist in returnedLists for item in sublist]

    # save the values to a file
    savepath = os.path.join(OUTPUTFILEPATH, "fieldMaxSquareInfo.txt")
    with open(savepath, "w") as outFile:
        outFile.write(f"fid maxWidth x y \n")
        for index, field in fieldLabels.iterrows():
            outFile.write(f"{field[0]}, {bestsList[index][0]}, {bestsList[index][1][0]}, {bestsList[index][1][1]}\n")
                

##################################################

if __name__ == "__main__":
    main()
