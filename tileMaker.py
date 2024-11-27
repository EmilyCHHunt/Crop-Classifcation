#####
# Author: Joseph Metcalfe
# Creation Date: 16/11/2023
# Purpose: Taking the outputs of ROIFinder and fieldTimeSeriesIO to create a balanced dataset of usable tiles for train/test
#####

import numpy as np
import matplotlib.pyplot as plt
import os
from skimage import filters
from skimage.util import random_noise

##################################################

# band documentation https://assets.planet.com/docs/Fusion-Tech-Spec_v1.0.0.pdf
INPUTIMAGEDIR = r"C:\Users\emily\OneDrive - Swansea University\Documents\Proj/OutputFields"
INPUTMETADATAPATH = r"C:\Users\emily\OneDrive - Swansea University\Documents\Proj/OutputFieldTilesMetadata"
INPUTROIINFO = r"C:\Users\emily\OneDrive - Swansea University\Documents\Proj/OutputFieldTilesMetadatafieldMaxSquareInfo.txt"
OUTPUTTILEPATH = r"C:\Users\emily\OneDrive - Swansea University\Documents\Proj/OutputFieldTiles"
OUTPUTMETADATAPATH = r"C:\Users\emily\OneDrive - Swansea University\Documents\Proj/OutputFieldTilesMetadata"

USEDCLASSES = ["Meadows", "Forage Crops", "Wheat", "Rye", "Corn", "Oil Seeds", "Barley", "Oats", "Root Crops"]
ROISIZE = 9 # width of square ROI to tile, sould always be odd
TILESPERCLASS = 600

##################################################
        
def getValidFieldList(inputROIsfile, inputMetadataDir, roiSize, usedClassList):
    """work out which fields are valid by the used list and which have large enough possible ROIs"""
    # start by working out valid fields to use by which have a possible ROI
    validFields = []
    with open(inputROIsfile, "r") as inputFile:
        for line in inputFile:
            splitLine = line.split(", ")
            # if len of split list isnt 4, is not valid data
            if len(splitLine) == 4:
                if float(splitLine[1]) >= roiSize:
                    validFields.append([splitLine[0],splitLine[2],splitLine[3]]) # field id, roi centre x, roi centre y
        inputFile.close()

    # now reference the individual field files of each valid field to find their class
    classedValidFields = [] # easier than list comprehension with this time frame
    classCounts = {key:0 for key in usedClassList}
    for field in validFields:
        fid = field[0]
        with open(os.path.join(inputMetadataDir, f"{fid}.txt"), "r") as inputFile:
            for line in inputFile: # only 1 line but this works
                splitLine = line.split(" ")
                # check if the line is one with the two word crop names
                if len(splitLine) == 7:
                    cropID = splitLine[1] # is useful to keep both name and id around in the data
                    cropName = splitLine[2]
                else:
                    cropID = splitLine[1]
                    cropName = splitLine[2] + " " + splitLine[3]
                # get centre point of the field for noise stratification
                minx = float(splitLine[-4])
                miny = float(splitLine[-3])
                maxx = float(splitLine[-2])
                maxy = float(splitLine[-1])
                fieldX = (minx + maxx) / 2
                fieldY = (miny + maxy) / 2
                # add in the data to the list if from a used crop type
                if cropName in usedClassList:
                    classedValidFields.append([fid, cropID, cropName, field[1], field[2], fieldX, fieldY]) # fid, id, name, roi centre x, roi centre y, field centre x, field centre y
                    # increment appropriate count
                    classCounts[cropName] += 1

    return classedValidFields, classCounts

def generateFieldTiles(inputFieldImages, usedClassList, ROISize, classedValidFields):
    """Turn valid fields into ROI tiles for each temporal spectral band"""
    validFieldTiles = {key:[] for key in usedClassList}
    fidDir = {}
    for field in classedValidFields:
        # load the appropriate npy file
        fullFieldArray = np.load(os.path.join(inputFieldImages, f"{field[0]}.npy"))
        # get the window coordinates - dont need to check for image edge as should already be handled by ROIFinder
        minX = int(int(field[3]) - (ROISize-1)/2)
        maxX = int(int(field[3]) + (ROISize-1)/2)
        minY = int(int(field[4]) - (ROISize-1)/2)
        maxY = int(int(field[4]) + (ROISize-1)/2)
        
        # cut out the ROI slices through all 
        fieldROI = fullFieldArray[:,:,minX:maxX+1, minY:maxY+1].copy()

        # because images are big
        del fullFieldArray

        # # display one of the individual spectral bands within a temporal band
        # plt.imshow(fieldROI[0][3], cmap = "gray", interpolation="nearest", vmin=0, vmax=10000)
        # plt.axis('off')
        # plt.show()

        # add the ROI with it's metadata to the list for its specific class (fid, field array, x, y, aug type, rot type)
        validFieldTiles[field[2]].append([field[0], fieldROI, field[5], field[6], None, None]) # fifth value is augmentation type and sixth counter-clockwise rotation

        fidDir[str(field[0])] = 0

    return validFieldTiles, fidDir


##### The Noise Functions #####

def applyNoiseOriginal(inputImageLayer):
    """This function exists to allow some rotations to have no noise applied, for balance"""
    outputImageLayer = inputImageLayer
    return outputImageLayer

def applyNoiseGaussian(inputImageLayer):
    # # display the image
    # plt.imshow(inputImageLayer, cmap = "gray", interpolation="nearest",  vmin = 0, vmax = 10000)
    # plt.axis('off')
    # plt.show()

    # convert to float
    multiplier = np.max(inputImageLayer)
    inputImageLayer = inputImageLayer / multiplier
    # apply noise
    outputImageLayer = random_noise(inputImageLayer, "gaussian", clip = True, var = 0.0002)
    # convert back to 1-10000
    outputImageLayer = outputImageLayer * multiplier

    # # display the image
    # plt.imshow(outputImageLayer, cmap = "gray", interpolation="nearest",  vmin = 0, vmax = 10000)
    # plt.axis('off')
    # plt.show()
    return outputImageLayer

def applyNoiseLocalvar(inputImageLayer):
    # # display the image
    # plt.imshow(inputImageLayer, cmap = "gray", interpolation="nearest",  vmin = 0, vmax = 10000)
    # plt.axis('off')
    # plt.show()

    # convert to float
    multiplier = np.max(inputImageLayer)
    inputImageLayer = inputImageLayer / multiplier
    # apply noise
    lvar = filters.gaussian(inputImageLayer, sigma=1)
    outputImageLayer = random_noise(inputImageLayer, "localvar", local_vars = lvar * 0.002)
    # convert back to 1-10000
    outputImageLayer = outputImageLayer * multiplier

    # # display the image
    # plt.imshow(outputImageLayer, cmap = "gray", interpolation="nearest",  vmin = 0, vmax = 10000)
    # plt.axis('off')
    # plt.show()

    return outputImageLayer

def applyNoisePoission(inputImageLayer):
    # # display the image
    # plt.imshow(inputImageLayer, cmap = "gray", interpolation="nearest",  vmin = 0, vmax = 10000)
    # plt.axis('off')
    # plt.show()

    # convert to float
    multiplier = np.max(inputImageLayer)
    inputImageLayer = inputImageLayer / multiplier
    # apply noise
    outputImageLayer = random_noise(inputImageLayer, "poisson", clip = True)
    # convert back to 1-10000
    outputImageLayer = outputImageLayer * multiplier

    # # display the image
    # plt.imshow(outputImageLayer, cmap = "gray", interpolation="nearest",  vmin = 0, vmax = 10000)
    # plt.axis('off')
    # plt.show()
    return outputImageLayer

def applyNoiseSalt(inputImageLayer):
    # # display the image
    # plt.imshow(inputImageLayer, cmap = "gray", interpolation="nearest",  vmin = 0, vmax = 10000)
    # plt.axis('off')
    # plt.show()

    # convert to float
    multiplier = np.max(inputImageLayer)
    inputImageLayer = inputImageLayer / multiplier
    # apply noise
    outputImageLayer = random_noise(inputImageLayer, "salt", amount = 0.06)
    # convert back to 1-10000
    outputImageLayer = outputImageLayer * multiplier

    # # display the image
    # plt.imshow(outputImageLayer, cmap = "gray", interpolation="nearest",  vmin = 0, vmax = 10000)
    # plt.axis('off')
    # plt.show()
    return outputImageLayer

def applyNoisePepper(inputImageLayer):
    # # display the image
    # plt.imshow(inputImageLayer, cmap = "gray", interpolation="nearest",  vmin = 0, vmax = 10000)
    # plt.axis('off')
    # plt.show()

    # convert to float
    multiplier = np.max(inputImageLayer)
    inputImageLayer = inputImageLayer / multiplier
    # apply noise
    outputImageLayer = random_noise(inputImageLayer, "pepper", amount = 0.02)
    # convert back to 1-10000
    outputImageLayer = outputImageLayer * multiplier

    # # display the image
    # plt.imshow(outputImageLayer, cmap = "gray", interpolation="nearest",  vmin = 0, vmax = 10000)
    # plt.axis('off')
    # plt.show()
    return outputImageLayer

def applyNoiseSaltAndPepper(inputImageLayer):
    # # display the image
    # plt.imshow(inputImageLayer, cmap = "gray", interpolation="nearest",  vmin = 0, vmax = 10000)
    # plt.axis('off')
    # plt.show()

    # convert to float
    multiplier = np.max(inputImageLayer)
    inputImageLayer = inputImageLayer / multiplier
    # apply noise
    outputImageLayer = random_noise(inputImageLayer, "s&p", amount = 0.08, salt_vs_pepper = 0.75)
    # convert back to 1-10000
    outputImageLayer = outputImageLayer * multiplier

    # # display the image
    # plt.imshow(outputImageLayer, cmap = "gray", interpolation="nearest",  vmin = 0, vmax = 10000)
    # plt.axis('off')
    # plt.show()
    return outputImageLayer

def applyNoiseSpeckle(inputImageLayer):
    # # display the image
    # plt.imshow(inputImageLayer, cmap = "gray", interpolation="nearest",  vmin = 0, vmax = 10000)
    # plt.axis('off')
    # plt.show()

    # convert to float
    multiplier = np.max(inputImageLayer)
    inputImageLayer = inputImageLayer / multiplier
    # apply noise
    outputImageLayer = random_noise(inputImageLayer, "speckle", clip = True, var = 0.0002)
    # convert back to 1-10000
    outputImageLayer = outputImageLayer * multiplier

    # # display the image
    # plt.imshow(outputImageLayer, cmap = "gray", interpolation="nearest",  vmin = 0, vmax = 10000)
    # plt.axis('off')
    # plt.show()
    return outputImageLayer

def applyNoiseUniform(inputImageLayer):
    # # display the image
    # plt.imshow(inputImageLayer, cmap = "gray", interpolation="nearest",  vmin = 0, vmax = 10000)
    # plt.axis('off')
    # plt.show()

    inputMin = np.min(inputImageLayer)
    inputMax = np.max(inputImageLayer)
    uniformNoise = np.random.uniform(inputMin, inputMax, np.shape(inputImageLayer))

    # shift the noise to be equally below and above the original image
    uniformNoise = uniformNoise - (inputMin + inputMax)/2

    # apply the noise
    outputImageLayer = inputImageLayer + uniformNoise

    # # display the image
    # plt.imshow(outputImageLayer, cmap = "gray", interpolation="nearest",  vmin = 0, vmax = 10000)
    # plt.axis('off')
    # plt.show()
    return outputImageLayer

##################################################


def main():
    """Taking input tile centre locations, field class info, and field arrays, make up one tile per usable field"""

    # get list of valid fields with their crop types and class counts
    classedValidFields, classCounts = getValidFieldList(INPUTROIINFO, INPUTMETADATAPATH, ROISIZE, USEDCLASSES)

    # # debug
    # print(classCounts)
    # exit()

    print("Valid Fields Calculated")

    # start by making up the natural tiles for the valid classes, and a dir of each fid for later augmentation counts
    validFieldTiles, fidDir = generateFieldTiles(INPUTIMAGEDIR, USEDCLASSES, ROISIZE, classedValidFields)

    print("Natural ROI Tiles Generated")

    # most noises generated by https://scikit-image.org/docs/stable/api/skimage.util.html#skimage.util.random_noise
    # generate dictionary of noises to apply to ROIs
    noiseDict = {
        "original": applyNoiseOriginal,
        "gaussian": applyNoiseGaussian,
        "localvar": applyNoiseLocalvar,
        "poission": applyNoisePoission,
        "salt": applyNoiseSalt,
        "pepper": applyNoisePepper,
        "s&p": applyNoiseSaltAndPepper,
        "speckle": applyNoiseSpeckle,
        "uniform": applyNoiseUniform
    }
    # and a similar list of noises 
    noises = ["original", "gaussian", "localvar", "poission", "salt", "pepper", "s&p", "speckle", "uniform"]

    # now to reducing overweight classes to the class target, and increasing underweight classes to the target by augmentation
    for crop in USEDCLASSES:
        # check if crop needs more or less tiles
        numTilesRequired = TILESPERCLASS - len(validFieldTiles[crop])
        if numTilesRequired < 0:
            # overweight, cut down, easy
            validFieldTiles[crop] = validFieldTiles[crop][0:TILESPERCLASS]
        elif numTilesRequired > 0:
            # augmentation required
            # first order by physical position in the dataset, longitude-wise
            validFieldTiles[crop].sort(key=lambda field: field[3])
            currentRotations = 0 # counter-clockwise
            noisePointer = 0 # used to select noise sequentially to spread it across the dataset
            # get number of fields for this crop before adding augmented ones 
            numOriginalFields = len(validFieldTiles[crop])
            for syntheticField in range(1,numTilesRequired+1):
                # pick an origin field to use
                originFieldIndex = syntheticField % numOriginalFields # loops through natural class
                originField = validFieldTiles[crop][originFieldIndex]
                # increment that field's augmentation counter
                fidDir[str(originField[0])] += 1
                # choose a noise
                usedNoise = noises[noisePointer]
                # create a frame for the synthetic field
                synthField = [f"{originField[0]}a{fidDir[str(originField[0])]}", None, originField[2], originField[3], usedNoise, currentRotations * 90]
                # rotate the origin data by current required amount (0 is possible) and place into the frame
                synthField[1] = np.rot90(originField[1].copy(), currentRotations, (2,3))

                # now go through every temporal and spectral band in the image (excluding qa layers) and apply the chosen noise type
                for temporalBand in range(0,len(synthField[1])):
                    for spectralBand in range(0,4):
                        synthField[1][temporalBand][spectralBand] = noiseDict[usedNoise](synthField[1][temporalBand][spectralBand])
                
                # add the synthetic field onto the class
                validFieldTiles[crop].append(synthField)

                # increment type of noise used
                noisePointer += 1
                noisePointer = noisePointer % len(noises)
                
                if syntheticField % (numOriginalFields * len(noises)) == 0: # every time every natural field is used for every type of noise :
                    currentRotations += 1 # rotate all images by a further 90 degrees

        print(f"Cropping or Augmentation of {crop} completed")

    # sanity checking
    print("Outputted Tile Counts:")
    for crop in USEDCLASSES:
        print(f"{crop}: {len(validFieldTiles[crop])}")

    # make sure output tiles dir exists
    if os.path.isdir(OUTPUTTILEPATH):
        print("Tile Output Directory Exists")
    else:
        os.mkdir(OUTPUTTILEPATH)
        print("Tile Output Directory Created")

    # with the tiles created for each crop type, write a metadata file with the info about all fields in, and then write tiles themselves
    outputFilePath = os.path.join(OUTPUTMETADATAPATH, "tileInfo.txt")
    with open(outputFilePath, "w") as outputFile:
        # go through all crop types in sequence
        for crop in USEDCLASSES:
            for field in validFieldTiles[crop]:
                # write the string itself as fid, crop type, centre x coord, centre y coord, noise type, degrees rotated counter-clockwise
                outputFile.write(f"{field[0]}, {crop}, {field[2]}, {field[3]}, {field[4]}, {field[5]}\n")
                # while we're here, do the .npy file too
                outTilePath = os.path.join(OUTPUTTILEPATH, field[0] + ".npy")
                np.save(outTilePath, field[1])

    print("Field Tiles and Metadata Written")


##################################################

if __name__ == "__main__":
    main()
