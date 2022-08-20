from cProfile import label
from cgitb import small
from matplotlib import pyplot as plt
from torch import equal
from tops.config import instantiate, LazyConfig
from ssd import utils
from tqdm import tqdm
import numpy as np

def get_config(config_path):
    cfg = LazyConfig.load(config_path)
    cfg.train.batch_size = 1
    return cfg


def get_dataloader(cfg, dataset_to_visualize):
    if dataset_to_visualize == "train":
        # Remove GroundTruthBoxesToAnchors transform
        cfg.data_train.dataset.transform.transforms = cfg.data_train.dataset.transform.transforms[:-1]
        data_loader = instantiate(cfg.data_train.dataloader)
    else:
        cfg.data_val.dataloader.collate_fn = utils.batch_collate
        data_loader = instantiate(cfg.data_val.dataloader)

    return data_loader


def analyzeHigherVSLower(dataloader, cfg):
    #allYMins = []
    wider = 0
    higher = 0
    equal = 0
    for batch in tqdm(dataloader):

        for boxPositions in batch["boxes"]:
            for boxPosition in boxPositions:
                xmin, ymin, xmax, ymax = boxPosition
                xDiff = (xmax-xmin)*1024
                yDiff = (ymax-ymin)*128
                if (xDiff > yDiff): #box is wider than it is high
                    wider += 1
                elif (yDiff > xDiff):
                    higher += 1
                else:
                    equal += 1

    print("HIGHER:", higher, "WIDER:", wider, "EQUAL:", equal)

def analyzeBoxRatios(dataloader, cfg):
    """counting number of each label"""
    numWiderThanHigher = 0
    numRatioOneTo1Point2 = 0
    numRatioOnePoint2ToPoint4 = 0
    numRatioOnePoint4ToPoint6 = 0
    numRest = 0

    for batch in tqdm(dataloader):

        for boxPositions in batch["boxes"]:
            for boxPosition in boxPositions:
                xmin, ymin, xmax, ymax = boxPosition
                xDiff = (xmax-xmin)*1024
                yDiff = (ymax-ymin)*128
                ratio = yDiff/xDiff 
                if (ratio <= 1): #box is wider than it is high
                    numWiderThanHigher += 1
                elif (ratio <= 1.5):
                    numRatioOneTo1Point2 += 1
                elif (ratio <= 2):
                    numRatioOnePoint2ToPoint4 += 1
                elif (ratio <= 3):
                    numRatioOnePoint4ToPoint6 += 1
                else:
                    numRest += 1

    print("numWiderThanHigher:", numWiderThanHigher, "numRatioOneTo1Point2:", numRatioOneTo1Point2, "numRatioOnePoint2ToPoint4:", numRatioOneTo1Point2, "numRatioOnePoint4ToPoint6", numRatioOnePoint4ToPoint6, "numRest", numRest)
    
    sizes = [numWiderThanHigher, numRatioOneTo1Point2, numRatioOneTo1Point2, numRatioOnePoint4ToPoint6, numRest]
   
    explode = (0, 0, 0, 0, 0.2)
    title = "Ratio of bounding boxes, given by height/width"
    labels = "Boxes with ratio <= 1 (wider than they are high)", "Ratio between 1 and 1.5", "Ratio between 1.5 and 2", "Ratio between 2 and 3", "Ratio higher than 3"

    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, explode=explode, autopct='%1.1f%%',
            shadow=True, startangle=90)
    ax1.axis('equal') 
    plt.legend(labels = labels)
    plt.title = title
    plt.show()    

def analyzeBoxWidthAndHeightWithSpreadPlot(dataloader, cfg):
    """counting number of each label"""
    allWidths = []
    allHeights = []

    for batch in tqdm(dataloader):

        for boxPositions in batch["boxes"]:
            for boxPosition in boxPositions:
                xmin, ymin, xmax, ymax = boxPosition
                xDiff = (xmax-xmin)*1024 # = width
                yDiff = (ymax-ymin)*128 # = height
                allWidths.append(xDiff)
                allHeights.append(yDiff)

    print("SM:", sum(allHeights)/sum(allWidths)) 
    plt.scatter(allWidths, allHeights, alpha=0.2, s=0.7)
    plt.xlabel("WIDTH")
    plt.ylabel("HEIGHT")
    plt.title("PLOT OF WIDTH AND HEIGH OF BOX SHAPES IN TRAIN DATASET")
    plt.show()

def analyzeBoxRatiosWithBoxPlot(dataloader, cfg):
    """counting number of each label"""
    ratios = []

    for batch in tqdm(dataloader):

        for boxPositions in batch["boxes"]:
            for boxPosition in boxPositions:
                xmin, ymin, xmax, ymax = boxPosition
                xDiff = (xmax-xmin)*1024 # = width
                yDiff = (ymax-ymin)*128 # = height
                ratio = yDiff/xDiff
                ratios.append(ratio.item()) #changing to a float, so it plots right

      
    fig = plt.figure(figsize =(10, 7))

    
    plt.boxplot(ratios, whis=10, meanline=True)
    low = round(np.percentile(ratios, 0), 2)
    firstPercentile = round(np.percentile(ratios, 25), 2)
    secondPercentile = round(np.percentile(ratios, 50), 2)
    thirdPercentile = round(np.percentile(ratios, 75), 2)
    high = round(np.percentile(ratios, 100), 2)

    font = {'family' : 'normal',
            'weight' : 'bold',
            'size'   : 18}

    plt.rc('font', **font)
    plt.xlabel(f"0th PERCENTILE: {low}, 25th PERCENTILE: {firstPercentile}, 50th PERCENTILE: {secondPercentile}, 75th PERCENTILE: {thirdPercentile}, 100th percentile: {high}",  fontsize=14)
    plt.ylabel("RATIO", fontsize=18)
    plt.title("PLOT OF RATIOS OF BOXES TRAIN DATASET")
    plt.grid()
    plt.show() 
    

def analyzeBoxRatiosForDifferentSizesWithBoxPlot(dataloader, cfg):
    ratios = []
    for batch in tqdm(dataloader):

        for boxPositions in batch["boxes"]:
            for boxPosition in boxPositions:
                xmin, ymin, xmax, ymax = boxPosition
                xDiff = (xmax-xmin)*1024 # = width
                yDiff = (ymax-ymin)*128 # = height
                ratio = yDiff/xDiff
                ratios.append(ratio.item()) #changing to a float, so it plots right

    ratios.sort()
    r1, r2, r3 = np.array_split(np.array(ratios), 3)
    ratios = [r1, r2, r3]
    
    print("r1 is:", r1.mean(), r1.min(), r1.max())
    print("r2 is:", r2.mean(), r2.min(), r2.max())
    print("r3 is:", r3.mean(), r3.min(), r3.max())
    #fi, ax = plt.figure(figsize =(10, 7))
    fig, ax = plt.subplots()
    ax.boxplot(ratios)


    font = {'family' : 'normal',
            'weight' : 'bold',
            'size'   : 12}

    plt.rc('font', **font)
    plt.xlabel(f"SMALL BOXES TO THE LEFT, MEDIUM IN MIDDLE, LARGE TO THE RIGHT. JUST DIVIDED ALL BOX SIZES INTO 3 AND PUT THE SMALLEST 3rd AS SMALL")
    plt.ylabel("RATIO", fontsize=18)
    plt.title("PLOT OF RATIOS OF BOXES TRAIN DATASET")
    plt.grid()
    labels = "smallest 3rd of boxe sizes", "middle 3rd of box sizes", "largest 3rd of box sizes"
    plt.legend(labels = labels)
    plt.show() 

def averageWidthAndAverageHeightForSmallMediumLargeBoxes(dataloader, cfg):
    allWidths = []
    allHeights = []
    for batch in tqdm(dataloader):

        for boxPositions in batch["boxes"]:
            for boxPosition in boxPositions:
                xmin, ymin, xmax, ymax = boxPosition
                xDiff = (xmax-xmin)*1024 # = width
                yDiff = (ymax-ymin)*128 # = height
                allWidths.append(xDiff)
                allHeights.append(yDiff)
    allWidths.sort()
    allHeights.sort()

    smallWidth, mediumWidth, largeWidth = np.array_split(np.array(allWidths), 3)
    smallHeight, mediumHeight, largeHeight = np.array_split(np.array(allHeights), 3)
    
    print("Average height of smallest third of boxes:", smallHeight.mean())
    print("Average width of smallest third of boxes:", smallWidth.mean())
    print("Average height of medium third of boxes:", mediumHeight.mean())
    print("Average width of medium third of boxes:", mediumWidth.mean())
    print("Average height of large third of boxes:", largeHeight.mean())
    print("Average width of large third of boxes:", largeWidth.mean())


def analyseSmallMediumLargeBoxRatio(dataloader, cfg):
    smallBoxRatios = [] #small box is under 16x16 px size
    mediumBoxRatios = [] #between 16x16 and 96x96
    largeBoxRatios = [] #over 96x96

    for batch in tqdm(dataloader):
    
        for boxPositions in batch["boxes"]:
            for boxPosition in boxPositions:
                xmin, ymin, xmax, ymax = boxPosition
                xDiff = (xmax-xmin)*1024 # = width
                yDiff = (ymax-ymin)*128 # = height
                ratio = yDiff/xDiff
                size = xDiff*yDiff
                if (size < 16*16):
                    smallBoxRatios.append(ratio)
                elif (size < 96*96):
                    mediumBoxRatios.append(ratio)
                else:
                    largeBoxRatios.append(ratio)

    smallBoxRatios, mediumBoxRatios, largeBoxRatios = np.array(smallBoxRatios), np.array(mediumBoxRatios), np.array(largeBoxRatios)
    ratios = [smallBoxRatios, mediumBoxRatios, largeBoxRatios]

    print(f"smallbox has {len(smallBoxRatios)} boxes, and is:", smallBoxRatios.mean(), smallBoxRatios.min(), smallBoxRatios.max())
    print(f"mediumbox has {len(mediumBoxRatios)} boxes, andis:", mediumBoxRatios.mean(), mediumBoxRatios.min(), mediumBoxRatios.max())
    print(f"large box has {len(largeBoxRatios)} boxes, and is:", largeBoxRatios.mean(), largeBoxRatios.min(), largeBoxRatios.max())
    fig, ax = plt.subplots()
    ax.boxplot(ratios)

    font = {'family' : 'normal',
            'weight' : 'bold',
            'size'   : 12}

    plt.rc('font', **font)
    #plt.xlabel(f"SMALL BOXES TO THE LEFT, MEDIUM IN MIDDLE, LARGE TO THE RIGHT")
    plt.ylabel("RATIO", fontsize=18)
    plt.title("PLOT OF RATIOS OF BOXES VAL DATASET")
    plt.grid()
    labels = "Small boxes to the left", "Medium boxes in middle", "Large boxes to the right"
    plt.legend(labels=labels)
    plt.show() 


def main():
    config_path = "configs/tdt4265.py"
    cfg = get_config(config_path)
    dataset_to_analyze = "val"  # or "val"

    print("Label map is:", cfg.label_map)

    dataloader = get_dataloader(cfg, dataset_to_analyze)


    #analyzeHigherVSLower(dataloader, cfg)
    #analyzeBoxRatios(dataloader, cfg)
    #analyzeBoxWidthAndHeightWithSpreadPlot(dataloader, cfg)
    #analyzeBoxRatiosWithBoxPlot(dataloader, cfg)
    #analyzeBoxRatiosForDifferentSizesWithBoxPlot(dataloader, cfg)
    #averageWidthAndAverageHeightForSmallMediumLargeBoxes(dataloader, cfg)
    analyseSmallMediumLargeBoxRatio(dataloader, cfg)


if __name__ == '__main__':
    main()
