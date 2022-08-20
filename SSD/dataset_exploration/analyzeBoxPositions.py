from cProfile import label
from matplotlib import pyplot as plt
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


def analyzeBoxLowerHalfPositions(dataloader, cfg):
    """counting number of each label"""
    allYMins = []
    totalCounter = 0
    lowerHalfCounter = 0
    for batch in tqdm(dataloader):

        for boxPositions in batch["boxes"]:
            for boxPosition in boxPositions:
                totalCounter+=1
                xmin, ymin, xmax, ymax = boxPosition
                allYMins.append(ymin)
                if (ymin >=0.5): #then the box is on the lower half of the image
                    lowerHalfCounter += 1

    print("TOTALCOUNTER:", totalCounter, "LOWERHALF:", lowerHalfCounter)
    print(lowerHalfCounter/totalCounter)

    sizes = []
    sizes.append(lowerHalfCounter)
    numTopHalf = totalCounter-lowerHalfCounter
    sizes.append(numTopHalf)
    explode = (0.2, 0)
    labels = "Boxes in lower half of picture", "Boxes in top half of picture"

    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
            shadow=True, startangle=90)
    ax1.axis('equal') 
    plt.show()    


def analyzeBoxIn5Positions(dataloader, cfg):
    """counting number of each label"""
    allYMins = []
    totalCounter = 0
    zeroToForty = 0
    FortyToSixty = 0
    SixtyToHundred = 0

    for batch in tqdm(dataloader):

        for boxPositions in batch["boxes"]:
            for boxPosition in boxPositions:
                totalCounter+=1
                xmin, ymin, xmax, ymax = boxPosition
                allYMins.append(ymin)
                if (ymin >= 0.6): #then the box is on the lower half of the image
                    zeroToForty += 1
                elif (ymin>=0.4):
                    FortyToSixty += 1
                else:
                    SixtyToHundred += 1


    sizes = [zeroToForty, FortyToSixty, SixtyToHundred]
    labels = "Number boxes where top point is in lower 40 percent of image", "Number of boxes where top point of image is between 40 and 60 percent of the height of the image", "Number of boxes where the top point is in the top 40 percent of the image"
    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, autopct='%1.1f%%',
            shadow=True, startangle=90)
    ax1.axis('equal') 
    plt.legend(labels = labels)
    plt.show()    

""" #just another way to check how many boxes of each class
def calcHowManyOfEachClass(dataloader, cfg):
    numOfEachClass = [0 for i in range(9)]
    for batch in tqdm(dataloader):
            for labels in batch["labels"]:
                for label in labels:
                    num = numOfEachClass[label]
                    num+=1
                    numOfEachClass[label] = num
    print("NUM OF EACH CLASS:",numOfEachClass)
"""



def main():
    config_path = "configs/tdt4265.py"
    cfg = get_config(config_path)
    dataset_to_analyze = "train"  # or "val"

    print("Label map is:", cfg.label_map)

    dataloader = get_dataloader(cfg, dataset_to_analyze)


    analyzeBoxLowerHalfPositions(dataloader, cfg)
    analyzeBoxIn5Positions(dataloader, cfg)
    #calcHowManyOfEachClass(dataloader, cfg)

if __name__ == '__main__':
    main()
