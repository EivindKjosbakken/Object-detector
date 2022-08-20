from tops.config import instantiate, LazyConfig
from ssd import utils
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

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


def analyzeBoxSizes(dataloader, cfg):
    boxSizes = []
    numSmallBoxes = 0
    boxesInEachImage = []
    c = 0
    #this analyzes the boxsize, and amount of small boxes (<100 px), also plots the box sizes
    for batch in tqdm(dataloader):
        
        for box in batch["boxes"][0]:
            boxSize = calculateBoxSizeInPx(box[0], box[1], box[2], box[3])
            boxSizes.append(boxSize) 
            if (boxSize < 100):
                numSmallBoxes+=1
    
    print("num small boxes: ", numSmallBoxes)
    boxSizes = np.array(boxSizes)
    #print("boxSizes: ", len(boxSizes))
    #print(np.average(boxSizes))
    #print("max: ", np.max(boxSizes))
    #print("min: ", np.min(boxSizes))

    print(len(boxSizes))
    sizes = [0, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
    plt.hist(boxSizes, bins=sizes, rwidth=0.8,alpha=0.9)
    plt.ylabel("Number of boxes")
    plt.xlabel("Pixel size")
    plt.title("Box sizes from val set")
    plt.show()  
   


def analyseNumSmallMediumLargeBoxes(dataloader, cfg):
    smallBox = 0 #small box is under 16x16 px size
    mediumBox = 0 #between 16x16 and 96x96
    largeBox = 0 #over 96x96

    for batch in tqdm(dataloader):
    
        for boxPositions in batch["boxes"]:
            for boxPosition in boxPositions:
                xmin, ymin, xmax, ymax = boxPosition
                xDiff = (xmax-xmin)*1024 # = width
                yDiff = (ymax-ymin)*128 # = height
                ratio = yDiff/xDiff
                size = xDiff*yDiff
                if (size < 16*16):
                    smallBox += 1
                elif (size < 96*96):
                    mediumBox += 1
                else:
                    largeBox += 1
    totalNumBoxes = smallBox + mediumBox + largeBox
    print(f"There are {smallBox} small boxes in the original dataset, that is {smallBox*100/totalNumBoxes}% of all boxes")
    print(f"There are {mediumBox} medium boxes in the original dataset, that is {mediumBox*100/totalNumBoxes}% of all boxes")
    print(f"There are {largeBox} large boxes in the original dataset, that is {largeBox*100/totalNumBoxes}% of all boxes")



def calculateBoxSizeInPx(xmin, ymin, xmax, ymax):
    x = (xmax-xmin)*1024 #since picture is 1024 px in x-direction
    y = (ymax-ymin)*128 #since picture in 128 px in y-direction
    return x*y


def main():
    config_path = "configs/tdt4265.py"
    cfg = get_config(config_path)
    dataset_to_analyze = "val"  # or "val"

    print("Label map is:", cfg.label_map)

    dataloader = get_dataloader(cfg, dataset_to_analyze)


    #analyzeBoxSizes(dataloader, cfg)
    analyseNumSmallMediumLargeBoxes(dataloader, cfg)

if __name__ == '__main__':
    main()
