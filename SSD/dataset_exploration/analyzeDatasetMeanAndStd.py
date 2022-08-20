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


def analyzeMeanAndStdOfDataset(dataloader, cfg):
    """calculates mean and std for dataset"""
    import numpy as np
    a, b, c = [], [], []
    for batchNum, batch in enumerate(tqdm(dataloader)):
        img = np.array(batch["image"]).squeeze() #remove useless batch dimension
        #print("BATCH:", batchNum)
        
        for idx, eachLayer in enumerate(img): #3 layers
            if (idx == 0):
                a.extend(eachLayer)
            elif (idx == 1):
                b.extend(eachLayer)
            elif (idx == 2):
                c.extend(eachLayer)
            else:
                raise Exception("noe feil")
        
    print(np.mean(np.array(a)), np.mean(np.array(b)), np.mean(np.array(c)))
    print(np.std(np.array(a)), np.std(np.array(b)), np.std(np.array(c)))

def analyzeMeanAndStdOfExtendedDataset(dataloader, cfg):
    """calculates mean and std for dataset (since the dataset is big, it calculates the mean of the means of each half of the dataset)"""
    import numpy as np
    a, b, c = [], [], []
    for batchNum, batch in enumerate(tqdm(dataloader)):
        img = np.array(batch["image"]).squeeze() #remove useless batch dimension
        #print("BATCH:", batchNum)
        for idx, eachLayer in enumerate(img): #3 layers
            if (idx == 0):
                a.extend(eachLayer)
            elif (idx == 1):
                b.extend(eachLayer)
            elif (idx == 2):
                c.extend(eachLayer)
            else:
                raise Exception("noe feil")
        if batchNum == 8000:
            aMean1, bMean1, cMean1 = np.mean(np.array(a)), np.mean(np.array(b)), np.mean(np.array(c))
            aStd1, bStd1, cStd1 = np.std(np.array(a)), np.std(np.array(b)), np.std(np.array(c))
            aMean2, bMean2, cMean2 = np.mean(np.array(a)), np.mean(np.array(b)), np.mean(np.array(c))
            aStd2, bStd2, cStd2 = np.std(np.array(a)), np.std(np.array(b)), np.std(np.array(c))
            a,b,c = [], [], []

    aMean2, bMean2, cMean2 = np.mean(np.array(a)), np.mean(np.array(b)), np.mean(np.array(c))
    aStd2, bStd2, cStd2 = np.std(np.array(a)), np.std(np.array(b)), np.std(np.array(c))

    aMean = (aMean1+aMean2)/2
    bMean = (bMean1+bMean2)/2
    cMean = (cMean1+cMean2)/2
    aStd = (aStd1+aStd2)/2
    bStd = (bStd1+bStd2)/2
    cStd = (cStd1+cStd2)/2
    print("MEANS:", aMean, bMean, cMean)
    print("Stds:", aStd, bStd, cStd)


def main():
    config_path = "configs/task2_2_only_horizontalFlip.py" #change this to configs/tdt4265.py to calculate on original dataset
    cfg = get_config(config_path)
    dataset_to_analyze = "train"  # or "val"

    print("Label map is:", cfg.label_map)

    dataloader = get_dataloader(cfg, dataset_to_analyze)
    #analyzeMeanAndStdOfDataset(dataloader, cfg)
    analyzeMeanAndStdOfExtendedDataset(dataloader, cfg)


if __name__ == '__main__':
    main()
