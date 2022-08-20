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


def countLabels(dataloader, cfg):
    """counting number of each label"""
    numLabels = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    for batch in tqdm(dataloader):
        
        for box in batch["labels"]:            
            for label in box:
                numLabels[label] +=1

    print(numLabels)




def main():
    config_path = "configs/tdt4265.py"
    cfg = get_config(config_path)
    dataset_to_analyze = "train"  # or "val"

    print("Label map is:", cfg.label_map)

    dataloader = get_dataloader(cfg, dataset_to_analyze)
    countLabels(dataloader, cfg)


if __name__ == '__main__':
    main()
