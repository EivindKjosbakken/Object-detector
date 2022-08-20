# if your dataset is in COCO format, this cell can be replaced by the following three lines:
from typing import Dict, List
from detectron2.data.datasets import register_coco_instances
# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random
import sys


# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog


register_coco_instances("dataset_train", {}, "./tdt4265_2022/train_annotations.json", "./tdt4265_2022") #get the datasets, easy since thet are in coco format
register_coco_instances("dataset_val", {}, "./tdt4265_2022/val_annotations.json", "./tdt4265_2022")





from detectron2.data import DatasetCatalog

# later, to access the data:
dataDict: List[Dict] = DatasetCatalog.get("dataset_train")

""" #show example image 
im = cv2.imread("./tdt4265_2022\images/train/trip007_glos_Video00009_86.png")
cv2.imshow("image",im)
cv2.waitKey(0) 
#closing all open windows 
cv2.destroyAllWindows() 
"""

trainMetaData = MetadataCatalog.get("dataset_train")
valMetaData = MetadataCatalog.get("dataset_val")

""" 
img = cv2.imread("./tdt4265_2022/images/train/trip007_glos_Video00000_0.jpg")
visualizer = Visualizer(img[:, :, ::-1], metadata=trainMetaData, scale=0.5)
out = visualizer.draw_dataset_dict(d)
cv2.imshow(out.get_image()[:, :, ::-1])
"""


#"""
def showImagesWithAnnotations(numberOfImages : int, dataset : str):
    if dataset.lower() == "train":
        dataDict: List[Dict] = DatasetCatalog.get("dataset_train")
    elif dataset.lower() == "val":
        dataDict: List[Dict] = DatasetCatalog.get("dataset_val")
    else:
        raise Exception("Have to choose either 'train' or 'val'")

    dataset_dicts = dataDict
    for d in random.sample(dataset_dicts, numberOfImages):
        print("D IS:", d)
        print("VALUE IS:", d["file_name"])
    
        img = cv2.imread(d["file_name"])
        visualizer = Visualizer(img[:, :, ::-1], metadata=trainMetaData, scale=0.5)
        out = visualizer.draw_dataset_dict(d)
        cv2.imshow("image", out.get_image()[:, :, ::-1])
        cv2.waitKey(0) 
        #closing all open windows 
        cv2.destroyAllWindows()  #must have these to view locally in cv2 (in google colab i wouldnt need them)
#"""
