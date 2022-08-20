from detectron2.data.datasets.coco import register_coco_instances
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2 import model_zoo
import os
import torch
# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

from detectron2.structures import BoxMode



register_coco_instances("dataset_train", {}, "./tdt4265_2022/train_annotations.json", "./tdt4265_2022") #get the datasets, easy since thet are in coco format
register_coco_instances("dataset_val", {}, "./tdt4265_2022/val_annotations.json", "./tdt4265_2022")


cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("dataset_train")
cfg.DATASETS.TEST = ("dataset_val")
cfg.DATALOADER.NUM_WORKERS = 0
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # pretrained weights
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025  # 
cfg.SOLVER.MAX_ITER = 300    # 300 iterations
cfg.SOLVER.STEPS = []        # do not decay learning rate #TODO legge til
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128 #(default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 9  
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg) 
trainer.resume_or_load(resume=False)
trainer.train()

cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "./model_final.pth")  # path to the model we just trained
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold
predictor = DefaultPredictor(cfg)

from detectron2.utils.visualizer import ColorMode, Visualizer
dataset_dicts = get_balloon_dicts("balloon/val")
for d in random.sample(dataset_dicts, 1):    
    im = cv2.imread(d["file_name"])
    outputs = predictor(im)
    print("TYPE:", type(outputs))
    #print("OUTPUT ARE:", outputs, type(outputs))
    v = Visualizer(im[:, :, ::-1],
                metadata=balloon_metadata, 
                scale=0.5, 
                instance_mode=ColorMode.IMAGE_BW
    )
    #out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    cv2.imshow("image",out.get_image()[:, :, ::-1])

    cv2.waitKey(0) 
    #closing all open windows 
    cv2.destroyAllWindows() 