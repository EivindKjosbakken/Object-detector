from detectron2.data.datasets.coco import register_coco_instances
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2 import model_zoo
import os

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