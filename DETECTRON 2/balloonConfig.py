from balloon import *

from detectron2.engine import DefaultTrainer

cfg = get_cfg()
shouldEvaluate = True
shouldTrain = True
if (shouldTrain):
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("balloon_train",)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 0
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
    cfg.SOLVER.MAX_ITER = 300    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
    cfg.SOLVER.STEPS = []        # do not decay learning rate
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
    # NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg) 
    trainer.resume_or_load(resume=True)
    trainer.train()



if (shouldEvaluate):
    
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