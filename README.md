# object-detection-torch
```
object-detection-torch
│   .gitignore
│   `analysis.py`
│   coco_eval.py
│   coco_utils.py
│   commands.txt
│   `detections.py`
│   Dockerfile
│   engine.py
│   folder_structure.txt
│   README.md
│   `run_training.sh`
│   `train_custom.py`
│   train_simple.py
│   transforms.py
│   utils.py
│
├───artifacts
│   │   .gitkeep
│   │
│   └───planes_sgd
│       │   FasterRCNN_ResNet50
│       │   FasterRCNN_ResNet50.pt
│       │
│       ├───analysis
│       │       annots_ground_truth.png
│       │       annots_raw.png
│       │       annots_thresh_0.5.png
│       │       Loss Graph.png
│       │
│       └───weights_at_eval
│               .gitkeep
│               20_epochs
│
├───datasets
│   └───xView
│       ├───images
│       │   └───train_images_tiled_planes
│       │
│       └───labels
│               coco_planes.pkl
└───detection
        data.py
        models.py
        setup.py
        trainer.py
        trainer_depricated.py
        __init__.py
```
