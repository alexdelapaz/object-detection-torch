﻿# object-detection-torch
[analysis.py](https://github.com/alexdelapaz/object-detection-torch/blob/main/analysis.py)

<span style="color:blue">some *blue* text</span>.

``![image text](https://github.com/alexdelapaz/object-detection-torch/blob/main/datasets/xView/sample_analysis/Loss%20Graph.png)``
``![image text](https://github.com/alexdelapaz/object-detection-torch/blob/main/datasets/xView/sample_analysis/annots_ground_truth.png)``
``![image text](https://github.com/alexdelapaz/object-detection-torch/blob/main/datasets/xView/sample_analysis/annots_raw.pngg)``
``![image text](https://github.com/alexdelapaz/object-detection-torch/blob/main/datasets/xView/sample_analysis/annots_thresh_0.5.png)``

```
object-detection-torch
│   analysis.py
│   commands.txt
│   detections.py
│   Dockerfile
│   run_training.sh
│   train_custom.py
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
├───datasets
│   └───xView
│       ├───images
│       │   └───train_images_tiled_planes
│       └───labels
│               coco_planes.pkl
└───detection
        data.py
        models.py
        setup.py
        trainer.py
```
