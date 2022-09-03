﻿# PyTorch Torchvision Object Detection for Satellite Imagery
## Author [Alex de la Paz](http://www.alexdelapaz.com)

#### The code can be opened directly in Colab and commands are setup [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/alexdelapaz/object-detection-torch/blob/main/object_detection_torch.ipynb)

#### Additionally pulling the repo to a workstation that has an NVIDIA enabled GPU, GPU Driver, and NVIDIA container toolkit.
- enter `./run_build.sh` within the `object-detection-torch folder` to pull and build the container image
- enter `./run_training.sh` to run an automated training detailed below

### Training
If you wish to manually run the python program to perform the same training cycle as the automated bash script here is an example:
- `python3 train_custom.py -d 'artifacts' -n 'planes_sgd' -bs 4 -e 20 -s 5 -lr 0.005`


### This repository is a light framework for object detection using [ Torchvision FasterRCNN](https://pytorch.org/vision/stable/models/faster_rcnn.html)

The code used here can be extended to use other labeled datasets if PyTorch dataset classes are created to ingest the annotations in the format the model (FasterRCNN here) requires.

A subset of xView categories are used to include {'Small Aircraft', 'Fixed-Wing', 'Cargo Plane'}

Considerations for the example dataset are training on a personal GPU such as a V100 available in Colab or consumer NVIDIA Geforce graphics cards.

### Analysis and Detections
If you wish to manually run the python programs to perform the analysis and inference as the automated bash script here are examples:
- `python3 analysis.py -d 'artifacts' -n planes_sgd`
- `python3 detections.py -d 'artifacts' -n planes_sgd`

#### Running `analysis.py`
- will save a matplotlib plot of the `training` and `validation` losses produced by the final epoch of training
- using `-d 'artifacts'` `-n 'planes_sgd'` to pass the `artifacts folder` and the `planes_sgd` experiment

<p align="center" width="100%">
<img src="https://github.com/alexdelapaz/object-detection-torch/blob/main/datasets/xView/sample_analysis/Loss%20Graph.png" width="600">
</p>

#### Running `detections.py`
- will save the ground truth, raw detection, and a confidence score filtered detection on a random image
- using `-d 'artifacts'` `-n 'planes_sgd'` to pass the `artifacts folder` and the `planes_sgd` experiment

<div id="image-table">
    <table>
            <th>Ground Truth</th>
            <th>Raw detection</th>
            <th>Score confidence threshold of 0.5</th>
	    <tr>
    	    <td style="padding:10px">
        <img src="https://github.com/alexdelapaz/object-detection-torch/blob/main/datasets/xView/sample_analysis/annots_ground_truth.png" width="300"/>
      	    </td>
            <td style="padding:10px">
        <img src="https://github.com/alexdelapaz/object-detection-torch/blob/main/datasets/xView/sample_analysis/annots_raw.png" width="300"/>
            </td>
            <td style="padding:10px">
        <img src="https://github.com/alexdelapaz/object-detection-torch/blob/main/datasets/xView/sample_analysis/annots_thresh_0.5.png" width="300"/>
            </td>
        </tr>
    </table>
</div>

The main folder when using git to pull the repo is `object-detection-torch`
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
