# object-detection-torch

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/alexdelapaz/object-detection-torch/blob/main/object_detection_torch.ipynb)

<span style="color:blue">some *blue* text</span>.
[analysis.py](https://github.com/alexdelapaz/object-detection-torch/blob/main/analysis.py)

<p align="center" width="100%">
<img src="https://github.com/alexdelapaz/object-detection-torch/blob/main/datasets/xView/sample_analysis/Loss%20Graph.png" width="600">
</p>

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
