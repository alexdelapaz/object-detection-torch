# PyTorch Torchvision Object Detection for Satellite Imagery
## Author [Alex de la Paz](http://www.alexdelapaz.com)

## Neural Network code:
This repository is a containerized environment and a python package I created named `detection` that extends [Torchvision FasterRCNN](https://pytorch.org/vision/stable/models/faster_rcnn.html) for object detection training, analysis, and simple inference as an example.

The code used here can be extended to use other labeled datasets if PyTorch dataset classes are created to ingest the annotations in the format the model (FasterRCNN here) requires.

## Data:
A subset of the [xView Dataset](https://github.com/DIUx-xView/xView1_baseline) categories are used (3 in total) to include {`Small Aircraft`, `Fixed-Wing`, `Cargo Plane`}

`Space complexity considerations` are the batch size and overall dataset sample size training on a personal GPU such as a P100 available in Colab or consumer NVIDIA Geforce graphics cards. A `batch size of 4` is used for the torch dataloader for the example training run and dataset.

`Deep learning network training considerations` are the dataset is small (`420 samples`). It is used as an example dataset. Further training can be extended and done with this set of tools easily. Training the full xView would require multiple GPUs or a very long training run with a P100 to achieve reasonable results. The dataset allows enough examples for the network to converge, but not enough for high accuracy without some weighted sampling or synthetic augmentation of the data.

## The notable `object-detection-torch` folders and files mentioned in the readme.
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


# (Option 1) The code can be opened directly in Colab and commands are setup
#### Open the ipynb included to work in Colab [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/alexdelapaz/object-detection-torch/blob/main/object_detection_torch.ipynb)


# (Option 2) Run the object detection training on Linux:
#### Clone the repo to a workstation that has an NVIDIA enabled GPU, GPU Driver, and NVIDIA container toolkit.

<br/>


## Build the deep learning dockerized environment

### Clone the repo:
- `git clone https://github.com/alexdelapaz/object-detection-torch`

<br/>


### Build and Run the container:

#### Running `run_build.sh`
- set `object-detection-torch` as the current working directory
- enter `./run_build.sh` within the `object-detection-torch folder` to pull and build the container image

#### Running `run_container.sh`
- (the shell script that starts the container passes a reference to the `current working directory` which should be `object-detection-torch`)
- enter `./run_container.sh` to run the object detection training container

<br/>


<hr/>
NOTE: this is a deep learning container intended for computer vision (which is time prohibitive to run on cpu)

- the container can be run with a cpu, the example dataset will take more than a few minutes to perform one epoch of training
- the bash script runs the docker container with `--gpus all`

To run the container on a system that does not have a gpu available the docker arg `--gpus all` can be removed

- `docker run -it --mount type=bind,source="$(pwd)"/,target=/workspace detection`
- use `ctrl+c` to exit the training loop if testing on a cpu becomes time prohibitive
<hr/>

<br/>


## Training the object detection network

### Run the training program within the container:
#### Running `run_training.sh`
- enter `./run_training.sh` to run an the example training strategy (saved in `artifacts/plaines_sgd`)

<br/>


### Running the python training program with custom training parameters:
Perform the same training cycle as the automated bash script `run_training.sh` with `train_custom.py`

#### Running `python3 train_custom.py -d 'artifacts' -n 'planes_sgd' -bs 4 -e 20 -s 5 -lr 0.005`

- `-d` is the directory `artifacts`

- `-n` is the `name` of the experiment and a `subfolder` within `artifacts` that represents all the saved info to organize the `training strategy`

- `-bs` is the batch size (`4` is used and tested to work with an NVIDIA P100 12gb VRAM)

- `-e` 20 epochs were ran for the example training run.

- `-s` is the `save frequency` that evaluation is performed on the test set and the model info is saved.

- `-lr` is the `learning rate` for the optimizer (0.005 is used here for Stochastic Gradient Descent)

- `-o` is the `optimizer` (sgd is used for better generalization, adam is the other option)

<br/>


## Analysis and Detections

<br/>

Run the `analysis.py` and `detections.py` python programs to perform the analysis and inference on the model artifacts stored in the `experiement folder (ex. planed_sgd)` created during a training run:
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