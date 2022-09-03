FROM pytorch/pytorch:latest

RUN conda update -n base -c defaults conda
RUN conda install -c conda-forge pycocotools