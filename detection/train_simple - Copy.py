import torchvision.transforms as T
from engine import train_one_epoch, evaluate
import utils
import os
import pathlib

from detection import data, models, trainer

# current working directory
dir_working = pathlib.Path().absolute()
print(dir_working)
# determing test_amount to keep seperate from training data
test_count = 20

dir_coco = 'datasets/xView/labels/'

xview_haul_trucks = data.load_coco_torchvision('haul_trucks', dir_coco)
xview_haul_trucks_train = xview_haul_trucks[:-test_count]
xview_haul_trucks_test = xview_haul_trucks[-test_count:]

print(len(xview_haul_trucks))
print(len(xview_haul_trucks_train))
print(len(xview_haul_trucks_test))

xview_train = data.split_data(xview_haul_trucks_train) # split_data(annots_real, annots_synth=None, annots_split={'real':100,'synth':0})
xview_test = xview_haul_trucks_test

print(len(xview_train))
print(len(xview_test))

def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())

    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))

    return T.Compose(transforms)


cwd = os.getcwd() 
print(cwd)

experiment = 'test'
dir_out = './test' #os.path.join('.', experiment)


torchvision_ds_train = data.COCO_Dataset(xview_train, get_transform(train=True))
torchvision_ds_test = data.COCO_Dataset(xview_test, get_transform(train=False))

model_name, model = models.load_fasterrcnn(1)

print(model_name)

trainer.train(torchvision_ds_train, torchvision_ds_test, model, 1, dir_out)