import torch
import torchvision.transforms as T
import os
import pathlib
import argparse
import sys

# adding torchvision vision tools folder to import modules
notebook_folders = ['vision/references/detection']
for folder in notebook_folders:
    sys.path.append(folder)
#sys.path.insert(0, '/vision')

# Import torchvision vision folder scripts from vision/references/detection
from engine import train_one_epoch, evaluate
import utils

# import light training framework for torch
from detection import data, models, trainer


# Define transforms function
def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())

    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))

    return T.Compose(transforms)


if __name__ == '__main__':	
	parser = argparse.ArgumentParser()
	parser.add_argument('-d', '--dir_out', type = str, required=True, help='Save directory')
	parser.add_argument('-n', '--exp_name', type = str, required=True, help='Experiment name')	
	parser.add_argument('-bs', '--batch_size', type = str, required=True, help='Train batch size')	
	parser.add_argument('-e', '--epochs', type = str, required=True, help='Epoch count')	
	parser.add_argument('-s', '--eval_freq', type = str, required=True, help='Eval frequency (also saves artifacts)')	
	parser.add_argument('-lr', '--learning_rate', type = str, required=True, help='Learning rate')	
	parser.add_argument('-o', '--opt', type = str, required=False, help='Optimizer (sgd or adam)')	

	args_passed = vars(parser.parse_args())
	

	# Output directory name, default is artifacts
	DIR_OUT = args_passed['dir_out']
    # Experiment name
	EXP_NAME = args_passed['exp_name']
	# BATCH_SIZE = 4 works for most 12-16 gb cards, images used in example are 512x512 pixels
	BATCH_SIZE = int(args_passed['batch_size'])


	# Determines test_amount to keep seperate from training data, hard coded as 20% here and used to separate the last 20 samples
	test_percent = 20

    # Select the class count, planes has 3 classes, will be passed to create background 0 + 1,2,3 in category ids for mlp head
	fasterrcnn_class_count = 1
    # The label directory for the dataset used
	dir_coco = 'datasets/xView/labels/'
    # The dataset used
	dataset = 'haul_trucks'#'planes'

    # Import and prepare dataset
    # List[dict] is created from coco annotations to aggregate each annotation per image
    # COCO style list[dict] has one dict per annotation, not one dict for all annotations of an image
	coco_dataset = data.load_coco_torchvision(dataset, dir_coco)
	test_percent /=100
	total_count = len(coco_dataset)
	test_count = int(total_count*test_percent)
	train_count = int(total_count-test_count)
    # Split data into train and test sets, the hard coded 20% samples are used for eval
	coco_dataset_train = coco_dataset[:train_count] # [:2]#
	coco_dataset_test = coco_dataset[-test_count:] # [-2:]#

    # Create torch dataset object for coco annotations, subset of xView coco dataset is used for planes example
	torchvision_ds_train = data.COCO_Dataset(coco_dataset_train, get_transform(train=True))
	torchvision_ds_test = data.COCO_Dataset(coco_dataset_test, get_transform(train=False))

	# define training and validation data loaders
	dl_train = torch.utils.data.DataLoader(
		torchvision_ds_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=2,
		collate_fn=utils.collate_fn)

	dl_test = torch.utils.data.DataLoader(
		torchvision_ds_test, batch_size=1, shuffle=False, num_workers=2,
		collate_fn=utils.collate_fn)

    # Store train,test dataloaders in a tuple to pass to training function
	loaders = (dl_train, dl_test)



	model_path = '{}/{}'.format(DIR_OUT,EXP_NAME)
	if not os.path.exists(model_path):
		os.mkdir(model_path)

	model_name, model = models.load_fasterrcnn(fasterrcnn_class_count)
	model_save = model_path + os.sep + model_name

	print('------------------------------------------------------')
	print('Training: {}'.format(model_name), end ='\n')
	print('\nSaving {} artifacts to: {}'.format(EXP_NAME, model_path), end='\n')
	print('------------------------------------------------------')

	print('Data [Total count]:\t{}'.format(total_count))
	print('Data [Train count]:\t{}'.format(train_count))
	print('Data [Test count] :\t{}'.format(test_count))


	EPOCHS = int(args_passed['epochs'])
	# Save after SAVE_FREQ epochs, 4 was used in example
	SAVE_FREQ = int(args_passed['eval_freq'])
	# Learning rate, lr = 0.005 was used in example
	LR = float(args_passed['learning_rate'])
	# Select optimizer, sgd or adam
	OPT = args_passed['opt']
	# If optimizer name sgd or adam is passed to -o cli arg, then it it passed to trainder.fasterrcnn
	# Default optimizer without passing the argument name is sgd
	if OPT:
		trainer.fasterrcnn(model, model_save, loaders, EPOCHS, SAVE_FREQ, LR, OPT)
	else:
		trainer.fasterrcnn(model, model_save, loaders, EPOCHS, SAVE_FREQ, LR)