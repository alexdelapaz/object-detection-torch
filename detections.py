import torch
import torchvision
import torchvision.transforms as T
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt
import os
import random
import argparse
import sys

from detection import data, models

# adding torchvision vision tools folder to import modules
notebook_folders = ['vision/references/detection']
for folder in notebook_folders:
    sys.path.append(folder)

#sys.path.insert(0, '/vision')


# Define transforms
def get_transform(train):
	transforms = []
	transforms.append(T.ToTensor())

	if train:
		transforms.append(T.RandomHorizontalFlip(0.5))

	return T.Compose(transforms)

# Define image and annotation save function
def save_img_plot(imgs, name, dir_save):
	plt.rcParams["figure.figsize"] = (10,10)
	plt.rcParams["savefig.bbox"] = 'tight'

	if not isinstance(imgs, list):
		imgs = [imgs]
	fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
	for i, img in enumerate(imgs):
		img = img.detach()
		img = F.to_pil_image(img)
		axs[0, i].imshow(np.asarray(img))
		axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

	analysis_path = '{}/{}'.format(dir_save, 'analysis')
	if not os.path.exists(analysis_path):
		os.mkdir(analysis_path)
	plot_name = '{}{}{}'.format('annots_', name,'.png')
	plot_path = '{}/{}'.format(analysis_path, plot_name)

	exp_name = analysis_path.split('/')[-1]
	plt.savefig(plot_path)


if __name__ == '__main__':	
	parser = argparse.ArgumentParser()
	parser.add_argument('-d', '--dir_saved', type = str, required=True, help='Saved model artifacts directory')
	parser.add_argument('-n', '--exp_name', type = str, required=True, help='Experiment name')	

	args_passed = vars(parser.parse_args())

	# Output directory name, default is artifacts
	DIR_OUT = args_passed['dir_saved']
	# Experiment name
	EXP_NAME = args_passed['exp_name']
	SAVE_PATH = '{}/{}'.format(DIR_OUT,EXP_NAME)


	# The same hard coded 20% amount used as the training run to ensure no training data is tested on
	test_percent = 20

	# Select the class count, planes has 3 classes, will be passed to create background 0 + 1,2,3 in category ids for mlp head
	fasterrcnn_class_count = 3
	# The label directory for the dataset used
	dir_coco = 'datasets/xView/labels/'
	# The dataset used
	dataset = 'planes'


	# Import and prepare dataset
	# List[dict] is created from coco annotations to aggregate each annotation per image
	# COCO style list[dict] has one dict per annotation, not one dict for all annotations of an image
	coco_dataset = data.load_coco_torchvision(dataset, dir_coco)
	test_percent /=100
	test_count = int(len(coco_dataset)*test_percent)
	coco_dataset_test = coco_dataset[-test_count:]

	ds = data.xViewDataset(coco_dataset_test, get_transform(train=False))


	# Load FasterRCNN_ResNet50
	model_name, model_loaded = models.load_fasterrcnn(fasterrcnn_class_count)
	# Get model path from experiement directory
	model_path = '{}/{}/{}'.format(DIR_OUT,EXP_NAME, model_name)

	# Prior training weights and information
	# Stored model information
	model_info = torch.load(model_path)
	# Model weights
	best_model_weights = model_info['weights']
	model_loaded.load_state_dict(best_model_weights)

	# model_loaded for loaded in cell directly above
	model_on_gpu = model_loaded.eval()
	model_on_gpu = model_on_gpu.cuda()

	# Randomly select index 
	test_idxs = len(coco_dataset_test)
	img_idx = random.randrange(test_idxs)
	img, target = ds[img_idx]
	img = img.cuda()


	# Prior training run epoch count
	current_epoch = model_info['epochs_trained']
	print('\n-------------------------------------------------------------------------------')
	print('Detections with |{}: {} epochs of training|'.format(model_name, current_epoch))
	print('-------------------------------------------------------------------------------')


	with torch.no_grad():
		predictions = model_on_gpu([img])


	# Return image to range of bit values and conver to unsigned 8 bit integer
	img_label = img*255
	img_label = img_label.type(torch.uint8)

	# trained model saved without scripting or tracing outputs predictions differently
	#labels = [str(lbl.item()) for lbl in predictions[0]['labels'].data]


	# **************** RAW PREDICTIONS ****************
	decoder = {'1': 'Fixed-wing Aircraft', '2': 'Small Aircraft', '3': 'Cargo Plane'}
	#labels = [decoder[label] for label in labels]
	labels = [decoder[str(label.item())] for label in predictions[0]['labels'].data]

	labeled_img_annots = torchvision.utils.draw_bounding_boxes(img_label, predictions[0]['boxes'], labels, colors='#D627D6')
	save_img_plot(labeled_img_annots, 'raw', SAVE_PATH)


    # **************** NMS PREDICTIONS ****************
	detect_threshold = 0.5
	idx_b = torch.where(predictions[0]['scores'] > detect_threshold)

	conf_filtered_boxes = predictions[0]['boxes'][idx_b]

	conf_filtered_labels = [str(label.item()) for label in predictions[0]['labels'][idx_b].data]
	decoder = {'1': 'Fixed-wing Aircraft', '2': 'Small Aircraft', '3': 'Cargo Plane'}
	conf_filtered_labels_names = [decoder[label] for label in conf_filtered_labels]

	labeled_img_annots = torchvision.utils.draw_bounding_boxes(img_label, conf_filtered_boxes, conf_filtered_labels_names, colors='#D627D6')
	save_img_plot(labeled_img_annots, 'thresh_{}'.format(detect_threshold), SAVE_PATH)


	# **************** GROUND TRUTH ****************
	decoder = {'1': 'Fixed-wing Aircraft', '2': 'Small Aircraft', '3': 'Cargo Plane'}
	labels = [str(lbl.item()) for lbl in target['labels'].data]
	labels_ground_truth = [decoder[label] for label in labels]

	labeled_img_annots = torchvision.utils.draw_bounding_boxes(img_label, target['boxes'], labels_ground_truth, colors='#D627D6')

	print('\n-------------------------------------------------------------------------------')
	print('Saving {} artifacts to: {}/{}/'.format('detection', SAVE_PATH, 'analysis'), end='\n')
	print('-------------------------------------------------------------------------------\n')
	save_img_plot(labeled_img_annots, 'ground_truth', SAVE_PATH)