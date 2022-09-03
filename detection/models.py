import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

def load_fasterrcnn(num_classes):

	# load a model pre-trained on COCO
	model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
	model_name = 'FasterRCNN_ResNet50'

	# 0 is background, all integers accumulation after represents class count
	num_classes = num_classes+1

	# classifier input features
	in_features = model.roi_heads.box_predictor.cls_score.in_features

	# replace output head
	model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

	return model_name, model