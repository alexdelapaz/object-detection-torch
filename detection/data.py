import pickle
import random
import torch
from PIL import Image
import os

def load_coco_torchvision(subset_class, dir_coco):
  '''
  Load pickle file containing list[dict] of subset of coco dataset by class for torchvision training
  '''

  # example subset_class = 'haul_trucks' => coco_haul_trucks.pkl

  pickle_file = dir_coco + 'coco_' + subset_class + '.pkl'
  with open(pickle_file, 'rb') as f:
   return(pickle.load(f))


def merge_data(annots_real, annots_synth=None, annots_split={'real':100,'synth':0}):
  '''
  Sample from real and synthetic (if passed), concatenate list[dicts], and return result
  '''

  if annots_synth is not None:
    count_real = len(annots_real)
    percent_real = annots_split['real']/100

    count_synth = len(annots_synth)
    percent_synth = annots_split['synth']/100

    count_total = count_real + count_synth

    end_idx_real = int(percent_real * count_real)
    end_idx_synth = int(count_real - end_idx_real)

    annots = annots_real[:end_idx_real] + annots_synth[:end_idx_synth]

  else:
    annots = annots_real

  random.shuffle(annots)
  return annots

class COCO_Dataset(torch.utils.data.Dataset):

  def __init__(self, COCO_input, transforms=None):

    # transforms to be applied to training data, not applied to validation data
    self.transforms = transforms

    # sorted list of image filepaths
    # self.imgs = list(sorted(os.listdir(path_xview_imgs_tiled)))
    # list[dict], depends on detectron2 dataset class, aggregates annotations per image
    self.annots = COCO_input

  def __getitem__(self, idx):

    # return path to image from list[dict] of annotations
    img_path = self.annots[idx]['file_name']
    # if not training, return rgb pil Image, opened from img path
    img = Image.open(img_path).convert("RGB")

    # for box in each annotation per unique image id, convert to x1,y1,x2,y2 format by adding height and width
    annot_boxes = [ [ annot['bbox'][0], annot['bbox'][1], annot['bbox'][0]+annot['bbox'][2], annot['bbox'][1]+annot['bbox'][3] ] for annot in self.annots[idx]['annotations'] ]
    # convert boxes to torch tensor
    boxes = torch.as_tensor(annot_boxes, dtype=torch.float32)

    # get the area
    area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

    # collect labels from category ids in list[dict] annotations
    annot_labels = [int(annot['category_id']) for annot in self.annots[idx]['annotations']]
    # convert labels to torch tensor
    labels = torch.as_tensor(annot_labels, dtype=torch.int64)
    
    # get the image id and convert to torch tensor
    image_id = torch.tensor([self.annots[idx]['image_id']])
    
    # get the iscrowd and convert to torch tensor
    iscrowd = torch.tensor([annot['iscrowd'] for annot in self.annots[idx]['annotations']])


    # create dict annotation to return with labels per image sample
    target = {'boxes': boxes, 'labels': labels, 'image_id': image_id, 'area': area, 'iscrowd': iscrowd}


    if self.transforms is not None:
      # if training data, convert images to tensors
      img = self.transforms(img)

    return img, target

  def __len__(self):
    return len(self.annots)

  def getimgpath(self, idx):
    return os.path.join(self.root,'/',self.annots[idx]['file_name'])



class xViewDataset(torch.utils.data.Dataset):

  def __init__(self, root, transforms=None):
    # transforms to be applied to training data, not applied to validation data
    self.transforms = transforms

    # list[dict], depends on detectron2 dataset class, aggregates annotations per image
    self.annots = root # changed xview_return()
    # or coco_return(path_xview_imgs_tiled, path_xview_lbls_tiled)
    # or custom_return(open(pickled subset haul truck list[dict]))

  def __getitem__(self, idx):

    # return path to image from list[dict] of annotations
    img_path = self.annots[idx]['file_name']
    # if not training, return rgb pil Image, opened from img path
    img = Image.open(img_path).convert("RGB")

    # for box in each annotation per unique image id, convert to x1,y1,x2,y2 format by adding height and width
    annot_boxes = [ [ annot['bbox'][0], annot['bbox'][1], annot['bbox'][0]+annot['bbox'][2], annot['bbox'][1]+annot['bbox'][3] ] for annot in self.annots[idx]['annotations'] ]
    # convert boxes to torch tensor
    boxes = torch.as_tensor(annot_boxes, dtype=torch.float32)

    # collect labels from category ids in list[dict] annotations
    annot_labels = [int(annot['category_id']) for annot in self.annots[idx]['annotations']]
    # convert labels to torch tensor
    labels = torch.as_tensor(annot_labels, dtype=torch.int64)
    
    # get the image id and convert to torch tensor
    image_id = torch.tensor([self.annots[idx]['image_id']])
    
    # create dict annotation to return with labels per image sample
    target = {'boxes': boxes, 'labels': labels, 'image_id': image_id}


    if self.transforms is not None:
      # if training data, convert images to tensors
      img = self.transforms(img)

    return img, target

  def __len__(self):
    return len(self.annots)

  def getimgpath(self, idx):
    return os.path.join(self.root,'/',self.annots[idx]['file_name'])