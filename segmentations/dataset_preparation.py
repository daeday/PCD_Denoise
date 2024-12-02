import torch.optim
from mmseg.apis import MMSegInferencer

from glob import glob
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2
from tqdm import tqdm
import math

from skimage.color import rgb2gray
# from skimage import data
from skimage.filters import gaussian
from skimage.segmentation import active_contour
from skimage.util import invert

from utils import *


img_train_path = "ADEChallengeData2016/images/training/*.*"
img_val_path = "ADEChallengeData2016/images/validation/*.*"

annot_train_path = "ADEChallengeData2016/annotations/training/*.*"
annot_val_path = "ADEChallengeData2016/annotations/validation/*.*"

annot_val_path_ls = glob(annot_val_path)
annot_val_path_ls[0]

file_name = './drive/MyDrive/Data/ADE20K/person_validation.txt'

person_validation_txtpath = 'VOCdevkit/VOC2012/ImageSets/Main/person_val.txt' ## Classification set for person in image
VOC_image_folder = 'VOCdevkit/VOC2012/JPEGImages/*.*'
VOC_seg_folder = 'VOCdevkit/VOC2012/SegmentationClass/*.*'

VOC_imgpath_root = 'VOCdevkit/VOC2012/JPEGImages/'
VOC_maskpath_root = 'VOCdevkit/VOC2012/SegmentationClass/'

# ## Extract image indexes that include human objects as segmentation masks
# person_list = []
# for idx, im_path in tqdm(enumerate(annot_val_path_ls)):
#   img = cv2.imread(im_path)
#   if 13 in set(img.flatten()):
#     person_list.append(im_path)

# ## Record the image indexes of validation images with 'person' class
# file_name = './drive/MyDrive/Data/ADE20K/person_validation.txt'
# with open(file_name, 'w+') as file:
#   file.write('\n'.join(person_list))

# # with open(file_name, 'w') as file:
# #   for num in person_list:
# #     file.write(str(num) + "\n")


## ADE20K - 'Person' 13
## Read the text record
person_list2 = []
with open(file_name, "r") as file:
  for i in file:
    person_list2.append(i.strip().split('validation/')[1].split('.')[0])

## Pick only selected images
ADE_impaths = []
for idx, im_path in enumerate(glob(img_val_path)):
  if im_path.split('validation/')[1].split('.')[0] in person_list2:
    ADE_impaths.append(im_path)

## Pick only selected annotations
annot_img = []
for idx, im_path in enumerate(glob(annot_val_path)):
  if im_path.split('validation/')[1].split('.')[0] in person_list2:
    annot_img.append(im_path)

ADE_impaths.sort()
annot_img.sort()

ADE_expected = person_class_extractor(annot_img)


##PASCAL VOC - Person '147'
## Take image index that we have to focus
## Check whether person class image of 'JPEGImages' is in 'SegmentationClass'
VOC_image_paths = glob(VOC_image_folder)
VOC_seg_paths = [path.split('Class/')[1].split('.png')[0] for path in glob(VOC_seg_folder)]

with open(person_validation_txtpath, 'r') as f:
  lines = f.readlines()

## Extract only human
person_image_index = []
for line in lines:
  if int(line.split()[-1]) == 1:
    person_image_index.append(line.split()[0])

VOC_person_impath = []
for im_path in VOC_image_paths:
  if im_path.split('Images/')[1].split('.jpg')[0] in person_image_index:  ## check whether 'person' class in image
    if im_path.split('Images/')[1].split('.jpg')[0] in VOC_seg_paths:  ## check whether 'person' class image has segmentation label
      VOC_person_impath.append(im_path.split('Images/')[1].split('.jpg')[0])

## Take path of image and segmentation masks
VOC_imgpaths = [VOC_imgpath_root+pths+'.jpg' for pths in VOC_person_impath]
VOC_maskpaths = [VOC_maskpath_root+pths+'.png' for pths in VOC_person_impath]

## Take expected outputs
VOC_expected = person_class_extractor(VOC_maskpaths, person_label=147)


