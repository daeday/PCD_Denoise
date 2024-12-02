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

## From gray-scale image(output of segmentation or mask data) To person_only output
# For list of path
def person_class_extractor(img_path_list, person_label=13):
  person_img_list = []
  for img_path in tqdm(img_path_list):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    temp_img = np.zeros(np.shape(img))
    for i, cv in enumerate(img):
      for c, v in enumerate(cv):
        if v == person_label:
          temp_img[i,c] = 1
    person_img_list.append(temp_img)
  return person_img_list

# For each image
def person_extractor(img, person_label=13):
  temp_img = np.zeros(np.shape(img))
  for i, cv in enumerate(img):
    for c, v in enumerate(cv):
      if v == person_label:
        temp_img[i,c] = 1
  return temp_img

## Expected inputs (infered_image(h*w), expected_output_image(h*w))
# mIoU = (Intersection/Union area)
def mIoU(inf_img, exp_img):
  mask_union = np.zeros(np.shape(inf_img))
  mask_inter = np.zeros(np.shape(inf_img))
  for r in range(len(inf_img)):
    for c in range(len(inf_img[r])):
      if inf_img[r][c] != 0 or exp_img[r][c] !=0:
        mask_union[r][c] = 1
      if inf_img[r][c] != 0 and exp_img[r][c] != 0:
        mask_inter[r][c] = 1
  area_union = mask_union.sum()
  area_inter = mask_inter.sum()
  return area_inter/area_union

# Target cover weighted mIoU extractor = ((Intersection/Target area)*(Intersection/Union area))^(1/2)
def weight_mIoU(inf_img, exp_img):
  mask_union = np.zeros(np.shape(inf_img))
  mask_inter = np.zeros(np.shape(inf_img))
  mask_target = np.zeros(np.shape(inf_img))
  for r in range(len(inf_img)):
    for c in range(len(inf_img[r])):
      if inf_img[r][c] != 0 or exp_img[r][c] !=0:
        mask_union[r][c] = 1
      if inf_img[r][c] != 0 and exp_img[r][c] != 0:
        mask_inter[r][c] = 1
      if exp_img[r][c] != 0:
        mask_target[r][c] = 1
  area_union = mask_union.sum()
  area_inter = mask_inter.sum()
  area_target = mask_target.sum()
  # print(area_inter, area_union, area_target)
  return math.sqrt((area_inter/area_union)*(area_inter/area_target))

## Performance checker
def segmentation_performance(input_img_path, exp_imgs, inferencer, person_label, expanded=False, metric='mIoU'):
  if metric == 'mIoU':
    scores = []
    for i in range(len(input_img_path)):
      inf_img = inferencer(input_img_path[i])['predictions']
      inf_img = person_extractor(inf_img, person_label = person_label)
      if expanded:
        inf_img = mask_expansion(inf_img)

      score = mIoU(inf_img, exp_imgs[i])
      scores.append(score)

  elif metric == 'weight_mIoU':
    scores = []
    for i in range(len(input_img_path)):
      inf_img = inferencer(input_img_path[i])['predictions']
      inf_img = person_extractor(inf_img, person_label = person_label)
      if expanded:
        inf_img = mask_expansion(inf_img)

      score = weight_mIoU(inf_img, exp_imgs[i])
      scores.append(score)

  elif metric == 'both':
    scores1 = []
    scores2 = []
    for i in range(len(input_img_path)):
      inf_img = inferencer(input_img_path[i])['predictions']
      inf_img = person_extractor(inf_img, person_label = person_label)
      if expanded:
        inf_img = mask_expansion(inf_img)

      score1 = mIoU(inf_img, exp_imgs[i])
      score2 = weight_mIoU(inf_img, exp_imgs[i])
      scores1.append(score1)
      scores2.append(score2)
      scores = [scores1, scores2]

  return scores

## Mask expansion
def mask_expansion(mask_img, kernel_size=2):
  expanded_mask = np.zeros(np.shape(mask_img))
  for r in range(len(mask_img)):
    for c in range(len(mask_img[r])):

      if mask_img[r][c] != 0:
        for exp_r in range(max(r-kernel_size,0),min(r+kernel_size,len(mask_img))):
          for exp_c in range(max(c-kernel_size,0),min(c+kernel_size,len(mask_img[r]))):
            expanded_mask[exp_r][exp_c] = 1
  return expanded_mask

def active_contour_mask(mask_img, sigma=3):
  h,w = np.shape(mask_img)
  rad = max(h, w) * 0.75

  s = np.linspace(0, 2 * np.pi, 500)
  r = w * 0.5 + rad * np.sin(s)
  c = h + rad * np.cos(s)
  init = np.array([r, c]).T

  snake = active_contour(
      gaussian(mask_img, sigma=sigma, preserve_range=False),
      init,
      alpha=0.01,
      beta=0.1,
      gamma=0.001,
  )

  temp = np.zeros((h, w, 3), dtype=np.uint8)

  snake[:,[1,0]] = snake[:,[0,1]]
  boundary_points = snake.astype(int)
  boundary_points = boundary_points.reshape((-1, 1, 2))

  color = (255, 255, 255)
  cv2.fillPoly(temp, [boundary_points], color)
  output = rgb2gray(temp)
  return output