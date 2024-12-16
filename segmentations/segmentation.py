import torch.optim
from mmseg.apis import MMSegInferencer

from glob import glob
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2
from tqdm import tqdm
import math
import pandas as pd

# from skimage.color import rgb2gray
# from skimage import data
# from skimage.filters import gaussian
# from skimage.segmentation import active_contour
# from skimage.util import invert

from segmentations.utils import *

import argparse

# ## Take parameters
# parser = argparse.ArgumentParser()
# parser.add_argument("--model", default='BEiT_ADE')
# parser.add_argument("--path", required=True) 

# args = parser.parse_args()

def segmentation_img(img_path, args):
    model_list= {'DeepLab_ADE':'deeplabv3plus_r101-d8_4xb4-160k_ade20k-512x512',
                'DeepLab_VOC':'deeplabv3plus_r101-d8_4xb4-40k_voc12aug-512x512',
                'BEiT_ADE':'beit-large_upernet_8xb1-amp-160k_ade20k-640x640',
                'SAN_COCO':'san-vit-l14_coco-stuff164k-640x640',
                'Segformer_ADE':'segformer_mit-b5_8xb2-160k_ade20k-640x640'}
    image_path = img_path
    model_name = args.model
    
    inferencer = MMSegInferencer(model=model_list[model_name])
    inference_image = inferencer(image_path)['predictions']

    if model_name in ['DeepLab_ADE', 'BEiT_ADE', 'Segformer_ADE']:
        person_label = 12
    elif model_name in ['DeepLab_VOC']:
        person_label = 15
    elif model_name in ['SAN_COCO']:
        person_label = 0
    human_image = person_extractor(inference_image, person_label=person_label)
    return human_image

def human_labeling(origin_img, mask_img, mask_color=(255,1,1)):
    for h in range(len(mask_img)):
        for w in range(len(mask_img[h])):
            if mask_img[h][w]==1:
                origin_img[h][w]=mask_color
    return origin_img

def mask_conversion(mask):
    org_mask = mask
    new_mask = np.zeros(np.shape(org_mask))
    for h in range(len(org_mask)):
        for w in range(len(org_mask[h])):
            if org_mask[h][w] == 0:
                new_mask[h][w] = np.uint8(255)
            else:
                new_mask[h][w] = np.uint8(0)
    # new_mask = new_mask[:,:,np.newaxis]
    new_mask = cv2.merge([new_mask, new_mask, new_mask])
    return new_mask

# if __name__ == '__main__':
#     out_img = segmentation_img(args)
#     cv2.imwrite('output_image.png')