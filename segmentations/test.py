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


if __name__ == "__main__":
    print("Experiment Ready")

    inferencer = MMSegInferencer(model='beit-large_upernet_8xb1-amp-160k_ade20k-640x640')
    impa = '360_rgb_image.png' ##VOC_imgpaths[5]
    inf_img = inferencer(impa)['predictions']
    
    plt.figure(figsize=(10,10))
    plt.subplot(311)
    inp_im = cv2.imread(impa)
    inp_im = cv2.cvtColor(inp_im, cv2.COLOR_BGR2RGB)
    plt.imshow(inp_im)

    plt.subplot(312)
    plt.imshow(inf_img)

    plt.subplot(313)
    inf_img = person_extractor(inf_img, person_label=12)
    plt.imshow(inf_img)
    plt.show()