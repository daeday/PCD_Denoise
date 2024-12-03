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

from skimage.color import rgb2gray
# from skimage import data
from skimage.filters import gaussian
from skimage.segmentation import active_contour
from skimage.util import invert

from utils import *


if __name__ == "__main__":
## Preparing performance check
    model_list= {'DeepLab_ADE':'deeplabv3plus_r101-d8_4xb4-160k_ade20k-512x512',
                'DeepLab_VOC':'deeplabv3plus_r101-d8_4xb4-40k_voc12aug-512x512',
                'BEiT_ADE':'beit-large_upernet_8xb1-amp-160k_ade20k-640x640',
                'SAN_COCO':'san-vit-l14_coco-stuff164k-640x640',
                'Segformer_ADE':'segformer_mit-b5_8xb2-160k_ade20k-640x640'}
    performance_record = {}
    for key, model_name in tqdm(model_list.items()):
        ##
        if key in ['DeepLab_ADE', 'BEiT_ADE', 'Segformer_ADE']:
            person_label = 12
        elif key in ['DeepLab_VOC']:
            person_label = 15
        elif key in ['SAN_COCO']:
            person_label = 0

        inferencer = MMSegInferencer(model=model_name)

        print("Experiment is started on {}".format(key))
        performance_ADE = segmentation_performance(ADE_impaths,ADE_expected,inferencer,person_label=person_label, expanded=True, metric='both') ## Check on ADE20k
        performance_record[key+'_ADEdataset_mIoU'] = performance_ADE[0]
        performance_record[key+'_ADEdataset_wmIoU'] = performance_ADE[1]

        performance_VOC = segmentation_performance(VOC_imgpaths,VOC_expected,inferencer,person_label=person_label, expanded=True, metric='both') ## Check on VOC
        performance_record[key+'_VOCdataset_mIoU'] = performance_VOC[0]
        performance_record[key+'_VOCdataset_wmIoU'] = performance_VOC[1]

    ## Test start    
    new_df = pd.DataFrame.from_dict(data={'DL_ADE':performance_record['DeepLab_ADE_ADEdataset_mIoU'],
                                'DL_VOC':performance_record['DeepLab_VOC_ADEdataset_mIoU'],
                                'BEiT_ADE':performance_record['BEiT_ADE_ADEdataset_mIoU'],
                                'SAN_COCO':performance_record['SAN_COCO_ADEdataset_mIoU'],
                                'Segformer_ADE':performance_record['Segformer_ADE_ADEdataset_mIoU']},
                        orient='columns')

    new_df1 = pd.DataFrame.from_dict(data={'DL_ADE':performance_record['DeepLab_ADE_ADEdataset_wmIoU'],
                                'DL_VOC':performance_record['DeepLab_VOC_ADEdataset_wmIoU'],
                                'BEiT_ADE':performance_record['BEiT_ADE_ADEdataset_wmIoU'],
                                'SAN_COCO':performance_record['SAN_COCO_ADEdataset_wmIoU'],
                                'Segformer_ADE':performance_record['Segformer_ADE_ADEdataset_wmIoU']},
                        orient='columns')

    new_df2 = pd.DataFrame.from_dict(data={'DL_ADE':performance_record['DeepLab_ADE_VOCdataset_mIoU'],
                                'DL_VOC':performance_record['DeepLab_VOC_VOCdataset_mIoU'],
                                'BEiT_ADE':performance_record['BEiT_ADE_VOCdataset_mIoU'],
                                'SAN_COCO':performance_record['SAN_COCO_VOCdataset_mIoU'],
                                'Segformer_ADE':performance_record['Segformer_ADE_VOCdataset_mIoU']},
                        orient='columns')

    new_df3 = pd.DataFrame.from_dict(data={'DL_ADE':performance_record['DeepLab_ADE_VOCdataset_wmIoU'],
                                'DL_VOC':performance_record['DeepLab_VOC_VOCdataset_wmIoU'],
                                'BEiT_ADE':performance_record['BEiT_ADE_VOCdataset_wmIoU'],
                                'SAN_COCO':performance_record['SAN_COCO_VOCdataset_wmIoU'],
                                'Segformer_ADE':performance_record['Segformer_ADE_VOCdataset_wmIoU']},
                        orient='columns')
    
    ## Performance write
    new_df.to_excel('./Performances_on_ADE20k_mIOU_eX.xlsx')
    new_df1.to_excel('./Performances_on_ADE20k_wmIOU_eX.xlsx')
    new_df2.to_excel('./Performances_on_PASCALVOC_mIOU_eX.xlsx')
    new_df3.to_excel('./Performances_on_PASCALVOC_wmIOU_eX.xlsx')