from segmentations.segmentation import segmentation_img
from PCDprocessing.pcd2img import project_PCD2ERP
import cv2

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model", default='BEiT_ADE')
parser.add_argument("--pcd_path", required=True) 
parser.add_argument("--img_height", default=512)
parser.add_argument("--img_width", default=1024)
args = parser.parse_args()

if __name__ == '__main__':
    ## PCD to ERP image before segmentation
    erp_img = project_PCD2ERP(args)
    cv2.imwrite('./temp/temp_img.jpg',erp_img)

    ## Segmentation on ERP image
    human_mask = segmentation_img('./temp/temp_img.jpg', args)

    ## Inpaint human area