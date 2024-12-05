from segmentations.segmentation import segmentation_img
from PCDprocessing.pcd2img import project_PCD2ERP
from PCDprocessing.img2pcd import project_ERP_to_PCD
import cv2
import json
import os

import argparse
from argparse import Namespace

# parser = argparse.ArgumentParser()
# parser.add_argument("--model", default='BEiT_ADE')
# parser.add_argument("--pcd_path", required=True) 
# parser.add_argument("--img_height", default=512)
# parser.add_argument("--img_width", default=1024)
# args = parser.parse_args()

def load_config_as_args(json_path):
    """Load configuration from a JSON file and return it as an argparse.Namespace object."""
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Configuration file not found: {json_path}")
    with open(json_path, 'r') as file:
        config = json.load(file)

    # Convert dictionary to Namespace
    args = Namespace(**config)
    return args

if __name__ == '__main__':

    ## Take parameters
    parser = argparse.ArgumentParser(description="Script to process files using a JSON configuration.")
    parser.add_argument('-c', '--config', type=str, required=True, help="Path to the JSON configuration file.")
    cli_args = parser.parse_args()
    args = load_config_as_args(cli_args.config)

    ## PCD to ERP image before segmentation
    erp_img = project_PCD2ERP(args)
    cv2.imwrite('./temp/temp_img.jpg',erp_img)

    ## Segmentation on ERP image
    human_mask = segmentation_img('./temp/temp_img.jpg', args)

    ## Inpaint human area

    ## ERP projection on PCD 
    recon_pcd = project_ERP_to_PCD(args, erp_img)