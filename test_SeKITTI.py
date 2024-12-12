from segmentations.segmentation import segmentation_img, mask_conversion
from PCDprocessing.pcd2img import project_PCD2ERP
from PCDprocessing.img2pcd import project_ERP_to_PCD
import cv2
import json
import os
import open3d as o3d
import glob
import struct

import argparse
import numpy as np
from argparse import Namespace
from pyinpaint import Inpaint

# parser = argparse.ArgumentParser()
# parser.add_argument("--model", default='BEiT_ADE')
# parser.add_argument("--pcd_path", required=True) 
# parser.add_argument("--img_height", default=512)
# parser.add_argument("--img_width", default=1024)
# args = parser.parse_args()

def BinToPcd(point_file):
    size_float = 4
    list_pcd = []
    file_to_open = point_file
    file_to_save = str(point_file)[:-3]+"pcd"
    with open(file_to_open, "rb") as f:
        byte = f.read(size_float*4)
        while byte:
            x,y,z,intensity = struct.unpack("ffff", byte)
            list_pcd.append([x,y,z])
            byte = f.read(size_float*4)
    np_pcd = np.asarray(list_pcd)
    pcd = o3d.geometry.PointCloud()
    v3d = o3d.utility.Vector3dVector
    pcd.points = v3d(np_pcd)
    o3d.io.write_point_cloud(file_to_save, pcd)

# # Read bin files
# point_files = sorted(glob.glob("velodyne_points/data/*.bin"))
# print(point_files)

if __name__ == '__main__':
    print("nothing yet")

    ## SemanticKITTI dataset load

    ## Set module
    # Image segmentation
    # Image projection on Point cloud
    # Point cloud labeling

    ## Calculate 3D mIoU module