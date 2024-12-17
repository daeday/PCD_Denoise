import os
import matplotlib.pyplot as plt
import open3d
import cv2
from segmentations.segmentation import human_labeling
import open3d as o3d
import argparse

from PCDprocessing.utils import *


def render_image_with_boxes(img, objects, calib):
    """
    Show image with 3D boxes
    """
    # projection matrix
    P_rect2cam2 = calib['P2'].reshape((3, 4))

    img1 = np.copy(img)
    for obj in objects:
        if obj.type == 'DontCare':
            continue
        box3d_pixelcoord = map_box_to_image(obj, P_rect2cam2)
        img1 = draw_projected_box3d(img1, box3d_pixelcoord)

    plt.imshow(img1)
    plt.yticks([])
    plt.xticks([])
    plt.show()

def render_lidar_with_boxes(pc_velo, objects, calib, img_width, img_height):
    # projection matrix (project from velo2cam2)
    proj_velo2cam2 = project_velo_to_cam2(calib)

    # apply projection
    pts_2d = project_to_image(pc_velo.transpose(), proj_velo2cam2)

    # Filter lidar points to be within image FOV
    inds = np.where((pts_2d[0, :] < img_width) & (pts_2d[0, :] >= 0) &
                    (pts_2d[1, :] < img_height) & (pts_2d[1, :] >= 0) &
                    (pc_velo[:, 0] > 0)
                    )[0]
    imgfov_pc_velo = pc_velo[inds, :]

    # create open3d point cloud and axis
    mesh_frame = open3d.geometry.TriangleMesh.create_coordinate_frame(size=2, origin=[0, 0, 0])
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(imgfov_pc_velo)
    entities_to_draw = [pcd, mesh_frame]

    # Projection matrix
    proj_cam2_2_velo = project_cam2_to_velo(calib)

    # Draw objects on lidar
    for obj in objects:
        if obj.type == 'DontCare':
            continue

        # Project boxes from camera to lidar coordinate
        boxes3d_pts = project_camera_to_lidar(obj.in_camera_coordinate(), proj_cam2_2_velo)

        # Open3d boxes
        boxes3d_pts = open3d.utility.Vector3dVector(boxes3d_pts.T)
        box = open3d.geometry.OrientedBoundingBox.create_from_points(boxes3d_pts)
        box.color = [1, 0, 0]
        entities_to_draw.append(box)

    # Draw
    open3d.visualization.draw_geometries([*entities_to_draw],
                                         front=[-0.9945, 0.03873, 0.0970],
                                         lookat=[38.4120, 0.6139, 0.48500],
                                         up=[0.095457, -0.0421, 0.99453],
                                         zoom=0.33799
                                         )

def render_lidar_on_image(pts_velo, img, calib, img_width, img_height):
    # projection matrix (project from velo2cam2)
    proj_velo2cam2 = project_velo_to_cam2(calib)

    # apply projection
    pts_2d = project_to_image(pts_velo.transpose(), proj_velo2cam2)

    # Filter lidar points to be within image FOV
    inds = np.where((pts_2d[0, :] < img_width) & (pts_2d[0, :] >= 0) &
                    (pts_2d[1, :] < img_height) & (pts_2d[1, :] >= 0) &
                    (pc_velo[:, 0] > 0)
                    )[0]

    # Filter out pixels points
    imgfov_pc_pixel = pts_2d[:, inds]

    # Retrieve depth from lidar
    imgfov_pc_velo = pts_velo[inds, :]
    imgfov_pc_velo = np.hstack((imgfov_pc_velo, np.ones((imgfov_pc_velo.shape[0], 1))))
    imgfov_pc_cam2 = proj_velo2cam2 @ imgfov_pc_velo.transpose()

    cmap = plt.cm.get_cmap('hsv', 256)
    cmap = np.array([cmap(i) for i in range(256)])[:, :3] * 255

    for i in range(imgfov_pc_pixel.shape[1]):
        depth = imgfov_pc_cam2[2, i]
        color = cmap[int(640.0 / depth), :]
        cv2.circle(img, (int(np.round(imgfov_pc_pixel[0, i])),
                         int(np.round(imgfov_pc_pixel[1, i]))),
                   2, color=tuple(color), thickness=-1)
    plt.imshow(img)
    plt.yticks([])
    plt.xticks([])
    plt.show()
    return img

def RGB_mapping2pcd(binary, origin_img, mask_img, P2, Tr, coloring=False, erase=False):
    # read raw data from binary
    scan = np.fromfile(binary, dtype=np.float32).reshape((-1, 4))
    points = scan[:, 0:3]  # lidar xyz (front, left, up)

    # Apply projection matrix
    velo = np.insert(points, 3, 1, axis=1).T
    cam = P2.dot(Tr.dot(velo))
    cam[:2] /= cam[2, :]

    # Preparation on image
    plt.figure(figsize=(12, 5), dpi=96, tight_layout=True)
    if coloring:
        label_color = (255,1,1)
        png = human_labeling(origin_img, mask_img, mask_color=label_color)
    else:
        png = origin_img
    IMG_H, IMG_W, _ = png.shape
    plt.axis([0, IMG_W, IMG_H, 0])

    # Filter point out of canvas
    u, v, z = cam
    u_out = np.logical_or(u < 0, u > IMG_W)
    v_out = np.logical_or(v < 0, v > IMG_H)
    outlier = np.logical_or(u_out, v_out)

    # generate color map from depth
    u, v, z = cam

    # Adding rgb data
    rgb_values = []
    for u_coord, v_coord, z_coord in zip(u, v, z):
        u_int, v_int = int(round(u_coord)), int(round(v_coord))
        if z_coord >= 0:
            if 0 <= u_int < IMG_W and 0 <= v_int < IMG_H:
                rgb_value = png[v_int, u_int]
                rgb_values.append(rgb_value)
            else:
                rgb_values.append((0, 0, 0))
        else:
            rgb_values.append((0, 0, 0))
    
    if coloring:
        ## Erase colored points
        erase_idx = []
        for idx, rgb in enumerate(rgb_values):
            if np.array_equal(rgb,label_color):
                erase_idx.append(idx)
        if erase:
            points = np.delete(points,erase_idx,axis=0)
            rgb_values = np.delete(rgb_values, erase_idx,axis=0)
            rgb_values = np.array(rgb_values)/255.
            u = np.delete(u, erase_idx)
            v = np.delete(v, erase_idx)
    else:
        erase_idx = []
    rgb_values = np.array(rgb_values)/255.
    
    # Create Open3D point cloud
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    point_cloud.colors = o3d.utility.Vector3dVector(rgb_values)

    return point_cloud, (u,v), rgb_values, erase_idx
def label_coloring2pcd(binary, origin_img, P2, Tr, labels, Person_label):
    # read raw data from binary
    scan = np.fromfile(binary, dtype=np.float32).reshape((-1, 4))
    points = scan[:, 0:3]  # lidar xyz (front, left, up)

    # Apply projection matrix
    velo = np.insert(points, 3, 1, axis=1).T
    cam = P2.dot(Tr.dot(velo))
    cam[:2] /= cam[2, :]

    # Preparation on image
    plt.figure(figsize=(12, 5), dpi=96, tight_layout=True)
    label_color = (255,1,1)
    png = origin_img
    IMG_H, IMG_W, _ = png.shape
    plt.axis([0, IMG_W, IMG_H, 0])

    # Filter point out of canvas
    u, v, z = cam
    u_out = np.logical_or(u < 0, u > IMG_W)
    v_out = np.logical_or(v < 0, v > IMG_H)
    outlier = np.logical_or(u_out, v_out)

    # generate color map from depth
    u, v, z = cam

    # Adding rgb data
    rgb_values = []
    for u_coord, v_coord, z_coord in zip(u, v, z):
        u_int, v_int = int(round(u_coord)), int(round(v_coord))
        if z_coord >= 0:
            if 0 <= u_int < IMG_W and 0 <= v_int < IMG_H:
                rgb_value = png[v_int, u_int]
                rgb_values.append(rgb_value)
            else:
                rgb_values.append((0, 0, 0))
        else:
            rgb_values.append((0, 0, 0))
    
    for idx, tmp_lbl in enumerate(labels):
        if tmp_lbl == Person_label:
            rgb_values[idx] = label_color
    rgb_values = np.array(rgb_values)/255.
    
    # Create Open3D point cloud
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    point_cloud.colors = o3d.utility.Vector3dVector(rgb_values)

    return point_cloud, (u,v), rgb_values


if __name__ == '__main__':
    # Load image, calibration file, label bbox
    rgb = cv2.cvtColor(cv2.imread(os.path.join('data/000114_image.png')), cv2.COLOR_BGR2RGB)
    img_height, img_width, img_channel = rgb.shape

    # Load calibration
    calib = read_calib_file('data/000114_calib.txt')

    # Load labels
    labels = load_label('data/000114_label.txt')

    # Load Lidar PC
    pc_velo = load_velo_scan('data/000114.bin')[:, :3]

    # render_image_with_boxes(rgb, labels, calib)
    render_lidar_with_boxes(pc_velo, labels, calib, img_width=img_width, img_height=img_height)
    # render_lidar_on_image(pc_velo, rgb, calib, img_width, img_height)
