import numpy as np
import e57

def project_ERP_to_PCD(args, erp_image):
    """
    Map RGB values from an ERP image to a point cloud (E57).
    
    Args:
        args: Arguments containing calibration and image dimensions.
        pcd: Point cloud loaded from E57 (with XYZ coordinates).
        erp_image: ERP image (height x width x 3) in numpy array format.
    
    Returns:
        point_cloud: Point cloud with updated RGB values (N x 6).
    """
    # ERP 이미지 크기
    img_height, img_width = args.img_height, args.img_width
    pcd = e57.read_points(args.pcd_path)

    # 포인트 클라우드의 XYZ 좌표 가져오기
    xyz = pcd.points  # (N, 3) 형태

    # XYZ 좌표를 구형 좌표계 (방위각, 고도각)으로 변환
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    norm = np.linalg.norm(xyz, axis=1)

    # 방위각 (theta)와 고도각 (phi) 계산
    theta = np.arctan2(y, x)  # [-π, π]
    phi = np.arcsin(np.clip(z / norm, -1, 1))  # [-π/2, π/2]

    # 방위각과 고도각을 이미지의 픽셀 좌표로 변환
    u = ((theta + np.pi) / (2 * np.pi) * img_width).astype(int)  # [0, width]
    v = ((1 - (phi + (np.pi / 2)) / np.pi) * img_height).astype(int)  # [0, height]

    # 이미지 경계 값으로 클리핑
    u = np.clip(u, 0, img_width - 1)
    v = np.clip(v, 0, img_height - 1)

    # ERP 이미지로부터 RGB 값을 가져와 포인트 클라우드에 매핑
    rgb = erp_image[v, u]  # 각 픽셀의 RGB 값 추출

    # XYZ와 새로 매핑된 RGB 값을 결합
    updated_point_cloud = np.hstack((xyz, rgb))
    
    return updated_point_cloud