import e57
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--pcd_path", required=True) 
parser.add_argument("--img_height", default=512)
parser.add_argument("--img_width", default=1024)
args = parser.parse_args()

def project_PCD2ERP(args):
    pcd = e57.read_points(args.pcd_path)
    rgb_pcd = np.concatenate((pcd.points, np.array(pcd.color*255,dtype=np.int8())),axis=1)
    prj_img = np.zeros((args.img_height, args.img_width, 3), dtype=np.uint8)

    # 포인트 위치와 RGB 값 분리
    xyz = rgb_pcd[:, :3]
    rgb = rgb_pcd[:, 3:].astype(np.uint8)

    # XYZ 좌표를 구형 좌표계 (방위각, 고도각)로 변환
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]

    # 방위각 (theta)와 고도각 (phi) 계산
    theta = np.arctan2(y, x)  # 방위각
    phi = np.arcsin(z / np.linalg.norm(xyz, axis=1))  # 고도각

    # 방위각과 고도각을 이미지의 픽셀 좌표로 변환
    u = ((theta + np.pi) / (2 * np.pi) * args.img_width).astype(int)  # [0, width] 범위로 매핑
    v = ((1 - (phi + (np.pi / 2)) / np.pi) * args.img_height).astype(int)  # [0, height] 범위로 매핑

    # 이미지 경계 값으로 클리핑
    u = np.clip(u, 0, args.img_width - 1)
    v = np.clip(v, 0, args.img_height - 1)

    # 각 픽셀에 RGB 값을 매핑
    for i in range(len(rgb_pcd)):
        prj_img[v[i], u[i]] = rgb[i]
    return prj_img

if __name__ == '__main__':
    project_PCD2ERP(args)