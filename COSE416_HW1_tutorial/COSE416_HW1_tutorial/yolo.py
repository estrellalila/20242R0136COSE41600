import open3d as o3d
import numpy as np
import torch
import cv2
import hdbscan

# Voxel Downsampling 수행
def voxel_downsampling(pcd, voxel_size=0.1):
    return pcd.voxel_down_sample(voxel_size)

# Statistical Outlier Removal (SOR) 적용
def statistical_outlier_removal(pcd, nb_neighbors=20, std_ratio=6.0):
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    return pcd.select_by_index(ind)

# RANSAC을 사용하여 평면 추정
def ransac_plane_estimation(pcd, distance_threshold=0.2, ransac_n=4, num_iterations=4000):
    plane_model, inliers = pcd.segment_plane(distance_threshold=distance_threshold,
                                             ransac_n=ransac_n,
                                             num_iterations=num_iterations)
    return plane_model, inliers

# 도로에 속하지 않는 포인트 (outliers) 추출
def extract_outliers(pcd, inliers):
    return pcd.select_by_index(inliers, invert=True)

# Z축 강조 가중치 적용
def apply_z_weighting(pcd, scale_factor=1.2):
    points = np.asarray(pcd.points)
    points[:, 2] *= scale_factor  # Z축에 가중치 추가
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd

# HDBSCAN 클러스터링
def hdbscan_clustering(pcd, min_cluster_size=20, min_samples=13):
    points = np.asarray(pcd.points)
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric='euclidean',
        cluster_selection_epsilon=0.1,
        allow_single_cluster=False
    )
    labels = clusterer.fit_predict(points)
    return labels

# BEV 이미지를 생성하는 함수
def project_to_bev(points, lidar_bounds, bev_size):
    # 정규화된 2D 이미지 좌표로 변환
    normalized_points = np.zeros_like(points, dtype=np.int32)
    normalized_points[:, 0] = np.clip(
        ((points[:, 0] - lidar_bounds[0]) / (lidar_bounds[3] - lidar_bounds[0]) * bev_size[0]).astype(np.int32),
        0, bev_size[0] - 1
    )
    normalized_points[:, 1] = np.clip(
        ((points[:, 1] - lidar_bounds[1]) / (lidar_bounds[4] - lidar_bounds[1]) * bev_size[1]).astype(np.int32),
        0, bev_size[1] - 1
    )
    
    # BEV 이미지 생성
    bev_image = np.zeros(bev_size, dtype=np.uint8)
    bev_image[normalized_points[:, 1], normalized_points[:, 0]] = 255
    return bev_image

# LiDAR 데이터를 3개의 평면으로 투영하여 BEV 이미지 생성
def lidar_to_projection_images(pcd, lidar_bounds, bev_size=(10000, 10000)):
    points = np.asarray(pcd.points)
    
    # XY 평면 (기존)
    xy_points = points[:, :2]  # X, Y 좌표만 사용
    xy_image = project_to_bev(xy_points, lidar_bounds, bev_size)
    
    # YZ 평면 (Y, Z 좌표)
    yz_points = points[:, 1:3]  # Y, Z 좌표만 사용
    yz_image = project_to_bev(yz_points, lidar_bounds, bev_size)
    
    # ZX 평면 (Z, X 좌표)
    zx_points = points[:, [2, 0]]  # Z, X 좌표만 사용
    zx_image = project_to_bev(zx_points, lidar_bounds, bev_size)
    
    return xy_image, yz_image, zx_image

# YOLO 탐지
def detect_with_yolo(bev_image, model):
    bev_image_rgb = cv2.cvtColor(bev_image, cv2.COLOR_GRAY2RGB)
    results = model(bev_image_rgb)
    return results.pandas().xyxy[0]  # Extract predictions as DataFrame

# LiDAR 범위 계산
def compute_lidar_bounds(pcd):
    points = np.asarray(pcd.points)
    return [
        np.min(points[:, 0]), np.min(points[:, 1]), np.min(points[:, 2]), 
        np.max(points[:, 0]), np.max(points[:, 1]), np.max(points[:, 2])
    ]

# YOLO 결과를 LiDAR 좌표로 매핑
def map_yolo_to_lidar(detections, lidar_bounds, bev_size, plane):
    x_min, x_max, y_min, y_max = lidar_bounds[:4]  # 4개의 값만 추출
    bounding_boxes = []
    
    for _, det in detections.iterrows():
        x_min_img, y_min_img, x_max_img, y_max_img = det[['xmin', 'ymin', 'xmax', 'ymax']]

        # Map BEV image coordinates back to LiDAR coordinates
        if plane == 'xy':
            x_min_lidar = x_min_img / bev_size[1] * (x_max - x_min) + x_min
            y_min_lidar = y_min_img / bev_size[0] * (y_max - y_min) + y_min
            x_max_lidar = x_max_img / bev_size[1] * (x_max - x_min) + x_min
            y_max_lidar = y_max_img / bev_size[0] * (y_max - y_min) + y_min
        elif plane == 'yz':
            x_min_lidar = y_min_img / bev_size[1] * (x_max - x_min) + x_min
            y_min_lidar = x_min_img / bev_size[0] * (y_max - y_min) + y_min
            x_max_lidar = y_max_img / bev_size[1] * (x_max - x_min) + x_min
            y_max_lidar = x_max_img / bev_size[0] * (y_max - y_min) + y_min
        elif plane == 'zx':
            x_min_lidar = y_min_img / bev_size[1] * (x_max - x_min) + x_min
            y_min_lidar = x_min_img / bev_size[0] * (y_max - y_min) + y_min
            x_max_lidar = y_max_img / bev_size[1] * (x_max - x_min) + x_min
            y_max_lidar = x_max_img / bev_size[0] * (y_max - y_min) + y_min

        bounding_boxes.append(((x_min_lidar, y_min_lidar), (x_max_lidar, y_max_lidar)))
    return bounding_boxes

# 3D 매핑 및 시각화
def visualize_3d_with_boxes(pcd, bounding_boxes):
    geometries = [pcd]  # LiDAR 포인트 클라우드를 리스트에 추가
    for box in bounding_boxes:
        bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=(box[0][0], box[0][1], -2), 
                                                   max_bound=(box[1][0], box[1][1], 2))
        bbox.color = (1, 0, 0)  # 빨간색으로 설정
        geometries.append(bbox)  # 박스를 geometries 리스트에 추가
    o3d.visualization.draw_geometries(geometries)

# --- 메인 실행 코드 --- 
if __name__ == "__main__":
    # LiDAR 포인트 클라우드 파일 경로
    file_path = "C:/Users/estre/OneDrive/Desktop/개발/20242R0136COSE41600/COSE416_HW1_tutorial/COSE416_HW1_tutorial/test_data/1727320101-665925967.pcd"
    
    # 1. LiDAR 포인트 클라우드 로드
    pcd = o3d.io.read_point_cloud(file_path)
    
    # 2. 전처리: Voxel Downsampling, SOR, RANSAC 평면 추정, Z축 가중치 적용
    downsampled_pcd = voxel_downsampling(pcd, voxel_size=0.1)
    sor_pcd = statistical_outlier_removal(downsampled_pcd)
    plane_model, inliers = ransac_plane_estimation(sor_pcd)
    final_point = extract_outliers(sor_pcd, inliers)
    weighted_pcd = apply_z_weighting(final_point)
    
    # 3. HDBSCAN 클러스터링
    labels = hdbscan_clustering(weighted_pcd)
    
    # 4. LiDAR 범위 계산
    lidar_bounds = compute_lidar_bounds(weighted_pcd)
    
    # 5. LiDAR 데이터를 3개의 평면으로 투영
    xy_image, yz_image, zx_image = lidar_to_projection_images(weighted_pcd, lidar_bounds)
    
    # 6. BEV 이미지 저장
    cv2.imwrite('xy_bev_image.png', xy_image)
    cv2.imwrite('yz_bev_image.png', yz_image)
    cv2.imwrite('zx_bev_image.png', zx_image)
    
    # 7. YOLO로 객체 탐지 (YOLOv5 모델 예시)
    model = torch.hub.load("ultralytics/yolov5:v7.0", "yolov5s")  # YOLOv5 small 모델 로드
    detections = detect_with_yolo(xy_image, model)  # BEV 이미지에 대해 YOLO 탐지 수행
    
    # 8. YOLO 결과를 LiDAR 좌표로 매핑
    bounding_boxes = map_yolo_to_lidar(detections, lidar_bounds, xy_image.shape, 'xy')
    
    # bounding_boxes 내용 출력
    print(f"Number of Bounding Boxes: {len(bounding_boxes)}")
    # 9. 3D 시각화
    visualize_3d_with_boxes(weighted_pcd, bounding_boxes)
