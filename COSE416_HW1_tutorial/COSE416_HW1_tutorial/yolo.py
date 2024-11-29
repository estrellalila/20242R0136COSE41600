import open3d as o3d
import numpy as np
import cv2
import torch
import hdbscan

# --- LiDAR 데이터 전처리 ---
def preprocess_lidar(file_path, voxel_size=0.05, min_cluster_size=20, min_samples=13):
    pcd = o3d.io.read_point_cloud(file_path)
    
    # 다운샘플링
    downsample_pcd = pcd.voxel_down_sample(voxel_size=voxel_size)

    # Statistical Outlier Removal
    _, ind = downsample_pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=6.0)
    filtered_pcd = downsample_pcd.select_by_index(ind)

    # 평면 제거
    plane_model, inliers = filtered_pcd.segment_plane(distance_threshold=0.2, ransac_n=5, num_iterations=4000)
    non_ground_pcd = filtered_pcd.select_by_index(inliers, invert=True)

    # Z축 강조 후 클러스터링
    weighted_points = np.asarray(non_ground_pcd.points)
    weighted_points[:, 2] *= 1.2  # Z축 강조
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples)
    labels = clusterer.fit_predict(weighted_points)
    
    return non_ground_pcd, labels

# --- LiDAR 데이터를 BEV 이미지로 변환 ---
def lidar_to_bev(point_cloud, lidar_bounds, bev_size=(10000, 10000)):
    # 1. 포인트 클라우드에서 x, y, z 추출
    points = np.asarray(point_cloud.points)

    # 2. 디버깅: 원본 포인트 범위 확인
    print(f"Point cloud X range: {points[:, 0].min()} to {points[:, 0].max()}")
    print(f"Point cloud Y range: {points[:, 1].min()} to {points[:, 1].max()}")

    # 3. 정규화
    normalized_points = np.zeros_like(points, dtype=np.int32)
    normalized_points[:, 0] = np.clip(
        ((points[:, 0] - lidar_bounds[0]) / (lidar_bounds[3] - lidar_bounds[0]) * bev_size[0]).astype(np.int32),
        0, bev_size[0] - 1
    )
    normalized_points[:, 1] = np.clip(
        ((points[:, 1] - lidar_bounds[1]) / (lidar_bounds[4] - lidar_bounds[1]) * bev_size[1]).astype(np.int32),
        0, bev_size[1] - 1
    )

    # 4. 디버깅: 정규화된 좌표 범위 출력
    print(f"Normalized X range: {normalized_points[:, 0].min()} to {normalized_points[:, 0].max()}")
    print(f"Normalized Y range: {normalized_points[:, 1].min()} to {normalized_points[:, 1].max()}")

    # 5. BEV 이미지 생성
    bev_image = np.zeros(bev_size, dtype=np.uint8)
    try:
        bev_image[normalized_points[:, 1], normalized_points[:, 0]] = 255
    except IndexError as e:
        print(f"IndexError: {e}")
        print(f"Out-of-bounds indices: X={normalized_points[:, 0].max()}, Y={normalized_points[:, 1].max()}")

    return bev_image, lidar_bounds

# --- YOLO 탐지 수행 ---
def detect_with_yolo(bev_image, model):
    bev_image_rgb = cv2.cvtColor(bev_image, cv2.COLOR_GRAY2RGB)
    results = model(bev_image_rgb)
    return results.pandas().xyxy[0]  # Extract predictions as DataFrame

# --- YOLO 결과를 3D로 매핑 ---
def map_yolo_to_lidar(detections, pcd, grid_size, lidar_bounds):
    x_min, x_max, y_min, y_max = lidar_bounds[:4]  # 4개의 값만 추출
    bounding_boxes = []
    for _, det in detections.iterrows():
        x_min_img, y_min_img, x_max_img, y_max_img = det[['xmin', 'ymin', 'xmax', 'ymax']]

        # Map BEV image coordinates back to LiDAR coordinates
        x_min_lidar = x_min_img / grid_size[1] * (x_max - x_min) + x_min
        y_min_lidar = y_min_img / grid_size[0] * (y_max - y_min) + y_min
        x_max_lidar = x_max_img / grid_size[1] * (x_max - x_min) + x_min
        y_max_lidar = y_max_img / grid_size[0] * (y_max - y_min) + y_min

        bounding_boxes.append(((x_min_lidar, y_min_lidar), (x_max_lidar, y_max_lidar)))
    return bounding_boxes

# --- 메인 실행 코드 ---
if __name__ == "__main__":
    # LiDAR 파일 경로
    file_path = "C:/Users/estre/OneDrive/Desktop/개발/20242R0136COSE41600/COSE416_HW1_tutorial/COSE416_HW1_tutorial/test_data/1727320101-665925967.pcd"

    # 1. LiDAR 데이터 전처리
    non_ground_pcd, labels = preprocess_lidar(file_path)

    # 2. LiDAR 범위 계산
    lidar_bounds = [
        np.min(np.asarray(non_ground_pcd.points)[:, 0]),
        np.min(np.asarray(non_ground_pcd.points)[:, 1]),
        np.min(np.asarray(non_ground_pcd.points)[:, 2]),
        np.max(np.asarray(non_ground_pcd.points)[:, 0]),
        np.max(np.asarray(non_ground_pcd.points)[:, 1]),
        np.max(np.asarray(non_ground_pcd.points)[:, 2])
    ]

    # 3. LiDAR 데이터를 BEV 이미지로 변환
    bev_image, _ = lidar_to_bev(non_ground_pcd, lidar_bounds)

    # BEV 이미지 저장 (확인용)
    cv2.imwrite("bev_image.png", bev_image)

    # 4. YOLO 초기화 및 탐지 수행
    yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # YOLO 모델 초기화
    detections = detect_with_yolo(bev_image, yolo_model)

    # 5. YOLO 결과를 LiDAR 좌표로 매핑
    bounding_boxes = map_yolo_to_lidar(detections, non_ground_pcd, bev_image.shape, lidar_bounds)

    # 6. 3D 시각화
    for box in bounding_boxes:
        bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=(box[0][0], box[0][1], -2), 
                                                   max_bound=(box[1][0], box[1][1], 2))
        bbox.color = (1, 0, 0)
        non_ground_pcd.paint_uniform_color([0, 0, 0])  # Reset color
        non_ground_pcd.bounding_box = bbox

    o3d.visualization.draw_geometries([non_ground_pcd])
