import os
import time
import open3d as o3d
import numpy as np
import hdbscan

# 설정
FRAME_RATE = 5
FRAME_INTERVAL = 1.0 / FRAME_RATE

# PCD 파일 경로 설정
pcd_dir = "C:/Users/estre/Downloads/COSE416_HW1_data_v1/data/01_straight_walk/pcd"  # PCD 파일 폴더 경로를 설정하세요.
pcd_files = sorted([os.path.join(pcd_dir, f) for f in os.listdir(pcd_dir) if f.endswith(".pcd")])

# 필터 및 바운딩 박스 설정
voxel_size = 0.1
min_points_in_cluster = 30
max_points_in_cluster = 150
min_z_value = -1.5
max_z_value = 1.5
min_height = 0.3
max_height = 2.0
max_distance = 120.0
max_ratio = 3.0


# 포인트 클라우드 및 바운딩 박스를 시각화하는 함수
def visualize_frame_with_bounding_boxes(vis, pcd, bounding_boxes):
    vis.clear_geometries()
    vis.add_geometry(pcd)
    for bbox in bounding_boxes:
        vis.add_geometry(bbox)
    vis.poll_events()
    vis.update_renderer()


# PCD 처리 함수
def process_pcd(file_path):
    # PCD 파일 읽기
    original_pcd = o3d.io.read_point_cloud(file_path)

    # 다운샘플링
    downsample_pcd = original_pcd.voxel_down_sample(voxel_size=voxel_size)

    # SOR 필터링
    _, ind = downsample_pcd.remove_statistical_outlier(nb_neighbors=30, std_ratio=12.0)
    sor_pcd = downsample_pcd.select_by_index(ind)

    # 평면 제거 (도로 영역 제거)
    _, inliers = sor_pcd.segment_plane(distance_threshold=0.2, ransac_n=4, num_iterations=4000)
    final_point = sor_pcd.select_by_index(inliers, invert=True)

    # Z축 가중치 적용
    weighted_points = np.asarray(final_point.points)
    weighted_points[:, 2] *= 1.2

    # HDBSCAN 클러스터링
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=15,
        min_samples=10,
        metric="euclidean",
        cluster_selection_epsilon=0.1,
        allow_single_cluster=False,
    )
    labels = clusterer.fit_predict(weighted_points)

    # 노이즈 포인트 색상
    colors = np.zeros((len(labels), 3))  # 기본 검정색 (노이즈)
    colors[labels >= 0] = [0, 0, 1]  # 파란색
    final_point.colors = o3d.utility.Vector3dVector(colors)

    # 클러스터 필터링 및 바운딩 박스 생성
    bboxes = []
    for i in range(labels.max() + 1):
        cluster_indices = np.where(labels == i)[0]
        if min_points_in_cluster <= len(cluster_indices) <= max_points_in_cluster:
            cluster_pcd = final_point.select_by_index(cluster_indices)
            points = np.asarray(cluster_pcd.points)
            z_values = points[:, 2]
            z_min = z_values.min()
            z_max = z_values.max()
            if min_z_value <= z_min and z_max <= max_z_value:
                height_diff = z_max - z_min
                if min_height <= height_diff <= max_height:
                    distances = np.linalg.norm(points, axis=1)
                    if distances.max() <= max_distance:
                        # 바운딩 박스 생성
                        bbox = cluster_pcd.get_axis_aligned_bounding_box()
                        extent = bbox.get_extent()
                        width, length = extent[0], extent[1]

                        # 비율 검증
                        if width / length > max_ratio or length / width > max_ratio:
                            continue
                        bbox.color = (1, 0, 0)
                        bboxes.append(bbox)
    return final_point, bboxes


# 실시간 시각화
def visualize_pcd_sequence():
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Pedestrian Detection in PCD Sequence")

    for idx, file_path in enumerate(pcd_files):
        start_time = time.time()

        # PCD 처리
        final_pcd, bounding_boxes = process_pcd(file_path)
        print(f"Processing frame {idx + 1}/{len(pcd_files)}: {file_path}")

        # 시각화 갱신
        visualize_frame_with_bounding_boxes(vis, final_pcd, bounding_boxes)

        # 프레임 속도 유지
        elapsed_time = time.time() - start_time
        sleep_time = max(0, FRAME_INTERVAL - elapsed_time)
        time.sleep(sleep_time)

    vis.destroy_window()


# 실행
visualize_pcd_sequence()
