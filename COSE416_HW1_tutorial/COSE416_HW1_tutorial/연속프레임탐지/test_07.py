# 시각화에 필요한 라이브러리 불러오기
import open3d as o3d
import numpy as np
import hdbscan
import os
import time
import glob

# PCD 처리 및 필터링 기준 설정
voxel_size = 0.1  # Voxel downsampling 크기
min_points_in_cluster = 50
max_points_in_cluster = 200
min_z_value = -2.0
max_z_value = 5.0
min_height = 0.3
max_height = 5.0
max_distance = 150.0
max_ratio = 4.0
threshold_ratio = 5.0
FRAME_INTERVAL = 0.1  # 프레임 간격 (초 단위)

# 클러스터 필터링 및 바운딩 박스 생성 함수
def filter_and_visualize_clusters(remaining_pcd, labels):
    bboxes = []
    for i in range(labels.max() + 1):
        cluster_indices = np.where(labels == i)[0]
        if min_points_in_cluster <= len(cluster_indices) <= max_points_in_cluster:
            cluster_pcd = remaining_pcd.select_by_index(cluster_indices)
            points = np.asarray(cluster_pcd.points)
            z_values = points[:, 2]
            z_min, z_max = z_values.min(), z_values.max()
            
            if min_z_value <= z_min and z_max <= max_z_value:
                height_diff = z_max - z_min
                if min_height <= height_diff <= max_height:
                    distances = np.linalg.norm(points, axis=1)
                    if distances.max() <= max_distance:
                        bbox = cluster_pcd.get_axis_aligned_bounding_box()
                        extent = bbox.get_extent()
                        width, length, height = extent[0], extent[1], extent[2]
                        
                        # 가로/세로 비율이 너무 큰 경우 제외
                        if width / length > max_ratio or length / width > max_ratio:
                            continue
                        # XY 면적 / Z 높이 비율 기반 필터링
                        xy_area = width * length
                        if xy_area / height <= threshold_ratio:
                            bbox.color = (1, 0, 0)  # 빨간색
                            bboxes.append(bbox)
    return bboxes

# PCD 처리 함수
def process_pcd(file_path):
    # PCD 파일 읽기
    original_pcd = o3d.io.read_point_cloud(file_path)

    # Downsampling 및 이상치 제거
    downsample_pcd = original_pcd.voxel_down_sample(voxel_size=voxel_size)
    cl, ind = downsample_pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=6.0)
    sor_pcd = downsample_pcd.select_by_index(ind)

    # 평면 추정 및 제거
    plane_model, inliers = sor_pcd.segment_plane(distance_threshold=0.1, ransac_n=5, num_iterations=4000)
    final_point = sor_pcd.select_by_index(inliers, invert=True)

    # 클러스터링
    weighted_points = np.asarray(final_point.points)
    weighted_points[:, 2] *= 1.3  # Z축 강조
    clusterer = hdbscan.HDBSCAN(min_cluster_size=15, min_samples=10, cluster_selection_epsilon=0.2, allow_single_cluster=False)
    labels = clusterer.fit_predict(weighted_points)
    # 클러스터 색상 설정 (모든 클러스터는 파란색)
    colors = np.zeros((len(labels), 3))  # 기본 검정색 (노이즈)
    colors[labels >= 0] = [0, 0, 1]  # 파란색으로 지정
    final_point.colors = o3d.utility.Vector3dVector(colors)

    # 클러스터 필터링 및 바운딩 박스 생성
    bounding_boxes = filter_and_visualize_clusters(final_point, labels)
    return final_point, bounding_boxes
# 전체 시각화 함수
def visualize_pcd_sequence(pcd_folder_path):
    # PCD 파일 경로 가져오기
    pcd_files = sorted(glob.glob(os.path.join(pcd_folder_path, "*.pcd")))
    if not pcd_files:
        print("No PCD files found in the specified folder.")
        return

    # Open3D Visualizer 생성
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Pedestrian Detection in PCD Sequence")

    # 모든 PCD 파일 처리 및 시각화
    for idx, file_path in enumerate(pcd_files):
        start_time = time.time()

        # PCD 처리
        final_pcd, bounding_boxes = process_pcd(file_path)
        print(f"Processing frame {idx + 1}/{len(pcd_files)}: {file_path}")

        # 시각화 갱신
        vis.clear_geometries()
        vis.add_geometry(final_pcd)
        for bbox in bounding_boxes:
            vis.add_geometry(bbox)

        # 카메라 설정
        view_ctl = vis.get_view_control()
        view_ctl.set_zoom(0.1)  # 줌 비율
        view_ctl.set_front([0, -1.5, 1])  # 카메라가 점군을 보는 방향
        view_ctl.set_lookat([0, 0, 3])  # 카메라가 보는 중심점
        view_ctl.set_up([0, 0.5, 0])     # 카메라의 위쪽 방향

        vis.poll_events()
        vis.update_renderer()

        # 프레임 속도 유지
        elapsed_time = time.time() - start_time
        sleep_time = max(0, FRAME_INTERVAL - elapsed_time)
        time.sleep(sleep_time)

    vis.destroy_window()

# 메인 실행
if __name__ == "__main__":
    # PCD 파일이 있는 폴더 경로를 설정하세요
    pcd_folder_path = "C:/Users/estre/Downloads/COSE416_HW1_data_v1/data/07_straight_walk/pcd"
    visualize_pcd_sequence(pcd_folder_path)
