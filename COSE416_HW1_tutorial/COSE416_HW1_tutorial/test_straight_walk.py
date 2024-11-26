# 시각화에 필요한 라이브러리 불러오기
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import hdbscan

# pcd 파일 불러오기, 필요에 맞게 경로 수정
#file_path = "C:/Users/estre/OneDrive/Desktop/개발/20242R0136COSE41600/COSE416_HW1_tutorial/COSE416_HW1_tutorial/test_data/1727320101-665925967.pcd"

#straight_walk
file_path = "C:/Users/estre/Downloads/COSE416_HW1_data_v1/data/01_straight_walk/pcd/pcd_000212.pcd"

#straight_crawl
#file_path = "C:/Users/estre/Downloads/COSE416_HW1_data_v1/data/03_straight_crawl/pcd/pcd_000844.pcd"

#straight_duck
#file_path = "C:/Users/estre/Downloads/COSE416_HW1_data_v1/data/05_straight_duck_walk/pcd/pcd_000577.pcd"

# PCD 파일 읽기
original_pcd = o3d.io.read_point_cloud(file_path)

# Voxel Downsampling 수행
voxel_size = 0.1  # 필요에 따라 voxel 크기를 조정하세요.
downsample_pcd = original_pcd.voxel_down_sample(voxel_size=voxel_size)

# SOR (Statistical Outlier Removal)
cl, ind = downsample_pcd.remove_statistical_outlier(nb_neighbors=30, std_ratio=12.0)
sor_pcd = downsample_pcd.select_by_index(ind)

# RANSAC을 사용하여 평면 추정
plane_model, inliers = sor_pcd.segment_plane(distance_threshold=0.4,
                                             ransac_n=12,
                                             num_iterations=4000)

# 도로에 속하지 않는 포인트 (outliers) 추출
final_point = sor_pcd.select_by_index(inliers, invert=True)

# Z축 강조 가중치 적용, 적당히 긴 사람 형상 위해서,,
weighted_points = np.asarray(final_point.points)
weighted_points[:, 2] *= 1.2  # Z축에 가중치 추가

# HDBSCAN 클러스터링
clusterer = hdbscan.HDBSCAN(
    min_cluster_size=15,      # 클러스터 내 최소 점 개수
    min_samples=10,           # 핵심 점으로 간주할 최소 이웃 수
    metric='euclidean',       # 거리 척도
    cluster_selection_epsilon=0.1,  # 클러스터 선택 임계값
    allow_single_cluster=False  # 하나의 클러스터 허용 비활성화
)
labels = clusterer.fit_predict(weighted_points)

# 노이즈 포인트는 검정색, 클러스터 포인트는 파란색으로 지정
colors = np.zeros((len(labels), 3))  # 기본 검정색 (노이즈)
colors[labels >= 0] = [0, 0, 1]  # 파란색으로 지정

final_point.colors = o3d.utility.Vector3dVector(colors)

# 필터링 기준 설정
min_points_in_cluster = 30   # 클러스터 내 최소 포인트 수
max_points_in_cluster = 150  # 클러스터 내 최대 포인트 수

# 수직(높이)
min_z_value = -1.5          # 클러스터 내 최소 Z값
max_z_value = 1.5           # 클러스터 내 최대 Z값

# 클러스터 자체의 높이
min_height = 0.2            # Z값 차이의 최소값
max_height = 2.0            # Z값 차이의 최대값

# 밀집도 기준
max_distance = 120.0        # 원점으로부터의 최대 거리

# 바운딩 박스 필터링 조건 추가
max_ratio = 3.0  # 가로/세로 비율의 최대 값 설정

# 1번, 2번, 3번 조건을 모두 만족하는 클러스터 필터링 및 바운딩 박스 생성
bboxes_1234 = []
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
                    bbox_color = (1, 0, 0)
                    
                    # 바운딩 박스의 크기 계산
                    extent = bbox.get_extent()
                    width = extent[0]  # X축 크기
                    length = extent[1]  # Y축 크기
                    height = extent[2]  # Z축 크기
                    
                    # 가로/세로 비율이 너무 큰 경우 제외
                    if width / length > max_ratio or length / width > max_ratio:
                        continue  # 비율이 너무 큰 경우, 해당 바운딩 박스를 제외
                    
                    bbox.color = bbox_color
                    bboxes_1234.append(bbox)

# 포인트 클라우드 및 바운딩 박스를 시각화하는 함수
def visualize_with_bounding_boxes(pcd, bounding_boxes, window_name="Filtered Clusters and Bounding Boxes", point_size=1.0):
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_name)
    
    # 포인트 클라우드와 바운딩 박스를 시각화
    vis.add_geometry(pcd)
    for bbox in bounding_boxes:
        vis.add_geometry(bbox)
    
    vis.get_render_option().point_size = point_size
    
    # # 카메라 시점과 확대 상태 설정
    # ctr = vis.get_view_control()
    
    # # 카메라 위치 설정을 위한 방향 및 시점 조정
    # ctr.set_lookat([0.0, 0.0, 0.0])  # 클러스터 중심을 바라봄
    # ctr.set_up([0.0, 0.0, 1.0])      # 위 방향 설정 (Z축 기준)
    # ctr.set_front([-0.5, -2.0, 1.0])  # 카메라의 바라보는 방향 설정 (눈 방향 벡터)
    
    # # 줌 레벨 설정 (0.0~1.0 범위로 조정)
    # ctr.set_zoom(0.08)                # 줌 레벨
    
    # 시각화 실행
    vis.run()
    vis.destroy_window()

# 시각화 (포인트 크기를 원하는 크기로 조절 가능)
visualize_with_bounding_boxes(final_point, bboxes_1234, point_size=1.0)
