# 시각화에 필요한 라이브러리 불러오기
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import hdbscan

# pcd 파일 불러오기, 필요에 맞게 경로 수정
#straight_walk
file_path = "C:/Users/estre/Downloads/COSE416_HW1_data_v1/data/05_straight_duck_walk/pcd/pcd_000416.pcd"

# PCD 파일 읽기
original_pcd = o3d.io.read_point_cloud(file_path)

# Voxel Downsampling 수행
voxel_size = 0.1  # 필요에 따라 voxel 크기를 조정하세요.
downsample_pcd = original_pcd.voxel_down_sample(voxel_size=voxel_size)

# Radius Outlier Removal (ROR) 적용
# cl, ind = downsample_pcd.remove_radius_outlier(nb_points=10, radius=2.0)
# ror_pcd = downsample_pcd.select_by_index(ind)

# SOR (Statistical Outlier Removal)
cl, ind = downsample_pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=6.0)
sor_pcd = downsample_pcd.select_by_index(ind)

# RANSAC을 사용하여 평면 추정
plane_model, inliers = sor_pcd.segment_plane(distance_threshold=0.1,
                                             ransac_n=5,
                                             num_iterations=4000)


# 도로에 속하지 않는 포인트 (outliers) 추출
final_point = sor_pcd.select_by_index(inliers, invert=True)


# DBSCAN 클러스터링 적용
# with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
#     labels = np.array(final_point.cluster_dbscan(eps=0.25, min_points=13, print_progress=True))


# clusterer = hdbscan.HDBSCAN(min_cluster_size=20, min_samples=15)
# labels = clusterer.fit_predict(np.asarray(final_point.points))


# Z축 강조 가중치 적용, 적당히 긴 사람 형상 위해서,,
weighted_points = np.asarray(final_point.points)
weighted_points[:, 2] *= 1.3  # Z축에 가중치 추가

# HDBSCAN 클러스터링
clusterer = hdbscan.HDBSCAN(
    #20이하인데 이게..(20)
    min_cluster_size=15,      # 클러스터 내 최소 점 개수
    #10-15 유지 (13)
    min_samples=10,            # 핵심 점으로 간주할 최소 이웃 수
    metric='euclidean',       # 거리 척도
    cluster_selection_epsilon=0.2,  # 클러스터 선택 임계값
    allow_single_cluster=False  # 하나의 클러스터 허용 비활성화
)
labels = clusterer.fit_predict(weighted_points)



# 노이즈 포인트는 검정색, 클러스터 포인트는 파란색으로 지정
colors = np.zeros((len(labels), 3))  # 기본 검정색 (노이즈)
colors[labels >= 0] = [0, 0, 1]  # 파란색으로 지정

final_point.colors = o3d.utility.Vector3dVector(colors)

# 필터링 기준 설정
min_points_in_cluster = 50   # 클러스터 내 최소 포인트 수
max_points_in_cluster = 500  # 클러스터 내 최대 포인트 수

# 수직(높이)
# 차량 위 센서 기준임 (사람키기준)
min_z_value = -2.0          # 클러스터 내 최소 Z값
# 이거 높이니까 되는데..?
max_z_value = 5.0         # 클러스터 내 최대 Z값

# 클러스터 자체의 높이
# 키가 낮은 객체(아이, 앉아 있는 사람 등)도 포함하려면 min_height를 낮추는 것이 좋습니다.
min_height = 0.3            # Z값 차이의 최소값
max_height = 5.0           # Z값 차이의 최대값

#밀집도 기준, 동적인 사람도 포함해야 한다(보폭 커질 떄 등)
max_distance = 150.0         # 원점으로부터의 최대 거리

# 바운딩 박스 필터링 조건 추가 (너무 넓은 바운딩 박스 제외)
max_ratio = 4.0  # 가로/세로 비율의 최대 값 설정

# # 1번, 2번, 3번 조건을 모두 만족하는 클러스터 필터링 및 바운딩 박스 생성


# 필터링 조건을 만족하는 객체 추출 및 바운딩 박스 생성
def filter_and_visualize_clusters(remaining_pcd, labels, threshold_ratio=3.0):
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
                            continue  # 비율이 너무 큰 경우, 해당 바운딩 박스를 제외
                        # XY 면적 / Z 높이 비율 기반 필터링
                        xy_area = width * length
                        if xy_area / height <= threshold_ratio:
                            bbox.color = (1, 0, 0)  # 빨간색
                            bboxes.append(bbox)
    return bboxes

# 클러스터 필터링 및 바운딩 박스 생성
filtered_bboxes = filter_and_visualize_clusters(final_point, labels, threshold_ratio=5.0)

# 시각화 함수
def visualize_with_bounding_boxes(pcd, bounding_boxes, window_name="Filtered Clusters and Bounding Boxes", point_size=1.0):
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_name)
    vis.add_geometry(pcd)
    for bbox in bounding_boxes:
        vis.add_geometry(bbox)
    vis.get_render_option().point_size = point_size
    
        # 뷰 컨트롤러 가져오기
    view_ctl = vis.get_view_control()

    # 카메라 줌 설정
    view_ctl.set_zoom(0.3)  # 줌 비율 (기본값은 1.0, 더 작은 값으로 확대 가능)
    view_ctl.set_front([0, -1.5, 1])  # 카메라가 점군을 보는 방향
    view_ctl.set_lookat([0, 0, 3])  # 카메라가 보는 중심점
    view_ctl.set_up([0, 0.5, 0])     # 카메라의 위쪽 방향
    
    vis.run()
    vis.destroy_window()

# 결과 시각화
visualize_with_bounding_boxes(final_point, filtered_bboxes, point_size=1.0)