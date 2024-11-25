import open3d as o3d
import numpy as np
import hdbscan
import time  # 프레임 간 시간 간격 조정을 위해 사용

# 주요 파라미터 설정
voxel_size = 0.1
min_points_in_cluster = 30
max_points_in_cluster = 150
min_z_value = -1.5
max_z_value = 1.5
min_height = 0.3
max_height = 2.0
max_distance = 120.0
max_ratio = 3.0
frame_rate = 5  # 초당 프레임 수 (1초에 5프레임)
time_interval = 1.0 / frame_rate  # 프레임 간 간격

# 포인트 클라우드 및 바운딩 박스를 시각화하는 함수
def visualize_with_bounding_boxes(pcd, bounding_boxes, window_name="Filtered Clusters and Bounding Boxes", point_size=1.0):
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_name)
    vis.add_geometry(pcd)
    for bbox in bounding_boxes:
        vis.add_geometry(bbox)
    vis.get_render_option().point_size = point_size
    vis.run()
    vis.destroy_window()

# 주요 처리 함수
def process_pcd(file_path):
    # PCD 파일 읽기
    original_pcd = o3d.io.read_point_cloud(file_path)
    
    # Voxel Downsampling
    downsample_pcd = original_pcd.voxel_down_sample(voxel_size=voxel_size)

    # Outlier Removal (SOR)
    cl, ind = downsample_pcd.remove_statistical_outlier(nb_neighbors=30, std_ratio=12.0)
    sor_pcd = downsample_pcd.select_by_index(ind)

    # Plane Segmentation (RANSAC)
    _, inliers = sor_pcd.segment_plane(distance_threshold=0.2, ransac_n=4, num_iterations=4000)
    final_point = sor_pcd.select_by_index(inliers, invert=True)

    # Z축 강조 가중치 적용
    weighted_points = np.asarray(final_point.points)
    weighted_points[:, 2] *= 1.2

    # HDBSCAN 클러스터링
    clusterer = hdbscan.HDBSCAN(min_cluster_size=15, min_samples=10, metric='euclidean', cluster_selection_epsilon=0.1)
    labels = clusterer.fit_predict(weighted_points)

    # 노이즈 및 클러스터 색상 지정
    colors = np.zeros((len(labels), 3))
    colors[labels >= 0] = [0, 0, 1]
    final_point.colors = o3d.utility.Vector3dVector(colors)

    # 바운딩 박스 필터링
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
                        bbox = cluster_pcd.get_axis_aligned_bounding_box()
                        extent = bbox.get_extent()
                        width, length = extent[0], extent[1]
                        if width / length > max_ratio or length / width > max_ratio:
                            continue
                        bbox.color = (1, 0, 0)
                        bboxes_1234.append(bbox)
    
    # 시각화
    visualize_with_bounding_boxes(final_point, bboxes_1234, window_name=f"PCD: {file_path}", point_size=2.0)

# 여러 PCD 파일 처리
pcd_files = [
"C:/Users/estre/OneDrive/Desktop/개발/20242R0136COSE41600/COSE416_HW1_tutorial/COSE416_HW1_tutorial/test_data/1727320101-665925967.pcd",
"C:/Users/estre/OneDrive/Desktop/개발/20242R0136COSE41600/COSE416_HW1_tutorial/COSE416_HW1_tutorial/test_data/1727320101-961578277.pcd",
"C:/Users/estre/OneDrive/Desktop/개발/20242R0136COSE41600/COSE416_HW1_tutorial/COSE416_HW1_tutorial/test_data/1727320102-53276943.pcd",
"C:/Users/estre/OneDrive/Desktop/개발/20242R0136COSE41600/COSE416_HW1_tutorial/COSE416_HW1_tutorial/test_data/1727320102-153284974.pcd",
"C:/Users/estre/OneDrive/Desktop/개발/20242R0136COSE41600/COSE416_HW1_tutorial/COSE416_HW1_tutorial/test_data/1727320102-253066301.pcd",

]

# 1초에 5프레임으로 PCD 처리
while True:  # 파일 리스트를 반복적으로 처리
    for pcd_file in pcd_files:
        process_pcd(pcd_file)
        time.sleep(time_interval)  # 프레임 간 시간 간격 유지
