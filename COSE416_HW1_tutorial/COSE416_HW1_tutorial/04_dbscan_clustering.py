# 시각화에 필요한 라이브러리 불러오기
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import hdbscan

# pcd 파일 불러오기, 필요에 맞게 경로 수정
file_path = "C:/Users/estre/Downloads/COSE416_HW1_tutorial/COSE416_HW1_tutorial/test_data/1727320101-665925967.pcd"
#file_path = "C:/Users/estre/Downloads/COSE416_HW1_tutorial/COSE416_HW1_tutorial/test_data/1727320101-961578277.pcd"

# PCD 파일 읽기
original_pcd = o3d.io.read_point_cloud(file_path)

# Voxel Downsampling 수행
voxel_size = 0.05  # 필요에 따라 voxel 크기를 조정
downsample_pcd = original_pcd.voxel_down_sample(voxel_size=voxel_size)

# Radius Outlier Removal (ROR) 적용
# cl, ind = downsample_pcd.remove_radius_outlier(nb_points=10, radius=2.0)
# ror_pcd = downsample_pcd.select_by_index(ind)

# SOR (Statistical Outlier Removal)
cl, ind = downsample_pcd.remove_statistical_outlier(nb_neighbors=50, std_ratio=2.0)
sor_pcd = downsample_pcd.select_by_index(ind)

# RANSAC을 사용하여 평면 추정
plane_model, inliers = sor_pcd.segment_plane(distance_threshold=0.2,
                                             ransac_n=4,
                                             num_iterations=4000)

# 도로에 속하지 않는 포인트 (outliers) 추출
final_point = sor_pcd.select_by_index(inliers, invert=True)

# DBSCAN 클러스터링 적용
# epsilon은 작아야 구별이 잘 되는 듯
# with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
#     labels = np.array(final_point.cluster_dbscan(eps=0.25, min_points=13, print_progress=True))


# clusterer = hdbscan.HDBSCAN(min_cluster_size=13, min_samples = 8)
# labels = clusterer.fit_predict(np.asarray(final_point.points))


# Z축 강조 가중치 적용
weighted_points = np.asarray(final_point.points)
weighted_points[:, 2] *= 2.0  # Z축에 가중치 추가

# HDBSCAN 클러스터링
clusterer = hdbscan.HDBSCAN(
    min_cluster_size=20,      # 클러스터 내 최소 점 개수
    min_samples=10,            # 핵심 점으로 간주할 최소 이웃 수
    metric='manhattan',       # 거리 척도
    cluster_selection_epsilon=0.1,  # 클러스터 선택 임계값
    allow_single_cluster=False  # 하나의 클러스터 허용 비활성화
)
labels = clusterer.fit_predict(weighted_points)

# 각 클러스터를 색으로 표시
max_label = labels.max()
print(f"point cloud has {max_label + 1} clusters")

# 노이즈를 제거하고 각 클러스터에 색상 지정
colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
colors[labels < 0] = 0  # 노이즈는 검정색으로 표시
final_point.colors = o3d.utility.Vector3dVector(colors[:, :3])


# 포인트 클라우드 시각화 함수
def visualize_point_cloud_with_point_size(pcd, window_name="Point Cloud Visualization", point_size=1.0):
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_name)
    vis.add_geometry(pcd)
    vis.get_render_option().point_size = point_size
    vis.run()
    vis.destroy_window()

# 시각화 (포인트 크기를 원하는 크기로 조절 가능)
visualize_point_cloud_with_point_size(final_point, 
                                      window_name="DBSCAN Clustered Points", point_size=2.0)

