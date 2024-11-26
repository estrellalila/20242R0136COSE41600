import open3d as o3d
import numpy as np
from joblib import Parallel, delayed

# 병렬 처리 기반 국소 평면 적합 함수
def compute_normal(kdtree, points, idx, radius):
    _, indices, _ = kdtree.search_radius_vector_3d(points[idx], radius)
    neighbors = points[indices]
    if len(neighbors) < 3:  # 최소한 3개의 이웃이 있어야 평면 계산 가능
        return np.array([0, 0, 0])
    cov_matrix = np.cov(neighbors.T)
    eigvals, eigvecs = np.linalg.eig(cov_matrix)
    normal = eigvecs[:, np.argmin(eigvals)]  # 가장 작은 고유값의 고유벡터
    return normal

def local_plane_fit_parallel(pcd, radius=1.0, n_jobs=-1):
    points = np.asarray(pcd.points)
    kdtree = o3d.geometry.KDTreeFlann(pcd)
    normals = Parallel(n_jobs=n_jobs)(
        delayed(compute_normal)(kdtree, points, i, radius) for i in range(len(points))
    )
    return np.array(normals)

# 포인트 클라우드 파일 경로
file_path = "C:/Users/estre/Downloads/COSE416_HW1_data_v1/data/05_straight_duck_walk/pcd/pcd_000577.pcd"

# 1. PCD 파일 읽기
original_pcd = o3d.io.read_point_cloud(file_path)

# 2. Voxel Downsampling (점 개수 줄이기)
voxel_size = 0.2  # 적절히 설정해 점 개수 감소
downsample_pcd = original_pcd.voxel_down_sample(voxel_size=voxel_size)

# 3. Statistical Outlier Removal (SOR)
nb_neighbors = 20
std_ratio = 6.0
_, ind = downsample_pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
sor_pcd = downsample_pcd.select_by_index(ind)

# 4. 국소 평면 적합 (법선 벡터 계산)
radius = 0.5  # 반경 설정 (최적화 필요)
normals = local_plane_fit_parallel(sor_pcd, radius=radius)
sor_pcd.normals = o3d.utility.Vector3dVector(normals)

# 5. 평면 영역 표시
# 법선 벡터를 색상으로 표시 (R, G, B)
colors = (normals - normals.min(axis=0)) / (normals.max(axis=0) - normals.min(axis=0))
sor_pcd.colors = o3d.utility.Vector3dVector(colors)

# 6. RANSAC 평면 추출
plane_model, inliers = sor_pcd.segment_plane(
    distance_threshold=0.2, ransac_n=6, num_iterations=1000
)

# 도로와 비도로 영역 분리
road_pcd = sor_pcd.select_by_index(inliers)
non_road_pcd = sor_pcd.select_by_index(inliers, invert=True)

# 7. 시각화 함수 정의
def visualize_point_clouds(pcd_list, window_name="Point Cloud Visualization", point_size=1.0):
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_name)
    for pcd in pcd_list:
        vis.add_geometry(pcd)
    vis.get_render_option().point_size = point_size
    vis.run()
    vis.destroy_window()

# 8. 시각화
# 법선 벡터를 색상으로 표시한 포인트 클라우드
visualize_point_clouds([sor_pcd], window_name="Normals Visualization", point_size=2.0)

# 도로 영역(빨강)과 비도로 영역(파랑) 시각화
road_pcd.paint_uniform_color([1, 0, 0])  # 빨강
non_road_pcd.paint_uniform_color([0, 0, 1])  # 파랑
visualize_point_clouds([road_pcd, non_road_pcd], window_name="Road and Non-Road", point_size=2.0)
