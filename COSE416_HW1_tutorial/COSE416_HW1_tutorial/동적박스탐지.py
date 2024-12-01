import os
import time
import open3d as o3d
import numpy as np
import hdbscan

# 특정 폴더에서 PCD 파일 목록 불러오기
def get_pcd_files_from_folder(folder_path):
    files = sorted([os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.pcd')])
    return files

# 이전 프레임과의 바운딩 박스 중심 비교
def filter_static_objects(current_bboxes, previous_bboxes, threshold=0.5):
    dynamic_bboxes = []
    for current_bbox in current_bboxes:
        current_center = current_bbox.get_center()  # 현재 바운딩 박스 중심
        
        is_static = False
        for prev_bbox in previous_bboxes:
            prev_center = prev_bbox.get_center()  # 이전 바운딩 박스 중심
            
            # 중심 거리 계산
            distance = np.linalg.norm(np.array(current_center) - np.array(prev_center))
            if distance < threshold:
                is_static = True
                break
        
        if not is_static:
            dynamic_bboxes.append(current_bbox)  # 동적 객체만 추가
    
    return dynamic_bboxes

# 클러스터 필터링 및 바운딩 박스 생성 함수
def filter_and_visualize_clusters(remaining_pcd, labels, threshold_ratio=0.5, max_side_area_ratio=2.5):
    bboxes = []
    for i in range(labels.max() + 1):
        cluster_indices = np.where(labels == i)[0]
        if 30 <= len(cluster_indices) <= 150:
            cluster_pcd = remaining_pcd.select_by_index(cluster_indices)
            points = np.asarray(cluster_pcd.points)
            z_values = points[:, 2]
            z_min, z_max = z_values.min(), z_values.max()
            
            if -1.5 <= z_min and z_max <= 1.5:
                height_diff = z_max - z_min
                if 0.3 <= height_diff <= 2.0:
                    distances = np.linalg.norm(points, axis=1)
                    if distances.max() <= 120.0:
                        bbox = cluster_pcd.get_axis_aligned_bounding_box()
                        extent = bbox.get_extent()
                        width, length, height = extent[0], extent[1], extent[2]
                        
                        # XY 면적 / Z 높이 비율 기반 필터링
                        xy_area = width * length
                        if xy_area / height <= threshold_ratio:
                            # 옆면적 비율 필터링
                            side_area_1 = width * height
                            side_area_2 = length * height
                            side_area_ratio = max(side_area_1, side_area_2) / min(side_area_1, side_area_2)
                            
                            if side_area_ratio <= max_side_area_ratio:
                                bbox.color = (1, 0, 0)  # 빨간색
                                bboxes.append(bbox)
    return bboxes

# 동영상처럼 프레임을 시각화
def visualize_frames_as_video(folder_path, fps=5, threshold=0.5):
    frame_files = get_pcd_files_from_folder(folder_path)
    if not frame_files:
        print("폴더에 PCD 파일이 없습니다.")
        return
    
    interval = 1.0 / fps  # 프레임 간 간격
    vis = o3d.visualization.Visualizer()
    vis.create_window("Dynamic Object Detection", width=1280, height=720)
    
    previous_bboxes = []
    geometry_added = False
    
    for file_path in frame_files:
        print(f"Processing: {file_path}")
        # PCD 데이터 로드 및 전처리
        original_pcd = o3d.io.read_point_cloud(file_path)
        downsample_pcd = original_pcd.voxel_down_sample(voxel_size=0.1)
        cl, ind = downsample_pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=6.0)
        sor_pcd = downsample_pcd.select_by_index(ind)
        plane_model, inliers = sor_pcd.segment_plane(distance_threshold=0.2, ransac_n=4, num_iterations=4000)
        final_point = sor_pcd.select_by_index(inliers, invert=True)
        weighted_points = np.asarray(final_point.points)
        weighted_points[:, 2] *= 1.2
        clusterer = hdbscan.HDBSCAN(min_cluster_size=20, min_samples=13, metric='euclidean')
        labels = clusterer.fit_predict(weighted_points)
        
        # 바운딩 박스 생성
        current_bboxes = filter_and_visualize_clusters(final_point, labels, threshold_ratio=5.0)
        dynamic_bboxes = filter_static_objects(current_bboxes, previous_bboxes, threshold=threshold)
        
        # 시각화 업데이트
        if not geometry_added:
            vis.add_geometry(final_point)
            for bbox in dynamic_bboxes:
                vis.add_geometry(bbox)
            geometry_added = True
        else:
            vis.update_geometry(final_point)
            for bbox in dynamic_bboxes:
                vis.update_geometry(bbox)
        
        vis.poll_events()
        vis.update_renderer()
        
        time.sleep(interval)  # FPS 맞추기
        
        # 이전 바운딩 박스 업데이트
        previous_bboxes = current_bboxes
    
    vis.destroy_window()

# PCD 파일이 저장된 폴더 경로
folder_path = "C:/Users/estre/Downloads/COSE416_HW1_data_v1/data/01_straight_walk/pcd"  # 여기에 PCD 파일 경로 입력
visualize_frames_as_video(folder_path, fps=5, threshold=0.5)
