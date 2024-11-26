import open3d as o3d
import numpy as np
import time
import os

# PCD 파일 경로 (PCD 파일이 연속적으로 있는 폴더)
pcd_folder_path = "C:/Users/estre/Downloads/COSE416_HW1_data_v1/data/01_straight_walk/pcd"  # 경로 수정

# PCD 파일 리스트
pcd_files = sorted([f for f in os.listdir(pcd_folder_path) if f.endswith(".pcd")])

# 포인트 클라우드를 시각화하는 함수
def visualize_continuous_with_zoom(pcd, window_name="Point Cloud", point_size=1.0):
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_name)
    vis.add_geometry(pcd)
        # 카메라 시점과 확대 상태 설정
    ctr = vis.get_view_control()
    
    # 카메라 위치 설정을 위한 방향 및 시점 조정
    ctr.set_lookat([0.0, 0.0, 0.0])  # 클러스터 중심을 바라봄
    ctr.set_up([0.0, 0.0, 1.0])      # 위 방향 설정 (Z축 기준)
    ctr.set_front([-0.5, -2.0, 1.0])  # 카메라의 바라보는 방향 설정 (눈 방향 벡터)
    
    # 줌 레벨 설정 (0.0~1.0 범위로 조정)
    ctr.set_zoom(0.08)                # 줌 레벨


    # 점 크기 조정
    vis.get_render_option().point_size = point_size
        # PCD 파일을 연속적으로 시각화
    for pcd_file in pcd_files:
        pcd_path = os.path.join(pcd_folder_path, pcd_file)
        
        # PCD 파일 읽기
        pcd = o3d.io.read_point_cloud(pcd_path)
        
        # 이전 포인트 클라우드를 지우고 새로운 클라우드 추가
        vis.clear_geometry()
        vis.add_geometry(pcd)
        
        # 시각화 실행
        vis.poll_events()
        vis.update_renderer()
        
        # 1초에 5프레임씩 시각화 (각 프레임은 0.2초마다 업데이트)
        time.sleep(0.2)  # 0.2초마다 새로운 프레임을 시각화

    vis.destroy_window()


# 시각화 함수 실행
visualize_continuous_with_zoom(pcd_folder_path, pcd_files)