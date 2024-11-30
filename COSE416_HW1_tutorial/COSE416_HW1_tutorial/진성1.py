import open3d as o3d

def visualize_single_frame(pcd_file, save_camera_params='camera_params.json'):
    pcd = o3d.io.read_point_cloud(pcd_file)
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Set Camera View", width=1920, height=1080)  # 창의 크기를 지정
    vis.add_geometry(pcd)
    vis.run()  # 시각화 창이 열리고, 카메라 시점을 조정할 수 있습니다.
    
    # 카메라 파라미터 저장
    param = vis.get_view_control().convert_to_pinhole_camera_parameters()
    o3d.io.write_pinhole_camera_parameters(save_camera_params, param)
    print(f"카메라 파라미터가 {save_camera_params}에 저장되었습니다.")

    vis.destroy_window()

if __name__ == '__main__':
    sample_pcd = "C:/Users/estre/OneDrive/Desktop/개발/20242R0136COSE41600/COSE416_HW1_tutorial/COSE416_HW1_tutorial/test_data/1727320101-665925967.pcd"
  # 예시 파일 경로
    visualize_single_frame(sample_pcd)

