import numpy as np
import open3d as o3d
import math

def load_point_cloud(pcd_path):
    points = np.load(pcd_path)
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points[:, :3])  # x, y, z 좌표 사용
    return point_cloud

def load_detections(result_file, score_threshold=0.2):
    pred_boxes = []
    with open(result_file, 'r') as f:
        for line in f:
            elements = line.strip().split()
            if len(elements) < 11:
                continue  # 잘못된 형식의 라인 스킵

            # x, y, z, dx, dy, dz, yaw, vx, vy, score, label 순서라고 가정
            box = list(map(float, elements[:7]))  # x, y, z, dx, dy, dz, yaw
            score = float(elements[9])
            label = int(elements[10])

            # 특정 조건: 보행자(label == 9)만 선택하고 SCORE 기준 적용
            if label == 9 and score >= score_threshold:
                pred_boxes.append(box)
                print(box)

    return np.array(pred_boxes)

def get_rotation_matrix(yaw):
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    rotation_matrix = np.array([
        [cos_yaw, -sin_yaw, 0],
        [sin_yaw, cos_yaw, 0],
        [0, 0, 1]
    ])
    return rotation_matrix

def visualize(point_cloud, pred_boxes, camera_params_path='camera_params.json'):
    vis = o3d.visualization.Visualizer()

    vis.create_window(window_name="Visualization", width=1920, height=1080)
    vis.add_geometry(point_cloud)

    for box in pred_boxes:
        center = box[:3]
        size = box[3:6]
        yaw = box[6]

        # 바운딩 박스 생성
        bbox = o3d.geometry.OrientedBoundingBox()
        bbox.center = center
        bbox.extent = size

        R = get_rotation_matrix(yaw)
        bbox.R = R
        bbox.color = (1, 0, 0)  # 빨간색

        vis.add_geometry(bbox)

    vis.get_render_option().point_size = 2.0  # 포인트 크기 조절

    # 저장된 카메라 파라미터 적용
    param = o3d.io.read_pinhole_camera_parameters(camera_params_path)
    ctr = vis.get_view_control()
    ctr.convert_from_pinhole_camera_parameters(param, allow_arbitrary=True)

    vis.run()
    vis.destroy_window()

def main():
    # 포인트 클라우드 및 결과 파일 경로 설정
    pcd_path = "C:/Users/estre/OneDrive/Desktop/개발/20242R0136COSE41600/COSE416_HW1_tutorial/COSE416_HW1_tutorial/test_data/1727320101-665925967.pcd"
    result_file = "C:/Users/estre/OneDrive/Desktop/개발/20242R0136COSE41600/COSE416_HW1_tutorial/COSE416_HW1_tutorial/test_data/1727320101-665925967.txt"

    # 포인트 클라우드 로드
    point_cloud = load_point_cloud(pcd_path)

    # 추론 결과 로드 (보행자만)
    pred_boxes = load_detections(result_file)

    # 시각화 실행 (저장된 카메라 파라미터 사용)
    visualize(point_cloud, pred_boxes, camera_params_path='camera_params.json')

if __name__ == '__main__':
    main()
