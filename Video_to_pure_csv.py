import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse


def calculate_angle(a, b, c):
    """세 점 사이의 각도를 계산하는 함수 (0-180도)"""
    a = np.array(a)  # 첫 번째 점
    b = np.array(b)  # 중간 점
    c = np.array(c)  # 세 번째 점

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    # 각도가 180도를 넘으면 360에서 빼서 작은 각을 반환
    if angle > 180.0:
        angle = 360 - angle
    return angle


def process_video(video_path, label):
    """
    비디오 파일을 분석하여 자세 데이터를 추출하고, 결과 영상과 데이터프레임을 반환합니다.

    Args:
        video_path (str): 분석할 비디오 파일의 경로.
        label (str): 'correct' 또는 'incorrect'와 같이 데이터에 부여할 라벨.

    Returns:
        pd.DataFrame: 추출된 자세 데이터가 담긴 데이터프레임.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"오류: '{video_path}' 파일을 찾을 수 없습니다.")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"오류: '{video_path}' 영상을 열 수 없습니다.")

    # 영상 기본 정보 설정
    h, w = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = cap.get(cv2.CAP_PROP_FPS)
    video_name = os.path.basename(video_path)
    output_video_path = f"{os.path.splitext(video_name)[0]}_output.mp4"

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (w, h))

    # Mediapipe Pose 모델 초기화
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    frame_idx = 0
    data_rows = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # BGR 이미지를 RGB로 변환
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False  # 성능 향상을 위해 이미지 쓰기 불가로 설정

        # 자세 추정 수행
        results = pose.process(image_rgb)

        # 다시 BGR로 변환하여 화면에 표시
        image_rgb.flags.writeable = True
        image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark

            # 주요 관절 좌표 추출
            shoulder = [lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * w,
                        lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * h]
            hip = [lm[mp_pose.PoseLandmark.RIGHT_HIP.value].x * w, lm[mp_pose.PoseLandmark.RIGHT_HIP.value].y * h]
            knee = [lm[mp_pose.PoseLandmark.RIGHT_KNEE.value].x * w, lm[mp_pose.PoseLandmark.RIGHT_KNEE.value].y * h]
            ankle = [lm[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x * w, lm[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y * h]

            # 각도 계산 (180도 기준)
            knee_angle = calculate_angle(hip, knee, ankle)
            hip_angle = calculate_angle(shoulder, hip, knee)

            # 데이터 저장
            data_rows.append(
                [video_name, frame_idx, 'RIGHT_SHOULDER', shoulder[0], shoulder[1], 'shoulder', None, label])
            data_rows.append([video_name, frame_idx, 'RIGHT_HIP', hip[0], hip[1], 'hip', round(hip_angle, 2), label])
            data_rows.append(
                [video_name, frame_idx, 'RIGHT_KNEE', knee[0], knee[1], 'knee', round(knee_angle, 2), label])
            data_rows.append([video_name, frame_idx, 'RIGHT_ANKLE', ankle[0], ankle[1], 'ankle', None, label])

            # 화면에 랜드마크 및 각도 정보 그리기
            mp.solutions.drawing_utils.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            cv2.putText(image, f'Knee: {int(knee_angle)}', tuple(np.int32(knee)), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(image, f'Hip: {int(hip_angle)}', tuple(np.int32(hip)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0),
                        2, cv2.LINE_AA)

        out.write(image)
        frame_idx += 1

    # 리소스 해제
    cap.release()
    out.release()
    pose.close()

    print(f"✅ 분석 완료! 영상 저장됨: {output_video_path}")

    df = pd.DataFrame(data_rows, columns=['idx', 'frame', 'joint', 'x', 'y', 'angle_type', 'angle', 'label'])
    return df


def visualize_angles(df):
    """데이터프레임을 받아 관절 각도 변화를 시각화합니다."""
    plt.figure(figsize=(12, 6))

    knee_df = df[df['angle_type'] == 'knee'].dropna()
    hip_df = df[df['angle_type'] == 'hip'].dropna()

    plt.plot(knee_df['frame'], knee_df['angle'], label='Knee Angle', color='green', marker='.', linestyle='-')
    plt.plot(hip_df['frame'], hip_df['angle'], label='Hip Angle', color='blue', marker='.', linestyle='-')

    plt.xlabel('Frame')
    plt.ylabel('Angle (degrees)')
    plt.title('Joint Angles Over Time')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # 명령어 인자 파서 설정
    parser = argparse.ArgumentParser(description="비디오에서 스쿼트 자세를 분석하여 CSV로 저장합니다.")
    parser.add_argument('--video', type=str, required=True, help="분석할 비디오 파일 경로")
    parser.add_argument('--label', type=str, required=True, choices=['correct', 'incorrect'],
                        help="데이터에 부여할 라벨 ('correct' 또는 'incorrect')")

    args = parser.parse_args()

    # 비디오 처리 및 데이터 추출
    extracted_data = process_video(args.video, args.label)

    # CSV 파일로 저장
    output_csv_path = f"{os.path.splitext(args.video)[0]}_output.csv"
    extracted_data.to_csv(output_csv_path, index=False)
    print(f"✅ CSV 저장 완료: {output_csv_path}")

    # 결과 시각화
    visualize_angles(extracted_data)
