import pandas as pd
import io


def transform_squat_data(input_file_path, output_file_path):
    """
    스쿼트 자세 데이터를 프레임별로 정제하여 새로운 CSV 파일로 저장합니다.

    Input Format (한 프레임에 여러 줄):
    - columns: idx, frame, joint, x, y, angle_type, angle, label

    Output Format (한 프레임에 한 줄):
    - columns: shoulder_horizon, shoulder_vertical, hip_horizon, hip_vertical,
               knee_horizon, knee_vertical, ankle_horizon, ankle_vertical,
               hip_angle, knee_angle

    Args:
        input_file_path (str or file-like object): 입력 CSV 파일 경로 또는 객체.
        output_file_path (str): 저장할 CSV 파일 경로.
    """
    try:
        print(f"'{input_file_path}' 파일에서 데이터를 읽고 있습니다...")
        df = pd.read_csv(input_file_path)
    except FileNotFoundError:
        print(f"오류: '{input_file_path}' 파일을 찾을 수 없습니다. 파일 경로를 확인해주세요.")
        return

    # 1. 좌표(x, y) 데이터 정제
    # 'frame'을 기준으로 'joint'의 x, y 좌표를 펼칩니다.
    coords_df = df.pivot_table(index='frame', columns='joint', values=['x', 'y'])

    # 생성된 다중 레벨 컬럼 이름을 단일 레벨로 변경합니다. (예: ('x', 'RIGHT_SHOULDER') -> 'x_RIGHT_SHOULDER')
    coords_df.columns = [f'{val}_{joint.lower()}' for val, joint in coords_df.columns]

    # 2. 각도(angle) 데이터 정제
    # 각도 값이 있는 행만 필터링합니다.
    angles_df = df.dropna(subset=['angle_type', 'angle'])
    # 'frame'을 기준으로 'angle_type'의 angle 값을 펼칩니다.
    angles_df = angles_df.pivot_table(index='frame', columns='angle_type', values='angle')

    # 3. 좌표 데이터와 각도 데이터 병합
    # frame 인덱스를 기준으로 두 데이터프레임을 합칩니다.
    final_df = coords_df.join(angles_df)

    # 4. 최종 컬럼 이름 변경 및 순서 지정
    # 요구사항에 맞게 컬럼 이름을 변경합니다.
    rename_map = {
        'x_right_shoulder': 'shoulder_horizon', 'y_right_shoulder': 'shoulder_vertical',
        'x_right_hip': 'hip_horizon', 'y_right_hip': 'hip_vertical',
        'x_right_knee': 'knee_horizon', 'y_right_knee': 'knee_vertical',
        'x_right_ankle': 'ankle_horizon', 'y_right_ankle': 'ankle_vertical',
        'hip': 'hip_angle',
        'knee': 'knee_angle'
    }
    final_df = final_df.rename(columns=rename_map)

    # 최종 컬럼 순서를 정의합니다.
    output_columns = [
        'shoulder_horizon', 'shoulder_vertical',
        'hip_horizon', 'hip_vertical',
        'knee_horizon', 'knee_vertical',
        'ankle_horizon', 'ankle_vertical',
        'hip_angle', 'knee_angle'
    ]

    # 누락된 컬럼이 있을 경우를 대비하여 확인 후, 순서를 맞춥니다.
    # (예: 특정 프레임에 특정 관절이 없는 경우)
    for col in output_columns:
        if col not in final_df.columns:
            final_df[col] = None  # 없는 컬럼은 빈 값으로 추가

    final_df = final_df[output_columns]

    # 5. CSV 파일로 저장
    final_df.to_csv(output_file_path, index=True)  # index=True로 'frame' 컬럼을 유지
    print(f"데이터 정제가 완료되었습니다. 결과가 '{output_file_path}' 파일에 저장되었습니다.")

    # 결과 미리보기 출력
    print("\n--- 정제된 데이터 미리보기 (상위 5개 행) ---")
    print(final_df.head())


# --- 메인 실행 부분 ---


# # --- 실제 파일 사용 ---
# # 위 테스트 부분을 주석 처리하고 아래 코드의 주석을 해제하여 사용하세요.
input_filename = 'dataset/csv/correct_pure_data.csv'
output_filename = 'dataset/csv/correct_manipulated_data.csv'
transform_squat_data(input_filename, output_filename)
