import pandas as pd
import numpy as np
from scipy.signal import find_peaks
import argparse
import os


def transform_raw_data_to_manipulated(input_path):
    """
    1단계: 원본 CSV 데이터를 읽어 프레임당 한 줄로 정제합니다.
    (코드 1의 역할)
    """
    print(f"--- 1단계: 원본 데이터 정제 시작 ---")
    print(f"'{input_path}' 파일에서 데이터를 읽고 있습니다...")
    try:
        df = pd.read_csv(input_path)
    except FileNotFoundError:
        print(f"오류: '{input_path}' 파일을 찾을 수 없습니다.")
        return None

    # 좌표(x, y) 데이터 정제
    coords_df = df.pivot_table(index='frame', columns='joint', values=['x', 'y'])
    coords_df.columns = [f'{val}_{joint.lower()}' for val, joint in coords_df.columns]

    # 각도(angle) 데이터 정제
    angles_df = df.dropna(subset=['angle_type', 'angle'])
    angles_df = angles_df.pivot_table(index='frame', columns='angle_type', values='angle')

    # 데이터 병합
    manipulated_df = coords_df.join(angles_df).reset_index()

    # 컬럼 이름 변경 및 순서 지정
    rename_map = {
        'x_right_shoulder': 'shoulder_horizon', 'y_right_shoulder': 'shoulder_vertical',
        'x_right_hip': 'hip_horizon', 'y_right_hip': 'hip_vertical',
        'x_right_knee': 'knee_horizon', 'y_right_knee': 'knee_vertical',
        'x_right_ankle': 'ankle_horizon', 'y_right_ankle': 'ankle_vertical',
        'hip': 'hip_angle',
        'knee': 'knee_angle'
    }
    manipulated_df = manipulated_df.rename(columns=rename_map)

    output_columns = [
        'frame', 'shoulder_horizon', 'shoulder_vertical', 'hip_horizon', 'hip_vertical',
        'knee_horizon', 'knee_vertical', 'ankle_horizon', 'ankle_vertical',
        'hip_angle', 'knee_angle'
    ]
    for col in output_columns:
        if col not in manipulated_df.columns:
            manipulated_df[col] = np.nan

    manipulated_df = manipulated_df[output_columns]
    print("✅ 1단계 완료: 데이터가 성공적으로 정제되었습니다.")
    return manipulated_df


def process_manipulated_data_to_final(input_df, output_path):
    """
    2단계: 정제된 데이터를 분석하여 최종 결과물을 생성합니다.
    (코드 2의 역할)
    """
    print(f"\n--- 2단계: 스쿼트 사이클 분석 및 정규화 시작 ---")

    # 'frame'을 인덱스로 설정
    input_df = input_df.set_index('frame')

    # 스쿼트 구간 판별
    y_coords_hip = input_df['hip_vertical'].values
    prominence_threshold = 10
    distance_threshold = 10
    troughs, _ = find_peaks(y_coords_hip, prominence=prominence_threshold, distance=distance_threshold)
    peaks, _ = find_peaks(-y_coords_hip, prominence=prominence_threshold, distance=distance_threshold)

    if len(peaks) < 2 or len(troughs) == 0:
        print("오류: 분석할 수 있는 완전한 스쿼트 사이클을 찾지 못했습니다.")
        return

    print(f"총 {len(peaks)}개의 최고점(선 자세)과 {len(troughs)}개의 최저점(앉은 자세)을 감지했습니다.")

    final_results = []
    squat_idx_counter = 0

    for i in range(len(peaks) - 1):
        start_peak_idx = peaks[i]
        end_peak_idx = peaks[i + 1]
        troughs_in_between = [t for t in troughs if start_peak_idx < t < end_peak_idx]

        if not troughs_in_between:
            continue

        squat_idx_counter += 1

        # 핵심 프레임 추출
        stand_frame_idx = start_peak_idx
        sit_frame_idx = max(troughs_in_between, key=lambda t: y_coords_hip[t])
        stand_frame = input_df.index[stand_frame_idx]
        sit_frame = input_df.index[sit_frame_idx]

        # 길이 계산용 사이클 데이터
        cycle_df = input_df.loc[input_df.index[start_peak_idx]:input_df.index[end_peak_idx]]
        min_shoulder_y = cycle_df['shoulder_vertical'].min()
        max_ankle_y = cycle_df['ankle_vertical'].max()
        vertical_length = max_ankle_y - min_shoulder_y
        max_knee_x = cycle_df['knee_horizon'].max()
        min_hip_x = cycle_df['hip_horizon'].min()
        horizon_length = max_knee_x - min_hip_x

        if vertical_length == 0 or horizon_length == 0:
            print(f"경고: {squat_idx_counter}번 스쿼트의 길이가 0입니다. 계산을 건너뜁니다.")
            continue

        stand_data = input_df.loc[stand_frame].copy()
        sit_data = input_df.loc[sit_frame].copy()
        processed_data = {'squat_idx': squat_idx_counter}

        # 좌표 변환 및 정규화
        for data_type, data_series in [('stand', stand_data), ('sit', sit_data)]:
            ankle_origin_h = data_series['ankle_horizon']
            ankle_origin_v = data_series['ankle_vertical']
            for joint in ['shoulder', 'hip', 'knee', 'ankle']:
                h_col, v_col = f'{joint}_horizon', f'{joint}_vertical'
                translated_h = data_series[h_col] - ankle_origin_h
                translated_v = data_series[v_col] - ankle_origin_v
                processed_data[f'{data_type}_{h_col}'] = translated_h / horizon_length
                processed_data[f'{data_type}_{v_col}'] = translated_v / vertical_length
            processed_data[f'{data_type}_hip_angle'] = data_series['hip_angle']
            processed_data[f'{data_type}_knee_angle'] = data_series['knee_angle']

        final_results.append(processed_data)

    if not final_results:
        print("최종적으로 처리된 데이터가 없습니다.")
        return

    # 최종 데이터프레임 생성 및 저장
    final_df = pd.DataFrame(final_results)
    final_columns = ['squat_idx']
    for data_type in ['stand', 'sit']:
        for joint in ['shoulder', 'hip', 'knee', 'ankle']:
            final_columns.append(f'{data_type}_{joint}_horizon')
            final_columns.append(f'{data_type}_{joint}_vertical')
        final_columns.append(f'{data_type}_hip_angle')
        final_columns.append(f'{data_type}_knee_angle')
    final_df = final_df[final_columns]

    final_df.to_csv(output_path, index=False)
    print(f"\n✅ 2단계 완료: 최종 분석 결과가 '{output_path}'에 저장되었습니다.")
    print("\n--- 최종 데이터 미리보기 (상위 5개 행) ---")
    print(final_df.head())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="스쿼트 영상 원본 데이터를 분석하여 최종 정규화된 데이터로 변환합니다.")
    parser.add_argument('--input', type=str, required=True, help="분석할 원본 CSV 파일 경로 (프레임당 여러 줄 형식)")
    parser.add_argument('--output', type=str, required=True, help="저장할 최종 CSV 파일 경로 (스쿼트 사이클당 한 줄 형식)")

    args = parser.parse_args()

    # 1단계: 원본 데이터 -> 정제된 데이터 (메모리)
    manipulated_dataframe = transform_raw_data_to_manipulated(args.input)

    # 2단계: 정제된 데이터 -> 최종 분석 데이터 (파일)
    if manipulated_dataframe is not None:
        process_manipulated_data_to_final(manipulated_dataframe, args.output)
    else:
        print("\n파이프라인이 중단되었습니다.")
