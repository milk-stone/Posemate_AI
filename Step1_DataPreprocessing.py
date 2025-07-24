import pandas as pd
import numpy as np
from scipy.signal import find_peaks
import io


def process_squat_data_pipeline(input_df, output_filename='incorrect_final_output.csv'):
    """
    스쿼트 데이터 분석의 전체 파이프라인을 실행합니다.
    1. 사이클 판별
    2. 길이 계산
    3. 핵심 프레임(stand, sit) 추출
    4. 좌표 변환 (발목 기준)
    5. 정규화
    6. 최종 CSV 파일 저장

    Args:
        input_df (pd.DataFrame): 'frame'을 인덱스로 하는 정제된 스쿼트 데이터.
        output_filename (str): 저장할 최종 CSV 파일 이름.
    """
    print("스쿼트 분석 파이프라인을 시작합니다...")

    # --- 1. 스쿼트 구간 판별 ---
    y_coords_hip = input_df['hip_vertical'].values
    # prominence와 distance는 데이터에 맞게 조정이 필요할 수 있습니다.
    prominence_threshold = 10
    distance_threshold = 10

    troughs, _ = find_peaks(y_coords_hip, prominence=prominence_threshold, distance=distance_threshold)
    peaks, _ = find_peaks(-y_coords_hip, prominence=prominence_threshold, distance=distance_threshold)

    if len(peaks) < 2 or len(troughs) == 0:
        print("분석할 수 있는 완전한 스쿼트 사이클을 찾지 못했습니다. 데이터나 파라미터를 확인해주세요.")
        return

    print(f"총 {len(peaks)}개의 최고점(선 자세)과 {len(troughs)}개의 최저점(앉은 자세)을 감지했습니다.")

    final_results = []
    squat_idx_counter = 0

    # 연속된 두 최고점(peak) 사이에 최저점(trough)이 있는지 확인하여 사이클을 정의
    for i in range(len(peaks) - 1):
        start_peak_idx = peaks[i]
        end_peak_idx = peaks[i + 1]

        # 두 최고점 사이에 있는 최저점들을 찾습니다.
        troughs_in_between = [t for t in troughs if start_peak_idx < t < end_peak_idx]

        if not troughs_in_between:
            continue

        squat_idx_counter += 1

        # --- 3. 제일 서있을 때와 앉았을 때 프레임만 남기기 ---
        stand_frame_idx = start_peak_idx
        # 최저점(가장 깊이 앉은 지점)은 hip_vertical이 최대인 지점입니다.
        sit_frame_idx = max(troughs_in_between, key=lambda t: y_coords_hip[t])

        stand_frame = input_df.index[stand_frame_idx]
        sit_frame = input_df.index[sit_frame_idx]

        # 사이클 전체 데이터 (길이 계산용)
        cycle_df = input_df.loc[input_df.index[start_peak_idx]:input_df.index[end_peak_idx]]

        # --- 2. 각 스쿼트 구간 별 길이 구하기 ---
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

        # 서 있을 때와 앉았을 때 데이터에 대해 변환 및 정규화 수행
        for data_type, data_series in [('stand', stand_data), ('sit', sit_data)]:

            # --- 4. 좌표 기준점 변환 (발목을 0,0으로) ---
            ankle_origin_h = data_series['ankle_horizon']
            ankle_origin_v = data_series['ankle_vertical']

            # --- 5. 정규화 및 데이터 정리 ---
            for joint in ['shoulder', 'hip', 'knee', 'ankle']:
                h_col = f'{joint}_horizon'
                v_col = f'{joint}_vertical'

                # 좌표 변환
                translated_h = data_series[h_col] - ankle_origin_h
                translated_v = data_series[v_col] - ankle_origin_v

                # 정규화
                normalized_h = translated_h / horizon_length
                normalized_v = translated_v / vertical_length

                processed_data[f'{data_type}_{h_col}'] = normalized_h
                processed_data[f'{data_type}_{v_col}'] = normalized_v

            # 각도 값 추가
            processed_data[f'{data_type}_hip_angle'] = data_series['hip_angle']
            processed_data[f'{data_type}_knee_angle'] = data_series['knee_angle']

        final_results.append(processed_data)

    if not final_results:
        print("최종적으로 처리된 데이터가 없습니다.")
        return

    # --- 6. 최종 CSV 파일 저장 ---
    final_df = pd.DataFrame(final_results)

    # 컬럼 순서 지정
    final_columns = ['squat_idx']
    for data_type in ['stand', 'sit']:
        for joint in ['shoulder', 'hip', 'knee', 'ankle']:
            final_columns.append(f'{data_type}_{joint}_horizon')
            final_columns.append(f'{data_type}_{joint}_vertical')
        final_columns.append(f'{data_type}_hip_angle')
        final_columns.append(f'{data_type}_knee_angle')

    final_df = final_df[final_columns]

    final_df.to_csv(output_filename, index=False)
    print(f"\n파이프라인이 성공적으로 완료되었습니다. 결과가 '{output_filename}'에 저장되었습니다.")
    print("\n--- 최종 데이터 미리보기 (상위 5개 행) ---")
    print(final_df.head())


# --- 메인 실행 부분 ---

# # 함수 실행
# process_squat_data_pipeline(input_df)

# --- 실제 파일 사용 ---
# 위 테스트 부분을 주석 처리하고 아래 코드의 주석을 해제하여 사용하세요.
try:
    input_df_from_file = pd.read_csv('dataset/correct_manipulated_data.csv', index_col='frame')
    process_squat_data_pipeline(input_df_from_file, 'dataset/correct_final_output.csv')
except FileNotFoundError:
    print("오류: 입력 파일을 찾을 수 없습니다. 파일 경로를 확인해주세요.")

