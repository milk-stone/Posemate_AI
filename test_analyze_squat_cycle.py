import pandas as pd
import numpy as np
from scipy.signal import find_peaks
import io
import matplotlib.pyplot as plt


def analyze_squat_cycles(df, prominence_threshold=50, distance_threshold=20):
    """
    정제된 스쿼트 데이터에서 사이클을 분석하고, 각 사이클의 세로 및 가로 길이를 계산합니다.

    Args:
        df (pd.DataFrame): 'frame'을 인덱스로 하고 각 관절의 좌표와 각도 컬럼을 포함하는 데이터프레임.
        prominence_threshold (int): 스쿼트 최저점/최고점 감지 민감도.
        distance_threshold (int): 스쿼트 사이의 최소 프레임 간격.

    Returns:
        list: 각 스쿼트 사이클의 분석 결과(딕셔너리)를 담은 리스트.
    """
    print("엉덩이(hip)의 수직 움직임을 기준으로 '선 자세 -> 앉은 자세 -> 선 자세' 사이클을 분석합니다.")
    print(f"설정값: Prominence > {prominence_threshold}, Min Frame Distance > {distance_threshold}\n")

    # 분석의 기준이 될 힙의 수직 좌표
    y_coords_hip = df['hip_vertical'].values

    # Y 좌표는 화면 위쪽이 0이므로, 값이 클수록 아래쪽에 위치합니다.
    # 최저점 (troughs, 앉은 자세)은 Y 좌표의 최댓값(peaks)입니다.
    troughs, _ = find_peaks(y_coords_hip, prominence=prominence_threshold, distance=distance_threshold)

    # 최고점 (peaks, 선 자세)은 Y 좌표의 최솟값이므로, 음수 데이터의 최댓값을 찾습니다.
    peaks, _ = find_peaks(-y_coords_hip, prominence=prominence_threshold, distance=distance_threshold)

    if len(peaks) < 2:
        print("스쿼트 사이클을 감지하기에 충분한 최고점(선 자세)을 찾지 못했습니다. (최소 2개 필요)")
        return [], peaks, troughs

    print(f"총 {len(peaks)}개의 최고점(선 자세)과 {len(troughs)}개의 최저점(앉은 자세)을 감지했습니다.\n")

    analysis_results = []
    rep_count = 0
    # 연속된 두 최고점(peak) 사이에 최저점(trough)이 있는지 확인하여 사이클을 정의합니다.
    for i in range(len(peaks) - 1):
        start_peak_idx = peaks[i]
        end_peak_idx = peaks[i + 1]

        # 두 최고점 사이에 있는 최저점을 찾습니다.
        troughs_in_between = [t for t in troughs if start_peak_idx < t < end_peak_idx]

        if troughs_in_between:
            rep_count += 1

            # 사이클의 시작과 끝 프레임을 가져옵니다.
            start_frame = df.index[start_peak_idx]
            end_frame = df.index[end_peak_idx]

            # 현재 사이클 구간에 해당하는 데이터만 추출합니다.
            cycle_df = df.loc[start_frame:end_frame]

            # --- 길이 계산 로직 ---
            # vertical_length: (가장 높은 어깨 좌표) - (가장 낮은 발목 좌표)
            # 화면 좌표계에서 '가장 높다'는 것은 y값이 가장 작은 것을 의미합니다.
            # 따라서 '어깨의 최소 y값'과 '발목의 최대 y값'을 사용합니다.
            min_shoulder_y = cycle_df['shoulder_vertical'].min()
            max_ankle_y = cycle_df['ankle_vertical'].max()
            vertical_length = max_ankle_y - min_shoulder_y

            # horizon_length: (가장 오른쪽 무릎 좌표) - (가장 왼쪽 엉덩이 좌표)
            max_knee_x = cycle_df['knee_horizon'].max()
            min_hip_x = cycle_df['hip_horizon'].min()
            horizon_length = max_knee_x - min_hip_x
            # --- 길이 계산 로직 끝 ---

            result = {
                'squat_number': rep_count,
                'start_frame': start_frame,
                'end_frame': end_frame,
                'vertical_length': vertical_length,
                'horizon_length': horizon_length
            }
            analysis_results.append(result)

            print(f"--- 스쿼트 {rep_count} ---")
            print(f"  - 구간: 프레임 {start_frame} ~ {end_frame}")
            print(f"  - Vertical Length: {vertical_length:.2f} pixels")
            print(f"  - Horizontal Length: {horizon_length:.2f} pixels")

    if not analysis_results:
        print("\n완전한 스쿼트 사이클을 감지하지 못했습니다.")

    return analysis_results, peaks, troughs


def plot_squat_analysis(df, peaks, troughs):
    """ 스쿼트 분석 결과를 시각화합니다. """
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(15, 6))

    hip_vertical = df['hip_vertical']

    ax.plot(df.index, hip_vertical, label='Hip Vertical Position', color='dodgerblue', alpha=0.8)
    ax.plot(df.index[troughs], hip_vertical.iloc[troughs], "v", color='red', markersize=10,
            label='Squat Bottom (Trough)')
    ax.plot(df.index[peaks], hip_vertical.iloc[peaks], "^", color='green', markersize=10,
            label='Standing Position (Peak)')

    ax.set_title('Squat Cycle Analysis', fontsize=16)
    ax.set_xlabel('Frame', fontsize=12)
    ax.set_ylabel('Y-coordinate (pixels)', fontsize=12)
    ax.legend()
    ax.invert_yaxis()
    plt.tight_layout()
    plt.show()


# --- 메인 실행 부분 ---

# CSV 데이터를 읽어 데이터프레임으로 변환하고 'frame'을 인덱스로 설정합니다.
input_df = pd.read_csv("dataset/csv/incorrect_manipulated_data.csv", index_col='frame')

# 1. 스쿼트 분석 함수 실행
# prominence와 distance는 데이터의 스케일과 스쿼트 속도에 따라 조절이 필요할 수 있습니다.
results, peaks, troughs = analyze_squat_cycles(input_df, prominence_threshold=50, distance_threshold=30)

# 2. 결과 시각화
if results:
    plot_squat_analysis(input_df, peaks, troughs)

