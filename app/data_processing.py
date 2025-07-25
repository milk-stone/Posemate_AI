import pandas as pd
import numpy as np
from scipy.signal import find_peaks


def transform_raw_data_to_manipulated(raw_df):
    """Step 1: Transforms the raw DataFrame to have one row per frame."""
    print("\n--- Step 1: Starting Raw Data Transformation ---")
    if raw_df is None or raw_df.empty:
        print("Error: No input data provided.")
        return None

    df = raw_df.copy()

    coords_df = df.pivot_table(index='frame', columns='joint', values=['x', 'y'])
    coords_df.columns = [f'{val}_{joint.lower()}' for val, joint in coords_df.columns]

    angles_df = df.dropna(subset=['angle_type', 'angle'])
    angles_df = angles_df.pivot_table(index='frame', columns='angle_type', values='angle')

    manipulated_df = coords_df.join(angles_df).reset_index()

    rename_map = {
        'x_right_shoulder': 'shoulder_horizon', 'y_right_shoulder': 'shoulder_vertical',
        'x_right_hip': 'hip_horizon', 'y_right_hip': 'hip_vertical',
        'x_right_knee': 'knee_horizon', 'y_right_knee': 'knee_vertical',
        'x_right_ankle': 'ankle_horizon', 'y_right_ankle': 'ankle_vertical',
        'hip': 'hip_angle',
        'knee': 'knee_angle'
    }
    manipulated_df = manipulated_df.rename(columns=rename_map)

    print("✅ Step 1 Complete: Data successfully transformed.")
    return manipulated_df


def process_manipulated_data_to_final(manipulated_df):
    """Step 2: Analyzes the transformed data to generate the final output."""
    print("\n--- Step 2: Starting Squat Cycle Analysis and Normalization ---")
    if manipulated_df is None or manipulated_df.empty:
        print("Error: No transformed data available.")
        return None

    input_df = manipulated_df.set_index('frame')
    y_coords_hip = input_df['hip_vertical'].values

    prominence_threshold = 10
    distance_threshold = 10
    troughs, _ = find_peaks(y_coords_hip, prominence=prominence_threshold, distance=distance_threshold)
    peaks, _ = find_peaks(-y_coords_hip, prominence=prominence_threshold, distance=distance_threshold)

    if len(peaks) < 2 or len(troughs) == 0:
        print("Error: Could not find any complete squat cycles to analyze.")
        return None

    final_results = []
    for i in range(len(peaks) - 1):
        start_peak_idx, end_peak_idx = peaks[i], peaks[i + 1]
        troughs_in_between = [t for t in troughs if start_peak_idx < t < end_peak_idx]
        if not troughs_in_between: continue

        stand_frame_idx = start_peak_idx
        sit_frame_idx = max(troughs_in_between, key=lambda t: y_coords_hip[t])

        cycle_df = input_df.iloc[start_peak_idx:end_peak_idx + 1]
        vertical_length = cycle_df['ankle_vertical'].max() - cycle_df['shoulder_vertical'].min()
        horizon_length = cycle_df['knee_horizon'].max() - cycle_df['hip_horizon'].min()
        if vertical_length == 0 or horizon_length == 0: continue

        stand_data = input_df.iloc[stand_frame_idx].copy()
        sit_data = input_df.iloc[sit_frame_idx].copy()
        processed_data = {'squat_idx': i + 1}

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
        print("Error: Failed to generate final analysis data.")
        return None

    final_df = pd.DataFrame(final_results)
    print(f"✅ Step 2 Complete: Detected a total of {len(final_df)} squat movements.")
    return final_df
