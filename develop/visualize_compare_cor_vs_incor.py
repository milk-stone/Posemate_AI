import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import io


def load_and_label_data(correct_path, incorrect_path):
    """두 CSV 파일을 로드하고, 'label' 컬럼을 추가한 뒤 병합합니다."""
    try:
        correct_df = pd.read_csv(correct_path)
        incorrect_df = pd.read_csv(incorrect_path)

        correct_df['label'] = 'correct'
        incorrect_df['label'] = 'incorrect'

        combined_df = pd.concat([correct_df, incorrect_df], ignore_index=True)
        print("데이터 로딩 및 병합 완료.")
        return combined_df
    except FileNotFoundError as e:
        print(f"오류: 파일을 찾을 수 없습니다. ({e.filename})")
        return None


def plot_average_poses(df):
    """
    Correct/Incorrect 그룹의 평균 앉은 자세와 변동성을 시각화합니다.
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(8, 10))

    connections = [('shoulder', 'hip'), ('hip', 'knee'), ('knee', 'ankle')]
    colors = {'correct': 'dodgerblue', 'incorrect': 'tomato'}

    for label, group_df in df.groupby('label'):
        mean_coords = {}
        std_coords = {}

        # 각 관절의 평균 및 표준편차 계산
        for joint in ['shoulder', 'hip', 'knee', 'ankle']:
            h_col = f'sit_{joint}_horizon'
            v_col = f'sit_{joint}_vertical'
            mean_coords[joint] = (group_df[h_col].mean(), group_df[v_col].mean())
            std_coords[joint] = (group_df[h_col].std(), group_df[v_col].std())

        # 평균 자세 그리기
        for start, end in connections:
            x = [mean_coords[start][0], mean_coords[end][0]]
            y = [mean_coords[start][1], mean_coords[end][1]]
            ax.plot(x, y, 'o-', color=colors[label], linewidth=3, markersize=10,
                    label=f'Average {label.capitalize()} Pose')

        # 관절 위치의 변동성을 원으로 표시
        for joint in ['shoulder', 'hip', 'knee']:
            # 표준편차의 평균을 반지름으로 사용
            radius = (std_coords[joint][0] + std_coords[joint][1]) / 2
            circle = plt.Circle(mean_coords[joint], radius, color=colors[label], alpha=0.15, zorder=0)
            ax.add_artist(circle)

    ax.set_title('Average Sit Pose Comparison (Correct vs. Incorrect)', fontsize=16)
    ax.set_xlabel('Normalized Horizontal Position')
    ax.set_ylabel('Normalized Vertical Position')
    ax.set_aspect('equal', adjustable='box')
    ax.legend()
    ax.invert_yaxis()
    plt.tight_layout()


def plot_feature_violins(df):
    """
    주요 특징들의 분포를 Violin Plot으로 비교합니다.
    """
    # 비교할 특징 선택
    features_to_compare = [
        'sit_hip_angle',
        'sit_knee_angle',
        'sit_shoulder_horizon',  # 상체 기울기 관련
        'sit_hip_vertical'  # 스쿼트 깊이 관련
    ]

    # 데이터를 long-form으로 변환
    melted_df = df.melt(id_vars=['label'], value_vars=features_to_compare, var_name='feature', value_name='value')

    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(14, 8))
    sns.violinplot(x='feature', y='value', hue='label', data=melted_df, split=True, inner='quart', palette='pastel')

    plt.title('Distribution Comparison of Key Squat Features', fontsize=16)
    plt.xlabel('Feature')
    plt.ylabel('Normalized Value / Angle (degrees)')
    plt.xticks(rotation=15)
    plt.tight_layout()


def plot_angle_kde(df):
    """
    앉은 자세의 각도 분포를 KDE 플롯으로 비교합니다.
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    sns.kdeplot(data=df, x='sit_hip_angle', hue='label', ax=axes[0], fill=True, common_norm=False)
    axes[0].set_title('Sit Hip Angle Distribution', fontsize=14)

    sns.kdeplot(data=df, x='sit_knee_angle', hue='label', ax=axes[1], fill=True, common_norm=False)
    axes[1].set_title('Sit Knee Angle Distribution', fontsize=14)

    fig.suptitle('Comparison of Angle Distributions at Full Squat', fontsize=18)
    plt.tight_layout(rect=[0, 0, 1, 0.95])


# --- 메인 실행 부분 ---

correct_filename = '../dataset/csv/correct_jaekwon_final_output.csv'
incorrect_filename = '../dataset/csv/incorrect_final_output.csv'
combined_data_from_files = load_and_label_data(correct_filename, incorrect_filename)
if combined_data_from_files is not None:
    plot_average_poses(combined_data_from_files)
    plot_feature_violins(combined_data_from_files)
    plot_angle_kde(combined_data_from_files)
    plt.show()

