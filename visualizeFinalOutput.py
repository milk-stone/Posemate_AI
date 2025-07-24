import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io


def plot_squat_poses(df):
    """
    모든 스쿼트의 '선 자세'와 '앉은 자세'를 겹쳐서 시각화합니다.
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(8, 10))

    # 관절 연결 순서
    connections = [
        ('shoulder', 'hip'),
        ('hip', 'knee'),
        ('knee', 'ankle')
    ]

    # 각 스쿼트 반복에 대해 루프
    for idx, row in df.iterrows():
        # 선 자세 그리기 (파란색)
        stand_coords = {
            'shoulder': (row['stand_shoulder_horizon'], row['stand_shoulder_vertical']),
            'hip': (row['stand_hip_horizon'], row['stand_hip_vertical']),
            'knee': (row['stand_knee_horizon'], row['stand_knee_vertical']),
            'ankle': (row['stand_ankle_horizon'], row['stand_ankle_vertical']),
        }
        for start_joint, end_joint in connections:
            x_coords = [stand_coords[start_joint][0], stand_coords[end_joint][0]]
            y_coords = [stand_coords[start_joint][1], stand_coords[end_joint][1]]
            # 첫 번째 스쿼트만 라벨을 추가하여 범례가 중복되지 않도록 함
            label = 'Stand Pose' if idx == df.index[0] else ""
            ax.plot(x_coords, y_coords, 'o-', color='cornflowerblue', alpha=0.3, label=label)

        # 앉은 자세 그리기 (빨간색)
        sit_coords = {
            'shoulder': (row['sit_shoulder_horizon'], row['sit_shoulder_vertical']),
            'hip': (row['sit_hip_horizon'], row['sit_hip_vertical']),
            'knee': (row['sit_knee_horizon'], row['sit_knee_vertical']),
            'ankle': (row['sit_ankle_horizon'], row['sit_ankle_vertical']),
        }
        for start_joint, end_joint in connections:
            x_coords = [sit_coords[start_joint][0], sit_coords[end_joint][0]]
            y_coords = [sit_coords[start_joint][1], sit_coords[end_joint][1]]
            label = 'Sit Pose' if idx == df.index[0] else ""
            ax.plot(x_coords, y_coords, 'o-', color='tomato', alpha=0.3, label=label)

    ax.set_title('Squat Pose Distribution (All Reps Overlayed)', fontsize=16)
    ax.set_xlabel('Normalized Horizontal Position')
    ax.set_ylabel('Normalized Vertical Position')
    ax.set_aspect('equal', adjustable='box')
    ax.legend()
    # Y축을 반전시켜 실제 자세처럼 보이게 함
    ax.invert_yaxis()
    plt.tight_layout()


def plot_angle_distributions(df):
    """
    선 자세와 앉은 자세의 고관절 및 무릎 각도 분포를 KDE 플롯으로 시각화합니다.
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 고관절 각도 분포
    sns.kdeplot(df['stand_hip_angle'], ax=axes[0], label='Stand', fill=True)
    sns.kdeplot(df['sit_hip_angle'], ax=axes[0], label='Sit', fill=True)
    axes[0].set_title('Hip Angle Distribution', fontsize=14)
    axes[0].set_xlabel('Hip Angle (degrees)')
    axes[0].legend()

    # 무릎 각도 분포
    sns.kdeplot(df['stand_knee_angle'], ax=axes[1], label='Stand', fill=True)
    sns.kdeplot(df['sit_knee_angle'], ax=axes[1], label='Sit', fill=True)
    axes[1].set_title('Knee Angle Distribution', fontsize=14)
    axes[1].set_xlabel('Knee Angle (degrees)')
    axes[1].legend()

    fig.suptitle('Angle Distributions for Stand vs. Sit Poses', fontsize=18)
    plt.tight_layout(rect=[0, 0, 1, 0.96])


def plot_angle_boxplots(df):
    """
    자세별 각도 분포를 Box Plot으로 비교합니다.
    """
    # Box plot을 그리기 위해 데이터를 'long' 형태로 변환
    stand_hip = pd.DataFrame({'angle': df['stand_hip_angle'], 'pose': 'Stand', 'joint': 'Hip'})
    sit_hip = pd.DataFrame({'angle': df['sit_hip_angle'], 'pose': 'Sit', 'joint': 'Hip'})
    stand_knee = pd.DataFrame({'angle': df['stand_knee_angle'], 'pose': 'Stand', 'joint': 'Knee'})
    sit_knee = pd.DataFrame({'angle': df['sit_knee_angle'], 'pose': 'Sit', 'joint': 'Knee'})

    plot_df = pd.concat([stand_hip, sit_hip, stand_knee, sit_knee])

    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(10, 7))
    sns.boxplot(x='joint', y='angle', hue='pose', data=plot_df, palette='pastel')
    plt.title('Angle Comparison: Stand vs. Sit', fontsize=16)
    plt.xlabel('Joint')
    plt.ylabel('Angle (degrees)')
    plt.tight_layout()


# --- 메인 실행 부분 ---

# 제공된 최종 데이터를 문자열로 저장하여 사용합니다.

df = pd.read_csv("dataset/incorrect_final_output.csv")

# 시각화 함수 호출
plot_squat_poses(df)
plot_angle_distributions(df)
plot_angle_boxplots(df)

# 모든 플롯을 화면에 표시
plt.show()

