import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import io

# SMOTE를 사용하기 위해 imbalanced-learn 라이브러리를 임포트합니다.
# 설치: pip install imbalanced-learn
try:
    from imblearn.over_sampling import SMOTE
except ImportError:
    print("오류: 'imbalanced-learn' 라이브러리를 찾을 수 없습니다.")
    print("터미널에서 'pip install imbalanced-learn' 명령어로 설치해주세요.")
    SMOTE = None


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


def create_balanced_merged_dataset(correct_path, incorrect_path, output_path, strategy='undersample'):
    """
    두 CSV 파일을 로드하고, 지정된 전략으로 데이터 균형을 맞춘 뒤,
    하나의 CSV 파일로 병합하여 저장합니다.
    """
    print(f"\n--- '{strategy}' 전략으로 병합 데이터셋 생성 시작 ---")
    combined_df = load_and_label_data(correct_path, incorrect_path)

    if combined_df is not None:
        # 데이터 균형 맞추기
        balanced_df = balance_data(combined_df, strategy=strategy)

        if balanced_df is None:  # SMOTE 라이브러리가 없을 경우
            return None

        try:
            # squat_idx 컬럼 삭제
            if 'squat_idx' in balanced_df.columns:
                balanced_df = balanced_df.drop(columns=['squat_idx'])
                print("'squat_idx' 컬럼을 삭제했습니다.")

            # 균형 잡힌 데이터를 저장
            balanced_df.to_csv(output_path, index=False)
            print(f"균형 조정된 데이터가 '{output_path}'에 성공적으로 저장되었습니다.")
            print("\n--- 병합된 데이터 미리보기 ---")
            print(balanced_df.head())
            return balanced_df
        except Exception as e:
            print(f"오류: 파일을 저장하는 중 문제가 발생했습니다 - {e}")
            return None
    return None


def balance_data(df, strategy='undersample'):
    """
    데이터 불균형을 처리하기 위해 리샘플링을 수행합니다.
    'oversample' 전략 선택 시 SMOTE를 사용합니다.
    """
    label_counts = df['label'].value_counts()
    majority_class_label = label_counts.idxmax()
    minority_class_label = label_counts.idxmin()

    print(f"원본 데이터 수: {majority_class_label}={label_counts.max()}, {minority_class_label}={label_counts.min()}")

    if strategy == 'undersample':
        print(f"언더샘플링 수행: 다수 클래스('{majority_class_label}') 데이터를 소수 클래스 개수({label_counts.min()})에 맞춥니다.")
        df_majority = df[df['label'] == majority_class_label]
        df_minority = df[df['label'] == minority_class_label]
        df_majority_downsampled = df_majority.sample(n=len(df_minority), random_state=42)
        balanced_df = pd.concat([df_majority_downsampled, df_minority])

    elif strategy == 'oversample':
        if SMOTE is None:
            print("SMOTE를 사용할 수 없어 오버샘플링을 진행할 수 없습니다.")
            return None

        print(f"SMOTE 오버샘플링 수행: 소수 클래스('{minority_class_label}')의 합성 데이터를 생성하여 다수 클래스 개수({label_counts.max()})에 맞춥니다.")

        # 특성(X)과 타겟(y) 분리
        X = df.drop(columns=['label', 'squat_idx'], errors='ignore')
        y = df['label']

        # SMOTE 적용
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)

        # 다시 데이터프레임으로 결합
        balanced_df = pd.DataFrame(X_resampled, columns=X.columns)
        balanced_df['label'] = y_resampled

    else:
        raise ValueError("전략은 'undersample' 또는 'oversample'이어야 합니다.")

    print(f"조정된 데이터 수: {balanced_df['label'].value_counts().to_dict()}")
    return balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)


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

# --- 실제 파일 사용 ---
# 위 샘플 데이터 부분을 주석 처리하고 아래 코드의 주석을 해제하여 사용하세요.
correct_filename = 'dataset/correct_final_output.csv'
incorrect_filename = 'dataset/incorrect_final_output.csv'

# 언더샘플링된 파일 생성
create_balanced_merged_dataset(correct_filename, incorrect_filename, 'dataset/undersampled_merged_data.csv', strategy='undersample')

# 오버샘플링된 파일 생성
create_balanced_merged_dataset(correct_filename, incorrect_filename, 'dataset/oversampled_merged_data.csv', strategy='oversample')

# 시각화 (오버샘플링된 데이터를 다시 불러와서 사용)
balanced_df_for_plot = pd.read_csv('dataset/oversampled_merged_data.csv')
plot_average_poses(balanced_df_for_plot)
plot_feature_violins(balanced_df_for_plot)
plot_angle_kde(balanced_df_for_plot)
plt.show()
