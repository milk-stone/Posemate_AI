import pandas as pd
import numpy as np
import io


def load_and_label_data(correct_path, incorrect_path):
    """두 CSV 파일을 로드하고, 'label' 컬럼을 추가한 뒤 병합합니다."""
    try:
        correct_df = pd.read_csv(correct_path)
        incorrect_df = pd.read_csv(incorrect_path)

        correct_df['label'] = 'correct'
        incorrect_df['label'] = 'incorrect'

        # 각 데이터프레임을 별도로 반환하도록 수정
        print("데이터 로딩 완료.")
        return correct_df, incorrect_df
    except FileNotFoundError as e:
        print(f"오류: 파일을 찾을 수 없습니다. ({e.filename})")
        return None, None


def create_custom_balanced_dataset(correct_path, incorrect_path, output_path, target_samples_per_class=550):
    """
    두 CSV 파일을 로드하고, 각 클래스를 지정된 수만큼 오버샘플링하여
    하나의 균형잡힌 CSV 파일로 저장합니다.
    """
    print(f"\n--- 각 클래스를 {target_samples_per_class}개로 맞추는 데이터셋 생성 시작 ---")
    correct_df, incorrect_df = load_and_label_data(correct_path, incorrect_path)

    if correct_df is None or incorrect_df is None:
        return

    print(f"원본 데이터 수: correct={len(correct_df)}, incorrect={len(incorrect_df)}")

    # 각 클래스를 목표 개수만큼 오버샘플링합니다.
    # replace=True 옵션은 데이터 복제를 허용합니다.
    correct_resampled = correct_df.sample(n=target_samples_per_class, replace=True, random_state=42)
    incorrect_resampled = incorrect_df.sample(n=target_samples_per_class, replace=True, random_state=42)

    print(f"오버샘플링 후 데이터 수: correct={len(correct_resampled)}, incorrect={len(incorrect_resampled)}")

    # 두 데이터프레임을 병합합니다.
    balanced_df = pd.concat([correct_resampled, incorrect_resampled])

    # 데이터를 무작위로 섞어줍니다.
    balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

    try:
        # squat_idx 컬럼이 있다면 삭제합니다.
        if 'squat_idx' in balanced_df.columns:
            balanced_df = balanced_df.drop(columns=['squat_idx'])
            print("'squat_idx' 컬럼을 삭제했습니다.")

        # 최종 데이터를 CSV 파일로 저장합니다.
        balanced_df.to_csv(output_path, index=False)
        print(f"총 {len(balanced_df)}개의 데이터가 '{output_path}'에 성공적으로 저장되었습니다.")
        print("\n--- 최종 데이터 미리보기 ---")
        print(balanced_df.head())

    except Exception as e:
        print(f"오류: 파일을 저장하는 중 문제가 발생했습니다 - {e}")


# --- 메인 실행 부분 ---

# --- 실제 파일 사용 ---
# 로컬에 'dataset' 폴더가 있고 그 안에 파일들이 있다고 가정합니다.
# 만약 파일이 다른 곳에 있다면 경로를 수정해주세요.
correct_filename = '../dataset/csv/correct_final_output.csv'
incorrect_filename = '../dataset/csv/incorrect_final_output.csv'
output_filename = '../dataset/csv/oversampled_merged_data.csv'

# 각 클래스별 550개, 총 1100개의 데이터셋 생성
create_custom_balanced_dataset(correct_filename, incorrect_filename, output_filename, target_samples_per_class=550)

