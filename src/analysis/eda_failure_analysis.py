import pandas as pd
import os

# 1. 파일 경로 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(BASE_DIR, '..', '..')
DATA_PATH = os.path.join(PROJECT_ROOT, 'data') 

# 모든 분석 파일은 DATA_PATH (data/ 폴더)에 있습니다.
analysis_path = os.path.join(DATA_PATH, "baseline_analysis.csv")
train_path = os.path.join(DATA_PATH, "train.csv")
submission_path = os.path.join(DATA_PATH, 'baseline_submission.csv') 

# 2. 데이터프레임 로드
try:
    analysis_df = pd.read_csv(analysis_path)
    train_df = pd.read_csv(train_path)
except FileNotFoundError as e:
    print(f"Error: 파일을 찾을 수 없습니다. 경로를 확인하세요: {e}")
    print(f"시도한 analysis_path: {analysis_path}")
    print(f"시도한 train_path: {train_path}")
    exit()

# ----------------------------------------------------------------------
# 3. 핵심 수정: train_df에는 'id' 컬럼이 없으므로, 인덱스를 기반으로 ID를 생성합니다.
# ----------------------------------------------------------------------

# train_df의 인덱스(0부터 시작)를 기반으로 분석용 'id' (1부터 시작)를 생성
analysis_df['id'] = train_df.index + 1 

# train_df의 type, 원문/정답 문장을 analysis_df의 동일 인덱스에 추가합니다.
analysis_df['type'] = train_df['type']
analysis_df['err_sentence'] = train_df['err_sentence']
analysis_df['cor_sentence_gold'] = train_df['cor_sentence']

# 모델 예측값 로드 
try:
    analysis_df['prediction'] = pd.read_csv(submission_path)['cor_sentence'] 
except FileNotFoundError as e:
    print(f"Error: baseline_submission.csv 파일을 찾을 수 없습니다. {submission_path}에 있는지 확인하세요.")
    exit()

# 필요한 컬럼만 선택
analysis_df = analysis_df[['id', 'type', 'err_sentence', 'cor_sentence_gold', 'prediction', 'tp', 'fp', 'fm', 'fr']]

print("✅ 데이터 로드 및 병합 완료.")

# ----------------------------------------------------
# --- 🔎 FN (놓친 교정) 최다 사례 확인 ---
# ----------------------------------------------------
print("\n--- 🔎 FN (놓친 교정) 최다 문장 Top 5 ---")
fn_top_5 = analysis_df.sort_values(by='fm', ascending=False).head(5)

# FN 분석 결과 출력 (FM이 높은 문장 5개)
for index, row in fn_top_5.iterrows():
    print(f"\n[ID]: {int(row['id'])}")
    print(f"[오류 유형]: {row['type']}")
    print(f"[FN 수]: {int(row['fm'])} 건 (모델이 {row['fm']}개의 교정을 놓쳤습니다)")
    print(f"[원문]: {row['err_sentence']}")
    print(f"[정답]: {row['cor_sentence_gold']}")
    print(f"[모델 예측]: {row['prediction']}")

# ----------------------------------------------------
# --- 🔴 FP/FR (잘못된/불필요한 수정) 최다 사례 확인 ---
# ----------------------------------------------------
print("\n--- 🔴 FP/FR (잘못된/불필요한 수정) 최다 문장 Top 5 ---")
# FP와 FR의 합계로 정렬
analysis_df['fp_fr_sum'] = analysis_df['fp'] + analysis_df['fr']
fp_fr_top_5 = analysis_df.sort_values(by='fp_fr_sum', ascending=False).head(5)

for index, row in fp_fr_top_5.iterrows():
    print(f"\n[ID]: {int(row['id'])}")
    print(f"[FP+FR 수]: {int(row['fp_fr_sum'])} 건")
    print(f"[원문]: {row['err_sentence']}")
    print(f"[모델 예측]: {row['prediction']}")