import pandas as pd
import argparse

def filter_fm(input_csv: str, output_csv: str):
    """
    입력 CSV에서 err_sentence와 cor_sentence가 완벽히 일치하는 행만 추출하여 새로운 CSV로 저장합니다.
    이 행들은 모델이 오류가 없다고 판단했지만, 실제로는 오류를 놓쳤을 수 있는 FM(False Negative) 후보군입니다.
    """
    print(f"Loading data from: {input_csv}")
    
    # NaN 값을 처리하기 위해 dtype을 str로 명시적으로 설정하여 로드
    df = pd.read_csv(input_csv, dtype={'err_sentence': str, 'cor_sentence': str})
    
    # 필수 컬럼 검사
    if not all(col in df.columns for col in ['id', 'err_sentence', 'cor_sentence']):
        raise ValueError("Input CSV must contain 'id', 'err_sentence', and 'cor_sentence' columns.")

    # 띄어쓰기나 미세한 공백 차이로 인해 불필요하게 필터링되지 않도록 양쪽 공백 제거
    df['err_sentence'] = df['err_sentence'].str.strip()
    df['cor_sentence'] = df['cor_sentence'].str.strip()
    
    # 두 컬럼의 내용이 완벽하게 일치하는 행을 필터링 (FM 후보군)
    # 이 문장들은 V36 모델이 교정하지 않은 문장들입니다.
    fm_candidates_df = df[df['err_sentence'] == df['cor_sentence']]
    
    # 재교정할 err_sentence와 id만 남김
    fm_candidates_for_retry = fm_candidates_df[['id', 'err_sentence']].rename(columns={'err_sentence': 'text_to_retry'})
    
    # output CSV는 재교정 스크립트의 input 형식(id, err_sentence)과 맞춰야 합니다.
    # 컬럼명을 원래대로 복원 (id, err_sentence)
    fm_candidates_for_retry.rename(columns={'text_to_retry': 'err_sentence'}, inplace=True)
    
    print(f"Total rows: {len(df)}")
    print(f"FM Candidates (rows to retry): {len(fm_candidates_for_retry)}")

    if len(fm_candidates_for_retry) > 0:
        # data 폴더에 저장
        fm_candidates_for_retry.to_csv(output_csv, index=False)
        print(f"✅ FM candidates saved to: {output_csv}")
    else:
        print("Warning: No FM candidates found. All sentences were either corrected or identical.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter False Negative (FM) candidates for re-correction.")
    parser.add_argument("--input", default="final_2.csv", help="Path to the submission CSV to filter (e.g., final_2.csv)")
    parser.add_argument("--output", default="data/fm_candidates_to_retry.csv", help="Path to save the filtered FM candidates.")
    args = parser.parse_args()
    
    # data 폴더가 없다면 생성
    import os
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    filter_fm(args.input, args.output)