import pandas as pd
import argparse
import os

def merge_results(base_csv: str, correction_csv: str, output_csv: str):
    """
    Merges re-corrected sentences from correction_csv into the base_csv.
    Only rows present in correction_csv will have their cor_sentence overwritten in base_csv.
    """
    print(f"Loading base submission: {base_csv}")
    base_df = pd.read_csv(base_csv)
    
    print(f"Loading re-corrected candidates: {correction_csv}")
    correction_df = pd.read_csv(correction_csv)
    
    # Ensure 'id' column is the same type for merging/lookup
    base_df['id'] = base_df['id'].astype(str)
    correction_df['id'] = correction_df['id'].astype(str)
    
    # Create a mapping dictionary: {id: new_cor_sentence}
    new_corrections_map = correction_df.set_index('id')['cor_sentence'].to_dict()
    
    # Apply the new corrections to the base dataframe
    updated_count = 0
    
    for index, row in base_df.iterrows():
        if row['id'] in new_corrections_map:
            new_cor = new_corrections_map[row['id']].strip()
            
            # *핵심 로직*: 새로운 교정 내용이 원본 err_sentence와 다를 경우에만 덮어씁니다.
            # 이는 CoT 재시도가 또다시 오류를 찾지 못한 경우(FM-Retry-Failure) Precision을 낭비하지 않기 위함입니다.
            if new_cor != row['err_sentence'].strip():
                base_df.loc[index, 'cor_sentence'] = new_cor
                updated_count += 1
            # 만약 new_cor == err_sentence라면, CoT도 오류를 못 찾았다는 뜻이므로 원본 cor_sentence(err_sentence와 같음)를 유지합니다.

    print(f"Total rows updated with new correction (Recall Boosted): {updated_count}")
    
    # Save the final merged dataframe
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    base_df.to_csv(output_csv, index=False)
    
    print(f"✅ Final merged submission saved to: {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge re-corrected FM candidates into the base submission file.")
    parser.add_argument("--base", default="final_2.csv", help="Path to the base submission CSV (e.g., final_2.csv).")
    parser.add_argument("--correction", default="data/fm_recorrected.csv", help="Path to the CSV with re-corrected sentences.")
    parser.add_argument("--output", default="submission/final_submission_fm_boosted.csv", help="Path to save the final merged submission.")
    args = parser.parse_args()
    
    merge_results(args.base, args.correction, args.output)