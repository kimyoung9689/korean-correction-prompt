import os
import argparse
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from openai import OpenAI
from openai import APIError

# 새로 추가된 PROMPT_RETRY_COT를 포함하도록 import (prompts.py 수정 필수)
from prompts import PROMPT_RETRY_COT 

# Load environment variables
load_dotenv()

SYSTEM_MESSAGE = "당신은 한국어 문장 교정 전문가입니다. 주어진 지시에 따라 정확하게 분석하고 교정 작업을 수행합니다."


def extract_correction(output_text: str, original_text: str) -> str:
    """CoT 출력에서 최종 교정 문장만 파싱합니다."""
    
    # 1. <분석> 태그 이후에 나오는 텍스트를 최종 교정 문장으로 간주
    parts = output_text.strip().split("</분석>")
    if len(parts) > 1:
        # 태그 이후의 내용을 줄바꿈 기준으로 정리하여 반환
        corrected = parts[-1].strip()
    else:
        # 분석 태그가 없다면 전체 텍스트를 교정 문장으로 간주 (최악의 경우 원문 그대로 반환)
        corrected = output_text.strip()

    # 혹시 모를 잔여 텍스트나 빈 출력 대비 (안전장치)
    if not corrected:
        return original_text
    
    return corrected.split('\n')[-1].strip() # 마지막 줄이 최종 교정 문장이라고 가정

def retry_correction(client: OpenAI, model: str, text: str) -> str:
    """CoT 기반 Single-Turn API 호출을 수행합니다."""
    
    prompt = PROMPT_RETRY_COT.format(text=text)
    
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_MESSAGE},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0, # 안정적인 출력을 위해 0.0 유지
            max_tokens=2000 # 2000 토큰 제한 내에서 최대 출력 허용 (FM 문장이 짧은 경우가 많으므로)
        )
        raw_output = resp.choices[0].message.content.strip()
        
        # 최종 교정 문장만 파싱
        return extract_correction(raw_output, text)
        
    except Exception as e:
        # API 오류 또는 파싱 오류 시 원문 반환
        print(f"\n[Error] API 호출 또는 처리 실패: {e}")
        return text


def main():
    parser = argparse.ArgumentParser(description="Re-generate corrections for FM candidates using a strong CoT prompt.")
    parser.add_argument("--input", default="data/fm_candidates_to_retry.csv", help="Input CSV path (FM candidates) to re-correct.")
    parser.add_argument("--output", default="data/fm_recorrected.csv", help="Output CSV path for re-corrected results.")
    parser.add_argument("--model", default="solar-pro2", help="Model name (default: solar-pro2)")
    args = parser.parse_args()

    # Load data
    df = pd.read_csv(args.input)
    
    if "err_sentence" not in df.columns:
        raise ValueError("Input CSV must contain 'err_sentence' column")

    # Setup Upstage client
    api_key = os.getenv("UPSTAGE_API_KEY")
    if not api_key:
        raise ValueError("UPSTAGE_API_KEY not found in environment variables. Please check your .env file.")
    
    try:
        client = OpenAI(api_key=api_key, base_url="https://api.upstage.ai/v1")
    except Exception as e:
        raise ValueError(f"Failed to initialize OpenAI client: {e}")

    
    print(f"Model: {args.model}")
    print(f"FM Candidates to retry: {len(df)}")
    print(f"Output: {args.output}")

    ids = df["id"].astype(str).tolist()
    err_sentences = df["err_sentence"].astype(str).tolist()
    cor_sentences = []
    
    # Process each sentence
    for i, text in enumerate(tqdm(err_sentences, desc="Re-correcting FM")):
        corrected = retry_correction(client, args.model, text)
        cor_sentences.append(corrected)

    # Save results
    out_df = pd.DataFrame({
        "id": ids,
        "err_sentence": err_sentences, 
        "cor_sentence": cor_sentences,
    })
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    out_df.to_csv(args.output, index=False)
    
    print(f"✅ Re-correction complete. Results saved to {args.output}")


if __name__ == "__main__":
    main()