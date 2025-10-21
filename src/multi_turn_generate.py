import os
import argparse
import re # 정규 표현식 모듈 추가

import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from openai import OpenAI
# 단일 프롬프트만 가져옵니다.
from prompts import single_turn_cot_prompt 

# Load environment variables
load_dotenv()

SYSTEM_MESSAGE = "당신은 한국어 문장 교정 전문가입니다. 주어진 지시에 따라 정확하게 분석하고 교정 작업을 수행합니다."

# 최종 교정 결과만 파싱하는 함수
def extract_correction(output_text: str, original_text: str) -> str:
    # 1. <교정> 태그 이후의 텍스트를 찾습니다.
    match = re.search(r'<교정>\s*(.*?)$', output_text, re.DOTALL)
    
    if match:
        # 찾은 텍스트의 앞뒤 공백을 제거하고 반환합니다.
        corrected = match.group(1).strip()
        # LLM이 간혹 <교정> 태그를 출력하지 않고 문장만 출력할 때가 있으므로, 
        # 최종 문장이 비어있지 않은지 확인합니다.
        return corrected if corrected else original_text
    
    # 2. <교정> 태그를 찾지 못했다면, 마지막 줄을 시도해봅니다. (최후의 보루)
    lines = output_text.strip().split('\n')
    if lines and not lines[-1].startswith('<사고 과정>'):
        return lines[-1].strip()

    # 3. 모든 파싱에 실패하면 원문을 반환합니다.
    return original_text


def main():
    parser = argparse.ArgumentParser(description="Generate corrected sentences using Upstage API with Single-turn CoT")
    parser.add_argument("--input", default="data/train.csv", help="Input CSV path containing err_sentence column")
    parser.add_argument("--output", default="submission/single_turn_cot_submission.csv", help="Output CSV path") 
    parser.add_argument("--model", default="solar-pro2", help="Model name (default: solar-pro2)")
    args = parser.parse_args()

    # Load data
    df = pd.read_csv(args.input)
    
    if "err_sentence" not in df.columns:
        raise ValueError("Input CSV must contain 'err_sentence' column")

    # Setup Upstage client
    api_key = os.getenv("UPSTAGE_API_KEY")
    if not api_key:
        raise ValueError("UPSTAGE_API_KEY not found in environment variables")
    
    client = OpenAI(api_key=api_key, base_url="https://api.upstage.ai/v1")
    
    print(f"Model: {args.model}")
    print(f"Output: {args.output}")

    ids = df["id"].astype(str).tolist() if "id" in df.columns else list(range(1, len(df) + 1))
    err_sentences = df["err_sentence"].astype(str).tolist()
    cor_sentences = []
    
    # Process each sentence
    for i, text in enumerate(tqdm(err_sentences, desc="Generating")):
        
        try:
            # -----------------------------------------------------------------
            # Single-turn CoT: One API Call
            # -----------------------------------------------------------------
            
            # 사고 과정까지 포함된 통합 프롬프트
            prompt_cot = single_turn_cot_prompt.format(text=text)
            
            resp = client.chat.completions.create(
                model=args.model,
                messages=[
                    {"role": "system", "content": SYSTEM_MESSAGE},
                    {"role": "user", "content": prompt_cot},
                ],
                temperature=0.0,
            )
            raw_output = resp.choices[0].message.content.strip()
            
            # 최종 교정 문장만 파싱
            corrected = extract_correction(raw_output, text)
            cor_sentences.append(corrected)
            
        except Exception as e:
            print(f"\nError processing ID {ids[i]} ({text[:50]}...): {e}")
            cor_sentences.append(text)  # 오류 발생 시 원문으로 대체
            

    # Save results with required column names
    out_df = pd.DataFrame({
        "id": ids,
        "cor_sentence": cor_sentences,
    })
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    out_df.to_csv(args.output, index=False)
    
    print(f"✅ Generation complete. Results saved to {args.output}")


if __name__ == "__main__":
    main()