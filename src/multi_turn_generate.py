import os
import argparse
import re
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from openai import OpenAI
from openai import APIError # API 오류 처리를 위해 추가

# <<<--- 변경 1: prompts.py 파일의 baseline_prompt를 사용하도록 변경 --->>>
from prompts import baseline_prompt as current_prompt 

# Load environment variables
load_dotenv()

SYSTEM_MESSAGE = "당신은 한국어 문장 교정 전문가입니다. 주어진 지시에 따라 정확하게 분석하고 교정 작업을 수행합니다."


def extract_correction(output_text: str, original_text: str) -> str:
    """<교정> 태그만 파싱합니다."""
    # <교정> 태그 이후의 텍스트를 찾습니다.
    # 대소문자 무시 (IGNORECASE)를 제거하고 STRICT하게 변경하여 Precision 향상 시도
    match = re.search(r'<교정>\s*(.*?)$', output_text, re.DOTALL)
    
    if match:
        corrected = match.group(1).strip()
        # 파싱 실패 시 원문 반환
        return corrected if corrected else original_text
    
    # 태그를 찾지 못했으나 문장만 출력했을 경우를 대비하여 마지막 줄을 확인합니다.
    lines = output_text.strip().split('\n')
    if lines and len(lines) == 1:
        return lines[0].strip()
        
    # 최종적으로 파싱에 실패하면 원문 반환
    return original_text


def main():
    parser = argparse.ArgumentParser(description="Generate corrected sentences using Upstage API with Clean Baseline Strategy")
    # <<<--- 변경 2: 기본 입력 파일을 test.csv로 변경 --->>>
    parser.add_argument("--input", default="data/test.csv", help="Input CSV path containing err_sentence column")
    # <<<--- 변경 3: 출력 파일명을 새로운 Baseline 측정 파일로 변경 --->>>
    parser.add_argument("--output", default="submission/initial_baseline_test_set.csv", help="Output CSV path") 
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
    print(f"Output: {args.output}")

    ids = df["id"].astype(str).tolist() if "id" in df.columns else list(range(1, len(df) + 1))
    err_sentences = df["err_sentence"].astype(str).tolist()
    cor_sentences = []
    
    # Process each sentence
    for i, text in enumerate(tqdm(err_sentences, desc="Generating")):
        
        try:
            # -----------------------------------------------------------------
            # Clean Baseline: Stable API Call
            # -----------------------------------------------------------------
            
            # current_prompt는 baseline_prompt (prompts.py에서 정의)
            prompt_final = current_prompt.format(text=text)
            
            resp = client.chat.completions.create(
                model=args.model,
                messages=[
                    {"role": "system", "content": SYSTEM_MESSAGE},
                    {"role": "user", "content": prompt_final},
                ],
                # 안정적인 출력을 위해 temperature=0.0 유지
                temperature=0.0, 
            )
            raw_output = resp.choices[0].message.content.strip()
            
            # 최종 교정 문장만 파싱
            corrected = extract_correction(raw_output, text)
            cor_sentences.append(corrected)
            
        except APIError as e:
            # API 호출 중 발생하는 오류 (키 만료, 권한 없음 등)
            print(f"\n!!! API ERROR on ID {ids[i]} ({text[:50]}...): {e}")
            raise e
        except Exception as e:
            # 기타 일반 오류
            print(f"\n!!! UNEXPECTED ERROR on ID {ids[i]} ({text[:50]}...): {e}")
            raise e


    # Save results with required column names
    out_df = pd.DataFrame({
        "id": ids,
        # 원본 문장(err_sentences) 컬럼 추가
        "err_sentence": err_sentences, 
        "cor_sentence": cor_sentences,
    })
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    out_df.to_csv(args.output, index=False)
    
    print(f"✅ Generation complete. Results saved to {args.output}")


if __name__ == "__main__":
    main()