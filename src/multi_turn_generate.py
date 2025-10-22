import os
import argparse
import re
import json # JSON 대신 XML을 사용하지만, 라이브러리는 그대로 둡니다.
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from openai import OpenAI
from openai import APIError

# <<<--- 변경: prompts.py에서 Multi-Turn 프롬프트 2개를 가져오도록 변경 --->>>
from prompts import PROMPT_STEP_1, PROMPT_STEP_2 

# Load environment variables
load_dotenv()

SYSTEM_MESSAGE = "당신은 한국어 문장 교정 전문가입니다. 주어진 지시에 따라 정확하게 분석하고 교정 작업을 수행합니다."


def extract_correction(output_text: str, original_text: str) -> str:
    """Multi-Turn 전략에서는 최종 교정 문장만 출력되므로, 간단히 마지막 라인을 반환합니다."""
    
    # 모델이 지시를 따라 오직 교정된 문장만 출력했다고 가정하고 파싱
    corrected = output_text.strip()
    
    # 혹시 모를 잔여 텍스트나 빈 출력 대비 (안전장치)
    if not corrected:
        return original_text
    
    # 2단계 프롬프트가 문장만 출력하도록 강력하게 강제했으므로, 그대로 반환
    return corrected

# <----------------- 핵심 로직: Multi-Turn API 호출 함수 구현 (XML 기반) ----------------->
def multi_turn_correction(client: OpenAI, model: str, text: str) -> str:
    """오류 식별 -> 최종 교정의 2단계 Multi-Turn API 호출을 수행합니다. (XML 기반)"""
    
    # 1단계 출력이 실패했을 때 2단계 입력을 위한 기본값 설정
    error_list_summary = "오류 식별 실패. 원문만 참고하여 교정하세요."
    
    # -----------------------------------------------------------------
    # 1. 1차 호출: 오류 식별 (Recall 향상 목적)
    # -----------------------------------------------------------------
    step1_prompt = PROMPT_STEP_1.format(text=text)
    
    try:
        resp_1 = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_MESSAGE},
                {"role": "user", "content": step1_prompt},
            ],
            temperature=0.0,
            max_tokens=512 # 오류 목록 XML을 생성하기 위한 충분한 토큰 제공
        )
        error_list_raw = resp_1.choices[0].message.content.strip()
        
        # <<<--- 변경된 핵심 로직: XML 태그 추출 --->>>
        # <오류목록> 태그만 추출하여 2단계에 전달 (XML 파싱 오류 방지)
        xml_match = re.search(r'<오류목록>(.*?)</오류목록>', error_list_raw, re.DOTALL)
        
        if xml_match:
            # 태그 내부 내용과 태그 자체를 2차 호출에 전달하여 모델이 오류 목록임을 명확히 인지하게 함
            error_list_summary = "<오류목록>" + xml_match.group(1).strip() + "</오류목록>"
        else:
            # 태그 추출 실패 시, raw 텍스트 그대로 2차에 전달하여 참고하도록 유도
            error_list_summary = f"식별된 오류 목록: {error_list_raw[:500]}..."
            
    except APIError as e:
        print(f"\n[Warning] 1차 API 호출 실패: {e}. 2차 호출은 기본 프롬프트로 진행됩니다.")
        
    except Exception as e:
        print(f"\n[Warning] 1차 XML 파싱/처리 실패: {e}. 2차 호출은 기본 프롬프트로 진행됩니다.")
        
    # -----------------------------------------------------------------
    # 2. 2차 호출: 최종 교정 (Precision 유지)
    # -----------------------------------------------------------------
    step2_prompt = PROMPT_STEP_2.format(
        error_list_from_step1=error_list_summary,
        original_text=text
    )

    try:
        resp_2 = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_MESSAGE},
                {"role": "user", "content": step2_prompt},
            ],
            temperature=0.0,
        )
        raw_output = resp_2.choices[0].message.content.strip()
        
        # 최종 교정 문장만 파싱 (2차 프롬프트는 문장만 출력하도록 강력하게 지시)
        return extract_correction(raw_output, text)
        
    except Exception as e:
        print(f"\n[Error] 2차 API 호출 실패: {e}")
        return text # 최종 실패 시 원문 반환


def main():
    parser = argparse.ArgumentParser(description="Generate corrected sentences using Upstage API with Multi-Turn Strategy (XML v2)")
    parser.add_argument("--input", default="data/test.csv", help="Input CSV path containing err_sentence column")
    # <<<--- 변경: 출력 파일명을 새로운 Multi-Turn XML 파일로 변경 --->>>
    parser.add_argument("--output", default="submission/final_submission_multi_turn_xml_v2.csv", help="Output CSV path") 
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
            # Multi-Turn Logic Call (2 API calls per sentence)
            # -----------------------------------------------------------------
            corrected = multi_turn_correction(client, args.model, text)
            cor_sentences.append(corrected)
            
        except APIError as e:
            # API 키 만료, 권한 오류 등 심각한 오류 시
            print(f"\n!!! API ERROR on ID {ids[i]} ({text[:50]}...): {e}")
            cor_sentences.append(text) 
            
        except Exception as e:
            # 기타 일반 오류
            print(f"\n!!! UNEXPECTED ERROR on ID {ids[i]} ({text[:50]}...): {e}")
            cor_sentences.append(text)


    # Save results with required column names
    out_df = pd.DataFrame({
        "id": ids,
        "err_sentence": err_sentences, 
        "cor_sentence": cor_sentences,
    })
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    out_df.to_csv(args.output, index=False)
    
    print(f"✅ Generation complete. Results saved to {args.output}")


if __name__ == "__main__":
    main()