import os
import argparse
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from openai import OpenAI, APIError
import tiktoken # 토큰 계산 라이브러리 추가

# 새로 추가된 Multi-Turn 프롬프트를 포함하도록 import
from prompts import PROMPT_STEP1_XML, PROMPT_STEP2_XML

# Load environment variables
load_dotenv()

# 토큰 계산기 초기화 (GPT-4용이지만 Solar Pro 2의 토큰 근사치 계산에 활용)
TOKEN_LIMIT = 2000
ENCODER = tiktoken.get_encoding("cl100k_base") 
SYSTEM_MESSAGE = "당신은 한국어 문장 교정 전문가이며, 지시에 따라 XML 형식을 준수하고 단계별 작업을 정확하게 수행합니다."


def count_tokens(messages: list) -> int:
    """메시지 리스트의 전체 토큰 수를 계산합니다."""
    total_tokens = 0
    for message in messages:
        # role, content, name 등의 토큰을 근사 계산
        if message.get("content"):
            total_tokens += len(ENCODER.encode(message["content"]))
        total_tokens += 4 # role, content, name 등의 오버헤드
    return total_tokens + 2 # 마지막 메시지의 오버헤드


def retry_correction_multi_turn_v2(client: OpenAI, model: str, text: str) -> str:
    """Multi-Turn 2-Step API 호출을 수행합니다 (2000 토큰 안전 로직 포함)."""
    
    # 1. Step 1: 오류 식별 (Recall 공격)
    messages_step1 = [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": PROMPT_STEP1_XML.format(text=text)},
    ]
    
    current_tokens = count_tokens(messages_step1)
    if current_tokens >= TOKEN_LIMIT:
        print(f"\n[Safety Skip] 입력 토큰 초과 ({current_tokens} >= 2000). 원문 유지.")
        return text

    try:
        # Step 1 출력 토큰 제한: 입력 토큰을 제외한 나머지 예산 내에서 출력되도록 설정
        max_output_tokens_step1 = TOKEN_LIMIT - current_tokens - 100 
        
        resp1 = client.chat.completions.create(
            model=model,
            messages=messages_step1,
            temperature=0.0,
            max_tokens=max_output_tokens_step1 if max_output_tokens_step1 > 256 else 256
        )
        step1_output = resp1.choices[0].message.content.strip()
        
        # Step 1 출력 토큰 업데이트
        current_tokens += len(ENCODER.encode(step1_output)) 

    except APIError as e:
        print(f"\n[Error] Step 1 API 호출 실패: {e}. 원문 유지.")
        return text 
    
    # 2. Step 2: 최종 교정 문장만 출력 (Precision 확보)
    messages_step2 = messages_step1 + [
        {"role": "assistant", "content": step1_output},
        {"role": "user", "content": PROMPT_STEP2_XML},
    ]

    # 전체 세션 토큰 재확인 (입력 + 출력1)
    # 현재 토큰은 Step 1의 입력 + 출력1 토큰 합산
    current_tokens = count_tokens(messages_step2)
    if current_tokens >= TOKEN_LIMIT:
        print(f"\n[Safety Skip] Step 2 입력 토큰 초과 ({current_tokens} >= 2000). 원문 유지.")
        return text
    
    try:
        # Step 2 출력 토큰 제한: 교정 문장의 최대 길이보다 조금 더 크게 설정
        max_output_tokens_step2 = TOKEN_LIMIT - current_tokens - 100
        
        resp2 = client.chat.completions.create(
            model=model,
            messages=messages_step2,
            temperature=0.0,
            max_tokens=max_output_tokens_step2 if max_output_tokens_step2 > 128 else 128
        )
        step2_output = resp2.choices[0].message.content.strip()
        
        # 교정 문장만 반환
        return step2_output if step2_output else text
        
    except Exception as e:
        print(f"\n[Error] Step 2 API 호출 또는 처리 실패: {e}. 원문 유지.")
        return text


def main():
    parser = argparse.ArgumentParser(description="Re-generate corrections for 2nd FM candidates using a 2-Step Multi-Turn XML prompt with 2000 token safety.")
    parser.add_argument("--input", default="data/fm_candidates_to_retry_v2.csv", help="Input CSV path (2nd FM candidates) to re-correct.")
    parser.add_argument("--output", default="data/fm_recorrected_v2.csv", help="Output CSV path for 2nd re-corrected results.")
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
    print(f"2nd FM Candidates to retry: {len(df)} (9개 문장)")
    print(f"Output: {args.output}")
    print(f"!!! 2-Step Multi-Turn 전략으로 2000 토큰 제한을 안전하게 준수합니다. !!!")


    ids = df["id"].astype(str).tolist()
    err_sentences = df["err_sentence"].astype(str).tolist()
    cor_sentences = []
    
    # Process each sentence
    for i, text in enumerate(tqdm(err_sentences, desc="2nd Re-correcting FM (2-Step Multi-Turn)")):
        corrected = retry_correction_multi_turn_v2(client, args.model, text)
        cor_sentences.append(corrected)

    # Save results
    out_df = pd.DataFrame({
        "id": ids,
        "err_sentence": err_sentences, 
        "cor_sentence": cor_sentences,
    })
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    out_df.to_csv(args.output, index=False)
    
    print(f"✅ 2nd Re-correction complete. Results saved to {args.output}")


if __name__ == "__main__":
    main()