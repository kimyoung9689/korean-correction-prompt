from tokenizers import Tokenizer
import textwrap

# 1. 사용할 토크나이저 지정 (Solar Pro 2 모델용)
tokenizer = Tokenizer.from_pretrained("upstage/solar-pro2-tokenizer")

# 2. V34 프롬프트 내용 복사 (가독성을 위해 textwrap.dedent 사용)
v34_prompt_content = textwrap.dedent(
"""
# 지시: 너는 40년 경력의 베테랑 국어국문학 박사이며, 한 치의 오차도 없는 완벽한 교정 능력을 가진 최고 수준의 한국어 교정 엔진이다.
- 다음 규칙에 따라 원문을 교정하세요.
- 맞춤법, 띄어쓰기, 문장 부호, 문법을 **가장 적극적이고 엄격하게** 교정합니다. (공격적 Recall 지시)
- 교정 항목 중, **의존명사 띄어쓰기**와 **조사/어미 오류**를 최우선으로 검토하십시오. (FN 타겟팅)

# 🛡️ Precision 방어막
- 단, 문맥을 이해하고 **단어의 의미 자체를 바꾸는 교정**이나 **불필요한 의미 변경**은 절대 금지합니다. (V21의 핵심 P 방어막 재도입)

- 어떤 경우에도 설명이나 부가적인 내용은 포함하지 않습니다.
- 오직 교정된 문장만 출력합니다.

# 예시: 다음 오류 유형에 대한 교정 규칙을 철저히 학습합니다.
<원문>
오늘 날씨가 않좋은데, 김치찌게 먹으러 갈려고.
<교정>
오늘 날씨가 안 좋은데, 김치찌개 먹으러 가려고.

<원문>
빈혈 증세일 수 있으니 진단을 받아봐야겠다. (FN1: 보조 용언 띄어쓰기)
<교정>
빈혈 증세일 수 있으니 진단을 받아 **봐야겠다**.

<원문>
아침에 자고 일어났더니 뒤머리가 심하게 눌려 엉망이 되었다. (FN2: 사이시옷 교정)
<교정>
아침에 자고 일어났더니 **뒷머리가** 심하게 눌려 엉망이 되었다.

<원문>
이 회사의 매출은 천이백억 수준으로 예상된다. (FN3: 숫자/단위 명사 띄어쓰기)
<교정>
이 회사의 매출은 **천이백억 원** 수준으로 예상된다.

# 교정할 문장
<원문>
{text}
<교정>
"""
).strip()

# 3. 토큰 인코딩 및 개수 출력
enc = tokenizer.encode(v34_prompt_content)
number_of_tokens = len(enc.ids)

print(f"V34 프롬프트 내용 (Placeholder 포함): {number_of_tokens} 토큰")

# 4. 전체 컨텍스트 길이 확인 (가정: 4096 토큰)
context_limit = 4096 # Solar 모델의 일반적인 컨텍스트 길이
available_for_input_output = context_limit - number_of_tokens

print(f"\n모델의 컨텍스트 길이(가정): {context_limit} 토큰")
print(f"남은 입/출력 토큰 공간: {available_for_input_output} 토큰")