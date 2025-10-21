# Prompthon Baseline Code

프롬프트 해커톤을 위한 기본 베이스라인 코드입니다.

## 📋 필수 파일 설명

```
baseline_code/
├── baseline_generate.py   # 교정 문장 생성 스크립트
├── evaluate.py            # 평가 스크립트
├── metrics.py             # 평가 메트릭 (Recall, Precision 계산)
├── prompts.py             # 프롬프트 템플릿 (이 파일을 수정하세요!)
├── pyproject.toml         # Python 의존성 관리
├── .python-version        # Python 버전 명시
├── .env.example           # 환경 변수 예시
└── data/                  # 데이터셋 디렉토리
    └── train_dataset.csv  # 학습 데이터 (여기에 넣으세요)
```

## 🚀 빠른 시작

### 1. 환경 설정

```bash
# uv 설치 (이미 설치되어 있다면 생략)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 의존성 설치
uv sync
```


### 2. API 키 설정

# .env 파일을 열어서 API 키 입력
# UPSTAGE_API_KEY=your_actual_api_key_here

Upstage API 키는 [https://console.upstage.ai/](https://console.upstage.ai/)에서 발급받을 수 있습니다.

> 💡 **팁**: `.env` 파일은 API 키 같은 민감한 정보를 저장하는 파일이므로 Git에 커밋되지 않도록 `.gitignore`에 포함되어 있습니다.

### 3. 데이터 준비

`data/train_dataset.csv` 파일을 준비합니다. 파일은 다음 컬럼을 포함해야 합니다:
- `err_sentence`: 교정이 필요한 원문
- `cor_sentence`: 교정된 정답 (평가 시 사용)

### 4. 교정 문장 생성

```bash
# 기본 실행
uv run python baseline_generate.py

# 옵션 지정
uv run python baseline_generate.py --input data/train_dataset.csv --output submission.csv --model solar-pro2
```

생성된 `submission.csv` 파일은 다음 컬럼을 포함합니다:
- `err_sentence`: 원문
- `cor_sentence`: AI가 교정한 문장

### 5. 평가

```bash
# 기본 실행
uv run python evaluate.py

# 옵션 지정
uv run python evaluate.py --true_df data/train_dataset.csv --pred_df submission.csv --output analysis.csv
```

평가 결과:
- Recall과 Precision이 콘솔에 출력됩니다
- `analysis.csv` 파일에 샘플별 상세 분석 결과가 저장됩니다

## 🎯 성능 개선 방법

### 프롬프트 수정

`prompts.py` 파일의 `baseline_prompt`를 수정하여 성능을 개선할 수 있습니다.

```python
baseline_prompt = (
"""
# 지시
- 여기에 더 나은 지시사항을 작성하세요
- 예시를 추가하거나 수정하세요
- 오류 유형별 가이드를 제공하세요

# 교정할 문장
<원문>
{text}
<교정>
"""
    .strip()
)
```

### 실험 예시

1. **Few-shot 예시 추가**: 다양한 오류 유형의 예시를 프롬프트에 추가
2. **오류 유형 명시**: 맞춤법, 띄어쓰기, 조사 오류 등을 명시적으로 나열
3. **CoT (Chain-of-Thought)**: 단계별 사고 과정을 유도하는 프롬프트 작성
4. **시스템 메시지 수정**: `baseline_generate.py`의 system message를 수정

## 📊 평가 메트릭

- **TP (True Positive)**: 정답과 예측이 모두 같은 위치에서 같은 교정을 수행
- **FP (False Positive)**: 예측이 잘못된 교정을 수행
- **FM (False Missing)**: 예측이 필요한 교정을 놓침
- **FR (False Redundant)**: 예측이 불필요한 교정을 수행

**Recall** = TP / (TP + FP + FM) × 100  
**Precision** = TP / (TP + FP + FR) × 100

## 💡 팁

1. **반복 실험**: 다양한 프롬프트를 시도하고 `analysis.csv`를 분석하여 개선점 찾기
2. **오류 분석**: `analysis.csv`에서 fp, fm, fr이 높은 샘플을 집중 분석
3. **모델 선택**: `--model` 옵션으로 다른 모델 시도 (예: solar-pro, solar-mini)
4. **Temperature 조정**: `baseline_generate.py`의 temperature 파라미터 조정 (기본값: 0.0)

## 🔧 문제 해결

### API 키 오류
```
ValueError: UPSTAGE_API_KEY not found in environment variables
```
→ `.env.example` 파일을 `.env`로 **이름을 변경**했는지 확인하세요!  
→ `.env` 파일에 실제 API 키가 입력되어 있는지 확인하세요.

### 컬럼 오류
```
ValueError: Input CSV must contain 'err_sentence' column
```
→ 데이터셋에 `err_sentence` 컬럼이 있는지 확인하세요.

### 길이 불일치 오류
```
ValueError: Length mismatch: truth=100 vs pred=99
```
→ 생성 과정에서 일부 샘플이 누락되었습니다. 에러 로그를 확인하세요.

## 📚 참고 자료

- Upstage API 문서: https://console.upstage.ai/docs/getting-started
- uv 문서: https://docs.astral.sh/uv/

---

**Good luck with your prompt engineering!** 🚀

