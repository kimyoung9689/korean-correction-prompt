# Grammar Error Correction Prompathon
한국어 GEC 프롬프톤

이 프로젝트는 Solar Pro API를 활용하여 오로지 프롬프트만으로 한국어 문장을 교정하여 성능을 올리는 프롬프톤입니다. 모델 튜닝이 아닌 프롬프트 엔지니어링만으로 교정 태스크를 해결하는 것이 목표입니다.


한국어는 띄어쓰기, 조사 사용, 문장 부호 등 다양한 문법 규칙을 가지고 있어 맞춤법 교정이 쉽지 않습니다. 특히 기사나 공식 문서 작성 시 정확한 맞춤법과 문법은 필수적입니다. 하지만 많은 사람들이 맞춤법과 문법에 어려움을 겪고 있으며, 전문가의 검토를 받기에는 시간과 비용이 많이 소요됩니다.



- 문서 작성 시간 단축

- 맞춤법 검토 비용 절감

- 일관된 문서 품질 유지

등의 효과를 기대합니다.


## 📚 Table of Contents
- [팀 구성원](#팀-구성원)
- [0. Overview](#0-overview)
- [1. Competiton Info](#1-competiton-info)
- [2. Components](#2-components)
- [3. Data descrption](#3-data-descrption)
- [4. Prompt Engineering](#4-prompt-engineering)
- [5. Result](#5-result)
- [6. How to Run](#6-how-to-run)
- [7. Reference](#7-reference)
- [ETC](#etc)



## 팀 구성원

| ![김영](https://avatars.githubusercontent.com/u/213391898?v=4&s=200) |
| :--------------------------------------------------------------: |
| [![GitHub](https://img.shields.io/badge/GitHub-김영-181717?style=flat&logo=github&logoColor=white)](https://github.com/kimyoung9689) |


## 0. Overview

프롬프트 엔지니어링을 시스템 아키텍처 관점으로 접근하여 
LLM의 Recall 한계를 극복했습니다



- 모델 아키텍처: $\text{Upstage Solar Pro 2}$를 중심으로, 데이터 필터링 모듈과 **다단계 추론(Multi-Prompt)**을 결합한 하이브리드 교정 시스템

- 핵심 전략: $\text{FN}$ 후보군 선별 $\rightarrow$ $\text{CoT}$ 재시도 $\rightarrow$ $\text{XML}$ 2-Step 재시도 $\rightarrow$ 조건부 병합(Precision-Guard)

- 평가 지표: $\text{F1-Score}$ (LCS 토큰 단위 평가)의 $\text{Precision}$ 및 $\text{Recall}$ 동시 최적화

- 최종 점수: $\text{F1-Score}$ 38.2979 달성

- 주요 활용 기술: $\text{Chain-of-Thought (CoT)}$, $\text{Multi-Turn (XML)}$, $\text{Precision}$ 방어 로직 설계



###  Environment & Requirements

프롬프트 엔지니어링 자동화 및 시스템 구현을 위한 환경입니다.

| 항목 | 상세 내용 |
| :--- | :--- |
| **OS** | Ubuntu 20.04.6 LTS  |
| **Python** | **3.12**  |
| **Core Model** | **Upstage Solar Pro 2** |
| **API Endpoint** | [Upstage Console](https://console.upstage.ai/) (UPSTAGE\_API\_KEY 사용) |
| **Dependencies** | pandas, tqdm, tiktoken |
| **Full List** | **See pyproject.toml** |



## Core Technologies (Visual Stack)

| 항목 |  | 역할 |
| :--- | :--- | :--- |
| **Python** | ![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=flat-square&logo=python&logoColor=white) | 시스템 구축 언어 |
| **Core Model** | ![Upstage Solar Pro 2](https://img.shields.io/badge/Solar%20Pro%202-FF6F00?style=flat-square&logo=openai&logoColor=white) | LLM 백본 |
| **API Wrapper** | ![Python Client](https://img.shields.io/badge/Python%20Client-353535?style=flat-square&logo=python&logoColor=white) | Upstage API 호출 (openai 패키지 활용) |
| **Data Filtering** | ![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white) | FN 후보 선별 및 결과 병합 로직 |
| **Progress Bar** | ![tqdm](https://img.shields.io/badge/tqdm-303130?style=flat-square&logo=tqdm&logoColor=white) | API 호출 진행 상황 시각화 |
| **Tokenization** | ![tiktoken](https://img.shields.io/badge/tiktoken-000000?style=flat-square) | API 비용 및 길이 제한 관리 |
| **Metrics** | ![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=flat-square&logo=scikitlearn&logoColor=white) | F1 스코어 평가 로직 |
| **Dependency Mgmt**| **![uv](https://img.shields.io/badge/uv-30A4C7?style=flat-square)** | 효율적인 의존성 관리 |







## 1. Competiton Info

### Overview

이 대회는 Solar Pro 2 모델과 프롬프트 설계만을 활용하여 한국어 문장의 띄어쓰기, 맞춤법, 문장 부호 오류를 정확히 교정하는 경진대회입니다. 

- **기간**: 2025년 10월 20일 ~ 2025년 10월 24일

- **주제**: 프롬프트만으로 한국어 문법 오류를 정확히 교정


- **데이터셋**: 업스테이지 고객사 실제 데이터 기반 (254문장)

- **주요 제약**: 외부 검색 및 데이터 활용 금지, 토큰 제한($2000$ 토큰)


- **평가지표**: Recall (재현율, %)


토큰 단위 비교
문장을 공백 기준으로 토큰(단어) 단위로 분할

정답과 예측 결과의 차이를 최장 공통 부분 수열 알고리즘을 활용해 비교

이를 통해, 실제로 어떤 부분이 올바르게 교정되었는지, 놓친 부분/불필요하게 수정한 부분을 정밀하게 판별

#### 평가 항목

True Positive (TP): 정답과 예측이 정확히 일치한 교정 건수

False Positive (FP): 예측이 잘못 수정한 건수

False Missing (FN): 정답에는 있지만 예측이 놓친 교정 건수

False Redundant (FR): 예측이 불필요하게 수정한 건수

Precision(정밀도) 계산식
![alt text](image-1.png)

TP: 올바른 교정

FP: 잘못된 교정

FN: 놓친 교정

FR: 불필요한 교정

Recall(재현율) 계산식
![alt text](image-2.png)

TP: 올바른 교정

FP: 잘못된 교정

FN: 놓친 교정

- **주요 목표**:

오류 유형을 분석하고, 해당 패턴을 해결하기 위한 RAG/CoT 등의 심화 프롬프트 기법을 적용

SW 개발 및 시스템 이해하기: 제한된 토큰(2000 토큰) 및 API 호출 횟수(최대 3회/케이스)와 같은 기술적 제약 조건 내에서 효율성을 극대화하는 프롬프트 및 호출 로직을 설계

---

## 2. Components

### Directory

프로젝트는 모듈성과 재사용성을 극대화하기 위해 다음과 같은 명확한 디렉토리 구조를 따릅니다.


```txt
.
├── src/
│   ├── analysis/
│   │   ├── eda_failure_analysis.py  # 데이터 탐색 및 실패 사례 분석
│   │   └── __init__.py
│   ├── baseline_generate.py         # 초기 Baseline 프롬프트 실행 및 결과 생성
│   ├── check_tokens.py              # 토큰 제한(2000 토큰) 검사 유틸리티
│   ├── evaluate.py                  # 모델 출력에 대한 성능(리콜 점수) 평가 스크립트
│   ├── filter_fm_candidates.py      # 최종 제출 후보 필터링 로직
│   ├── metrics.py                   # 성능 지표(Metrics) 계산 로직
│   ├── merge_final_submission.py    # 최종 제출 파일을 병합하는 스크립트
│   ├── multi_turn_generate.py       # 멀티턴(Multi-turn) 전략 적용 프롬프트 실행
│   ├── prompts.py                   # 프롬프트 템플릿 및 관련 함수 정의
│   ├── retry_generate.py            # 실패 케이스 재시도 로직 (구 버전)
│   └── retry_generate_v2.py         # 개선된 실패 케이스 재시도 로직 (버전 2)
├── data/                            # 🚫 대회 데이터셋 (gitignore 처리됨)
│   ├── train.csv                    # 학습 데이터 파일
│   ├── test.csv                     # 테스트 데이터 파일
│   └── sample_submission.csv        # 제출 양식 파일
├── submission/                      # 최종 제출 파일 생성 위치
├── .env                             # 🚫 API 키 등 환경 변수 (gitignore 처리됨)
├── .gitignore
├── .python-version
├── pyproject.toml
└── README.md
```

---

## 3. Data descrption

### Dataset overview

- 이 대회는 Upstage 고객사 데이터를 활용하여 LLM이 합성한 한국어 데이터셋을 기반으로 진행되었습니다. 참가자는 프롬프트 엔지니어링을 통해 이 데이터셋에 포함된 한국어 문장의 맞춤법, 띄어쓰기, 문장 부호 오류를 정확하게 교정하는 것을 목표로 합니다.

때문에 대회 규정상 일반적인 정보만 언급, EDA결과나 데이터내용 작성이 금지됩니다. 

- 데이터 출처: Upstage 고객사 데이터를 활용한 LLM 합성 데이터셋

- 데이터 개수: 254문장

- 주요 오류 유형: 맞춤법, 띄어쓰기, 문장 부호 등 (조사 오류, 사이시옷, 고유명사 등 고질적 오류 포함)

### Data Processing

EDA (Exploratory Data Analysis)본 프로젝트는 대회 규칙을 준수하기 위해 데이터셋 자체를 외부에 공개하거나 시각화 결과를 공유하는 행위는 엄격히 금지하고 있습니다. 

따라서 EDA는 프롬프트 설계 전략을 수립하고 Recall 점수 극대화를 위한 인사이트를 얻는 내부적인 목적으로만 수행되었습니다.

1. EDA 목적 및 방법론

목적: 학습 데이터(train.csv)에 포함된 254문장의 주요 오류 패턴을 정성적/정량적으로 파악하여, Solar Pro 2 모델의 교정 성공률을 높이는 최적의 프롬프트 전략을 수립하기 위함입니다.

분석 대상 컬럼: 오류 유형(type), 오류 부분(original_target_part), 교정 부분(golden_target_part) 등 학습 데이터에만 존재하는 정보를 중점적으로 분석했습니다2.도구: src/analysis/eda_failure_analysis.py 스크립트를 활용하여 오류 분포 및 모델의 실패 케이스를 분석했습니다.

2. 프롬프트 전략과의 연계:Few-Shot 예시 구성: EDA를 통해 빈번하게 발생하는 조사 오류, 띄어쓰기 오류, 맞춤법 오류 등의 대표 유형을 식별하고, 해당 유형의 고품질 예시를 프롬프트에 포함하여 모델의 성능 편향을 유도했습니다

3. Multi-turn 전략 설계: 복합적인 오류가 포함된 케이스를 파악하여, 오류 유형 분류 후 **단계별 문제 해결 (CoT)**을 유도하는 Multi-turn 프롬프트 전략 설계에 활용했습니다

4. Recall 최적화: 평가 지표인 Recall($\text{Recall} = \text{TP} / (\text{TP} + \text{FP} + \text{FM})$)의 분모(TP + FP + FM)를 놓치지 않기 위해, 모델이 놓치기 쉬운 False Missing (FM) 오류 유형을 집중적으로 분석하여 프롬프트에 반영했습니다.

## 4. Prompt Engineering

Recall 점수 극대화를 위해 프롬프트 작성 규칙 을 철저히 준수하며 다음과 같은 고급 프롬프트 설계 기법을 적용했습니다.

#### Multi-turn 전략을 활용한 단계적 교정:

단일 API 호출의 한계를 극복하기 위해, 하나의 케이스 교정 시 최대 3회 이내 복수 API 호출 을 활용하는 Multi-turn 프롬프트 구조를 설계했습니다.

이는 오류 유형 분류 후 문제 해결 단계를 구분하는 등 CoT (Chain-of-Thought) 원리를 적용하여 교정의 정확도와 일관성을 높이는 전략입니다.

#### Few-Shot 및 데이터 인사이트 반영:

EDA를 통해 파악한 데이터셋의 **고질적 오류 유형 (조사, 사이시옷, 띄어쓰기 등)**을 해결하기 위한 맞춤형 Few-Shot 예시를 프롬프트에 포함하여 모델의 교정 방향성을 명확히 제시했습니다.


토큰 제한 (전체 세션 2000 토큰) 을 엄격히 준수하며 효율적인 예시 구성을 유지했습니다.

#### Function Calling 및 RAG (선별적 활용):

Function Calling 및 RAG 접근 방식이 허용되지만, 대회 제공 데이터셋 내에서의 활용에 한정된다는 규칙 을 준수했습니다.


외부 검색 및 외부 DB 호출은 엄격히 금지되므로, 오직 제공된 데이터셋 정보만을 이용하는 범위 내에서 교정의 견고함을 더하는 보조 전략으로 활용 가능성을 탐구했습니다.

#### 파라미터 최적화:

Temperature 등 API 호출 파라미터를 자유롭게 변경하며, 가장 안정적이고 높은 Recall 점수를 확보할 수 있는 최적의 조합을 반복 실험을 통해 도출했습니다.


---

## 5. Result

### Leader Board

![alt text](image-3.png)

---

## 6. How to Run 

### Setup

#### 1. 환경 설정

본 프로젝트는 Python 3.12 이상 환경에서 실행하는 것을 권장하며, 필요한 패키지는 pyproject.toml을 통해 관리됩니다.

/코드
# Python 버전 확인
$ python --version

# 필요한 라이브러리 설치 (Poetry 등 패키지 매니저 사용 시)
# $ poetry install


2. API 키 설정 (Solar Pro 2)
Upstage Solar Pro 2 API 호출을 위해서는 API Key 설정이 필수입니다. 보안 유지를 위해 환경 변수 파일을 사용하여 관리하며, 이 파일은 Git 추적에서 제외됩니다 (.gitignore에 명시됨).

프로젝트 루트 경로에 .env 파일을 생성합니다.

다음 형식에 맞춰 Solar Pro 2 API 키를 입력합니다.

/코드
# .env 파일 내용
SOLAR_API_KEY="[여기에 발급받은 실제 API 키 입력]"

3. 데이터셋 준비
대회 참여를 위해 제공받은 데이터셋 파일을 data/ 디렉토리 내에 위치시킵니다.

코드
# data 디렉토리 구조 (로컬 환경 기준)
.
└── data/
    ├── train.csv
    ├── test.csv
    └── sample_submission.csv


[주의] 제공된 데이터셋은 (주)업스테이지에 귀속되며 , 유출, 복사, 공유가 엄격히 금지되어 있습니다. 본 프로젝트 실행을 위해서만 사용해야 하며, data/ 디렉토리는 .gitignore를 통해 Git/GitHub 추적에서 제외됩니다.


---

## ETC

#### Acknowledgement
본 프로젝트는 Upstage Kernel Academy에서 주최한 Grammar Error Correction Promptathon에 참가하며 개발되었습니다. 대회 기간 동안 학습 및 실험 환경을 제공해주신 Upstage와 멘토님들께 감사드립니다.


#### License & Compliance (라이선스 및 규정 준수)
본 프로젝트에서 사용된 데이터셋, 평가 코드, 평가 로직 일체는 (주)업스테이지에 귀속되어 있습니다.

- 데이터셋 및 로직 공유 금지: 데이터셋, 평가 코드, 평가 로직의 일부 또는 전부를 복사, 복제, 판매, 재판매, 공개, 공유 등을 할 수 없습니다.


- 결과 재현 의무: 대회 최종 순위권에 해당하는 경우, 결과 재현을 위해 필요한 모든 산출물(코드 및 데이터셋 전체 파일)을 정리하여 제출해야 하는 규정을 철저히 준수했습니다.


### Future Work

- 실시간 오류 진단 RAG 구현: 현재는 제공된 데이터셋 내에서만 RAG 접근 방식을 탐구했으나, 일반적인 한국어 문법 및 띄어쓰기 규칙 기반의 외부 Knowledge Base를 (규정상 허용될 경우) 구축하여 LLM의 교정 근거를 강화하고 Recall을 안정화합니다.

- 파라미터 기반 A/B 테스트 자동화: 현재 수동으로 진행된 파라미터(Temperature, Top-P) 튜닝을 자동화된 실험 환경에서 진행하여, 모델의 안정적인 성능을 보장하는 최적의 조합을 더욱 빠르게 확보합니다.

- 복합 Multi-turn 로직 심화: 현재의 3회 API 호출 제한을 넘어서, 오류 유형별로 API 호출 횟수 및 파라미터를 동적으로 변경하는 더욱 정교하고 복합적인 Multi-turn 전략을 개발합니다.


### Reference

본 프로젝트 수행 및 전략 개발에 참고한 자료와 대회 공식 자료는 다음과 같습니다.


## 7. Reference

### 1. 기술 및 학술 참고 자료 (Academic References)

| 구분 | 논문/자료명 | 출처 (arXiv, GitHub 등) |
| :--- | :--- | :--- |
| **Chain-of-Thought (CoT)** | Chain-of-Thought Prompting Elicits Reasoning in Large Language Models | [arXiv:2201.11903](https://arxiv.org/abs/2201.11903) |
| **Retrieval-Augmented Generation (RAG)** | Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks | [arXiv:2005.11401](https://arxiv.org/abs/2005.11401) |
| **한국어 교정 연구** | LLM 모델을 활용한 한국어 맞춤법 교정 성능 최적화 방안 연구 | [Google Sites](https://sites.google.com/pusan.ac.kr/jjk801/%ED%99%88) |

### 2. API 및 방법론 참고 자료

* **Upstage Solar Pro API 문서:** Solar Pro 2 모델의 API 호출 규격, 파라미터(Parameter) 정보 및 사용 가이드라인 (**Upstage AI Developer Documentation**)
* **Chain-of-Thought/RAG 관련 공개 구현체:** CoT와 RAG의 프롬프팅 로직을 코드에 적용하는 방식을 이해하기 위해 공개된 GitHub 저장소 및 블로그 자료 (LangChain, LlamaIndex 등 **LLM 프레임워크 커뮤니티 자료**)
* **LLM 파라미터 튜닝 가이드:** Temperature, Top-P 등의 파라미터가 모델 출력에 미치는 영향을 분석하고 최적화하는 데 활용된 LLM 서비스 제공사의 기술 문서 및 **관련 연구 자료**











