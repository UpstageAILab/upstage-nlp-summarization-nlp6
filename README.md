# Diaglogue Summarization | 일상 대화 요약
## Team NLP 4조

| ![김윤겸](https://github.com/gyeom-yee.png) | ![남영진](https://github.com/NamisMe.png) | ![노균호](https://github.com/devguno.png) | ![윤수인](https://github.com/suinY00N.png) | ![정다슬](https://avatars.githubusercontent.com/u/156163982?v=4) |
| :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: |
|            [김윤겸](https://github.com/gyeom-yee)               |            [남영진](https://github.com/NamisMe)             |            [노균호](https://github.com/devguno)             |            [윤수인](https://github.com/suinY00N)             |            [정다슬](https://github.com/UpstageAILab)             |
|                         Modeling, 후처리                        |                            Data-Centric, 후처리                      |                         Modeling, Augmentation                 |                            Modeling, 전처리                      |                       Modeling,  hyperparameter tuning         |

## 0. Overview
### Environment
- 컴퓨팅 환경
    - 서버를 VSCode와 SSH로 연결하여 사용 
    - NVIDIA GeForce RTX 3090
    - CUDA Version 12.2
- 협업환경
  * Github, WandB
- 의사소통
  * Slack, Zoom

### Requirements
- pandas==2.1.4
- numpy==1.23.5
- wandb==0.16.1
- tqdm==4.66.1
- pytorch_lightning==2.1.2
- transformers[torch]==4.35.2
- rouge==1.0.1
- jupyter==1.0.0
- jupyterlab==4.0.9

## 1. Competiton Info

### Overview
- **Dialogue Summarization 경진대회**

  주어진 데이터를 활용하여 일상 대화에 대한 요약을 효과적으로 생성하는 모델을 개발하는 대회

- **배경**
    - 일상생활에서 대화는 **항상** 이루어지고 있습니다. 회의나 토의는 물론이고, 사소한 일상 대화 중에도 서로 다양한 주제와 입장들을 주고 받습니다. 나누는 대화를 녹음해두더라도 대화 전체를 항상 다시 들을 수는 없기 때문에 요약이 필요하고, 이를 위한 통화 비서와 같은 서비스들도 등장하고 있습니다.
    - 그러나 하나의 대화에서도 관점, 주제별로 정리하면 수 많은 요약을 만들 수 있습니다. 대화를 하는 도중에 이를 요약하게 되면 대화에 집중할 수 없으며, 대화 이후에 기억에 의존해 요약하게 되면 오해나 누락이 추가되어 주관이 많이 개입되게 됩니다.
    - 이를 돕기 위해, 우리는 이번 대회에서 **일상 대화를 바탕으로 요약문을 생성하는 모델**을 구축합니다!

- **대회 목표**

    - 경진대회의 목표는 정확하고 일반화된 모델을 개발하여 요약문을 생성하는 것입니다. 나누는 많은 대화에서 핵심적인 부분만 모델이 요약해주니, 업무 효율은 물론이고 관계도 개선될 수 있습니다. 또한, 참가자들은 모델의 성능을 평가하고 대화문과 요약문의 관계를 심층적으로 이해함으로써 자연어 딥러닝 모델링 분야에서의 실전 경험을 쌓을 수 있습니다.
    - 본 대회는 결과물 csv 확장자 파일을 제출하게 됩니다.
        - input : 249개의 대화문
        - output : 249개의 대화 요약문

### Timeline

- 24/03/08

  : 대회 시작
- 24/03/11 ~ 24/03/15
    - Baseline 코드를 통한 모델링 과정 이해
    - EDA를 통한 인사이트 도출
    - Tokenizer 수정 실험
    - 다양한 Model 사용 실험
- 24/03/16 ~ 24/03/19
    - Data Augmentation
    - Hyperparameter Tuning
    - 후처리를 통한 Score 향상 시도
- 24/03/20

  : 대회 종료

## 2. Components

### Directory

e.g.

```
├── code
│   ├── jupyter_notebooks
│   │   └── model_train.ipynb
│   └── train.py
├── docs
│   ├── pdf
│   │   └── (Template) [패스트캠퍼스] Upstage AI Lab 1기_그룹 스터디 .pptx
│   └── paper
└── input
    └── data
        ├── eval
        └── train
```

## 3. Data descrption

### Dataset overview
제공되는 데이터셋은 오직 **"** **대화문과 요약문** **"** 입니다. 회의, 일상 대화 등 다양한 주제를 가진 대화문과, 이에 대한 요약문을 포함하고 있습니다.
  
  <img src="https://aistages-api-public-prod.s3.amazonaws.com/app/Files/38e20522-3af8-438f-8039-c5547212b8db.png" height="150px" width="500px">

- 데이터 정보
    - train : 12457
    - dev : 499
    - test : 250
    - hidden-test : 249
      
- 데이터 예시

  <img src="https://aistages-api-public-prod.s3.amazonaws.com/app/Files/c0c1a6e2-6fa1-448e-9fc7-f9db45701022.png" height="200px" width="600px">
    
    - fname : 대화 고유번호 입니다. 중복되는 번호가 없습니다.
    - dialogue : 최소 2명에서 최대 7명이 등장하여 나누는 대화 내용입니다. 각각의 발화자를 구분하기 위해#Person”N”#: 을 사용하며, 발화자의 대화가 끝나면 \n 으로 구분합니다. 이 구분자를 기준으로 하여 대화에 몇 명의 사람이 등장하는지 확인해보는 부분은 [EDA](https://colab.research.google.com/drive/1O3ZAcHR9q7dccasRcxvNhCZD-gIlasGV#scrollTo=usQutfBFqtuk)에서 다루고 있습니다.
    - summary : 해당 대화를 바탕으로 작성된 요약문입니다.
  
### EDA

- _Describe your EDA process and step-by-step conclusion_

### Data Processing

- _Describe data processing process (e.g. Data Labeling, Data Cleaning..)_

## 4. Modeling

### Model descrition

- _Write model information and why your select this model_

### Modeling Process

- _Write model train and test process with capture_

## 5. Result

### Leader Board

- _Insert Leader Board Capture_
- _Write rank and score_

### Presentation

- _Insert your presentaion file(pdf) link_

## etc

### Meeting Log

- _Insert your meeting log link like Notion or Google Docs_

### Reference

- _Insert related reference_
