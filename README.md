# HAI! - Hecto AI Challenge : 딥페이크 탐지 AI 모델

> **DACON** | HAI(하이)! - Hecto AI Challenge : 2025 하반기 헥토 채용 AI 경진대회  
> 주제: 딥페이크 탐지 AI 모델 개발 (이진 분류)

---

## 📌 대회 개요

| 항목 | 내용 |
|------|------|
| 주최/주관 | 헥토(Hecto) / 데이콘 |
| 대회 링크 | [DACON 대회 페이지](https://dacon.io/competitions/official/236628/overview/description) |
| 태스크 | 이미지/영상 기반 딥페이크(Fake) vs 실제(Real) 이진 분류 |
| 평가 지표 | Accuracy |
| 학습 데이터 | 공식 제공 없음 (자체 구축 필요) |

---


## 🧠 접근 방법

### 데이터
- **FaceForensics++ (FF++)** 데이터셋을 자체 구축하여 학습에 활용
- 영상에서 얼굴 크롭(crop) 이미지를 추출하여 사용
- 영상당 균등 간격으로 **8개 프레임**을 샘플링

### 모델
- **EfficientNet-B4** (`timm` 라이브러리, ImageNet pretrained)
- 학습 시: 단일 프레임 입력 → 이진 분류
- 추론 시: 영상의 8개 프레임 예측값을 평균하여 최종 확률 산출

### 학습 설정

| 항목 | 값 |
|------|----|
| 입력 해상도 | 380 × 380 |
| 배치 크기 | 16 |
| 에폭 | 5 |
| 학습률 | 3e-4 |
| 옵티마이저 | AdamW |
| 손실 함수 | BCEWithLogitsLoss |

### 데이터 증강
- RandomResizedCrop, RandomHorizontalFlip, ColorJitter (학습 시)
- Resize only (검증/추론 시)

---

## ⚙️ 실행 방법

### 환경 설치

```bash
pip install torch torchvision timm opencv-python scikit-learn tqdm pillow pandas
```

### 1단계: 프레임 인덱스 추출

```bash
python 1_preprocess.py
```
- `train_data/ff++_datacrop/{real,fake}/` 경로의 영상을 읽어 각 영상에서 샘플링할 프레임 인덱스를 `.npy` 파일로 저장합니다.

### 2단계: 모델 학습

```bash
python 2_train.py
```
- 프레임 이미지(`.jpg`)를 로드하여 EfficientNet-B4를 학습합니다.
- 검증 정확도 기준으로 최적 모델을 `models/best_model.pth`에 저장합니다.

### 3단계: 추론 및 제출 파일 생성

```bash
python 3_inference.py
```
- `test_data/` 경로의 영상을 읽어 딥페이크 확률을 계산합니다.
- 결과를 `submission/submission.csv`로 저장합니다.

---

## 📁 출력 파일 형식

`submission.csv`

| video | prediction |
|-------|------------|
| sample_001.mp4 | 0.9231 |
| sample_002.mp4 | 0.0412 |

- `prediction`: 딥페이크일 확률 (0~1), 0.5 이상이면 Fake로 분류

---

## 📦 주요 라이브러리

| 라이브러리 | 용도 |
|-----------|------|
| `timm` | EfficientNet-B4 모델 |
| `torch`, `torchvision` | 딥러닝 학습/추론 |
| `opencv-python` | 영상 프레임 추출 |
| `scikit-learn` | Train/Val 분할 |
| `Pillow` | 이미지 로드 |
| `pandas` | 제출 파일 생성 |
# Hecto_Ai_Challenge
