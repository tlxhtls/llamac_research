# PPG-only 실시간 감정 category 분류 성능개선 목표

작성일: 2026-05-31

## 목표

LLaMAC 데이터셋에서 PPG 단일 센서만 사용해 `ReportedType` 5-class 감정 category를 예측한다. 목표 지표는 top-1 accuracy와 top-3 accuracy를 별도 리더보드로 최대화하되, class collapse를 막기 위해 macro F1, balanced accuracy, Cohen's kappa를 guard metric으로 함께 본다.

`IntendedType`은 원 논문/자극 의도 비교용 보조 진단 target으로만 사용한다. 제품 목표와 논문 주장에는 `ReportedType`을 우선한다.

## 프로덕션 프로토콜

후보 모델은 다음 조건을 만족해야 eligible 결과로 인정한다.

- 입력은 `band_*.csv`의 PPG 신호 하나에서 나온 정보만 사용한다.
- 예측 단위는 실시간 sliding-window에 대응되는 causal window다.
- 기본 평가 window는 단일 30초 window다.
- 제품 강건성 평가는 5초, 10초, 20초, 30초 window를 별도 stress test로 수행한다.
- full trial 전체 신호, trial 이후 구간, 미래 window, 다중 window test-time aggregation은 primary leaderboard에 넣지 않는다.
- split은 participant-grouped train/validation/test다.
- validation으로 모델과 ensemble을 선택하고, test는 선택 완료 후 한 번만 본다.
- 사용자별 calibration, subject-specific baseline, 사용자가 감정을 맞춰 알려주는 방식은 primary 결과에서 제외한다.
- Android 배포 후보는 ONNX 또는 Android에서 재현 가능한 추론 경로를 명시해야 한다.

## 현재 시작점

현재 strict 30초 waveform baseline은 raw PPG window를 resample/normalize한 뒤 5-class cross entropy로 학습했다. 로컬 artifact는 ignored `artifacts/` 아래에 저장되며 커밋하지 않는다.

| Target | Model | Val macro F1 | Test top-1 | Test top-3 | Test macro F1 | Kappa |
|---|---|---:|---:|---:|---:|---:|
| `ReportedType` | CNN | 0.1886 | 0.2162 | 0.6400 | 0.2017 | 0.0105 |
| `ReportedType` | CNN-LSTM | 0.2048 | 0.2313 | 0.6312 | 0.1921 | 0.0110 |
| `ReportedType` | ID-CNN | 0.2258 | 0.2387 | 0.6325 | 0.2230 | 0.0309 |
| `ReportedType` | CNN+Transformer encoder | 0.2306 | 0.2437 | 0.6262 | 0.2188 | 0.0411 |
| `IntendedType` | ID-CNN diagnostic | 0.2185 | 0.2175 | 0.6225 | 0.2131 | 0.0219 |

주의: 이전 trial-level tabular PPG 결과는 비교 참고용이지 primary production leaderboard가 아니다. 기존 trial-level PPG grouped 결과는 대략 top-1 0.22-0.26, top-3 0.62-0.66, macro F1 0.21-0.235 수준이었다. 이 값은 window-causal production 조건보다 느슨하다.

## 후보군

### 1. Raw waveform DNN

- CNN baseline
- ID-CNN / TCN
- CNN-LSTM / CNN-GRU
- CNN + Transformer encoder
- ResNet1D 계열
- validation-selected probability ensemble

### 2. Window-level PPG feature + classical/tree model

모든 feature는 해당 causal window 내부에서만 계산한다.

- Logistic regression / calibrated linear model
- SVC-RBF
- Random Forest / ExtraTrees
- HistGradientBoosting
- LightGBM / XGBoost
- MiniROCKET 또는 ROCKET류가 추가될 경우 PPG window 단위만 허용

### 3. Ensemble

- probability averaging
- validation macro F1 또는 top-3 기반 가중 averaging
- top-1 전용 ensemble과 top-3 전용 ensemble을 분리
- stacking은 out-of-fold prediction으로만 허용한다. test prediction을 meta-learner 학습에 쓰면 무효다.

## 리더보드

리더보드는 두 개를 분리한다.

1. **Top-1 leaderboard**
   - 1차 정렬: validation top-1
   - tie-break: validation macro F1, then validation kappa
   - test 보고: top-1, top-2, top-3, macro F1, balanced accuracy, kappa

2. **Top-3 leaderboard**
   - 1차 정렬: validation top-3
   - tie-break: validation macro F1, then validation top-1
   - test 보고: top-1, top-2, top-3, macro F1, balanced accuracy, kappa

Guard rule:

- prior/majority baseline보다 top-1 또는 top-3만 높고 macro F1이 0.15 미만이면 실용 후보로 채택하지 않는다.
- top-3 최적화 모델도 macro F1과 kappa가 현저히 낮으면 "ranking-only diagnostic"으로 분리한다.

## 종료조건

무한 탐색을 막기 위해 다음 중 하나를 만족하면 현재 round를 종료한다.

1. **후보 예산 도달**
   - eligible candidate 40개 평가 완료.
   - 또는 DNN 20개 + tabular/tree/ensemble 20개 평가 완료.

2. **연속 미개선 중단**
   - validation top-1과 top-3 어느 쪽도 `0.005` 이상 개선하지 못한 eligible candidate가 8개 연속 나오면 중단한다.
   - top-1 leaderboard와 top-3 leaderboard 중 하나라도 `0.005` 이상 개선되면 연속 미개선 카운트를 0으로 되돌린다.

3. **시간 예산 도달**
   - 단일 round의 총 GPU 학습 시간 24시간 도달.
   - 또는 실제 경과 시간 48시간 도달.

4. **목표치 도달 후 freeze**
   - validation top-1 `>= 0.30` 또는 validation top-3 `>= 0.70`을 달성하고,
   - validation macro F1 `>= 0.23`, kappa `> 0.04`를 동시에 만족하면,
   - 그 시점의 후보군으로 ensemble 검증 1회를 수행한 뒤 test를 freeze하고 round를 종료한다.

5. **품질 중단**
   - DNN 학습에서 NaN/inf가 2회 이상 재현되면 해당 설정 family를 중단한다.
   - validation 성능은 높지만 confusion matrix가 한두 class에 과도하게 몰리면 collapse 후보로 표시하고 leaderboard eligible에서 제외한다.

Test set은 round 종료 직전 선택된 후보에 대해서만 사용한다. test 결과를 본 뒤 hyperparameter, feature, ensemble weight를 고치면 새 round로 기록해야 하며, 기존 test 점수는 모델 선택에 사용하지 않는다.

## 산출물

각 round는 다음 산출물을 남긴다.

- `artifacts/results/` 아래 candidate별 result JSON
- candidate registry CSV 또는 JSON
- top-1 / top-3 리더보드
- validation-only selection 로그
- 최종 후보 model card
- Android 후보일 경우 ONNX 또는 Android 추론 경로와 입력 전처리 명세

## 기본 실행 예시

```bash
uv run python scripts/train_ppg_waveform_emotion.py \
  --model-arch idcnn_multihead \
  --label-column ReportedType \
  --min-window-seconds 30 \
  --max-window-seconds 30 \
  --eval-window-seconds 30 \
  --train-windows-per-trial 4 \
  --eval-windows-per-trial 1 \
  --seed 777 \
  --split-seed 42 \
  --learning-rate 1e-4 \
  --input-clip-value 8 \
  --device auto \
  --no-amp
```
