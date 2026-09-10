# DCC 2026 Mission 3 — Colab GPU

상담 텍스트에서 9개 증상을 찾는 다중 라벨 분류 코드입니다. 기본 모델은 `klue/roberta-large`이며, 긴 문서는 겹치는 청크로 나눈 뒤 문서 단위로 결합합니다.

## 가장 쉬운 실행 방법

1. Colab에서 `DCC2026_Mission3_Colab.ipynb`를 엽니다.
2. 런타임을 GPU로 변경합니다. (`런타임 → 런타임 유형 변경`)
3. Google Drive의 `train.csv`, `val.csv` 경로만 수정합니다.
4. 셀을 위에서부터 실행합니다.

학습 설정과 임계값은 **Training 내부 검증셋만** 사용해 결정합니다. `val.csv`는 최종 후보의 예측 파일 생성과 공식 평가에만 사용하세요.

## 기본 실험

```bash
python train.py \
  --train_csv /content/drive/MyDrive/DCC/train.csv \
  --output_dir /content/drive/MyDrive/DCC/checkpoints/klue_roberta_large
```

```bash
python predict.py \
  --input_csv /content/drive/MyDrive/DCC/val.csv \
  --checkpoint_dir /content/drive/MyDrive/DCC/checkpoints/klue_roberta_large \
  --output_csv /content/drive/MyDrive/DCC/val_predictions.csv
```

## 주요 기본 설정

- 모델: `klue/roberta-large`
- 손실함수: 클래스 불균형을 반영한 weighted BCE
- 최대 길이/stride: 384/96
- 문서 결합: 청크별 확률 max pooling
- AMP(fp16), gradient accumulation, gradient checkpointing 사용
- 평가 및 라벨별 임계값 선택: Training 80:20 내부 분할만 사용

T4에서 메모리가 부족하면 `--batch_size 1 --max_length 256`으로 낮추세요. L4/A100을 권장합니다.

## 주의

- 데이터와 모델 체크포인트는 GitHub에 올리지 않습니다.
- `val.csv`의 정답은 학습, 모델 선택, 임계값 조정에 사용하지 않습니다.
- 입력에는 `text`만 사용합니다.
데이터AI 크리에이터 캠프 공모전 미션3
