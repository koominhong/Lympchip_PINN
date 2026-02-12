# Lymph-chip PINN Simulator

림프칩 약물 분포 예측 시뮬레이터 - 피하주사 후 약물이 Blood, Lymph, ECM으로 분포되는 비율을 예측합니다.

## 주요 기능

- **실시간 예측**: 7개 파라미터 조절로 약물 분포 예측
- **시계열 시각화**: 0-72시간 동안의 약물 분포 변화
- **Decay 모델링**: 약물 분해 과정 포함

## 파라미터

| 파라미터 | 설명 | 영향도 |
|---------|------|-------|
| D_gel | 겔 확산계수 | 최고 (1.0) |
| Lp_ve | 혈관 투과도 | 높음 (0.74) |
| P_oncotic | 종양압 | 높음 (0.74) |
| MW | 분자량 | 중간 (0.60) |
| kdecay | 분해율 | 중간 (0.50) |
| sigma_ve | 반사계수 | 낮음 (0.24) |
| K | 조직 투과도 | 최저 (0.20) |

## 설치

```bash
pip install -r requirements.txt
```

## 실행

```bash
streamlit run app.py
```

## 기술 스택

- **Backend**: Python, PyTorch
- **Frontend**: Streamlit
- **Interpolation**: Inverse Distance Weighting (IDW)
- **Visualization**: Plotly

## 모델 버전

- v2.1: MW(분자량) 파라미터 추가, 파라미터 가중치 적용
- 평균 오차율: 2.17%
