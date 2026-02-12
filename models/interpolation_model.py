"""
보간 기반 시계열 예측 모델
실제 시뮬레이션 데이터를 기반으로 파라미터 공간에서 보간
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d, RBFInterpolator
from typing import Dict, List, Tuple, Optional
from pathlib import Path


class CaseInterpolator:
    """
    케이스 기반 보간 예측기

    각 케이스의 실제 시뮬레이션 데이터를 저장하고,
    새로운 파라미터에 대해 가장 가까운 케이스들을 가중 평균
    """

    # 파라미터별 영향력 가중치 (실제 데이터 분석 기반)
    # 영향력 순위: D_gel(118%p) > Lp_ve(87.5%p) > P_oncotic(87.3%p) > sigma_ve(28.3%p) > K(24.1%p)
    # 정규화된 가중치 (최대 영향력 = 1.0 기준)
    PARAM_IMPORTANCE_WEIGHTS = {
        'Lp_ve': 0.74,      # 87.5 / 118.3 = 0.74 (2위)
        'K': 0.20,          # 24.1 / 118.3 = 0.20 (5위, 가장 둔감)
        'P_oncotic': 0.74,  # 87.3 / 118.3 = 0.74 (3위)
        'sigma_ve': 0.24,   # 28.3 / 118.3 = 0.24 (4위)
        'D_gel': 1.0,       # 118.3 / 118.3 = 1.00 (1위, 가장 민감)
        'kdecay': 0.50,     # Decay 영향 (추정치)
        'MW': 0.60,         # 분자량 영향 (sigma_ve와 연동, 추정치)
    }

    def __init__(self, use_param_weights: bool = True):
        self.cases = {}  # case_name -> {'params': np.array, 'time_series': DataFrame}
        self.param_names = ['Lp_ve', 'K', 'P_oncotic', 'sigma_ve', 'D_gel', 'kdecay', 'MW']
        self.time_max = 72.0
        self.use_param_weights = use_param_weights

        # 파라미터 순서에 맞춰 가중치 배열 생성
        self.param_weights = np.array([
            self.PARAM_IMPORTANCE_WEIGHTS.get(name, 0.5)
            for name in self.param_names
        ])

    def add_case(self, case_name: str, params_normalized: np.ndarray, time_series: pd.DataFrame):
        """케이스 데이터 추가"""
        # 시간 보간 함수 생성 (0-72시간)
        time_hours = time_series['time_hour'].values
        blood = time_series['Blood'].values
        lymph = time_series['Lymph'].values
        ecm = time_series['ECM'].values
        decay = time_series['Decay'].values if 'Decay' in time_series.columns else np.zeros_like(blood)

        self.cases[case_name] = {
            'params': params_normalized,
            'interp_blood': interp1d(time_hours, blood, kind='cubic', fill_value='extrapolate'),
            'interp_lymph': interp1d(time_hours, lymph, kind='cubic', fill_value='extrapolate'),
            'interp_ecm': interp1d(time_hours, ecm, kind='cubic', fill_value='extrapolate'),
            'interp_decay': interp1d(time_hours, decay, kind='cubic', fill_value='extrapolate'),
            'final_blood': blood[-1],
            'final_lymph': lymph[-1],
            'final_ecm': ecm[-1],
            'final_decay': decay[-1]
        }

    def _compute_weights(self, query_params: np.ndarray, k: int = 3) -> List[Tuple[str, float]]:
        """
        쿼리 파라미터에 대한 각 케이스의 가중치 계산
        거리 기반 역가중치 (Inverse Distance Weighting)

        파라미터별 영향력 가중치 적용:
        - D_gel(1.0) > Lp_ve(0.74) > P_oncotic(0.74) > sigma_ve(0.24) > K(0.20)
        - 영향력이 큰 파라미터의 차이가 거리 계산에서 더 크게 반영됨
        """
        distances = []
        for case_name, case_data in self.cases.items():
            diff = query_params - case_data['params']

            # 파라미터 차원 맞추기
            if len(diff) > len(self.param_weights):
                weights = np.concatenate([self.param_weights, np.ones(len(diff) - len(self.param_weights)) * 0.5])
            elif len(diff) < len(self.param_weights):
                weights = self.param_weights[:len(diff)]
            else:
                weights = self.param_weights

            # 가중치 적용된 거리 계산
            if self.use_param_weights:
                weighted_diff = diff * weights
                dist = np.linalg.norm(weighted_diff)
            else:
                dist = np.linalg.norm(diff)

            distances.append((case_name, dist))

        # 거리순 정렬
        distances.sort(key=lambda x: x[1])

        # 상위 k개 선택
        top_k = distances[:k]

        # 역거리 가중치
        weights = []
        total_weight = 0

        for case_name, dist in top_k:
            if dist < 1e-6:  # 거의 정확히 일치
                return [(case_name, 1.0)]
            w = 1.0 / (dist ** 2)  # 거리 제곱의 역수
            weights.append((case_name, w))
            total_weight += w

        # 정규화
        weights = [(name, w / total_weight) for name, w in weights]

        return weights

    def predict(self, params_normalized: np.ndarray, time_hours: np.ndarray) -> np.ndarray:
        """
        주어진 파라미터와 시간에 대한 Blood/Lymph/ECM/Decay 예측

        Args:
            params_normalized: (6,) 정규화된 파라미터 [Lp_ve, K, P_oncotic, sigma_ve, D_gel, kdecay]
            time_hours: (n_times,) 시간 배열 (hours)

        Returns:
            (n_times, 4) 예측 비율 [Blood, Lymph, ECM, Decay]
        """
        weights = self._compute_weights(params_normalized)

        blood = np.zeros(len(time_hours))
        lymph = np.zeros(len(time_hours))
        ecm = np.zeros(len(time_hours))
        decay = np.zeros(len(time_hours))

        for case_name, w in weights:
            case = self.cases[case_name]
            blood += w * case['interp_blood'](time_hours)
            lymph += w * case['interp_lymph'](time_hours)
            ecm += w * case['interp_ecm'](time_hours)
            decay += w * case['interp_decay'](time_hours)

        # 음수 방지 및 정규화
        blood = np.maximum(blood, 0)
        lymph = np.maximum(lymph, 0)
        ecm = np.maximum(ecm, 0)
        decay = np.maximum(decay, 0)

        total = blood + lymph + ecm + decay
        total = np.maximum(total, 1e-6)

        blood = blood / total * 100
        lymph = lymph / total * 100
        ecm = ecm / total * 100
        decay = decay / total * 100

        return np.stack([blood, lymph, ecm, decay], axis=-1)

    def predict_single(self, params_normalized: np.ndarray, time_hour: float) -> np.ndarray:
        """단일 시점 예측 - [Blood, Lymph, ECM, Decay] 반환"""
        return self.predict(params_normalized, np.array([time_hour]))[0]


class HybridModel(nn.Module):
    """
    보간 + 신경망 하이브리드 모델

    보간 결과를 기본으로 하고, 신경망으로 미세 조정
    """

    def __init__(self, interpolator: CaseInterpolator, param_dim: int = 7, hidden_dim: int = 32):
        super().__init__()
        self.interpolator = interpolator
        self.param_dim = param_dim

        # 잔차 보정 네트워크 (작은 조정만)
        self.residual_net = nn.Sequential(
            nn.Linear(param_dim + 1, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 3),
            nn.Tanh()  # -1 ~ 1 출력
        )

        # 잔차 스케일 (최대 ±5% 조정)
        self.residual_scale = 0.05

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, 1 + param_dim) - [time_normalized, params...]
        """
        batch_size = x.shape[0]
        t_norm = x[:, 0].detach().cpu().numpy()
        params = x[:, 1:].detach().cpu().numpy()

        t_hours = t_norm * 72.0

        # 보간 예측
        interp_preds = []
        for i in range(batch_size):
            pred = self.interpolator.predict_single(params[i], t_hours[i])
            interp_preds.append(pred / 100.0)  # 0-1로 정규화

        interp_preds = torch.FloatTensor(interp_preds).to(x.device)

        # 잔차 보정
        residual = self.residual_net(x) * self.residual_scale

        # 합치기
        output = interp_preds + residual

        # 범위 제한 및 정규화
        output = torch.clamp(output, 0, 1)
        output = output / (output.sum(dim=-1, keepdim=True) + 1e-8)

        return output


def create_interpolator_from_data(data_dir: str = None) -> CaseInterpolator:
    """데이터에서 보간기 생성 (Injection site + sol765 데이터)"""
    from data.timeseries_preprocessor import TimeSeriesPreprocessor

    preprocessor = TimeSeriesPreprocessor(data_dir)

    # Injection site 데이터 로드 (kdecay = 0)
    case_data = preprocessor.load_case_data()

    # sol765 Decay 데이터 로드
    decay_data = preprocessor.load_decay_data()

    interpolator = CaseInterpolator()

    # Injection site 케이스 추가
    injection_count = 0
    for case_name, data in case_data.items():
        interpolator.add_case(
            case_name=case_name,
            params_normalized=data['params_normalized'],
            time_series=data['time_series']
        )
        injection_count += 1

    # sol765 Decay 케이스 추가
    decay_count = 0
    for case_name, data in decay_data.items():
        interpolator.add_case(
            case_name=f"sol765_{case_name}",
            params_normalized=data['params_normalized'],
            time_series=data['time_series']
        )
        decay_count += 1

    print(f"보간기 생성: Injection site {injection_count}개 + sol765 {decay_count}개 = 총 {injection_count + decay_count}개 케이스")

    return interpolator


def save_interpolator(interpolator: CaseInterpolator, save_path: str):
    """보간기 저장"""
    import pickle

    # interp1d 함수는 직접 저장할 수 없으므로 원본 데이터 저장
    save_data = {
        'param_names': interpolator.param_names,
        'time_max': interpolator.time_max,
        'cases': {}
    }

    for case_name, case_data in interpolator.cases.items():
        save_data['cases'][case_name] = {
            'params': case_data['params'],
            'final_blood': case_data['final_blood'],
            'final_lymph': case_data['final_lymph'],
            'final_ecm': case_data['final_ecm'],
            'final_decay': case_data['final_decay']
        }

    with open(save_path, 'wb') as f:
        pickle.dump(save_data, f)


if __name__ == "__main__":
    print("보간 모델 테스트 (MW 파라미터 포함)...")

    # 보간기 생성
    interpolator = create_interpolator_from_data()

    times = np.array([0, 12, 24, 48, 72])

    # 파라미터 순서: [Lp_ve, K, P_oncotic, sigma_ve, D_gel, kdecay, MW]

    print("\n=== 테스트 1: Injection site Case 1 (IgG 유사, MW=150) ===")
    # Case 1: kdecay=0, MW=150 (IgG와 유사)
    params_case1 = np.array([0, 0, 0, 0, 0, -1, 1])  # MW=150 → normalized ≈ 1
    preds = interpolator.predict(params_case1, times)

    print("시간(h)  Blood   Lymph   ECM     Decay")
    for i, t in enumerate(times):
        print(f"{t:4.0f}h   {preds[i,0]:5.1f}%  {preds[i,1]:5.1f}%  {preds[i,2]:5.1f}%  {preds[i,3]:5.1f}%")

    print("\n=== 테스트 2: sol765 IgG kdecay=0 (MW=150, sigma_ve=0.9) ===")
    # IgG: MW=150 (high), sigma_ve=0.9 (high)
    params_igg = np.array([0, 0, 0, 1, 0, -1, 1])  # sigma_ve=high, MW=high
    preds_igg = interpolator.predict(params_igg, times)

    print("시간(h)  Blood   Lymph   ECM     Decay")
    for i, t in enumerate(times):
        print(f"{t:4.0f}h   {preds_igg[i,0]:5.1f}%  {preds_igg[i,1]:5.1f}%  {preds_igg[i,2]:5.1f}%  {preds_igg[i,3]:5.1f}%")

    print("\n=== 테스트 3: sol765 INS kdecay=0 (MW=5.8, sigma_ve=0.1) ===")
    # INS: MW=5.8 (low), sigma_ve=0.1 (low)
    params_ins = np.array([0, 0, 0, -1, 0, -1, -1])  # sigma_ve=low, MW=low
    preds_ins = interpolator.predict(params_ins, times)

    print("시간(h)  Blood   Lymph   ECM     Decay")
    for i, t in enumerate(times):
        print(f"{t:4.0f}h   {preds_ins[i,0]:5.1f}%  {preds_ins[i,1]:5.1f}%  {preds_ins[i,2]:5.1f}%  {preds_ins[i,3]:5.1f}%")

    print("\n=== 테스트 4: sol765 ALB kdecay=0 (MW=66.5, sigma_ve=0.7) ===")
    # ALB: MW=66.5 (mid), sigma_ve=0.7 (중상)
    params_alb = np.array([0, 0, 0, 0.5, 0, -1, 0])  # sigma_ve=중상, MW=mid
    preds_alb = interpolator.predict(params_alb, times)

    print("시간(h)  Blood   Lymph   ECM     Decay")
    for i, t in enumerate(times):
        print(f"{t:4.0f}h   {preds_alb[i,0]:5.1f}%  {preds_alb[i,1]:5.1f}%  {preds_alb[i,2]:5.1f}%  {preds_alb[i,3]:5.1f}%")

    print("\n=== 테스트 5: High kdecay + IgG ===")
    params_high_decay = np.array([0, 0, 0, 1, 0, 1, 1])  # kdecay=high, MW=high
    preds_high_decay = interpolator.predict(params_high_decay, times)

    print("시간(h)  Blood   Lymph   ECM     Decay")
    for i, t in enumerate(times):
        print(f"{t:4.0f}h   {preds_high_decay[i,0]:5.1f}%  {preds_high_decay[i,1]:5.1f}%  {preds_high_decay[i,2]:5.1f}%  {preds_high_decay[i,3]:5.1f}%")
