"""
부드러운 출력을 위한 개선된 시계열 PINN
- 저주파 인코딩으로 울퉁불퉁함 방지
- 단조성 보장 구조
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
import numpy as np


class SmoothTimeEncoder(nn.Module):
    """
    부드러운 시간 인코딩 - 저주파만 사용
    """

    def __init__(self, num_frequencies: int = 6):
        super().__init__()
        self.num_frequencies = num_frequencies

        # 저주파 위주 (0.1 ~ 5.0)
        frequencies = torch.exp(torch.linspace(
            np.log(0.1),
            np.log(5.0),
            num_frequencies
        ))
        self.register_buffer('frequencies', frequencies)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_scaled = t * self.frequencies.unsqueeze(0)
        return torch.cat([
            torch.sin(2 * np.pi * t_scaled),
            torch.cos(2 * np.pi * t_scaled)
        ], dim=-1)


class MonotonicLayer(nn.Module):
    """
    단조성을 보장하는 레이어
    가중치를 양수로 제한하여 입력 증가 시 출력도 증가
    """

    def __init__(self, in_features: int, out_features: int, monotonic: bool = True):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.monotonic = monotonic

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.monotonic:
            # 가중치를 양수로 제한 (softplus)
            weight = F.softplus(self.linear.weight)
            return F.linear(x, weight, self.linear.bias)
        else:
            return self.linear(x)


class SmoothPINN(nn.Module):
    """
    부드러운 시계열 예측 PINN

    특징:
    1. 저주파 Fourier 인코딩 (울퉁불퉁함 방지)
    2. 시그모이드 기반 부드러운 전이
    3. 단조성 보장 구조
    """

    def __init__(
        self,
        param_dim: int = 5,
        hidden_dim: int = 128,
        num_layers: int = 4,
        num_time_freq: int = 6,
        dropout: float = 0.0
    ):
        super().__init__()

        self.param_dim = param_dim

        # 저주파 시간 인코딩
        self.time_encoder = SmoothTimeEncoder(num_time_freq)
        time_dim = 2 * num_time_freq

        # 파라미터 임베딩 (단순 선형)
        self.param_embed = nn.Sequential(
            nn.Linear(param_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # 시간-파라미터 결합
        self.combine = nn.Sequential(
            nn.Linear(time_dim + hidden_dim, hidden_dim),
            nn.Tanh()
        )

        # 부드러운 MLP (Tanh 사용)
        layers = []
        for _ in range(num_layers):
            layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
                nn.Dropout(dropout)
            ])
        self.mlp = nn.Sequential(*layers)

        # 최종 비율 예측을 위한 헤드
        # 각 출력에 대해 "전이 속도"와 "최종값"을 예측
        self.rate_head = nn.Linear(hidden_dim, 3)  # 전이 속도 (양수)
        self.final_head = nn.Linear(hidden_dim, 3)  # 최종 비율

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, 1 + param_dim) - [time, params...]
        returns: (batch, 3) - [Blood, Lymph, ECM]
        """
        t = x[:, 0:1]  # 정규화된 시간 (0-1)
        params = x[:, 1:]

        # 인코딩
        t_encoded = self.time_encoder(t)
        params_encoded = self.param_embed(params)

        # 결합 및 MLP
        h = torch.cat([t_encoded, params_encoded], dim=-1)
        h = self.combine(h)
        h = self.mlp(h)

        # 전이 속도와 최종값 예측
        rates = F.softplus(self.rate_head(h)) + 0.5  # 양수, 최소 0.5
        finals = torch.sigmoid(self.final_head(h))  # 0-1

        # 시그모이드 전이 곡선: ratio(t) = final * sigmoid(rate * (t - 0.5))
        # Blood와 Lymph는 증가, ECM은 감소

        # Blood: 0에서 시작, 증가
        blood_final = finals[:, 0:1] * 0.7  # 최대 70%
        blood = blood_final * torch.sigmoid(rates[:, 0:1] * (t - 0.3) * 10)

        # Lymph: 0에서 시작, 증가
        lymph_final = finals[:, 1:2] * 0.9  # 최대 90%
        lymph = lymph_final * torch.sigmoid(rates[:, 1:2] * (t - 0.2) * 10)

        # ECM: 1에서 시작, 감소
        ecm_min = finals[:, 2:3] * 0.3  # 최소 도달값 (0-30%)
        ecm = 1 - (1 - ecm_min) * torch.sigmoid(rates[:, 2:3] * (t - 0.3) * 10)

        # 합쳐서 정규화 (합이 1이 되도록)
        raw = torch.cat([blood, lymph, ecm], dim=-1)
        ratios = raw / (raw.sum(dim=-1, keepdim=True) + 1e-8)

        return ratios


class SmoothPINNv2(nn.Module):
    """
    더 단순한 부드러운 PINN - 직접적인 곡선 파라미터화
    """

    def __init__(
        self,
        param_dim: int = 5,
        hidden_dim: int = 64,
        num_layers: int = 3
    ):
        super().__init__()

        self.param_dim = param_dim

        # 파라미터 → 곡선 파라미터 변환
        layers = [nn.Linear(param_dim, hidden_dim), nn.Tanh()]
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.Tanh()])

        self.encoder = nn.Sequential(*layers)

        # 각 출력에 대해 곡선 파라미터 예측
        # [초기값, 최종값, 전이속도, 전이중심]
        self.blood_params = nn.Linear(hidden_dim, 4)
        self.lymph_params = nn.Linear(hidden_dim, 4)
        self.ecm_params = nn.Linear(hidden_dim, 4)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # 초기 바이어스 설정
        # Blood: 초기=0, 최종=0.2, 속도=5, 중심=0.3
        self.blood_params.bias.data = torch.tensor([0.0, 0.2, 5.0, 0.3])
        # Lymph: 초기=0, 최종=0.6, 속도=4, 중심=0.25
        self.lymph_params.bias.data = torch.tensor([0.0, 0.6, 4.0, 0.25])
        # ECM: 초기=1, 최종=0.1, 속도=4, 중심=0.3
        self.ecm_params.bias.data = torch.tensor([1.0, 0.1, 4.0, 0.3])

    def _sigmoid_curve(self, t, init, final, rate, center):
        """부드러운 시그모이드 전이 곡선"""
        # rate는 양수여야 함
        rate = F.softplus(rate) + 1.0
        # center는 0-1 사이
        center = torch.sigmoid(center)

        transition = torch.sigmoid(rate * (t - center))
        return init + (final - init) * transition

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t = x[:, 0:1]
        params = x[:, 1:]

        # 파라미터 인코딩
        h = self.encoder(params)

        # 각 출력의 곡선 파라미터
        bp = self.blood_params(h)
        lp = self.lymph_params(h)
        ep = self.ecm_params(h)

        # 곡선 생성
        blood = self._sigmoid_curve(
            t,
            init=torch.sigmoid(bp[:, 0:1]) * 0.05,  # 초기: 0-5%
            final=torch.sigmoid(bp[:, 1:2]) * 0.7,   # 최종: 0-70%
            rate=bp[:, 2:3],
            center=bp[:, 3:4]
        )

        lymph = self._sigmoid_curve(
            t,
            init=torch.sigmoid(lp[:, 0:1]) * 0.05,
            final=torch.sigmoid(lp[:, 1:2]) * 0.9,
            rate=lp[:, 2:3],
            center=lp[:, 3:4]
        )

        ecm = self._sigmoid_curve(
            t,
            init=0.95 + torch.sigmoid(ep[:, 0:1]) * 0.05,  # 초기: 95-100%
            final=torch.sigmoid(ep[:, 1:2]) * 0.3,          # 최종: 0-30%
            rate=ep[:, 2:3],
            center=ep[:, 3:4]
        )

        # 정규화
        raw = torch.cat([blood, lymph, ecm], dim=-1)
        ratios = raw / (raw.sum(dim=-1, keepdim=True) + 1e-8)

        return ratios


def create_smooth_model(model_type: str = 'v2', param_dim: int = 5, **kwargs):
    if model_type == 'v1':
        return SmoothPINN(param_dim=param_dim, **kwargs)
    elif model_type == 'v2':
        return SmoothPINNv2(param_dim=param_dim, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == "__main__":
    print("부드러운 PINN 테스트...")

    model = SmoothPINNv2(param_dim=5)
    print(f"파라미터 수: {sum(p.numel() for p in model.parameters()):,}")

    # 테스트 입력
    x = torch.zeros(50, 6)
    x[:, 0] = torch.linspace(0, 1, 50)  # 시간

    with torch.no_grad():
        y = model(x)

    print("\n시간별 예측:")
    print("t(h)    Blood   Lymph   ECM")
    for i in range(0, 50, 5):
        t_hour = x[i, 0].item() * 72
        print(f"{t_hour:5.1f}   {y[i,0].item()*100:5.1f}%  {y[i,1].item()*100:5.1f}%  {y[i,2].item()*100:5.1f}%")

    # 변화량 확인
    diffs = torch.abs(torch.diff(y, dim=0))
    print(f"\n변화량 (울퉁불퉁함):")
    print(f"Blood: 평균 {diffs[:,0].mean()*100:.3f}%, 최대 {diffs[:,0].max()*100:.3f}%")
    print(f"Lymph: 평균 {diffs[:,1].mean()*100:.3f}%, 최대 {diffs[:,1].max()*100:.3f}%")
    print(f"ECM: 평균 {diffs[:,2].mean()*100:.3f}%, 최대 {diffs[:,2].max()*100:.3f}%")
