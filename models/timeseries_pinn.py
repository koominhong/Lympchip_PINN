"""
시계열 예측용 Physics-Informed Neural Network
시간과 파라미터를 입력받아 Blood/Lymph/ECM 비율을 예측
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import numpy as np


class FourierFeatures(nn.Module):
    """
    Fourier Feature Encoding
    시간과 파라미터를 고주파 특성으로 인코딩
    """

    def __init__(self, input_dim: int, num_frequencies: int = 32, scale: float = 10.0):
        super().__init__()
        self.num_frequencies = num_frequencies

        # 학습 가능한 주파수 행렬
        B = torch.randn(input_dim, num_frequencies) * scale
        self.register_buffer('B', B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, input_dim)
        returns: (batch, 2 * num_frequencies)
        """
        x_proj = 2 * np.pi * x @ self.B
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class TimeEncoder(nn.Module):
    """
    시간 특화 인코딩
    여러 스케일의 시간 특성 추출
    """

    def __init__(self, num_frequencies: int = 16):
        super().__init__()
        self.num_frequencies = num_frequencies

        # 다양한 스케일의 주파수 (빠른 변화 ~ 느린 변화)
        frequencies = torch.exp(torch.linspace(
            np.log(0.1),    # 저주파: 전체적인 트렌드
            np.log(100.0),  # 고주파: 빠른 변화
            num_frequencies
        ))
        self.register_buffer('frequencies', frequencies)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        t: (batch, 1) 정규화된 시간 (0-1)
        returns: (batch, 2 * num_frequencies)
        """
        t_scaled = t * self.frequencies.unsqueeze(0)  # (batch, num_freq)
        return torch.cat([
            torch.sin(2 * np.pi * t_scaled),
            torch.cos(2 * np.pi * t_scaled)
        ], dim=-1)


class ResidualMLP(nn.Module):
    """잔차 연결이 있는 MLP 블록"""

    def __init__(self, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x + self.net(x))


class TimeSeriesPINN(nn.Module):
    """
    시계열 예측 PINN

    입력: [t, Lp_ve, K, P_oncotic, sigma_ve, D_gel] (정규화됨)
    출력: [Blood, Lymph, ECM] 비율 (0-1)

    물리적 제약:
    1. Blood + Lymph + ECM = 1 (질량 보존)
    2. t=0에서 Blood≈0, Lymph≈0, ECM≈1 (초기 조건)
    3. 시간에 따른 부드러운 변화 (연속성)
    """

    def __init__(
        self,
        param_dim: int = 5,          # 파라미터 수
        hidden_dim: int = 256,        # 히든 레이어 차원
        num_layers: int = 6,          # 레이어 수
        num_time_freq: int = 32,      # 시간 인코딩 주파수
        num_param_freq: int = 32,     # 파라미터 인코딩 주파수
        dropout: float = 0.0,
        use_fourier: bool = True      # Fourier 특성 사용 여부
    ):
        super().__init__()

        self.param_dim = param_dim
        self.use_fourier = use_fourier

        # 시간 인코딩
        self.time_encoder = TimeEncoder(num_time_freq)
        time_encoded_dim = 2 * num_time_freq

        # 파라미터 인코딩
        if use_fourier:
            self.param_encoder = FourierFeatures(param_dim, num_param_freq)
            param_encoded_dim = 2 * num_param_freq
        else:
            self.param_encoder = nn.Sequential(
                nn.Linear(param_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Linear(hidden_dim // 2, hidden_dim // 2)
            )
            param_encoded_dim = hidden_dim // 2

        # 입력 결합 레이어
        total_input_dim = time_encoded_dim + param_encoded_dim
        self.input_layer = nn.Sequential(
            nn.Linear(total_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )

        # 잔차 블록
        self.residual_blocks = nn.ModuleList([
            ResidualMLP(hidden_dim, dropout)
            for _ in range(num_layers)
        ])

        # 출력 헤드 (3개 비율)
        self.output_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 3)
        )

        # 가중치 초기화
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, 1 + param_dim) - [time, params...]
        returns: (batch, 3) - [Blood, Lymph, ECM] 비율 (합=1)
        """
        # 시간과 파라미터 분리
        t = x[:, 0:1]
        params = x[:, 1:]

        # 인코딩
        t_encoded = self.time_encoder(t)
        params_encoded = self.param_encoder(params)

        # 결합
        h = torch.cat([t_encoded, params_encoded], dim=-1)
        h = self.input_layer(h)

        # 잔차 블록 통과
        for block in self.residual_blocks:
            h = block(h)

        # 출력 (softmax로 합이 1)
        logits = self.output_head(h)
        ratios = F.softmax(logits, dim=-1)

        return ratios

    def predict_with_gradients(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        예측값과 시간에 대한 기울기 반환 (물리 손실용)

        Returns:
            ratios: (batch, 3)
            d_ratios_dt: (batch, 3) 시간에 대한 각 비율의 기울기
        """
        x = x.clone().requires_grad_(True)
        ratios = self.forward(x)

        # 각 출력에 대한 기울기 계산
        gradients = []
        for i in range(3):
            grad_outputs = torch.zeros_like(ratios)
            grad_outputs[:, i] = 1.0

            grad = torch.autograd.grad(
                outputs=ratios,
                inputs=x,
                grad_outputs=grad_outputs,
                create_graph=True,
                retain_graph=True
            )[0]

            # 시간(첫 번째 열)에 대한 기울기
            gradients.append(grad[:, 0:1])

        d_ratios_dt = torch.cat(gradients, dim=-1)  # (batch, 3)

        return ratios, d_ratios_dt


class TimeSeriesPINNWithBranches(nn.Module):
    """
    브랜치 구조를 가진 PINN
    각 출력(Blood, Lymph, ECM)에 대해 별도의 서브네트워크
    """

    def __init__(
        self,
        param_dim: int = 5,
        hidden_dim: int = 128,
        num_shared_layers: int = 3,
        num_branch_layers: int = 2,
        num_time_freq: int = 24,
        dropout: float = 0.0
    ):
        super().__init__()

        self.param_dim = param_dim

        # 시간 인코딩
        self.time_encoder = TimeEncoder(num_time_freq)
        time_encoded_dim = 2 * num_time_freq

        # 파라미터 임베딩
        self.param_embed = nn.Sequential(
            nn.Linear(param_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # 공유 인코더
        self.shared_encoder = nn.Sequential(
            nn.Linear(time_encoded_dim + hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )

        self.shared_layers = nn.ModuleList([
            ResidualMLP(hidden_dim, dropout)
            for _ in range(num_shared_layers)
        ])

        # Blood 브랜치
        self.blood_branch = nn.ModuleList([
            ResidualMLP(hidden_dim, dropout)
            for _ in range(num_branch_layers)
        ])
        self.blood_head = nn.Linear(hidden_dim, 1)

        # Lymph 브랜치
        self.lymph_branch = nn.ModuleList([
            ResidualMLP(hidden_dim, dropout)
            for _ in range(num_branch_layers)
        ])
        self.lymph_head = nn.Linear(hidden_dim, 1)

        # ECM 브랜치
        self.ecm_branch = nn.ModuleList([
            ResidualMLP(hidden_dim, dropout)
            for _ in range(num_branch_layers)
        ])
        self.ecm_head = nn.Linear(hidden_dim, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, 1 + param_dim)
        returns: (batch, 3) - [Blood, Lymph, ECM]
        """
        t = x[:, 0:1]
        params = x[:, 1:]

        # 인코딩
        t_encoded = self.time_encoder(t)
        params_encoded = self.param_embed(params)

        # 공유 레이어
        h = torch.cat([t_encoded, params_encoded], dim=-1)
        h = self.shared_encoder(h)

        for layer in self.shared_layers:
            h = layer(h)

        # Blood 브랜치
        h_blood = h
        for layer in self.blood_branch:
            h_blood = layer(h_blood)
        blood = torch.sigmoid(self.blood_head(h_blood))

        # Lymph 브랜치
        h_lymph = h
        for layer in self.lymph_branch:
            h_lymph = layer(h_lymph)
        lymph = torch.sigmoid(self.lymph_head(h_lymph))

        # ECM 브랜치
        h_ecm = h
        for layer in self.ecm_branch:
            h_ecm = layer(h_ecm)
        ecm = torch.sigmoid(self.ecm_head(h_ecm))

        # 합쳐서 정규화
        raw_output = torch.cat([blood, lymph, ecm], dim=-1)
        ratios = raw_output / (raw_output.sum(dim=-1, keepdim=True) + 1e-8)

        return ratios


def create_timeseries_model(
    model_type: str = 'standard',
    param_dim: int = 5,
    **kwargs
) -> nn.Module:
    """모델 팩토리 함수"""

    if model_type == 'standard':
        return TimeSeriesPINN(param_dim=param_dim, **kwargs)
    elif model_type == 'branched':
        return TimeSeriesPINNWithBranches(param_dim=param_dim, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == "__main__":
    print("시계열 PINN 모델 테스트...")

    # 표준 모델 테스트
    model = TimeSeriesPINN(param_dim=5, hidden_dim=128, num_layers=4)
    print(f"\n모델 파라미터 수: {sum(p.numel() for p in model.parameters()):,}")

    # 입력: [time, Lp_ve, K, P_oncotic, sigma_ve, D_gel]
    x = torch.randn(32, 6)
    x[:, 0] = torch.rand(32)  # 시간 0-1

    y = model(x)
    print(f"출력 shape: {y.shape}")
    print(f"비율 합계: {y.sum(dim=-1)[:5]}")  # 모두 1이어야 함

    # 기울기 테스트
    y, dy_dt = model.predict_with_gradients(x)
    print(f"\n기울기 shape: {dy_dt.shape}")

    # 브랜치 모델 테스트
    branched_model = TimeSeriesPINNWithBranches(param_dim=5)
    y_branched = branched_model(x)
    print(f"\n브랜치 모델 출력 shape: {y_branched.shape}")
    print(f"비율 합계: {y_branched.sum(dim=-1)[:5]}")
