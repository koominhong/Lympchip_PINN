"""
Physics-Informed Neural Network 모델
림프칩 시뮬레이션을 위한 PINN 구현
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import numpy as np


class SinusoidalPositionalEncoding(nn.Module):
    """시간 변수를 위한 위치 인코딩 (Fourier Features)"""

    def __init__(self, num_frequencies: int = 10):
        super().__init__()
        self.num_frequencies = num_frequencies
        # 주파수 스케일
        self.frequencies = nn.Parameter(
            torch.linspace(0, np.log(1000), num_frequencies).exp(),
            requires_grad=False
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        t: (batch, 1) 시간 값
        returns: (batch, 2 * num_frequencies) 인코딩된 시간
        """
        t = t.unsqueeze(-1) if t.dim() == 1 else t
        # sin과 cos 특성
        encoded = torch.cat([
            torch.sin(t * self.frequencies),
            torch.cos(t * self.frequencies)
        ], dim=-1)
        return encoded


class ResidualBlock(nn.Module):
    """잔차 연결이 있는 MLP 블록"""

    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.SiLU()  # Swish activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm1(x)
        x = self.activation(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x + residual


class LymphChipPINN(nn.Module):
    """
    림프칩 시뮬레이션을 위한 Physics-Informed Neural Network

    입력: [t, parameters...]
    출력: [concentration] 또는 [blood_ratio, lymph_ratio, ecm_ratio]
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 6,
        output_dim: int = 1,
        use_time_encoding: bool = True,
        num_time_frequencies: int = 10,
        dropout: float = 0.1,
        output_type: str = 'concentration'  # 'concentration' or 'ratios'
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.output_type = output_type
        self.use_time_encoding = use_time_encoding

        # 시간 인코딩
        if use_time_encoding:
            self.time_encoder = SinusoidalPositionalEncoding(num_time_frequencies)
            encoded_time_dim = 2 * num_time_frequencies
            actual_input_dim = input_dim - 1 + encoded_time_dim  # 시간 1개를 인코딩으로 대체
        else:
            actual_input_dim = input_dim

        # 입력 레이어
        self.input_layer = nn.Sequential(
            nn.Linear(actual_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU()
        )

        # 잔차 블록들
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, dropout)
            for _ in range(num_layers)
        ])

        # 출력 레이어
        if output_type == 'ratios':
            # 비율 출력: softmax로 합이 1이 되도록
            self.output_layer = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, output_dim)
            )
        else:
            # 농도 출력: 양수 보장
            self.output_layer = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, output_dim),
                nn.Softplus()  # 양수 출력 보장
            )

        # 가중치 초기화
        self._init_weights()

    def _init_weights(self):
        """Xavier 초기화"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, input_dim) - 첫 번째 열이 시간 t
        returns: (batch, output_dim)
        """
        if self.use_time_encoding:
            # 시간과 나머지 특성 분리
            t = x[:, 0:1]
            other_features = x[:, 1:]

            # 시간 인코딩
            t_encoded = self.time_encoder(t)

            # 결합
            x = torch.cat([t_encoded, other_features], dim=-1)

        # 네트워크 통과
        x = self.input_layer(x)

        for block in self.residual_blocks:
            x = block(x)

        output = self.output_layer(x)

        # 비율 출력인 경우 softmax 적용
        if self.output_type == 'ratios':
            output = F.softmax(output, dim=-1)

        return output

    def predict_with_gradients(
        self,
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        출력값과 시간에 대한 기울기 반환 (물리 손실 계산용)

        returns: (output, dC/dt)
        """
        x = x.clone().requires_grad_(True)
        output = self.forward(x)

        # 시간에 대한 기울기 계산
        grad_outputs = torch.ones_like(output)
        gradients = torch.autograd.grad(
            outputs=output,
            inputs=x,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True
        )[0]

        # 첫 번째 열(시간)에 대한 기울기
        dC_dt = gradients[:, 0:1]

        return output, dC_dt


class MultiTaskPINN(nn.Module):
    """
    다중 작업 PINN: 시계열 농도 + 최종 비율 동시 예측

    두 개의 서브네트워크를 공유 인코더와 함께 사용
    """

    def __init__(
        self,
        ts_input_dim: int,      # 시계열 입력 차원
        ratio_input_dim: int,    # 비율 입력 차원
        hidden_dim: int = 256,
        num_shared_layers: int = 3,
        num_task_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()

        self.ts_input_dim = ts_input_dim
        self.ratio_input_dim = ratio_input_dim

        # 시계열 브랜치
        self.ts_encoder = nn.Sequential(
            nn.Linear(ts_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU()
        )

        # 비율 브랜치
        self.ratio_encoder = nn.Sequential(
            nn.Linear(ratio_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU()
        )

        # 공유 레이어
        self.shared_layers = nn.ModuleList([
            ResidualBlock(hidden_dim, dropout)
            for _ in range(num_shared_layers)
        ])

        # 시계열 전용 레이어
        self.ts_layers = nn.ModuleList([
            ResidualBlock(hidden_dim, dropout)
            for _ in range(num_task_layers)
        ])

        # 비율 전용 레이어
        self.ratio_layers = nn.ModuleList([
            ResidualBlock(hidden_dim, dropout)
            for _ in range(num_task_layers)
        ])

        # 출력 헤드
        self.ts_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1),
            nn.Softplus()
        )

        self.ratio_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 3)  # blood, lymph, ecm
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward_ts(self, x: torch.Tensor) -> torch.Tensor:
        """시계열 농도 예측"""
        h = self.ts_encoder(x)

        for layer in self.shared_layers:
            h = layer(h)

        for layer in self.ts_layers:
            h = layer(h)

        return self.ts_head(h)

    def forward_ratios(self, x: torch.Tensor) -> torch.Tensor:
        """최종 비율 예측"""
        h = self.ratio_encoder(x)

        for layer in self.shared_layers:
            h = layer(h)

        for layer in self.ratio_layers:
            h = layer(h)

        logits = self.ratio_head(h)
        return F.softmax(logits, dim=-1)

    def forward(
        self,
        x_ts: Optional[torch.Tensor] = None,
        x_ratio: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        두 작업 모두 또는 하나만 수행

        returns: {'concentration': ..., 'ratios': ...}
        """
        result = {}

        if x_ts is not None:
            result['concentration'] = self.forward_ts(x_ts)

        if x_ratio is not None:
            result['ratios'] = self.forward_ratios(x_ratio)

        return result


class DeepONet(nn.Module):
    """
    DeepONet: 연산자 학습을 위한 네트워크

    파라미터 함수 -> 농도 프로파일 전체를 학습
    """

    def __init__(
        self,
        branch_input_dim: int,   # 파라미터 차원
        trunk_input_dim: int = 1,  # 시간 (1차원)
        hidden_dim: int = 128,
        num_branch_layers: int = 4,
        num_trunk_layers: int = 4,
        output_dim: int = 1
    ):
        super().__init__()

        # Branch network: 파라미터 인코딩
        branch_layers = [nn.Linear(branch_input_dim, hidden_dim), nn.Tanh()]
        for _ in range(num_branch_layers - 1):
            branch_layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.Tanh()])
        self.branch_net = nn.Sequential(*branch_layers)

        # Trunk network: 시간/위치 인코딩
        trunk_layers = [nn.Linear(trunk_input_dim, hidden_dim), nn.Tanh()]
        for _ in range(num_trunk_layers - 1):
            trunk_layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.Tanh()])
        self.trunk_net = nn.Sequential(*trunk_layers)

        # 출력 bias
        self.bias = nn.Parameter(torch.zeros(output_dim))

    def forward(
        self,
        params: torch.Tensor,
        t: torch.Tensor
    ) -> torch.Tensor:
        """
        params: (batch, branch_input_dim) - 시뮬레이션 파라미터
        t: (batch, trunk_input_dim) - 시간 포인트

        returns: (batch, 1) - 농도
        """
        branch_out = self.branch_net(params)  # (batch, hidden_dim)
        trunk_out = self.trunk_net(t)          # (batch, hidden_dim)

        # 내적 + bias
        output = torch.sum(branch_out * trunk_out, dim=-1, keepdim=True) + self.bias

        return F.softplus(output)  # 양수 보장


def create_model(
    model_type: str,
    input_dim: int,
    **kwargs
) -> nn.Module:
    """모델 팩토리 함수"""

    if model_type == 'pinn':
        return LymphChipPINN(input_dim=input_dim, **kwargs)
    elif model_type == 'multitask':
        return MultiTaskPINN(
            ts_input_dim=kwargs.get('ts_input_dim', input_dim),
            ratio_input_dim=kwargs.get('ratio_input_dim', 6),
            **{k: v for k, v in kwargs.items() if k not in ['ts_input_dim', 'ratio_input_dim']}
        )
    elif model_type == 'deeponet':
        return DeepONet(
            branch_input_dim=kwargs.get('branch_input_dim', input_dim - 1),
            **{k: v for k, v in kwargs.items() if k != 'branch_input_dim'}
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == "__main__":
    # 모델 테스트
    print("PINN 모델 테스트...")

    # 기본 PINN
    model = LymphChipPINN(input_dim=10, hidden_dim=128, num_layers=4)
    x = torch.randn(32, 10)
    y = model(x)
    print(f"PINN output shape: {y.shape}")

    # 기울기 테스트
    y, dC_dt = model.predict_with_gradients(x)
    print(f"dC/dt shape: {dC_dt.shape}")

    # 다중작업 PINN
    multitask = MultiTaskPINN(ts_input_dim=10, ratio_input_dim=6)
    x_ts = torch.randn(32, 10)
    x_ratio = torch.randn(8, 6)
    outputs = multitask(x_ts=x_ts, x_ratio=x_ratio)
    print(f"MultiTask concentration: {outputs['concentration'].shape}")
    print(f"MultiTask ratios: {outputs['ratios'].shape}")
    print(f"Ratios sum: {outputs['ratios'].sum(dim=-1)}")  # Should be 1

    # DeepONet
    deeponet = DeepONet(branch_input_dim=9)
    params = torch.randn(32, 9)
    t = torch.randn(32, 1)
    out = deeponet(params, t)
    print(f"DeepONet output: {out.shape}")
