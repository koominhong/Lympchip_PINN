"""
물리 기반 손실 함수
PINN 학습을 위한 다양한 손실 함수 정의
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import numpy as np


class PhysicsLoss(nn.Module):
    """
    물리 기반 손실 함수들의 기본 클래스

    물리적 제약:
    1. 질량 보존: 총 물질량은 보존되어야 함
    2. 단조 감소: 농도는 시간에 따라 감소 (decay가 있는 경우)
    3. 경계 조건: 초기 농도 C(0) = C0, 최종 농도 C(∞) → 0
    4. Soft PDE: 근사적인 수송 방정식 만족
    """

    def __init__(
        self,
        lambda_data: float = 1.0,
        lambda_physics: float = 0.1,
        lambda_conservation: float = 0.1,
        lambda_monotonic: float = 0.05,
        lambda_boundary: float = 0.1,
        lambda_smoothness: float = 0.01
    ):
        super().__init__()

        # 손실 가중치
        self.lambda_data = lambda_data
        self.lambda_physics = lambda_physics
        self.lambda_conservation = lambda_conservation
        self.lambda_monotonic = lambda_monotonic
        self.lambda_boundary = lambda_boundary
        self.lambda_smoothness = lambda_smoothness

    def data_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """데이터 손실 (MSE)"""
        return F.mse_loss(pred, target)

    def conservation_loss(
        self,
        ratios: torch.Tensor
    ) -> torch.Tensor:
        """
        질량 보존 손실: Blood + Lymph + ECM 비율의 합 = 1

        ratios: (batch, 3) - [blood, lymph, ecm] 비율
        """
        ratio_sum = ratios.sum(dim=-1)
        return F.mse_loss(ratio_sum, torch.ones_like(ratio_sum))

    def monotonic_loss(
        self,
        concentration: torch.Tensor,
        time: torch.Tensor,
        k_decay: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        단조 감소 손실: decay가 있으면 농도는 시간에 따라 감소해야 함

        concentration: (batch, 1)
        time: (batch, 1)
        k_decay: (batch, 1) or scalar
        """
        if k_decay is None or (isinstance(k_decay, torch.Tensor) and k_decay.max() == 0):
            return torch.tensor(0.0, device=concentration.device)

        # 시간 순서로 정렬
        sorted_idx = torch.argsort(time.squeeze())
        sorted_conc = concentration[sorted_idx]

        # 농도 증가분 (양수면 위반)
        diff = sorted_conc[1:] - sorted_conc[:-1]
        violations = F.relu(diff)  # 증가하는 경우만 페널티

        return violations.mean()

    def boundary_loss(
        self,
        model: nn.Module,
        x_initial: torch.Tensor,
        c0: float = 1.0
    ) -> torch.Tensor:
        """
        경계 조건 손실: C(t=0) = C0

        x_initial: 초기 시간(t=0)에서의 입력
        c0: 초기 농도 (정규화된 값)
        """
        pred_initial = model(x_initial)
        target = torch.full_like(pred_initial, c0)
        return F.mse_loss(pred_initial, target)

    def smoothness_loss(
        self,
        dC_dt: torch.Tensor
    ) -> torch.Tensor:
        """
        평활도 손실: dC/dt가 급격히 변하지 않도록

        dC_dt: 시간에 대한 농도 기울기
        """
        return (dC_dt ** 2).mean()

    def soft_pde_loss(
        self,
        dC_dt: torch.Tensor,
        concentration: torch.Tensor,
        k_decay: torch.Tensor,
        flux_term: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Soft PDE 손실: dC/dt + k_decay * C ≈ -flux

        단순화된 1차 decay 방정식: dC/dt = -k_decay * C
        """
        # 기본 decay 방정식
        residual = dC_dt + k_decay * concentration

        if flux_term is not None:
            residual = residual - flux_term

        return (residual ** 2).mean()


class LymphChipLoss(PhysicsLoss):
    """림프칩 시뮬레이션 특화 손실 함수"""

    def __init__(
        self,
        lambda_data: float = 1.0,
        lambda_physics: float = 0.1,
        lambda_conservation: float = 0.5,
        lambda_monotonic: float = 0.05,
        lambda_boundary: float = 0.1,
        lambda_smoothness: float = 0.01,
        lambda_ratio_constraint: float = 0.2
    ):
        super().__init__(
            lambda_data=lambda_data,
            lambda_physics=lambda_physics,
            lambda_conservation=lambda_conservation,
            lambda_monotonic=lambda_monotonic,
            lambda_boundary=lambda_boundary,
            lambda_smoothness=lambda_smoothness
        )
        self.lambda_ratio_constraint = lambda_ratio_constraint

    def ratio_constraint_loss(
        self,
        ratios: torch.Tensor
    ) -> torch.Tensor:
        """
        비율 제약 손실:
        - 모든 비율은 0 이상
        - 모든 비율은 1 이하
        - 합은 정확히 1
        """
        # 음수 비율 페널티
        negative_penalty = F.relu(-ratios).mean()

        # 1 초과 페널티
        excess_penalty = F.relu(ratios - 1).mean()

        # 합이 1이 되도록
        sum_penalty = self.conservation_loss(ratios)

        return negative_penalty + excess_penalty + sum_penalty

    def lymph_physics_loss(
        self,
        ratios: torch.Tensor,
        params: torch.Tensor,
        param_names: list
    ) -> torch.Tensor:
        """
        림프칩 물리학 기반 손실:

        알려진 관계:
        - high Lp → high lymph ratio
        - high pBV → high blood ratio
        - high oncotic pressure → more lymph retention
        """
        loss = torch.tensor(0.0, device=ratios.device)

        blood_ratio = ratios[:, 0]
        lymph_ratio = ratios[:, 1]

        # 파라미터 인덱스 찾기
        param_dict = {name: i for i, name in enumerate(param_names)}

        # Lp와 lymph 관계 (양의 상관관계)
        if 'Lp' in param_dict:
            Lp = params[:, param_dict['Lp']]
            # Lp가 높으면 lymph_ratio도 높아야 함
            correlation = -torch.mean(Lp * lymph_ratio)  # 최대화하려면 음수
            loss = loss + 0.1 * F.relu(-correlation)  # correlation이 음수면 페널티

        # pBV와 blood 관계 (양의 상관관계)
        if 'pBV' in param_dict:
            pBV = params[:, param_dict['pBV']]
            correlation = -torch.mean(pBV * blood_ratio)
            loss = loss + 0.1 * F.relu(-correlation)

        return loss

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        model: Optional[nn.Module] = None,
        x: Optional[torch.Tensor] = None,
        task_type: str = 'concentration',
        params: Optional[torch.Tensor] = None,
        param_names: Optional[list] = None,
        k_decay: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        전체 손실 계산

        task_type: 'concentration' or 'ratios'
        """
        losses = {}

        # 데이터 손실
        losses['data'] = self.lambda_data * self.data_loss(pred, target)

        if task_type == 'ratios':
            # 비율 작업의 손실
            losses['conservation'] = self.lambda_conservation * self.conservation_loss(pred)
            losses['ratio_constraint'] = self.lambda_ratio_constraint * self.ratio_constraint_loss(pred)

            if params is not None and param_names is not None:
                losses['physics'] = self.lambda_physics * self.lymph_physics_loss(
                    pred, params, param_names
                )

        elif task_type == 'concentration' and model is not None and x is not None:
            # 농도 작업의 손실

            # 기울기 기반 손실
            x_grad = x.clone().requires_grad_(True)
            pred_grad = model(x_grad)

            grad_outputs = torch.ones_like(pred_grad)
            gradients = torch.autograd.grad(
                outputs=pred_grad,
                inputs=x_grad,
                grad_outputs=grad_outputs,
                create_graph=True,
                retain_graph=True
            )[0]

            dC_dt = gradients[:, 0:1]

            # 평활도 손실
            losses['smoothness'] = self.lambda_smoothness * self.smoothness_loss(dC_dt)

            # Soft PDE 손실 (k_decay가 주어진 경우)
            if k_decay is not None:
                if not isinstance(k_decay, torch.Tensor):
                    k_decay = torch.tensor(k_decay, device=pred.device)
                if k_decay.dim() == 0:
                    k_decay = k_decay.expand(pred.shape[0], 1)

                losses['physics'] = self.lambda_physics * self.soft_pde_loss(
                    dC_dt, pred_grad, k_decay
                )

        # 총 손실
        losses['total'] = sum(losses.values())

        return losses


class AdaptiveLossWeights(nn.Module):
    """
    학습 가능한 손실 가중치 (Uncertainty Weighting)

    Kendall et al. "Multi-Task Learning Using Uncertainty to Weigh Losses"
    """

    def __init__(self, num_losses: int):
        super().__init__()
        # log(σ²) 학습
        self.log_vars = nn.Parameter(torch.zeros(num_losses))

    def forward(self, losses: list) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        losses: 개별 손실값들의 리스트

        returns: (total_loss, {각 손실의 가중치})
        """
        total = 0
        weights = {}

        for i, loss in enumerate(losses):
            precision = torch.exp(-self.log_vars[i])
            total = total + precision * loss + self.log_vars[i]
            weights[f'weight_{i}'] = precision.item()

        return total, weights


class FocalLoss(nn.Module):
    """
    Focal Loss for handling imbalanced data

    어려운 샘플에 더 집중
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mse = (pred - target) ** 2
        # 오차가 큰 샘플에 더 가중치
        focal_weight = (mse.detach() ** self.gamma)
        return (focal_weight * mse).mean()


def compute_relative_l2_error(
    pred: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-8
) -> torch.Tensor:
    """상대 L2 오차 계산"""
    numerator = torch.norm(pred - target, p=2)
    denominator = torch.norm(target, p=2) + eps
    return numerator / denominator


def compute_metrics(
    pred: torch.Tensor,
    target: torch.Tensor
) -> Dict[str, float]:
    """평가 지표 계산"""
    with torch.no_grad():
        mse = F.mse_loss(pred, target).item()
        mae = F.l1_loss(pred, target).item()
        rel_l2 = compute_relative_l2_error(pred, target).item()

        # R² score
        ss_res = ((target - pred) ** 2).sum()
        ss_tot = ((target - target.mean()) ** 2).sum()
        r2 = (1 - ss_res / (ss_tot + 1e-8)).item()

    return {
        'mse': mse,
        'mae': mae,
        'relative_l2': rel_l2,
        'r2': r2
    }


if __name__ == "__main__":
    # 손실 함수 테스트
    print("손실 함수 테스트...")

    loss_fn = LymphChipLoss()

    # 비율 예측 테스트
    pred_ratios = torch.softmax(torch.randn(8, 3), dim=-1)
    target_ratios = torch.tensor([
        [0.18, 0.48, 0.34],
        [0.52, 0.30, 0.18],
        [0.08, 0.69, 0.22],
        [0.27, 0.29, 0.43],
        [0.23, 0.38, 0.39],
        [0.18, 0.48, 0.34],
        [0.13, 0.51, 0.36],
        [0.24, 0.45, 0.31]
    ])

    losses = loss_fn(
        pred=pred_ratios,
        target=target_ratios,
        task_type='ratios'
    )

    print("\n비율 예측 손실:")
    for name, value in losses.items():
        print(f"  {name}: {value.item():.6f}")

    # 지표 테스트
    metrics = compute_metrics(pred_ratios, target_ratios)
    print("\n평가 지표:")
    for name, value in metrics.items():
        print(f"  {name}: {value:.6f}")
