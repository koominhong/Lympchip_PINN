"""
시계열 PINN을 위한 물리 기반 손실 함수
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import numpy as np


class TimeSeriesPhysicsLoss(nn.Module):
    """
    시계열 PINN 학습을 위한 물리 기반 손실 함수

    물리적 제약:
    1. 질량 보존: Blood + Lymph + ECM = 1
    2. 초기 조건: t=0에서 ECM≈1, Blood≈0, Lymph≈0
    3. 단조성: Blood/Lymph는 증가, ECM은 감소 경향
    4. 평활성: 시간에 따른 급격한 변화 억제
    5. 파라미터-출력 관계: 물리적 인과관계 반영
    """

    def __init__(
        self,
        # 손실 가중치
        lambda_data: float = 1.0,
        lambda_conservation: float = 0.5,
        lambda_initial: float = 0.5,
        lambda_monotonic: float = 0.1,
        lambda_smoothness: float = 0.05,
        lambda_physics: float = 0.1,
        lambda_boundary: float = 0.2,
        # 초기 조건 값
        initial_ecm: float = 0.998,
        initial_blood: float = 0.0,
        initial_lymph: float = 0.002,
    ):
        super().__init__()

        # 가중치
        self.lambda_data = lambda_data
        self.lambda_conservation = lambda_conservation
        self.lambda_initial = lambda_initial
        self.lambda_monotonic = lambda_monotonic
        self.lambda_smoothness = lambda_smoothness
        self.lambda_physics = lambda_physics
        self.lambda_boundary = lambda_boundary

        # 초기 조건
        self.initial_ecm = initial_ecm
        self.initial_blood = initial_blood
        self.initial_lymph = initial_lymph

    def data_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        데이터 손실 (MSE)

        Args:
            pred: (batch, 3) 예측 비율
            target: (batch, 3) 실제 비율
            weights: (batch,) 샘플별 가중치 (선택적)
        """
        mse = (pred - target) ** 2

        if weights is not None:
            mse = mse * weights.unsqueeze(-1)

        return mse.mean()

    def conservation_loss(self, ratios: torch.Tensor) -> torch.Tensor:
        """
        질량 보존 손실: Blood + Lymph + ECM = 1

        Args:
            ratios: (batch, 3) [Blood, Lymph, ECM]
        """
        ratio_sum = ratios.sum(dim=-1)
        return F.mse_loss(ratio_sum, torch.ones_like(ratio_sum))

    def initial_condition_loss(
        self,
        model: nn.Module,
        params: torch.Tensor,
        device: torch.device
    ) -> torch.Tensor:
        """
        초기 조건 손실: t=0에서 ECM≈1, Blood≈0, Lymph≈0

        Args:
            model: PINN 모델
            params: (batch, param_dim) 파라미터
            device: 디바이스
        """
        batch_size = params.shape[0]

        # t=0 입력 생성
        t_zero = torch.zeros(batch_size, 1, device=device)
        x_initial = torch.cat([t_zero, params], dim=-1)

        # 예측
        pred_initial = model(x_initial)

        # 초기 조건 타겟
        target_initial = torch.tensor([
            [self.initial_blood, self.initial_lymph, self.initial_ecm]
        ], device=device).expand(batch_size, -1)

        return F.mse_loss(pred_initial, target_initial)

    def boundary_condition_loss(
        self,
        model: nn.Module,
        params: torch.Tensor,
        device: torch.device
    ) -> torch.Tensor:
        """
        경계 조건 손실: t=1(72시간)에서의 조건

        Args:
            model: PINN 모델
            params: (batch, param_dim) 파라미터
        """
        batch_size = params.shape[0]

        # t=1 입력 생성
        t_final = torch.ones(batch_size, 1, device=device)
        x_final = torch.cat([t_final, params], dim=-1)

        # 예측
        pred_final = model(x_final)

        # 경계 조건:
        # 1. ECM은 0 이상
        # 2. Blood + Lymph는 일정 비율 이상
        ecm = pred_final[:, 2]
        circulation = pred_final[:, 0] + pred_final[:, 1]

        # ECM이 음수가 되면 페널티
        ecm_penalty = F.relu(-ecm).mean()

        # 순환계 비율이 너무 낮으면 페널티 (최소 50% 이상)
        # circulation_penalty = F.relu(0.5 - circulation).mean()

        return ecm_penalty  # + circulation_penalty

    def monotonicity_loss(
        self,
        model: nn.Module,
        x: torch.Tensor,
        dt: float = 0.01
    ) -> torch.Tensor:
        """
        단조성 손실: Blood/Lymph는 증가, ECM은 감소 경향

        작은 시간 간격에서 변화 방향 확인
        """
        t = x[:, 0:1]
        params = x[:, 1:]

        # 현재 시점과 약간 뒤 시점
        x_now = x
        x_later = torch.cat([t + dt, params], dim=-1)

        # 예측
        pred_now = model(x_now)
        pred_later = model(x_later)

        # 변화량
        delta = pred_later - pred_now

        # Blood (index 0): 증가해야 함 → delta > 0
        # Lymph (index 1): 증가해야 함 → delta > 0
        # ECM (index 2): 감소해야 함 → delta < 0

        # 초기(t < 0.1)와 후기(t > 0.8)에서는 완화
        weight = torch.where(
            (t > 0.05) & (t < 0.9),
            torch.ones_like(t),
            torch.zeros_like(t)
        )

        blood_violation = F.relu(-delta[:, 0:1]) * weight  # Blood 감소 시 페널티
        lymph_violation = F.relu(-delta[:, 1:2]) * weight  # Lymph 감소 시 페널티
        ecm_violation = F.relu(delta[:, 2:3]) * weight      # ECM 증가 시 페널티

        return (blood_violation + lymph_violation + ecm_violation).mean()

    def smoothness_loss(
        self,
        d_ratios_dt: torch.Tensor
    ) -> torch.Tensor:
        """
        평활성 손실: 시간에 따른 변화가 부드러워야 함

        Args:
            d_ratios_dt: (batch, 3) 시간에 대한 기울기
        """
        # 기울기의 크기 제한
        return (d_ratios_dt ** 2).mean()

    def physics_correlation_loss(
        self,
        ratios: torch.Tensor,
        params: torch.Tensor
    ) -> torch.Tensor:
        """
        물리적 상관관계 손실

        알려진 관계:
        - Lp ↑ → Lymph ↑
        - K ↑ → 전체 이동 증가
        - sigma ↑ → 막 투과 감소
        - P_oncotic ↑ → 혈관 내 유지

        Args:
            ratios: (batch, 3) [Blood, Lymph, ECM]
            params: (batch, 5) [Lp_ve, K, P_oncotic, sigma_ve, D_gel]
        """
        blood = ratios[:, 0]
        lymph = ratios[:, 1]
        ecm = ratios[:, 2]

        # 파라미터 추출 (정규화된 값, -1 ~ 1)
        Lp = params[:, 0]      # Lp_ve
        K = params[:, 1]       # K
        P_onc = params[:, 2]   # P_oncotic
        sigma = params[:, 3]   # sigma_ve
        D = params[:, 4]       # D_gel

        loss = torch.tensor(0.0, device=ratios.device)

        # Lp가 높으면 Lymph가 높아야 함
        # 상관계수 기반 손실
        Lp_lymph_corr = torch.corrcoef(torch.stack([Lp, lymph]))[0, 1]
        if not torch.isnan(Lp_lymph_corr):
            # 양의 상관관계 유도
            loss = loss + F.relu(-Lp_lymph_corr)

        # sigma가 높으면 막 투과가 어려워 ECM에 더 남음
        sigma_ecm_corr = torch.corrcoef(torch.stack([sigma, ecm]))[0, 1]
        if not torch.isnan(sigma_ecm_corr):
            loss = loss + F.relu(-sigma_ecm_corr)

        return loss

    def collocation_loss(
        self,
        model: nn.Module,
        collocation_time: torch.Tensor,
        collocation_params: torch.Tensor
    ) -> torch.Tensor:
        """
        콜로케이션 포인트에서의 물리 법칙 만족도

        Args:
            model: PINN 모델
            collocation_time: (n_points, 1) 콜로케이션 시간
            collocation_params: (n_points, param_dim) 콜로케이션 파라미터
        """
        x_colloc = torch.cat([collocation_time, collocation_params], dim=-1)

        # 예측 및 기울기 계산
        if hasattr(model, 'predict_with_gradients'):
            ratios, d_ratios_dt = model.predict_with_gradients(x_colloc)
        else:
            x_colloc.requires_grad_(True)
            ratios = model(x_colloc)
            # 수동 기울기 계산
            d_ratios_dt = torch.autograd.grad(
                ratios.sum(), x_colloc, create_graph=True
            )[0][:, 0:1].expand(-1, 3)

        # 물리 법칙:
        # 1. 질량 보존
        conservation = self.conservation_loss(ratios)

        # 2. 평활성
        smoothness = self.smoothness_loss(d_ratios_dt)

        # 3. 비율 범위 (0-1)
        range_loss = F.relu(-ratios).mean() + F.relu(ratios - 1).mean()

        return conservation + 0.1 * smoothness + range_loss

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        model: Optional[nn.Module] = None,
        x: Optional[torch.Tensor] = None,
        d_ratios_dt: Optional[torch.Tensor] = None,
        collocation_time: Optional[torch.Tensor] = None,
        collocation_params: Optional[torch.Tensor] = None,
        epoch: int = 0,
        max_epochs: int = 1000
    ) -> Dict[str, torch.Tensor]:
        """
        전체 손실 계산

        Args:
            pred: (batch, 3) 예측 비율
            target: (batch, 3) 타겟 비율
            model: PINN 모델 (물리 손실용)
            x: (batch, 1 + param_dim) 입력 [time, params...]
            d_ratios_dt: (batch, 3) 시간 기울기 (선택적)
            collocation_*: 콜로케이션 포인트 (선택적)
            epoch: 현재 에폭 (가중치 조정용)
            max_epochs: 최대 에폭
        """
        losses = {}
        device = pred.device

        # 1. 데이터 손실
        losses['data'] = self.lambda_data * self.data_loss(pred, target)

        # 2. 질량 보존 손실
        losses['conservation'] = self.lambda_conservation * self.conservation_loss(pred)

        # 3. 초기 조건 손실
        if model is not None and x is not None:
            params = x[:, 1:]
            losses['initial'] = self.lambda_initial * self.initial_condition_loss(
                model, params, device
            )

        # 4. 경계 조건 손실
        if model is not None and x is not None:
            params = x[:, 1:]
            losses['boundary'] = self.lambda_boundary * self.boundary_condition_loss(
                model, params, device
            )

        # 5. 단조성 손실 (에폭 후반에 강화)
        if model is not None and x is not None:
            monotonic_weight = min(1.0, epoch / (max_epochs * 0.5))
            losses['monotonic'] = self.lambda_monotonic * monotonic_weight * \
                                  self.monotonicity_loss(model, x)

        # 6. 평활성 손실
        if d_ratios_dt is not None:
            losses['smoothness'] = self.lambda_smoothness * self.smoothness_loss(d_ratios_dt)

        # 7. 물리 상관관계 손실 (에폭 후반에 강화)
        if x is not None:
            params = x[:, 1:]
            physics_weight = min(1.0, epoch / (max_epochs * 0.3))
            losses['physics'] = self.lambda_physics * physics_weight * \
                               self.physics_correlation_loss(pred, params)

        # 8. 콜로케이션 손실 (선택적)
        if model is not None and collocation_time is not None and collocation_params is not None:
            losses['collocation'] = 0.1 * self.collocation_loss(
                model, collocation_time, collocation_params
            )

        # 총 손실
        losses['total'] = sum(losses.values())

        return losses


class AdaptiveWeightedLoss(nn.Module):
    """
    Uncertainty Weighting을 통한 적응형 가중치 학습
    """

    def __init__(self, num_losses: int = 6):
        super().__init__()
        # log(σ²) 파라미터
        self.log_vars = nn.Parameter(torch.zeros(num_losses))

    def forward(self, losses: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            losses: 손실 딕셔너리 (total 제외)

        Returns:
            total_loss: 가중합
            weights: 학습된 가중치
        """
        loss_names = [k for k in losses.keys() if k != 'total']
        total = 0
        weights = {}

        for i, name in enumerate(loss_names):
            if i < len(self.log_vars):
                precision = torch.exp(-self.log_vars[i])
                total = total + precision * losses[name] + self.log_vars[i]
                weights[name] = precision.item()

        return total, weights


def compute_timeseries_metrics(
    pred: torch.Tensor,
    target: torch.Tensor
) -> Dict[str, float]:
    """시계열 예측 평가 지표"""
    with torch.no_grad():
        # 전체 MSE/MAE
        mse = F.mse_loss(pred, target).item()
        mae = F.l1_loss(pred, target).item()

        # 성분별 MSE
        blood_mse = F.mse_loss(pred[:, 0], target[:, 0]).item()
        lymph_mse = F.mse_loss(pred[:, 1], target[:, 1]).item()
        ecm_mse = F.mse_loss(pred[:, 2], target[:, 2]).item()

        # R² score
        ss_res = ((target - pred) ** 2).sum()
        ss_tot = ((target - target.mean()) ** 2).sum()
        r2 = (1 - ss_res / (ss_tot + 1e-8)).item()

        # 성분별 R²
        def r2_score(p, t):
            ss_res = ((t - p) ** 2).sum()
            ss_tot = ((t - t.mean()) ** 2).sum()
            return (1 - ss_res / (ss_tot + 1e-8)).item()

        blood_r2 = r2_score(pred[:, 0], target[:, 0])
        lymph_r2 = r2_score(pred[:, 1], target[:, 1])
        ecm_r2 = r2_score(pred[:, 2], target[:, 2])

    return {
        'mse': mse,
        'mae': mae,
        'r2': r2,
        'blood_mse': blood_mse,
        'lymph_mse': lymph_mse,
        'ecm_mse': ecm_mse,
        'blood_r2': blood_r2,
        'lymph_r2': lymph_r2,
        'ecm_r2': ecm_r2
    }


if __name__ == "__main__":
    print("시계열 손실 함수 테스트...")

    loss_fn = TimeSeriesPhysicsLoss()

    # 테스트 데이터
    pred = torch.softmax(torch.randn(32, 3), dim=-1)
    target = torch.softmax(torch.randn(32, 3), dim=-1)
    x = torch.randn(32, 6)
    x[:, 0] = torch.rand(32)  # 시간 0-1

    # 손실 계산
    losses = loss_fn(pred, target, x=x)

    print("\n손실 값:")
    for name, value in losses.items():
        print(f"  {name}: {value.item():.6f}")

    # 메트릭 계산
    metrics = compute_timeseries_metrics(pred, target)
    print("\n평가 지표:")
    for name, value in metrics.items():
        print(f"  {name}: {value:.6f}")
