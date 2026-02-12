#!/usr/bin/env python3
"""
부드러운 시계열 PINN 학습 스크립트
울퉁불퉁함 없이 부드러운 곡선 예측
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from datetime import datetime

from data.timeseries_preprocessor import TimeSeriesPreprocessor
from models.smooth_pinn import SmoothPINNv2, create_smooth_model


def smoothness_loss(pred: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
    """
    예측의 부드러움 측정
    연속된 시간에서 급격한 변화가 없도록
    """
    # 시간순 정렬
    sorted_idx = torch.argsort(time.squeeze())
    sorted_pred = pred[sorted_idx]

    # 1차 미분 (변화량)
    diff1 = torch.diff(sorted_pred, dim=0)

    # 2차 미분 (변화량의 변화)
    diff2 = torch.diff(diff1, dim=0)

    # 둘 다 작아야 함
    return diff1.pow(2).mean() + 0.5 * diff2.pow(2).mean()


def monotonicity_loss(pred: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
    """
    단조성 손실
    Blood/Lymph는 증가, ECM은 감소해야 함
    """
    sorted_idx = torch.argsort(time.squeeze())
    sorted_pred = pred[sorted_idx]

    diff = torch.diff(sorted_pred, dim=0)

    # Blood (index 0): 증가 (diff > 0)
    blood_violation = torch.relu(-diff[:, 0]).mean()

    # Lymph (index 1): 증가
    lymph_violation = torch.relu(-diff[:, 1]).mean()

    # ECM (index 2): 감소 (diff < 0)
    ecm_violation = torch.relu(diff[:, 2]).mean()

    return blood_violation + lymph_violation + ecm_violation


def train_smooth_pinn(
    num_epochs: int = 500,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    lambda_smooth: float = 0.5,
    lambda_mono: float = 0.3,
    verbose: bool = True
):
    """부드러운 PINN 학습"""

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"디바이스: {device}")

    # 데이터 로드
    print("\n[1/4] 데이터 로딩...")
    preprocessor = TimeSeriesPreprocessor()
    loaders = preprocessor.create_dataloaders(batch_size=batch_size, test_cases=None)

    train_loader = loaders['train_loader']
    test_loader = loaders['test_loader']
    metadata = loaders['metadata']

    print(f"  학습 샘플: {len(train_loader.dataset)}")
    print(f"  테스트 샘플: {len(test_loader.dataset) if test_loader else 0}")

    # 모델 생성
    print("\n[2/4] 모델 생성...")
    model = SmoothPINNv2(
        param_dim=len(metadata['param_names']),
        hidden_dim=64,
        num_layers=3
    ).to(device)

    print(f"  파라미터 수: {sum(p.numel() for p in model.parameters()):,}")

    # 옵티마이저
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

    # 학습
    print(f"\n[3/4] 학습 시작 ({num_epochs} 에폭)...")

    best_loss = float('inf')
    best_state = None

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0
        total_data_loss = 0
        total_smooth_loss = 0
        total_mono_loss = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            t = x[:, 0:1]

            optimizer.zero_grad()

            pred = model(x)

            # 데이터 손실
            data_loss = nn.functional.mse_loss(pred, y)

            # 부드러움 손실
            smooth_loss = smoothness_loss(pred, t)

            # 단조성 손실
            mono_loss = monotonicity_loss(pred, t)

            # 총 손실
            loss = data_loss + lambda_smooth * smooth_loss + lambda_mono * mono_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            total_data_loss += data_loss.item()
            total_smooth_loss += smooth_loss.item()
            total_mono_loss += mono_loss.item()

        scheduler.step()

        avg_loss = total_loss / len(train_loader)

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_state = model.state_dict().copy()

        if verbose and (epoch % 50 == 0 or epoch == 1):
            print(f"Epoch {epoch:4d} | Loss: {avg_loss:.6f} (data: {total_data_loss/len(train_loader):.6f}, "
                  f"smooth: {total_smooth_loss/len(train_loader):.6f}, mono: {total_mono_loss/len(train_loader):.6f})")

    # 베스트 모델 복원
    model.load_state_dict(best_state)

    # 평가
    print("\n[4/4] 평가...")
    model.eval()

    if test_loader:
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(device)
                pred = model(x)
                all_preds.append(pred.cpu())
                all_targets.append(y)

        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)

        mse = nn.functional.mse_loss(all_preds, all_targets).item()
        print(f"  테스트 MSE: {mse:.6f}")

        # R² 계산
        ss_res = ((all_targets - all_preds) ** 2).sum()
        ss_tot = ((all_targets - all_targets.mean()) ** 2).sum()
        r2 = (1 - ss_res / ss_tot).item()
        print(f"  테스트 R²: {r2:.4f}")

    # 부드러움 테스트
    print("\n  부드러움 테스트 (Case 1 파라미터)...")
    test_x = torch.zeros(100, 6).to(device)
    test_x[:, 0] = torch.linspace(0, 1, 100)

    with torch.no_grad():
        test_pred = model(test_x).cpu()

    diffs = torch.abs(torch.diff(test_pred, dim=0)) * 100
    print(f"  Blood 변화: 평균 {diffs[:,0].mean():.3f}%, 최대 {diffs[:,0].max():.3f}%")
    print(f"  Lymph 변화: 평균 {diffs[:,1].mean():.3f}%, 최대 {diffs[:,1].max():.3f}%")
    print(f"  ECM 변화: 평균 {diffs[:,2].mean():.3f}%, 최대 {diffs[:,2].max():.3f}%")

    # 모델 저장
    save_path = Path(__file__).parent / "checkpoints" / "smooth_pinn.pt"
    save_path.parent.mkdir(exist_ok=True)

    checkpoint = {
        'model_state_dict': model.state_dict(),
        'model_config': {
            'model_type': 'v2',
            'param_dim': len(metadata['param_names']),
            'hidden_dim': 64,
            'num_layers': 3
        },
        'param_names': metadata['param_names'],
        'time_max': metadata['time_max'],
        'timestamp': datetime.now().isoformat()
    }

    torch.save(checkpoint, save_path)
    print(f"\n모델 저장: {save_path}")

    return model


if __name__ == "__main__":
    print("=" * 60)
    print("부드러운 시계열 PINN 학습")
    print("=" * 60)

    model = train_smooth_pinn(
        num_epochs=500,
        batch_size=32,
        learning_rate=1e-3,
        lambda_smooth=0.5,
        lambda_mono=0.3
    )

    print("\n" + "=" * 60)
    print("완료!")
    print("=" * 60)
