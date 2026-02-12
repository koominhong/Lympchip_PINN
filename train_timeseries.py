#!/usr/bin/env python3
"""
시계열 PINN 학습 스크립트
0-72시간 동안 Blood/Lymph/ECM 비율 변화 예측 모델 학습
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau
import numpy as np
from datetime import datetime
import json

from data.timeseries_preprocessor import TimeSeriesPreprocessor
from models.timeseries_pinn import TimeSeriesPINN, create_timeseries_model
from models.timeseries_losses import TimeSeriesPhysicsLoss, compute_timeseries_metrics


def get_device():
    """최적의 디바이스 선택"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')


def train_epoch(
    model: nn.Module,
    train_loader,
    optimizer,
    loss_fn,
    device,
    epoch: int,
    max_epochs: int,
    collocation_time=None,
    collocation_params=None
):
    """단일 에폭 학습"""
    model.train()
    total_losses = {}

    for batch_idx, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()

        # 예측
        pred = model(x)

        # 기울기 계산 (물리 손실용)
        if hasattr(model, 'predict_with_gradients'):
            _, d_ratios_dt = model.predict_with_gradients(x)
        else:
            d_ratios_dt = None

        # 손실 계산
        losses = loss_fn(
            pred=pred,
            target=y,
            model=model,
            x=x,
            d_ratios_dt=d_ratios_dt,
            collocation_time=collocation_time.to(device) if collocation_time is not None else None,
            collocation_params=collocation_params.to(device) if collocation_params is not None else None,
            epoch=epoch,
            max_epochs=max_epochs
        )

        # 역전파
        losses['total'].backward()

        # 그래디언트 클리핑
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        # 손실 누적
        for name, value in losses.items():
            if name not in total_losses:
                total_losses[name] = 0
            total_losses[name] += value.item()

    # 평균 손실
    num_batches = len(train_loader)
    for name in total_losses:
        total_losses[name] /= num_batches

    return total_losses


def evaluate(model: nn.Module, test_loader, loss_fn, device):
    """평가"""
    model.eval()

    all_preds = []
    all_targets = []
    total_loss = 0

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)

            loss = loss_fn.data_loss(pred, y)
            total_loss += loss.item()

            all_preds.append(pred.cpu())
            all_targets.append(y.cpu())

    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)

    metrics = compute_timeseries_metrics(all_preds, all_targets)
    metrics['loss'] = total_loss / len(test_loader)

    return metrics, all_preds, all_targets


def train_timeseries_pinn(
    # 모델 설정
    hidden_dim: int = 256,
    num_layers: int = 6,
    num_time_freq: int = 32,
    model_type: str = 'standard',
    # 학습 설정
    num_epochs: int = 500,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-5,
    # 손실 가중치
    lambda_data: float = 1.0,
    lambda_conservation: float = 0.5,
    lambda_initial: float = 0.5,
    lambda_physics: float = 0.1,
    # 기타
    use_collocation: bool = True,
    n_collocation: int = 500,
    save_path: str = None,
    verbose: bool = True
):
    """
    시계열 PINN 학습

    Returns:
        model: 학습된 모델
        history: 학습 히스토리
    """
    device = get_device()
    print(f"디바이스: {device}")

    # 데이터 로드
    print("\n[1/5] 데이터 로딩...")
    preprocessor = TimeSeriesPreprocessor()
    loaders = preprocessor.create_dataloaders(
        batch_size=batch_size,
        test_cases=None  # 랜덤 분할 사용
    )

    train_loader = loaders['train_loader']
    test_loader = loaders['test_loader']
    metadata = loaders['metadata']

    print(f"  학습 샘플: {len(train_loader.dataset)}")
    if test_loader:
        print(f"  테스트 샘플: {len(test_loader.dataset)}")
    print(f"  파라미터: {metadata['param_names']}")
    print(f"  케이스 수: {metadata['num_cases']}")

    # 콜로케이션 포인트 생성
    collocation_time = None
    collocation_params = None
    if use_collocation:
        collocation_time, collocation_params = preprocessor.get_collocation_points(
            n_time=50,
            n_param_samples=n_collocation // 50
        )
        print(f"  콜로케이션 포인트: {len(collocation_time)}")

    # 모델 생성
    print("\n[2/5] 모델 생성...")
    param_dim = len(metadata['param_names'])

    model = create_timeseries_model(
        model_type=model_type,
        param_dim=param_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_time_freq=num_time_freq
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"  모델 타입: {model_type}")
    print(f"  파라미터 수: {num_params:,}")

    # 손실 함수
    loss_fn = TimeSeriesPhysicsLoss(
        lambda_data=lambda_data,
        lambda_conservation=lambda_conservation,
        lambda_initial=lambda_initial,
        lambda_physics=lambda_physics
    )

    # 옵티마이저
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )

    # 스케줄러
    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=50,
        T_mult=2,
        eta_min=1e-6
    )

    # 학습 히스토리
    history = {
        'train_loss': [],
        'test_loss': [],
        'test_r2': [],
        'best_epoch': 0,
        'best_r2': 0
    }

    best_r2 = -float('inf')
    best_state = None

    # 학습 시작
    print(f"\n[3/5] 학습 시작 ({num_epochs} 에폭)...")

    for epoch in range(1, num_epochs + 1):
        # 학습
        train_losses = train_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            epoch=epoch,
            max_epochs=num_epochs,
            collocation_time=collocation_time,
            collocation_params=collocation_params
        )

        history['train_loss'].append(train_losses['total'])

        # 평가
        if test_loader is not None:
            metrics, _, _ = evaluate(model, test_loader, loss_fn, device)
            history['test_loss'].append(metrics['loss'])
            history['test_r2'].append(metrics['r2'])

            # 베스트 모델 저장
            if metrics['r2'] > best_r2:
                best_r2 = metrics['r2']
                best_state = model.state_dict().copy()
                history['best_epoch'] = epoch
                history['best_r2'] = best_r2

        # 스케줄러 업데이트
        scheduler.step()

        # 로그 출력
        if verbose and (epoch % 20 == 0 or epoch == 1):
            log_str = f"Epoch {epoch:4d} | Train Loss: {train_losses['total']:.6f}"

            if test_loader is not None:
                log_str += f" | Test R²: {metrics['r2']:.4f}"
                log_str += f" (Blood: {metrics['blood_r2']:.3f}, Lymph: {metrics['lymph_r2']:.3f}, ECM: {metrics['ecm_r2']:.3f})"

            log_str += f" | LR: {scheduler.get_last_lr()[0]:.2e}"
            print(log_str)

    # 베스트 모델 복원
    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"\n[4/5] 베스트 모델 복원 (Epoch {history['best_epoch']}, R² = {history['best_r2']:.4f})")

    # 최종 평가
    print("\n[5/5] 최종 평가...")
    if test_loader is not None:
        final_metrics, _, _ = evaluate(model, test_loader, loss_fn, device)
        print(f"  전체 R²: {final_metrics['r2']:.4f}")
        print(f"  Blood R²: {final_metrics['blood_r2']:.4f}")
        print(f"  Lymph R²: {final_metrics['lymph_r2']:.4f}")
        print(f"  ECM R²: {final_metrics['ecm_r2']:.4f}")
        print(f"  MSE: {final_metrics['mse']:.6f}")

    # 모델 저장
    if save_path is None:
        save_dir = Path(__file__).parent / "checkpoints"
        save_dir.mkdir(exist_ok=True)
        save_path = save_dir / "timeseries_pinn.pt"

    checkpoint = {
        'model_state_dict': model.state_dict(),
        'model_config': {
            'model_type': model_type,
            'param_dim': param_dim,
            'hidden_dim': hidden_dim,
            'num_layers': num_layers,
            'num_time_freq': num_time_freq
        },
        'history': history,
        'param_names': metadata['param_names'],
        'time_max': metadata['time_max'],
        'timestamp': datetime.now().isoformat()
    }

    torch.save(checkpoint, save_path)
    print(f"\n모델 저장: {save_path}")

    return model, history


def quick_train():
    """빠른 테스트용 학습"""
    print("=" * 60)
    print("시계열 PINN 빠른 학습 테스트")
    print("=" * 60)

    model, history = train_timeseries_pinn(
        hidden_dim=256,
        num_layers=6,
        num_time_freq=32,
        num_epochs=300,
        batch_size=32,
        learning_rate=5e-4,
        weight_decay=1e-6,
        lambda_data=1.0,
        lambda_conservation=0.3,
        lambda_initial=0.3,
        lambda_physics=0.05,
        use_collocation=False,
        verbose=True
    )

    print("\n" + "=" * 60)
    print("학습 완료!")
    print("=" * 60)

    return model


def full_train():
    """전체 학습"""
    print("=" * 60)
    print("시계열 PINN 전체 학습")
    print("=" * 60)

    model, history = train_timeseries_pinn(
        hidden_dim=256,
        num_layers=6,
        num_time_freq=32,
        num_epochs=500,
        batch_size=64,
        learning_rate=1e-3,
        lambda_data=1.0,
        lambda_conservation=0.5,
        lambda_initial=0.5,
        lambda_physics=0.1,
        use_collocation=True,
        n_collocation=500,
        verbose=True
    )

    print("\n" + "=" * 60)
    print("학습 완료!")
    print("=" * 60)

    return model


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="시계열 PINN 학습")
    parser.add_argument('--mode', type=str, default='quick', choices=['quick', 'full'],
                        help='학습 모드: quick(빠른 테스트) 또는 full(전체 학습)')
    parser.add_argument('--epochs', type=int, default=None, help='에폭 수 (선택)')
    parser.add_argument('--hidden', type=int, default=None, help='히든 차원 (선택)')

    args = parser.parse_args()

    if args.mode == 'quick':
        quick_train()
    else:
        full_train()
