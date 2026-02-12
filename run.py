#!/usr/bin/env python3
"""
간단한 실행 스크립트
데이터 전처리부터 학습, 평가까지 한 번에 실행
"""

import sys
from pathlib import Path

# 경로 설정
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from data.preprocessor import LymphChipPreprocessor, LymphChipDataset
from models.pinn import LymphChipPINN, create_model
from models.losses import LymphChipLoss, compute_metrics


def run_quick_training():
    """빠른 학습 테스트"""

    print("=" * 60)
    print("림프칩 PINN 빠른 학습 테스트")
    print("=" * 60)

    # 디바이스 설정
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    print(f"\n디바이스: {device}")

    # 데이터 로드
    print("\n[1/4] 데이터 로딩...")
    data_dir = Path(__file__).parent.parent
    preprocessor = LymphChipPreprocessor(str(data_dir))

    data = preprocessor.prepare_training_data(
        include_time_series=True,
        include_final_ratios=True,
        test_split=0.2
    )

    # 시계열 데이터로 학습
    if 'time_series' in data:
        ts_data = data['time_series']
        print(f"  시계열 데이터: Train {ts_data['X_train'].shape}, Test {ts_data['X_test'].shape}")

        # 데이터셋 생성
        train_dataset = LymphChipDataset(ts_data['X_train'], ts_data['y_train'])
        test_dataset = LymphChipDataset(ts_data['X_test'], ts_data['y_test'])

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        # 모델 생성
        print("\n[2/4] 모델 생성...")
        input_dim = ts_data['X_train'].shape[1]
        model = LymphChipPINN(
            input_dim=input_dim,
            hidden_dim=128,
            num_layers=4,
            output_dim=1,
            output_type='concentration'
        ).to(device)

        print(f"  입력 차원: {input_dim}")
        print(f"  파라미터 수: {sum(p.numel() for p in model.parameters()):,}")

        # 손실 함수 및 옵티마이저
        loss_fn = LymphChipLoss()
        optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)

        # 학습
        print("\n[3/4] 학습 시작 (20 에폭)...")
        model.train()

        for epoch in range(1, 21):
            total_loss = 0
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)

                optimizer.zero_grad()
                pred = model(x)
                losses = loss_fn(pred, y, model=model, x=x, task_type='concentration')
                losses['total'].backward()
                optimizer.step()

                total_loss += losses['total'].item()

            avg_loss = total_loss / len(train_loader)

            if epoch % 5 == 0:
                print(f"  Epoch {epoch:2d}: Loss = {avg_loss:.6f}")

        # 평가
        print("\n[4/4] 평가...")
        model.eval()

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

        metrics = compute_metrics(all_preds, all_targets)

        print("\n평가 결과:")
        print(f"  MSE: {metrics['mse']:.6f}")
        print(f"  MAE: {metrics['mae']:.6f}")
        print(f"  R²:  {metrics['r2']:.4f}")

    # 비율 데이터 테스트
    if 'final_ratios' in data:
        print("\n" + "-" * 40)
        print("비율 예측 테스트")
        print("-" * 40)

        ratio_data = data['final_ratios']
        print(f"  데이터: {ratio_data['X'].shape}")
        print(f"  조건: {ratio_data['conditions']}")

        # 모델 생성
        input_dim = ratio_data['X'].shape[1]
        ratio_model = LymphChipPINN(
            input_dim=input_dim,
            hidden_dim=64,
            num_layers=3,
            output_dim=3,
            output_type='ratios',
            use_time_encoding=False
        ).to(device)

        optimizer = optim.AdamW(ratio_model.parameters(), lr=1e-2)

        # 학습 (데이터가 적으므로 많은 에폭)
        X_ratio = torch.FloatTensor(ratio_data['X']).to(device)
        y_ratio = torch.FloatTensor(ratio_data['y']).to(device)

        print("\n학습 중 (500 에폭)...")
        ratio_model.train()

        for epoch in range(1, 501):
            optimizer.zero_grad()
            pred = ratio_model(X_ratio)
            loss = nn.functional.mse_loss(pred, y_ratio)

            # 합이 1이 되도록 제약
            sum_constraint = ((pred.sum(dim=-1) - 1) ** 2).mean()
            total_loss = loss + 0.5 * sum_constraint

            total_loss.backward()
            optimizer.step()

            if epoch % 100 == 0:
                print(f"  Epoch {epoch}: Loss = {total_loss.item():.6f}")

        # 평가
        ratio_model.eval()
        with torch.no_grad():
            pred_ratios = ratio_model(X_ratio).cpu().numpy()

        print("\n예측 결과 (%):")
        print(f"{'조건':<20} {'Blood':>8} {'Lymph':>8} {'ECM':>8}")
        print("-" * 48)

        for i, cond in enumerate(ratio_data['conditions']):
            true = ratio_data['y'][i] * 100
            pred = pred_ratios[i] * 100
            print(f"{cond:<20} {pred[0]:>7.1f}% {pred[1]:>7.1f}% {pred[2]:>7.1f}%")
            print(f"{'  (실제)':<20} {true[0]:>7.1f}% {true[1]:>7.1f}% {true[2]:>7.1f}%")

    print("\n" + "=" * 60)
    print("테스트 완료!")
    print("=" * 60)

    return model


if __name__ == "__main__":
    run_quick_training()
