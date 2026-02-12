"""
PINN 학습 스크립트
림프칩 시뮬레이션 데이터를 사용한 PINN 학습
"""

import argparse
import json
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau
from torch.utils.data import DataLoader

# 로컬 모듈
import sys
sys.path.insert(0, str(Path(__file__).parent))

from data.preprocessor import LymphChipPreprocessor, LymphChipDataset
from models.pinn import LymphChipPINN, MultiTaskPINN, create_model
from models.losses import LymphChipLoss, compute_metrics


class Trainer:
    """PINN 학습 클래스"""

    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: Optional[object] = None,
        device: str = 'cpu',
        checkpoint_dir: str = './checkpoints',
        log_interval: int = 10
    ):
        self.model = model.to(device)
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_interval = log_interval

        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_metrics': [],
            'val_metrics': [],
            'learning_rates': []
        }

        self.best_val_loss = float('inf')
        self.epoch = 0

    def train_epoch(
        self,
        train_loader: DataLoader,
        task_type: str = 'concentration'
    ) -> Dict[str, float]:
        """한 에폭 학습"""
        self.model.train()
        total_losses = {}
        num_batches = 0

        for batch_idx, (x, y) in enumerate(train_loader):
            x = x.to(self.device)
            y = y.to(self.device)

            self.optimizer.zero_grad()

            # 순전파
            pred = self.model(x)

            # 손실 계산
            losses = self.loss_fn(
                pred=pred,
                target=y,
                model=self.model,
                x=x,
                task_type=task_type
            )

            # 역전파
            losses['total'].backward()

            # 기울기 클리핑
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            # 손실 누적
            for key, value in losses.items():
                if key not in total_losses:
                    total_losses[key] = 0.0
                total_losses[key] += value.item()

            num_batches += 1

            # 로그
            if batch_idx % self.log_interval == 0:
                print(f"  Batch {batch_idx}/{len(train_loader)}, Loss: {losses['total'].item():.6f}")

        # 평균 손실
        avg_losses = {k: v / num_batches for k, v in total_losses.items()}

        return avg_losses

    @torch.no_grad()
    def evaluate(
        self,
        val_loader: DataLoader,
        task_type: str = 'concentration'
    ) -> Dict[str, float]:
        """검증/평가"""
        self.model.eval()

        all_preds = []
        all_targets = []
        total_losses = {}
        num_batches = 0

        for x, y in val_loader:
            x = x.to(self.device)
            y = y.to(self.device)

            pred = self.model(x)

            # 손실 계산 (물리 손실 제외)
            losses = self.loss_fn(
                pred=pred,
                target=y,
                task_type=task_type
            )

            for key, value in losses.items():
                if key not in total_losses:
                    total_losses[key] = 0.0
                total_losses[key] += value.item()

            all_preds.append(pred.cpu())
            all_targets.append(y.cpu())
            num_batches += 1

        # 평균 손실
        avg_losses = {k: v / num_batches for k, v in total_losses.items()}

        # 전체 메트릭 계산
        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        metrics = compute_metrics(all_preds, all_targets)

        return {**avg_losses, **metrics}

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        num_epochs: int = 100,
        task_type: str = 'concentration',
        early_stopping_patience: int = 20
    ):
        """전체 학습 루프"""
        print(f"\n{'='*60}")
        print(f"학습 시작: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Device: {self.device}")
        print(f"Task: {task_type}")
        print(f"Epochs: {num_epochs}")
        print(f"{'='*60}\n")

        patience_counter = 0

        for epoch in range(1, num_epochs + 1):
            self.epoch = epoch
            print(f"\nEpoch {epoch}/{num_epochs}")
            print("-" * 40)

            # 학습
            train_losses = self.train_epoch(train_loader, task_type)
            self.history['train_loss'].append(train_losses['total'])

            print(f"Train Loss: {train_losses['total']:.6f}")

            # 검증
            if val_loader is not None:
                val_results = self.evaluate(val_loader, task_type)
                self.history['val_loss'].append(val_results['total'])
                self.history['val_metrics'].append(val_results)

                print(f"Val Loss: {val_results['total']:.6f}, R²: {val_results.get('r2', 0):.4f}")

                # 최적 모델 저장
                if val_results['total'] < self.best_val_loss:
                    self.best_val_loss = val_results['total']
                    self.save_checkpoint('best_model.pt')
                    patience_counter = 0
                    print("  [Best model saved]")
                else:
                    patience_counter += 1

                # Early stopping
                if patience_counter >= early_stopping_patience:
                    print(f"\nEarly stopping at epoch {epoch}")
                    break

            # 스케줄러 업데이트
            if self.scheduler is not None:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    if val_loader is not None:
                        self.scheduler.step(val_results['total'])
                else:
                    self.scheduler.step()

            current_lr = self.optimizer.param_groups[0]['lr']
            self.history['learning_rates'].append(current_lr)

            # 주기적 체크포인트
            if epoch % 50 == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch}.pt')

        # 최종 모델 저장
        self.save_checkpoint('final_model.pt')
        print(f"\n학습 완료: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        return self.history

    def save_checkpoint(self, filename: str):
        """체크포인트 저장"""
        checkpoint = {
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'history': self.history
        }
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        torch.save(checkpoint, self.checkpoint_dir / filename)

    def load_checkpoint(self, filename: str):
        """체크포인트 로드"""
        checkpoint = torch.load(self.checkpoint_dir / filename, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        self.history = checkpoint['history']

        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        print(f"Checkpoint loaded: epoch {self.epoch}, best_val_loss: {self.best_val_loss:.6f}")


def main():
    parser = argparse.ArgumentParser(description='림프칩 PINN 학습')

    # 데이터 관련
    parser.add_argument('--data_dir', type=str, default='..',
                        help='데이터 디렉토리 경로')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='배치 크기')
    parser.add_argument('--test_split', type=float, default=0.2,
                        help='테스트 데이터 비율')

    # 모델 관련
    parser.add_argument('--model_type', type=str, default='pinn',
                        choices=['pinn', 'multitask', 'deeponet'],
                        help='모델 유형')
    parser.add_argument('--hidden_dim', type=int, default=256,
                        help='은닉층 차원')
    parser.add_argument('--num_layers', type=int, default=6,
                        help='레이어 수')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='드롭아웃 비율')

    # 학습 관련
    parser.add_argument('--num_epochs', type=int, default=200,
                        help='학습 에폭 수')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='학습률')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='가중치 감쇠')
    parser.add_argument('--task', type=str, default='concentration',
                        choices=['concentration', 'ratios', 'both'],
                        help='학습 작업')

    # 손실 함수 가중치
    parser.add_argument('--lambda_data', type=float, default=1.0)
    parser.add_argument('--lambda_physics', type=float, default=0.1)
    parser.add_argument('--lambda_conservation', type=float, default=0.5)

    # 기타
    parser.add_argument('--device', type=str, default='auto',
                        help='디바이스 (auto/cpu/cuda/mps)')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                        help='체크포인트 저장 디렉토리')
    parser.add_argument('--seed', type=int, default=42,
                        help='랜덤 시드')

    args = parser.parse_args()

    # 시드 설정
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # 디바이스 설정
    if args.device == 'auto':
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    else:
        device = args.device

    print(f"Using device: {device}")

    # 데이터 로드
    print("\n데이터 로딩 중...")
    preprocessor = LymphChipPreprocessor(args.data_dir)
    data = preprocessor.prepare_training_data(
        include_time_series=(args.task in ['concentration', 'both']),
        include_final_ratios=(args.task in ['ratios', 'both']),
        test_split=args.test_split,
        random_state=args.seed
    )

    # 모델 생성
    if args.task == 'concentration' and 'time_series' in data:
        input_dim = data['time_series']['X_train'].shape[1]
        output_dim = 1

        model = create_model(
            model_type=args.model_type,
            input_dim=input_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            output_dim=output_dim,
            dropout=args.dropout,
            output_type='concentration'
        )

        # 데이터로더
        train_dataset = LymphChipDataset(
            data['time_series']['X_train'],
            data['time_series']['y_train']
        )
        test_dataset = LymphChipDataset(
            data['time_series']['X_test'],
            data['time_series']['y_test']
        )

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

        task_type = 'concentration'

    elif args.task == 'ratios' and 'final_ratios' in data:
        input_dim = data['final_ratios']['X'].shape[1]
        output_dim = 3

        model = create_model(
            model_type=args.model_type,
            input_dim=input_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            output_dim=output_dim,
            dropout=args.dropout,
            output_type='ratios'
        )

        # 전체 데이터를 학습에 사용 (데이터가 적으므로)
        train_dataset = LymphChipDataset(
            data['final_ratios']['X'],
            data['final_ratios']['y']
        )

        train_loader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=True)
        test_loader = None  # 데이터가 적어서 검증 세트 없음

        task_type = 'ratios'

    else:
        raise ValueError(f"Invalid task or missing data: {args.task}")

    print(f"\n모델: {args.model_type}")
    print(f"입력 차원: {input_dim}")
    print(f"출력 차원: {output_dim}")
    print(f"파라미터 수: {sum(p.numel() for p in model.parameters()):,}")

    # 손실 함수
    loss_fn = LymphChipLoss(
        lambda_data=args.lambda_data,
        lambda_physics=args.lambda_physics,
        lambda_conservation=args.lambda_conservation
    )

    # 옵티마이저
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    # 스케줄러
    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=50,
        T_mult=2,
        eta_min=1e-6
    )

    # 트레이너
    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        checkpoint_dir=args.checkpoint_dir
    )

    # 학습
    history = trainer.train(
        train_loader=train_loader,
        val_loader=test_loader,
        num_epochs=args.num_epochs,
        task_type=task_type
    )

    # 학습 기록 저장
    with open(Path(args.checkpoint_dir) / 'training_history.json', 'w') as f:
        # numpy 배열을 리스트로 변환
        history_serializable = {}
        for key, value in history.items():
            if isinstance(value, list):
                history_serializable[key] = [
                    v if not isinstance(v, dict) else
                    {k: float(vv) if isinstance(vv, (np.floating, float)) else vv
                     for k, vv in v.items()}
                    for v in value
                ]
            else:
                history_serializable[key] = value

        json.dump(history_serializable, f, indent=2)

    print("\n학습 완료!")
    print(f"체크포인트 저장 위치: {args.checkpoint_dir}")


if __name__ == "__main__":
    main()
