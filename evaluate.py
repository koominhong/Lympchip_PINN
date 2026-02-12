"""
평가 및 시각화 스크립트
학습된 PINN 모델의 성능 평가 및 결과 시각화
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # GUI 없이 실행

import sys
sys.path.insert(0, str(Path(__file__).parent))

from data.preprocessor import LymphChipPreprocessor
from models.pinn import create_model, LymphChipPINN
from models.losses import compute_metrics


def load_model(
    checkpoint_path: str,
    model_type: str,
    input_dim: int,
    output_dim: int,
    hidden_dim: int = 256,
    num_layers: int = 6,
    device: str = 'cpu'
) -> nn.Module:
    """체크포인트에서 모델 로드"""
    output_type = 'ratios' if output_dim == 3 else 'concentration'

    model = create_model(
        model_type=model_type,
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        output_dim=output_dim,
        output_type=output_type
    )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    return model


def plot_training_history(history_path: str, save_path: str = None):
    """학습 히스토리 시각화"""
    with open(history_path, 'r') as f:
        history = json.load(f)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 손실 곡선
    ax = axes[0, 0]
    ax.plot(history['train_loss'], label='Train Loss', color='blue')
    if history.get('val_loss'):
        ax.plot(history['val_loss'], label='Val Loss', color='orange')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training & Validation Loss')
    ax.legend()
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)

    # 학습률
    ax = axes[0, 1]
    if history.get('learning_rates'):
        ax.plot(history['learning_rates'], color='green')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Learning Rate')
        ax.set_title('Learning Rate Schedule')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)

    # R² 점수
    ax = axes[1, 0]
    if history.get('val_metrics'):
        r2_scores = [m.get('r2', 0) for m in history['val_metrics']]
        ax.plot(r2_scores, color='purple')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('R² Score')
        ax.set_title('Validation R² Score')
        ax.axhline(y=0.9, color='r', linestyle='--', alpha=0.5, label='0.9 threshold')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # MAE
    ax = axes[1, 1]
    if history.get('val_metrics'):
        mae_scores = [m.get('mae', 0) for m in history['val_metrics']]
        ax.plot(mae_scores, color='red')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('MAE')
        ax.set_title('Validation MAE')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"학습 히스토리 그래프 저장: {save_path}")
    else:
        plt.show()

    plt.close()


def plot_ratio_predictions(
    model: nn.Module,
    X: np.ndarray,
    y_true: np.ndarray,
    conditions: List[str],
    save_path: str = None,
    device: str = 'cpu'
):
    """비율 예측 결과 시각화"""
    model.eval()

    with torch.no_grad():
        X_tensor = torch.FloatTensor(X).to(device)
        y_pred = model(X_tensor).cpu().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(15, 6))

    compartments = ['Blood', 'Lymph', 'ECM']
    colors = ['#e74c3c', '#3498db', '#2ecc71']

    for i, (comp, color) in enumerate(zip(compartments, colors)):
        ax = axes[i]

        # 바 차트
        x = np.arange(len(conditions))
        width = 0.35

        bars1 = ax.bar(x - width/2, y_true[:, i] * 100, width,
                       label='True', color=color, alpha=0.7)
        bars2 = ax.bar(x + width/2, y_pred[:, i] * 100, width,
                       label='Predicted', color=color, alpha=0.4, hatch='//')

        ax.set_xlabel('Condition')
        ax.set_ylabel('Ratio (%)')
        ax.set_title(f'{comp} Distribution')
        ax.set_xticks(x)
        ax.set_xticklabels(conditions, rotation=45, ha='right', fontsize=8)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"비율 예측 그래프 저장: {save_path}")
    else:
        plt.show()

    plt.close()


def plot_concentration_predictions(
    model: nn.Module,
    X_test: np.ndarray,
    y_test: np.ndarray,
    scaler=None,
    save_path: str = None,
    device: str = 'cpu'
):
    """농도 예측 결과 시각화"""
    model.eval()

    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_test).to(device)
        y_pred = model(X_tensor).cpu().numpy()

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. 예측 vs 실제 산점도
    ax = axes[0, 0]
    ax.scatter(y_test.flatten(), y_pred.flatten(), alpha=0.5, s=10)
    max_val = max(y_test.max(), y_pred.max())
    ax.plot([0, max_val], [0, max_val], 'r--', label='Perfect prediction')
    ax.set_xlabel('True Concentration')
    ax.set_ylabel('Predicted Concentration')
    ax.set_title('Prediction vs True')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. 오차 분포
    ax = axes[0, 1]
    errors = (y_pred - y_test).flatten()
    ax.hist(errors, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    ax.axvline(x=0, color='red', linestyle='--')
    ax.set_xlabel('Prediction Error')
    ax.set_ylabel('Frequency')
    ax.set_title(f'Error Distribution (Mean: {errors.mean():.2e}, Std: {errors.std():.2e})')
    ax.grid(True, alpha=0.3)

    # 3. 시간에 따른 예측 (첫 번째 특성이 시간이라고 가정)
    ax = axes[1, 0]

    # 시간 순서로 정렬
    if scaler is not None:
        # 역변환하여 실제 시간 값 얻기
        time_idx = 0
        time_values = X_test[:, time_idx]
    else:
        time_values = X_test[:, 0]

    sorted_idx = np.argsort(time_values)
    ax.plot(time_values[sorted_idx], y_test.flatten()[sorted_idx],
            'b-', label='True', alpha=0.7)
    ax.plot(time_values[sorted_idx], y_pred.flatten()[sorted_idx],
            'r--', label='Predicted', alpha=0.7)
    ax.set_xlabel('Time (normalized)')
    ax.set_ylabel('Concentration')
    ax.set_title('Concentration vs Time')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. 상대 오차
    ax = axes[1, 1]
    relative_errors = np.abs(errors) / (np.abs(y_test.flatten()) + 1e-8) * 100
    ax.hist(relative_errors[relative_errors < 100], bins=50,
            color='coral', edgecolor='black', alpha=0.7)
    ax.set_xlabel('Relative Error (%)')
    ax.set_ylabel('Frequency')
    ax.set_title(f'Relative Error Distribution (Median: {np.median(relative_errors):.1f}%)')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"농도 예측 그래프 저장: {save_path}")
    else:
        plt.show()

    plt.close()


def plot_parameter_sensitivity(
    model: nn.Module,
    base_params: np.ndarray,
    param_names: List[str],
    param_idx: int,
    param_range: np.ndarray,
    save_path: str = None,
    device: str = 'cpu'
):
    """파라미터 민감도 분석 시각화"""
    model.eval()

    # 기본 파라미터에서 하나의 파라미터만 변경
    X_sweep = np.tile(base_params, (len(param_range), 1))
    X_sweep[:, param_idx] = param_range

    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_sweep).to(device)
        y_pred = model(X_tensor).cpu().numpy()

    fig, ax = plt.subplots(figsize=(10, 6))

    if y_pred.shape[1] == 3:  # 비율 예측
        ax.plot(param_range, y_pred[:, 0] * 100, 'r-', label='Blood', linewidth=2)
        ax.plot(param_range, y_pred[:, 1] * 100, 'b-', label='Lymph', linewidth=2)
        ax.plot(param_range, y_pred[:, 2] * 100, 'g-', label='ECM', linewidth=2)
        ax.set_ylabel('Ratio (%)')
    else:  # 농도 예측
        ax.plot(param_range, y_pred.flatten(), 'b-', linewidth=2)
        ax.set_ylabel('Concentration')

    ax.set_xlabel(param_names[param_idx])
    ax.set_title(f'Sensitivity to {param_names[param_idx]}')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"민감도 분석 그래프 저장: {save_path}")
    else:
        plt.show()

    plt.close()


def generate_report(
    model: nn.Module,
    test_data: Dict,
    task_type: str,
    save_dir: str,
    device: str = 'cpu'
):
    """종합 평가 리포트 생성"""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    model.eval()

    report = {
        'task_type': task_type,
        'metrics': {},
        'predictions': {}
    }

    if task_type == 'concentration':
        X_test = test_data['X_test']
        y_test = test_data['y_test']

        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_test).to(device)
            y_pred = model(X_tensor).cpu().numpy()

        # 메트릭 계산
        metrics = compute_metrics(
            torch.FloatTensor(y_pred),
            torch.FloatTensor(y_test)
        )
        report['metrics'] = metrics

        # 시각화
        plot_concentration_predictions(
            model, X_test, y_test,
            save_path=str(save_dir / 'concentration_predictions.png'),
            device=device
        )

    elif task_type == 'ratios':
        X = test_data['X']
        y = test_data['y']
        conditions = test_data.get('conditions', [f'Case {i}' for i in range(len(X))])

        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(device)
            y_pred = model(X_tensor).cpu().numpy()

        # 메트릭 계산
        metrics = compute_metrics(
            torch.FloatTensor(y_pred),
            torch.FloatTensor(y)
        )
        report['metrics'] = metrics

        # 시각화
        plot_ratio_predictions(
            model, X, y, conditions,
            save_path=str(save_dir / 'ratio_predictions.png'),
            device=device
        )

        # 상세 예측 결과
        report['predictions'] = {
            'conditions': conditions,
            'true_blood': y[:, 0].tolist(),
            'true_lymph': y[:, 1].tolist(),
            'true_ecm': y[:, 2].tolist(),
            'pred_blood': y_pred[:, 0].tolist(),
            'pred_lymph': y_pred[:, 1].tolist(),
            'pred_ecm': y_pred[:, 2].tolist()
        }

    # 리포트 저장
    with open(save_dir / 'evaluation_report.json', 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\n{'='*60}")
    print("평가 결과")
    print('='*60)
    for key, value in report['metrics'].items():
        print(f"  {key}: {value:.6f}")
    print(f"\n리포트 저장 위치: {save_dir}")

    return report


def main():
    parser = argparse.ArgumentParser(description='PINN 모델 평가')

    parser.add_argument('--checkpoint', type=str, required=True,
                        help='체크포인트 파일 경로')
    parser.add_argument('--data_dir', type=str, default='..',
                        help='데이터 디렉토리')
    parser.add_argument('--output_dir', type=str, default='./results',
                        help='결과 저장 디렉토리')
    parser.add_argument('--task', type=str, default='concentration',
                        choices=['concentration', 'ratios'],
                        help='평가 작업')
    parser.add_argument('--model_type', type=str, default='pinn',
                        help='모델 유형')
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--num_layers', type=int, default=6)
    parser.add_argument('--device', type=str, default='cpu')

    args = parser.parse_args()

    # 데이터 로드
    preprocessor = LymphChipPreprocessor(args.data_dir)
    data = preprocessor.prepare_training_data(
        include_time_series=(args.task == 'concentration'),
        include_final_ratios=(args.task == 'ratios')
    )

    # 모델 설정
    if args.task == 'concentration' and 'time_series' in data:
        input_dim = data['time_series']['X_test'].shape[1]
        output_dim = 1
        test_data = data['time_series']
    elif args.task == 'ratios' and 'final_ratios' in data:
        input_dim = data['final_ratios']['X'].shape[1]
        output_dim = 3
        test_data = data['final_ratios']
    else:
        raise ValueError(f"Invalid task or missing data: {args.task}")

    # 모델 로드
    model = load_model(
        args.checkpoint,
        args.model_type,
        input_dim,
        output_dim,
        args.hidden_dim,
        args.num_layers,
        args.device
    )

    # 학습 히스토리 시각화
    history_path = Path(args.checkpoint).parent / 'training_history.json'
    if history_path.exists():
        plot_training_history(
            str(history_path),
            save_path=str(Path(args.output_dir) / 'training_history.png')
        )

    # 종합 리포트 생성
    generate_report(
        model=model,
        test_data=test_data,
        task_type=args.task,
        save_dir=args.output_dir,
        device=args.device
    )


if __name__ == "__main__":
    main()
