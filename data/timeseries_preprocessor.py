"""
시계열 PINN 학습을 위한 데이터 전처리기
Excel Case 시트에서 시간별 Blood/Lymph/ECM 비율 데이터 추출
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import re


class TimeSeriesDataset(Dataset):
    """시계열 PINN 학습용 데이터셋"""

    def __init__(
        self,
        time: np.ndarray,
        params: np.ndarray,
        targets: np.ndarray,
        param_names: List[str] = None
    ):
        """
        Args:
            time: (N,) 시간 배열 (hours)
            params: (N, num_params) 파라미터 배열
            targets: (N, 3) [Blood, Lymph, ECM] 비율 배열
            param_names: 파라미터 이름 리스트
        """
        self.time = torch.FloatTensor(time).unsqueeze(-1)
        self.params = torch.FloatTensor(params)
        self.targets = torch.FloatTensor(targets)
        self.param_names = param_names

    def __len__(self):
        return len(self.time)

    def __getitem__(self, idx):
        # 입력: [time, params...]
        x = torch.cat([self.time[idx], self.params[idx]], dim=-1)
        y = self.targets[idx]
        return x, y


class TimeSeriesPreprocessor:
    """시계열 데이터 전처리기"""

    # 파라미터 정규화 범위 (실제 값 → 정규화)
    PARAM_RANGES = {
        'Lp_ve': {'low': 4e-12, 'mid': 8e-12, 'high': 1.6e-11, 'scale': 'log'},
        'K': {'low': 1e-17, 'mid': 1e-15, 'high': 1e-13, 'scale': 'log'},
        'P_oncotic': {'low': 3145, 'mid': 3590, 'high': 3815, 'scale': 'linear'},
        'sigma_ve': {'low': 0.1, 'mid': 0.5, 'high': 0.9, 'scale': 'linear'},
        'D_gel': {'low': 1e-11, 'mid': 3e-11, 'high': 1e-10, 'scale': 'log'},
        'kdecay': {'low': 0, 'mid': 1.7e-6, 'high': 1.5e-5, 'scale': 'log_zero'},
    }

    PARAM_ORDER = ['Lp_ve', 'K', 'P_oncotic', 'sigma_ve', 'D_gel', 'kdecay']

    def __init__(self, data_dir: str = None):
        if data_dir is None:
            self.data_dir = Path(__file__).parent.parent.parent
        else:
            self.data_dir = Path(data_dir)

        self.time_scaler = MinMaxScaler(feature_range=(0, 1))
        self.target_scaler = None  # 비율은 이미 0-100%

    def normalize_param(self, value: float, param_name: str) -> float:
        """파라미터를 -1 ~ 1 범위로 정규화"""
        import math

        config = self.PARAM_RANGES.get(param_name)
        if config is None:
            return 0.0

        low, mid, high = config['low'], config['mid'], config['high']

        if config['scale'] == 'log':
            log_val = math.log10(value)
            log_low = math.log10(low)
            log_mid = math.log10(mid)
            log_high = math.log10(high)

            if value <= mid:
                normalized = (log_val - log_mid) / (log_mid - log_low)
            else:
                normalized = (log_val - log_mid) / (log_high - log_mid)
        elif config['scale'] == 'log_zero':
            # 0을 허용하는 특수 로그 스케일 (kdecay용)
            if value <= 0 or value < 1e-10:
                normalized = -1.0
            else:
                log_val = math.log10(value)
                log_mid = math.log10(mid)
                log_high = math.log10(high)

                if value <= mid:
                    # 0 ~ mid 사이: -1 ~ 0으로 선형 매핑
                    normalized = -1.0 + (value / mid)
                else:
                    # mid ~ high 사이: 로그 스케일
                    normalized = (log_val - log_mid) / (log_high - log_mid)
        else:
            if value <= mid:
                normalized = (value - mid) / (mid - low)
            else:
                normalized = (value - mid) / (high - mid)

        return max(-1, min(1, normalized))

    def load_case_data(self, file_path: str = None) -> Dict[str, Dict]:
        """
        Case별 시계열 데이터 로드

        Returns:
            Dict[case_name, {
                'params': Dict[param_name, value],
                'params_normalized': np.array,
                'time_series': pd.DataFrame with [time_hour, Blood, Lymph, ECM, Decay]
            }]
        """
        if file_path is None:
            file_path = self.data_dir / "251103 (revised)_Injection site results v2 (수정).xlsx"

        xl = pd.ExcelFile(file_path)
        case_data = {}

        for sheet_name in xl.sheet_names:
            if 'Case' not in sheet_name:
                continue

            df = pd.read_excel(file_path, sheet_name=sheet_name, header=None)

            # 파라미터 추출 (column 2 헤더에서)
            params = {}
            for col in df.iloc[0]:
                if pd.notna(col):
                    col_str = str(col)
                    for match in re.finditer(r'(\w+)=([0-9.E+-]+)', col_str):
                        try:
                            params[match.group(1)] = float(match.group(2))
                        except ValueError:
                            pass

            # 정규화된 파라미터 배열 생성
            params_normalized = []
            for param_name in self.PARAM_ORDER:
                if param_name in params:
                    norm_val = self.normalize_param(params[param_name], param_name)
                elif param_name == 'kdecay':
                    norm_val = -1.0  # injection site는 decay 없음
                else:
                    norm_val = 0.0  # mid 값
                params_normalized.append(norm_val)

            # 시계열 데이터 추출 (columns 14-19)
            # Column 14: Time (h), Column 15: Total Mass, Column 16: Lymph, Column 17: Blood, Column 19: ECM
            # 처음 72시간에 도달하는 지점까지만 사용
            time_series_data = []
            prev_time = -1
            reached_72h = False

            for idx in range(1, len(df)):
                row = df.iloc[idx]

                time_val = row[14]
                if pd.isna(time_val):
                    continue

                try:
                    time_hour = float(time_val)

                    # 시간이 감소하면 다른 데이터셋 시작 → 중단
                    if time_hour < prev_time - 0.1:  # 약간의 여유
                        break

                    # 72시간 초과하면 중단
                    if time_hour > 72:
                        reached_72h = True
                        break

                    prev_time = time_hour

                    lymph = float(row[16]) if pd.notna(row[16]) else 0
                    blood = float(row[17]) if pd.notna(row[17]) else 0
                    decay = float(row[18]) if pd.notna(row[18]) else 0
                    ecm = float(row[19]) if pd.notna(row[19]) else 0

                    time_series_data.append({
                        'time_hour': time_hour,
                        'Blood': blood,
                        'Lymph': lymph,
                        'ECM': ecm,
                        'Decay': decay
                    })

                    # 72시간에 도달하면 중단
                    if time_hour >= 71.9:
                        reached_72h = True
                        break

                except (ValueError, TypeError):
                    continue

            if time_series_data:
                ts_df = pd.DataFrame(time_series_data)
                ts_df = ts_df.sort_values('time_hour').reset_index(drop=True)

                # 최종 유효성 검사
                if len(ts_df) > 0:
                    final_row = ts_df.iloc[-1]
                    total_ratio = final_row['Blood'] + final_row['Lymph'] + final_row['ECM'] + final_row['Decay']

                    if total_ratio > 90:  # 유효한 데이터만 포함
                        case_data[sheet_name] = {
                            'params': params,
                            'params_normalized': np.array(params_normalized),
                            'time_series': ts_df
                        }
                    else:
                        print(f"경고: {sheet_name} 케이스 제외 (총 비율: {total_ratio:.1f}%)")

        return case_data

    def load_decay_data(self) -> Dict[str, Dict]:
        """
        sol765 파일에서 Decay 학습용 데이터 로드
        약물 종류(IgG, INS, ALB)와 decay rate별 시뮬레이션 데이터

        Returns:
            Dict[case_name, {
                'drug_type': str (IgG, INS, ALB),
                'kdecay': float,
                'time_series': pd.DataFrame with [time_hour, Blood, Lymph, ECM, Decay]
            }]
        """
        file_path = self.data_dir / "251103 (revised)_sol765.xlsx"

        if not file_path.exists():
            print(f"경고: Decay 데이터 파일을 찾을 수 없습니다: {file_path}")
            return {}

        xl = pd.ExcelFile(file_path)
        decay_data = {}

        for sheet_name in xl.sheet_names:
            df = pd.read_excel(file_path, sheet_name=sheet_name, header=None)

            # 시트 이름에서 약물 종류와 kdecay 추출
            drug_type = None
            kdecay = 0.0

            if 'IgG' in sheet_name:
                drug_type = 'IgG'
            elif 'INS' in sheet_name:
                drug_type = 'INS'
            elif 'ALB' in sheet_name:
                drug_type = 'ALB'

            if 'kdecay 0' in sheet_name or 'decay 0' in sheet_name.replace('k', ''):
                kdecay = 0.0
            elif '2.9e-7' in sheet_name:
                kdecay = 2.9e-7
            elif '1.7e-6' in sheet_name:
                kdecay = 1.7e-6
            elif '1.5e-5' in sheet_name:
                kdecay = 1.5e-5

            # 시계열 데이터 추출
            time_series_data = []
            prev_time = -1

            for idx in range(1, len(df)):
                row = df.iloc[idx]
                time_val = row[14]
                if pd.isna(time_val):
                    continue

                try:
                    time_hour = float(time_val)

                    if time_hour < prev_time - 0.1:
                        break
                    if time_hour > 72:
                        break

                    prev_time = time_hour

                    lymph = float(row[16]) if pd.notna(row[16]) else 0
                    blood = float(row[17]) if pd.notna(row[17]) else 0
                    decay = float(row[18]) if pd.notna(row[18]) else 0
                    ecm = float(row[19]) if pd.notna(row[19]) else 0

                    time_series_data.append({
                        'time_hour': time_hour,
                        'Blood': blood,
                        'Lymph': lymph,
                        'ECM': ecm,
                        'Decay': decay
                    })

                    if time_hour >= 71.9:
                        break

                except (ValueError, TypeError):
                    continue

            if time_series_data and drug_type:
                ts_df = pd.DataFrame(time_series_data)
                ts_df = ts_df.sort_values('time_hour').reset_index(drop=True)

                # 정규화된 파라미터 생성
                params_normalized = []
                for param_name in self.PARAM_ORDER:
                    if param_name == 'kdecay':
                        norm_val = self.normalize_param(kdecay, 'kdecay')
                    else:
                        norm_val = 0.0  # mid 값
                    params_normalized.append(norm_val)

                decay_data[sheet_name] = {
                    'drug_type': drug_type,
                    'kdecay': kdecay,
                    'params_normalized': np.array(params_normalized),
                    'time_series': ts_df
                }

        return decay_data

    def prepare_training_data(
        self,
        test_cases: List[str] = None,
        test_ratio: float = 0.2,
        normalize_time: bool = True,
        random_state: int = 42
    ) -> Dict:
        """
        학습용 데이터 준비

        Args:
            test_cases: 테스트용으로 분리할 케이스 이름 리스트 (None이면 랜덤 분할)
            test_ratio: 테스트 데이터 비율 (test_cases가 None일 때 사용)
            normalize_time: 시간을 0-1로 정규화할지 여부
            random_state: 랜덤 시드

        Returns:
            {
                'train': {'time', 'params', 'targets', 'case_names'},
                'test': {'time', 'params', 'targets', 'case_names'},
                'param_names': list,
                'time_max': float
            }
        """
        np.random.seed(random_state)
        case_data = self.load_case_data()

        if test_cases is None:
            # 랜덤으로 테스트 케이스 선택
            all_cases = list(case_data.keys())
            n_test = max(1, int(len(all_cases) * test_ratio))
            test_cases = list(np.random.choice(all_cases, n_test, replace=False))

        train_time = []
        train_params = []
        train_targets = []
        train_case_names = []

        test_time = []
        test_params = []
        test_targets = []
        test_case_names = []

        time_max = 72.0

        for case_name, data in case_data.items():
            ts = data['time_series']
            params = data['params_normalized']

            times = ts['time_hour'].values
            targets = ts[['Blood', 'Lymph', 'ECM']].values / 100.0  # 0-1로 정규화

            # 각 시간 포인트에 대해 파라미터 복제
            params_repeated = np.tile(params, (len(times), 1))

            if case_name in test_cases:
                test_time.extend(times.tolist())
                test_params.extend(params_repeated.tolist())
                test_targets.extend(targets.tolist())
                test_case_names.extend([case_name] * len(times))
            else:
                train_time.extend(times.tolist())
                train_params.extend(params_repeated.tolist())
                train_targets.extend(targets.tolist())
                train_case_names.extend([case_name] * len(times))

        # NumPy 배열로 변환
        train_time = np.array(train_time)
        train_params = np.array(train_params)
        train_targets = np.array(train_targets)

        test_time = np.array(test_time) if test_time else np.array([]).reshape(0)
        test_params = np.array(test_params) if test_params else np.array([]).reshape(0, len(self.PARAM_ORDER))
        test_targets = np.array(test_targets) if test_targets else np.array([]).reshape(0, 3)

        # 시간 정규화
        if normalize_time:
            train_time = train_time / time_max
            if len(test_time) > 0:
                test_time = test_time / time_max

        return {
            'train': {
                'time': train_time,
                'params': train_params,
                'targets': train_targets,
                'case_names': train_case_names
            },
            'test': {
                'time': test_time,
                'params': test_params,
                'targets': test_targets,
                'case_names': test_case_names
            },
            'param_names': self.PARAM_ORDER,
            'time_max': time_max,
            'num_cases': len(case_data),
            'case_names': list(case_data.keys())
        }

    def create_dataloaders(
        self,
        batch_size: int = 64,
        test_cases: List[str] = None,
        shuffle_train: bool = True
    ) -> Dict:
        """DataLoader 생성"""

        data = self.prepare_training_data(test_cases=test_cases)

        train_dataset = TimeSeriesDataset(
            time=data['train']['time'],
            params=data['train']['params'],
            targets=data['train']['targets'],
            param_names=data['param_names']
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=shuffle_train
        )

        test_loader = None
        if len(data['test']['time']) > 0:
            test_dataset = TimeSeriesDataset(
                time=data['test']['time'],
                params=data['test']['params'],
                targets=data['test']['targets'],
                param_names=data['param_names']
            )
            test_loader = DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False
            )

        return {
            'train_loader': train_loader,
            'test_loader': test_loader,
            'metadata': data
        }

    def get_collocation_points(
        self,
        n_time: int = 100,
        n_param_samples: int = 50
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        물리 손실 계산을 위한 콜로케이션 포인트 생성

        Returns:
            time: (n_time * n_param_samples, 1)
            params: (n_time * n_param_samples, num_params)
        """
        # 시간 포인트 (0-1)
        time_points = np.linspace(0, 1, n_time)

        # 파라미터 샘플링 (Latin Hypercube 또는 균등 샘플링)
        param_samples = []
        for _ in range(n_param_samples):
            sample = []
            for param_name in self.PARAM_ORDER:
                # -1 ~ 1 범위에서 균등 샘플링
                sample.append(np.random.uniform(-1, 1))
            param_samples.append(sample)

        param_samples = np.array(param_samples)

        # 모든 조합 생성
        all_time = []
        all_params = []

        for t in time_points:
            for params in param_samples:
                all_time.append(t)
                all_params.append(params)

        return (
            torch.FloatTensor(all_time).unsqueeze(-1),
            torch.FloatTensor(all_params)
        )


if __name__ == "__main__":
    print("시계열 데이터 전처리 테스트...")

    preprocessor = TimeSeriesPreprocessor()
    data = preprocessor.prepare_training_data()

    print(f"\n총 케이스 수: {data['num_cases']}")
    print(f"케이스 목록: {data['case_names']}")
    print(f"\n학습 데이터:")
    print(f"  시간 포인트: {len(data['train']['time'])}")
    print(f"  파라미터 차원: {data['train']['params'].shape}")
    print(f"  타겟 차원: {data['train']['targets'].shape}")

    print(f"\n테스트 데이터:")
    print(f"  시간 포인트: {len(data['test']['time'])}")

    # DataLoader 테스트
    loaders = preprocessor.create_dataloaders(batch_size=32)
    for x, y in loaders['train_loader']:
        print(f"\n배치 샘플:")
        print(f"  입력 x shape: {x.shape}")  # [batch, 1 + num_params]
        print(f"  출력 y shape: {y.shape}")  # [batch, 3]
        break
