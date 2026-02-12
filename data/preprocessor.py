"""
데이터 전처리 모듈
엑셀 파일에서 시뮬레이션 데이터를 추출하고 PINN 학습용으로 변환
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import re


class LymphChipDataset(Dataset):
    """림프칩 시뮬레이션 데이터셋"""

    def __init__(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        feature_names: List[str] = None,
        target_names: List[str] = None
    ):
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)
        self.feature_names = feature_names
        self.target_names = target_names

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]


class LymphChipPreprocessor:
    """림프칩 시뮬레이션 데이터 전처리기"""

    def __init__(self, data_dir: str = None):
        self.data_dir = Path(data_dir) if data_dir else Path(__file__).parent
        self.feature_scaler = StandardScaler()
        self.target_scaler = MinMaxScaler()

        # 약물 특성 (MW in kDa)
        self.drug_properties = {
            'IgG': {'MW': 150, 'type_idx': 0},
            'INS': {'MW': 5.8, 'type_idx': 1},  # 인슐린
            'ALB': {'MW': 66.5, 'type_idx': 2}  # 알부민
        }

    def parse_column_parameters(self, col_name: str) -> Dict:
        """컬럼명에서 파라미터 추출"""
        params = {}
        col_str = str(col_name)

        # 정규표현식으로 파라미터 추출
        patterns = {
            'MW': r'MW=([0-9.E+-]+)',
            'sigma_ve': r'sigma_ve=([0-9.E+-]+)',
            'sigma_le': r'sigma_le=([0-9.E+-]+)',
            'p_ve': r'p_ve=([0-9.E+-]+)',
            'p_le': r'p_le=([0-9.E+-]+)',
            'D_gel': r'D_gel=([0-9.E+-]+)',
            'kf_m': r'kf_m=([0-9.E+-]+)',
            'kr_m': r'kr_m=([0-9.E+-]+)',
            'k_decay': r'k_decay=([0-9.E+-]+)',
            'Lp_ve': r'Lp_ve=([0-9.E+-]+)',
            'K': r'K=([0-9.E+-]+)',
            'P_oncotic': r'P_oncotic=([0-9.E+-]+)',
        }

        for param, pattern in patterns.items():
            match = re.search(pattern, col_str, re.IGNORECASE)
            if match:
                try:
                    params[param] = float(match.group(1))
                except ValueError:
                    pass

        return params

    def load_sol765_data(self, file_path: str = None) -> pd.DataFrame:
        """sol765.xlsx 파일 로드 - 약물별 시뮬레이션"""
        if file_path is None:
            file_path = self.data_dir / "251103 (revised)_sol765.xlsx"

        xl = pd.ExcelFile(file_path)
        all_data = []

        for sheet_name in xl.sheet_names:
            # 시트명에서 약물 종류와 decay 상수 추출
            drug_type = None
            for drug in ['IgG', 'INS', 'ALB']:
                if drug in sheet_name:
                    drug_type = drug
                    break

            if drug_type is None:
                continue

            # k_decay 값 추출
            k_decay = 0.0
            decay_match = re.search(r'decay\s*([0-9.E+-]+)', sheet_name, re.IGNORECASE)
            if decay_match:
                try:
                    k_decay = float(decay_match.group(1))
                except ValueError:
                    k_decay = 0.0

            # 데이터 읽기
            df = pd.read_excel(file_path, sheet_name=sheet_name)

            # Time 컬럼 찾기
            time_col_idx = None
            for i, col in enumerate(df.columns):
                if 'Time' in str(col):
                    time_col_idx = i
                    break

            if time_col_idx is None:
                continue

            # 파라미터 추출 (첫 번째 데이터 컬럼에서)
            if len(df.columns) > time_col_idx + 1:
                params = self.parse_column_parameters(str(df.columns[time_col_idx + 1]))
            else:
                params = {}

            # 시간 데이터
            time_data = pd.to_numeric(df.iloc[:, time_col_idx], errors='coerce')

            # 농도 데이터 컬럼들 찾기 (Concentration 포함된 컬럼)
            conc_cols = []
            for i, col in enumerate(df.columns):
                col_str = str(col)
                if 'Concentration' in col_str and i > time_col_idx:
                    conc_cols.append(i)

            # Cumulative flux 컬럼들 찾기
            flux_cols = []
            for i, col in enumerate(df.columns):
                col_str = str(col)
                if 'Cumulative' in col_str and 'flux' in col_str.lower():
                    flux_cols.append(i)

            # 데이터 추출
            for idx in range(len(df)):
                time_val = time_data.iloc[idx]
                if pd.isna(time_val):
                    continue

                row_data = {
                    'time': time_val,
                    'drug_type': drug_type,
                    'MW': self.drug_properties[drug_type]['MW'],
                    'k_decay': k_decay,
                    **params
                }

                # 농도값 추출 (여러 위치의 평균 또는 개별)
                if conc_cols:
                    conc_values = df.iloc[idx, conc_cols].values
                    conc_numeric = pd.to_numeric(pd.Series(conc_values), errors='coerce')
                    row_data['concentration_mean'] = conc_numeric.mean()
                    row_data['concentration_std'] = conc_numeric.std()

                # Flux 값 추출
                if flux_cols:
                    flux_values = df.iloc[idx, flux_cols].values
                    flux_numeric = pd.to_numeric(pd.Series(flux_values), errors='coerce')
                    row_data['cumulative_flux'] = flux_numeric.mean()

                all_data.append(row_data)

        return pd.DataFrame(all_data)

    def load_injection_site_data(self, file_path: str = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Injection site results 파일 로드 - Case별 + Summary"""
        if file_path is None:
            file_path = self.data_dir / "251103 (revised)_Injection site results v2 (수정).xlsx"

        xl = pd.ExcelFile(file_path)

        # Summary 데이터 로드
        summary_df = pd.read_excel(file_path, sheet_name='Summary', header=None)

        # Summary 파싱 (Blood, Lymph, ECM 비율)
        summary_data = []
        for idx in range(len(summary_df)):
            row = summary_df.iloc[idx]
            condition = str(row[2]) if pd.notna(row[2]) else ''

            if condition and condition != 'nan' and 'NaN' not in condition:
                blood = row[3] if pd.notna(row[3]) else None
                lymph = row[4] if pd.notna(row[4]) else None
                ecm = row[5] if pd.notna(row[5]) else None

                if blood is not None and lymph is not None and ecm is not None:
                    summary_data.append({
                        'condition': condition.strip(),
                        'blood_ratio': float(blood),
                        'lymph_ratio': float(lymph),
                        'ecm_ratio': float(ecm)
                    })

        summary_result = pd.DataFrame(summary_data)

        # Case별 시계열 데이터 로드
        case_data = []
        for sheet_name in xl.sheet_names:
            if 'Case' not in sheet_name:
                continue

            df = pd.read_excel(file_path, sheet_name=sheet_name, header=None)

            # 파라미터 추출 (헤더에서)
            params = {}
            for col in df.iloc[0]:
                if pd.notna(col):
                    parsed = self.parse_column_parameters(str(col))
                    params.update(parsed)

            # Case 이름에서 파라미터 변화 정보 추출
            case_info = {
                'case_name': sheet_name,
                **params
            }

            # 시계열 데이터 추출
            time_col_idx = 1  # 일반적으로 두 번째 컬럼이 시간

            for idx in range(1, len(df)):
                time_val = df.iloc[idx, time_col_idx]
                if pd.isna(time_val) or not isinstance(time_val, (int, float)):
                    continue

                row_data = {
                    'time': float(time_val),
                    **case_info
                }
                case_data.append(row_data)

        case_result = pd.DataFrame(case_data) if case_data else pd.DataFrame()

        return summary_result, case_result

    def prepare_training_data(
        self,
        include_time_series: bool = True,
        include_final_ratios: bool = True,
        test_split: float = 0.2,
        random_state: int = 42
    ) -> Dict:
        """학습용 데이터 준비"""

        np.random.seed(random_state)

        # 데이터 로드
        sol765_df = self.load_sol765_data()
        summary_df, case_df = self.load_injection_site_data()

        print(f"sol765 데이터: {len(sol765_df)} 행")
        print(f"Summary 데이터: {len(summary_df)} 행")

        result = {}

        # 1. 시계열 농도 예측용 데이터
        if include_time_series and len(sol765_df) > 0:
            # 특성 컬럼
            feature_cols = ['time', 'MW', 'k_decay']
            optional_cols = ['sigma_ve', 'sigma_le', 'p_ve', 'p_le', 'D_gel', 'kf_m', 'kr_m']

            for col in optional_cols:
                if col in sol765_df.columns and sol765_df[col].notna().any():
                    feature_cols.append(col)

            # 약물 종류 원-핫 인코딩
            drug_dummies = pd.get_dummies(sol765_df['drug_type'], prefix='drug')

            # 특성 행렬
            X_ts = sol765_df[feature_cols].copy()
            X_ts = pd.concat([X_ts, drug_dummies], axis=1)

            # 결측치 처리
            X_ts = X_ts.fillna(X_ts.median())

            # 타겟 (농도)
            y_ts = sol765_df[['concentration_mean']].copy()
            y_ts = y_ts.fillna(0)

            # 유효한 데이터만 필터링
            valid_mask = X_ts.notna().all(axis=1) & y_ts.notna().all(axis=1)
            X_ts = X_ts[valid_mask].values
            y_ts = y_ts[valid_mask].values

            # Train/Test 분할
            n_samples = len(X_ts)
            indices = np.random.permutation(n_samples)
            split_idx = int(n_samples * (1 - test_split))

            train_idx = indices[:split_idx]
            test_idx = indices[split_idx:]

            # 스케일링
            X_train_ts = self.feature_scaler.fit_transform(X_ts[train_idx])
            X_test_ts = self.feature_scaler.transform(X_ts[test_idx])

            y_train_ts = y_ts[train_idx]
            y_test_ts = y_ts[test_idx]

            # 특성 이름 생성
            all_feature_names = feature_cols + list(drug_dummies.columns)

            result['time_series'] = {
                'X_train': X_train_ts,
                'X_test': X_test_ts,
                'y_train': y_train_ts,
                'y_test': y_test_ts,
                'feature_names': all_feature_names,
                'scaler': self.feature_scaler
            }

        # 2. 최종 비율 예측용 데이터 (Summary)
        if include_final_ratios and len(summary_df) > 0:
            # Condition을 파라미터로 변환 (예: low Lp -> Lp=-1, high Lp -> Lp=1)
            condition_features = []
            for _, row in summary_df.iterrows():
                cond = row['condition'].lower()
                feat = {
                    'Lp': 0, 'K': 0, 'sigma': 0, 'oncotic': 0, 'pBV': 0, 'D': 0
                }

                if 'lp' in cond:
                    feat['Lp'] = -1 if 'low' in cond else 1
                if 'k' in cond and 'oncotic' not in cond:
                    feat['K'] = -1 if 'low' in cond else 1
                if 'sigma' in cond or 'σ' in cond:
                    feat['sigma'] = -1 if 'low' in cond else 1
                if 'oncotic' in cond or 'pπ' in cond or 'π' in cond:
                    feat['oncotic'] = -1 if 'low' in cond else 1
                if 'pbv' in cond:
                    feat['pBV'] = -1 if 'low' in cond else 1
                if cond.endswith('d') or 'd)' in cond:
                    feat['D'] = -1 if 'low' in cond else 1
                if 'representative' in cond:
                    pass  # 모든 파라미터 0 (중간값)

                condition_features.append(feat)

            X_ratio = pd.DataFrame(condition_features).values
            y_ratio = summary_df[['blood_ratio', 'lymph_ratio', 'ecm_ratio']].values / 100.0  # 0-1로 정규화

            result['final_ratios'] = {
                'X': X_ratio,
                'y': y_ratio,
                'feature_names': ['Lp', 'K', 'sigma', 'oncotic', 'pBV', 'D'],
                'target_names': ['blood_ratio', 'lymph_ratio', 'ecm_ratio'],
                'conditions': summary_df['condition'].tolist()
            }

        return result

    def create_dataloaders(
        self,
        batch_size: int = 32,
        test_split: float = 0.2
    ) -> Dict[str, DataLoader]:
        """PyTorch DataLoader 생성"""

        data = self.prepare_training_data(test_split=test_split)
        loaders = {}

        if 'time_series' in data:
            ts = data['time_series']
            train_dataset = LymphChipDataset(
                ts['X_train'], ts['y_train'],
                feature_names=ts.get('feature_names'),
                target_names=['concentration']
            )
            test_dataset = LymphChipDataset(
                ts['X_test'], ts['y_test'],
                feature_names=ts.get('feature_names'),
                target_names=['concentration']
            )

            loaders['train_ts'] = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            loaders['test_ts'] = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        if 'final_ratios' in data:
            fr = data['final_ratios']
            ratio_dataset = LymphChipDataset(
                fr['X'], fr['y'],
                feature_names=fr.get('feature_names'),
                target_names=fr.get('target_names')
            )
            # 데이터가 적으므로 전체를 사용
            loaders['ratios'] = DataLoader(ratio_dataset, batch_size=len(fr['X']), shuffle=True)

        loaders['metadata'] = data

        return loaders


if __name__ == "__main__":
    # 테스트
    preprocessor = LymphChipPreprocessor()

    print("데이터 로드 테스트...")
    data = preprocessor.prepare_training_data()

    if 'time_series' in data:
        print(f"\n시계열 데이터:")
        print(f"  Train: {data['time_series']['X_train'].shape}")
        print(f"  Test: {data['time_series']['X_test'].shape}")

    if 'final_ratios' in data:
        print(f"\n최종 비율 데이터:")
        print(f"  X: {data['final_ratios']['X'].shape}")
        print(f"  y: {data['final_ratios']['y'].shape}")
        print(f"  Conditions: {data['final_ratios']['conditions']}")
