"""
림프칩 PINN 에이전트 서비스
Streamlit 기반 웹 애플리케이션 - 시간별 농도 변화 시각화
"""

import streamlit as st
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import math
import re
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import sys
sys.path.insert(0, str(Path(__file__).parent))

from data.preprocessor import LymphChipPreprocessor
from data.timeseries_preprocessor import TimeSeriesPreprocessor
from models.pinn import LymphChipPINN
from models.timeseries_pinn import TimeSeriesPINN, create_timeseries_model
from models.smooth_pinn import SmoothPINNv2, create_smooth_model
from models.interpolation_model import CaseInterpolator, create_interpolator_from_data
from models.losses import compute_metrics

# 데이터 경로
DATA_DIR = Path("/Users/guminhong/lympchip_agent")

# 페이지 설정
st.set_page_config(
    page_title="림프칩 PINN 에이전트",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS 스타일
st.markdown("""
<style>
    .main-header {
        font-size: 10rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: left;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.3rem;
        color: #666;
        text-align: left;
        margin-bottom: 2rem;
    }
    .result-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)


# ============== 파라미터 정의 ==============

PARAM_RANGES = {
    'Lp_ve': {
        'name': 'Lp (혈관 투과도)',
        'unit': 'm/(Pa·s)',
        'low': 4e-12,
        'mid': 8e-12,
        'high': 1.6e-11,
        'scale': 'log',
        'description': '혈관벽의 수력 투과도'
    },
    'K': {
        'name': 'K (수력 전도도)',
        'unit': 'm²',
        'low': 1e-17,
        'mid': 1e-15,
        'high': 1e-13,
        'scale': 'log',
        'description': 'ECM의 수력 전도도'
    },
    'P_oncotic': {
        'name': 'Pπ (삼투압)',
        'unit': 'Pa',
        'low': 3145,
        'mid': 3590,
        'high': 3815,
        'scale': 'linear',
        'description': '혈장 삼투압'
    },
    'sigma_ve': {
        'name': 'σ (반사 계수)',
        'unit': '무차원',
        'low': 0.1,
        'mid': 0.5,
        'high': 0.9,
        'scale': 'linear',
        'description': '용질의 막 투과 반사 계수 (분자량에 따라 결정)'
    },
    'D_gel': {
        'name': 'D (확산 계수)',
        'unit': 'm²/s',
        'low': 1e-11,
        'mid': 3e-11,
        'high': 1e-10,
        'scale': 'log',
        'description': 'ECM 내 약물 확산 계수'
    },
    'kdecay': {
        'name': 'kdecay (분해 속도)',
        'unit': '1/s',
        'low': 0,
        'mid': 1.7e-6,
        'high': 1.5e-5,
        'scale': 'log_zero',  # 0을 허용하는 특수 로그 스케일
        'description': '약물 분해 속도 상수 (0=분해 없음)'
    },
    'MW': {
        'name': 'MW (분자량)',
        'unit': 'kDa',
        'low': 5.8,
        'mid': 66.5,
        'high': 150,
        'scale': 'log',
        'description': '약물 분자량 (INS=5.8, ALB=66.5, IgG=150 kDa)'
    }
}

# 파라미터 순서 (보간기와 일치 - MW 추가)
PARAM_ORDER = ['Lp_ve', 'K', 'P_oncotic', 'sigma_ve', 'D_gel', 'kdecay', 'MW']

# 시계열 예측용 파라미터 순서 (보간기와 동일)
TS_PARAM_ORDER = ['Lp_ve', 'K', 'P_oncotic', 'sigma_ve', 'D_gel', 'kdecay', 'MW']


# ============== 데이터 로드 함수 ==============

@st.cache_data
def load_case_time_series():
    """Case별 시계열 데이터 로드 (실제 시뮬레이션 결과)

    Excel 구조:
    - Column 14: Time (h)
    - Column 15: Total Mass (%)
    - Column 16: Lymph (%)
    - Column 17: Blood (%)
    - Column 18: Decay
    - Column 19: ECM (%)

    중요: 일부 시트에는 여러 데이터셋이 있으므로
    처음 72시간에 도달하는 지점까지만 사용 (시간이 감소하면 중단)
    """
    file_path = DATA_DIR / "251103 (revised)_Injection site results v2 (수정).xlsx"

    if not file_path.exists():
        return None

    xl = pd.ExcelFile(file_path)

    case_data = {}

    for sheet_name in xl.sheet_names:
        if 'Case' not in sheet_name:
            continue

        df = pd.read_excel(file_path, sheet_name=sheet_name, header=None)

        # 파라미터 추출 (column 2의 헤더에서)
        params = {}
        for col in df.iloc[0]:
            if pd.notna(col):
                col_str = str(col)
                for match in re.finditer(r'(\w+)=([0-9.E+-]+)', col_str):
                    try:
                        params[match.group(1)] = float(match.group(2))
                    except ValueError:
                        pass

        # 시계열 데이터 추출 (columns 14-19)
        # Column 14: Time (h), Column 16: Lymph, Column 17: Blood, Column 19: ECM
        # 처음 72시간에 도달하는 지점까지만 사용 (시간이 감소하면 다른 데이터셋 시작)
        time_series_data = []
        prev_time = -1

        for idx in range(1, len(df)):
            row = df.iloc[idx]

            # Time (h) - column 14
            time_val = row[14]
            if pd.isna(time_val):
                continue

            try:
                time_hour = float(time_val)

                # 시간이 감소하면 다른 데이터셋 시작 → 중단
                if time_hour < prev_time - 0.1:
                    break

                # 72시간 초과하면 중단
                if time_hour > 72:
                    break

                prev_time = time_hour

                # Extract Blood, Lymph, ECM, Decay percentages
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
                    break

            except (ValueError, TypeError):
                continue

        if time_series_data:
            ts_df = pd.DataFrame(time_series_data)
            ts_df = ts_df.sort_values('time_hour').reset_index(drop=True)

            case_data[sheet_name] = {
                'params': params,
                'time_series': ts_df
            }

    return case_data


@st.cache_data
def load_simulation_summary():
    """Summary 데이터 로드"""
    file_path = DATA_DIR / "251103 (revised)_Injection site results v2 (수정).xlsx"

    if not file_path.exists():
        return None

    df = pd.read_excel(file_path, sheet_name='Summary', header=None)

    summary_data = []
    for idx in range(len(df)):
        row = df.iloc[idx]
        condition = str(row[2]) if pd.notna(row[2]) else ''

        if condition and condition != 'nan' and 'NaN' not in condition:
            blood = row[3] if pd.notna(row[3]) else None
            lymph = row[4] if pd.notna(row[4]) else None
            ecm = row[5] if pd.notna(row[5]) else None

            if blood is not None and lymph is not None and ecm is not None:
                summary_data.append({
                    'condition': condition.strip(),
                    'Blood': float(blood),
                    'Lymph': float(lymph),
                    'ECM': float(ecm)
                })

    return pd.DataFrame(summary_data)


def generate_time_series_prediction(params_normalized, total_hours=72):
    """
    PINN 모델 기반 시계열 예측 생성
    파라미터에 따른 시간별 Blood/Lymph/ECM 비율 변화 시뮬레이션
    """
    # 시간 포인트 생성 (0-72시간)
    time_points = np.linspace(0, total_hours, 200)

    # 최종 비율 (모델 예측 또는 추정)
    # 간단한 물리 기반 모델로 시뮬레이션
    Lp, K, sigma, oncotic, pBV, D = params_normalized

    # 기본 비율 설정 (Representative 기준)
    base_blood = 18.4
    base_lymph = 47.9
    base_ecm = 33.6

    # 파라미터 영향 계산
    blood_factor = 1 + 0.3 * pBV + 0.1 * sigma
    lymph_factor = 1 + 0.3 * Lp - 0.1 * oncotic

    final_blood = base_blood * blood_factor
    final_lymph = base_lymph * lymph_factor
    final_ecm = 100 - final_blood - final_lymph

    # 정규화
    total = final_blood + final_lymph + final_ecm
    final_blood = final_blood / total * 100
    final_lymph = final_lymph / total * 100
    final_ecm = final_ecm / total * 100

    # 시간에 따른 변화 곡선 생성 (지수 함수 기반)
    # 확산 계수에 따른 속도 조절
    rate_factor = 1 + 0.5 * D  # D가 높으면 빠르게 평형 도달

    blood_curve = final_blood * (1 - np.exp(-rate_factor * time_points / 20))
    lymph_curve = final_lymph * (1 - np.exp(-rate_factor * time_points / 15))
    ecm_curve = 100 - blood_curve - lymph_curve

    return pd.DataFrame({
        'time_hour': time_points,
        'Blood': blood_curve,
        'Lymph': lymph_curve,
        'ECM': ecm_curve
    })


# ============== 유틸리티 함수 ==============

def normalize_param(value, param_name):
    """실제 파라미터 값을 -1 ~ 1로 정규화"""
    config = PARAM_RANGES[param_name]
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
        # 0 -> -1, mid -> 0, high -> 1
        if value <= 0 or value < 1e-10:
            normalized = -1.0
        else:
            log_val = math.log10(value)
            log_mid = math.log10(mid)
            log_high = math.log10(high)

            if value <= mid:
                # 0과 mid 사이: 선형 보간 (-1 ~ 0)
                normalized = -1.0 + (value / mid)
            else:
                # mid와 high 사이: 로그 스케일 (0 ~ 1)
                normalized = (log_val - log_mid) / (log_high - log_mid)
    else:
        if value <= mid:
            normalized = (value - mid) / (mid - low)
        else:
            normalized = (value - mid) / (high - mid)

    return max(-1, min(1, normalized))


def format_scientific(value):
    """과학적 표기법으로 포맷"""
    if abs(value) < 0.001 or abs(value) >= 10000:
        return f"{value:.2e}"
    else:
        return f"{value:.4f}"


# ============== 세션 상태 ==============

def init_session_state():
    if 'model' not in st.session_state:
        st.session_state.model = None
    if 'ts_model' not in st.session_state:
        st.session_state.ts_model = None
    if 'ts_checkpoint' not in st.session_state:
        st.session_state.ts_checkpoint = None
    if 'interpolator' not in st.session_state:
        st.session_state.interpolator = None
    if 'trained' not in st.session_state:
        st.session_state.trained = False
    if 'ts_trained' not in st.session_state:
        st.session_state.ts_trained = False


def load_pretrained_model():
    """저장된 비율 예측 모델 로드"""
    checkpoint_path = Path(__file__).parent / "checkpoints" / "final_model.pt"

    if not checkpoint_path.exists():
        return None

    try:
        model = LymphChipPINN(
            input_dim=6,
            hidden_dim=64,
            num_layers=3,
            output_dim=3,
            output_type='ratios',
            use_time_encoding=False
        )

        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        return model
    except Exception as e:
        return None


def load_timeseries_model():
    """시계열 PINN 모델 로드 (부드러운 모델 우선)"""

    # 1. 먼저 부드러운 모델 시도
    smooth_path = Path(__file__).parent / "checkpoints" / "smooth_pinn.pt"

    if smooth_path.exists():
        try:
            checkpoint = torch.load(smooth_path, map_location='cpu')
            config = checkpoint['model_config']

            model = create_smooth_model(
                model_type=config['model_type'],
                param_dim=config['param_dim'],
                hidden_dim=config['hidden_dim'],
                num_layers=config['num_layers']
            )

            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()

            return model, checkpoint
        except Exception as e:
            pass  # 실패하면 기존 모델 시도

    # 2. 기존 시계열 모델
    checkpoint_path = Path(__file__).parent / "checkpoints" / "timeseries_pinn.pt"

    if not checkpoint_path.exists():
        return None, None

    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        config = checkpoint['model_config']

        model = create_timeseries_model(
            model_type=config['model_type'],
            param_dim=config['param_dim'],
            hidden_dim=config['hidden_dim'],
            num_layers=config['num_layers'],
            num_time_freq=config['num_time_freq']
        )

        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        return model, checkpoint
    except Exception as e:
        st.error(f"시계열 모델 로드 실패: {e}")
        return None, None


def predict_timeseries_pinn(model, params_normalized, time_max=72.0, n_points=100):
    """
    시계열 PINN 모델로 0-72시간 예측

    Args:
        model: 시계열 PINN 모델
        params_normalized: 정규화된 파라미터 리스트 [Lp_ve, K, P_oncotic, sigma_ve, D_gel]
        time_max: 최대 시간 (hours)
        n_points: 시간 포인트 수

    Returns:
        pd.DataFrame with [time_hour, Blood, Lymph, ECM]
    """
    model.eval()

    # 시간 포인트 생성 (0-1로 정규화)
    time_normalized = np.linspace(0, 1, n_points)
    time_hours = time_normalized * time_max

    # 입력 생성: [time, params...]
    params_array = np.array(params_normalized)
    inputs = []
    for t in time_normalized:
        inputs.append(np.concatenate([[t], params_array]))

    inputs = torch.FloatTensor(inputs)

    with torch.no_grad():
        predictions = model(inputs).numpy()

    # 비율을 퍼센트로 변환
    return pd.DataFrame({
        'time_hour': time_hours,
        'Blood': predictions[:, 0] * 100,
        'Lymph': predictions[:, 1] * 100,
        'ECM': predictions[:, 2] * 100
    })


def predict_with_interpolator(interpolator, params_normalized, time_max=72.0, n_points=100):
    """
    보간 모델로 0-72시간 예측 (가장 정확)

    Args:
        interpolator: CaseInterpolator 객체
        params_normalized: 정규화된 파라미터 리스트 [Lp_ve, K, P_oncotic, sigma_ve, D_gel, kdecay]
        time_max: 최대 시간 (hours)
        n_points: 시간 포인트 수

    Returns:
        pd.DataFrame with [time_hour, Blood, Lymph, ECM, Decay]
    """
    time_hours = np.linspace(0, time_max, n_points)
    params_array = np.array(params_normalized)

    # 호환성: 보간기가 5개 파라미터만 기대하는 경우 kdecay 제외
    expected_dim = len(interpolator.param_names)
    if len(params_array) > expected_dim:
        params_array = params_array[:expected_dim]
    elif len(params_array) < expected_dim:
        # 파라미터가 부족하면 0으로 패딩
        params_array = np.concatenate([params_array, np.zeros(expected_dim - len(params_array))])

    predictions = interpolator.predict(params_array, time_hours)

    # 호환성: 3열(이전 버전) 또는 4열(새 버전) 처리
    if predictions.shape[1] >= 4:
        decay = predictions[:, 3]
    else:
        decay = np.zeros(len(time_hours))

    return pd.DataFrame({
        'time_hour': time_hours,
        'Blood': predictions[:, 0],
        'Lymph': predictions[:, 1],
        'ECM': predictions[:, 2],
        'Decay': decay
    })


INTERPOLATOR_VERSION = "v2.1_MW"  # 버전 변경 시 캐시 갱신 (MW 파라미터 추가)

@st.cache_resource(show_spinner="보간기 로딩 중...")
def load_interpolator(version=INTERPOLATOR_VERSION):
    """보간기 로드 (캐시됨)"""
    try:
        interpolator = create_interpolator_from_data(str(DATA_DIR))
        return interpolator
    except Exception as e:
        st.error(f"보간기 로드 실패: {e}")
        return None


def predict_ratios(model, params_normalized):
    """정규화된 파라미터로 비율 예측"""
    model.eval()
    with torch.no_grad():
        x = torch.FloatTensor([params_normalized])
        pred = model(x).numpy()[0]
    return {
        'Blood': pred[0] * 100,
        'Lymph': pred[1] * 100,
        'ECM': pred[2] * 100
    }


# ============== 차트 함수 ==============

def create_time_series_chart(df, title="시간에 따른 농도 변화"):
    """시간별 농도 변화 라인 차트"""
    fig = go.Figure()

    # Blood - 빨강
    fig.add_trace(go.Scatter(
        x=df['time_hour'],
        y=df['Blood'],
        mode='lines',
        name='Blood',
        line=dict(color='#e74c3c', width=3),
        fill='tozeroy',
        fillcolor='rgba(231, 76, 60, 0.1)'
    ))

    # Lymph - 초록
    fig.add_trace(go.Scatter(
        x=df['time_hour'],
        y=df['Lymph'],
        mode='lines',
        name='Lymph',
        line=dict(color='#27ae60', width=3),
        fill='tozeroy',
        fillcolor='rgba(39, 174, 96, 0.1)'
    ))

    # ECM - 주황
    fig.add_trace(go.Scatter(
        x=df['time_hour'],
        y=df['ECM'],
        mode='lines',
        name='ECM',
        line=dict(color='#f39c12', width=3),
        fill='tozeroy',
        fillcolor='rgba(243, 156, 18, 0.1)'
    ))

    # Decay - 파랑 (있는 경우만)
    if 'Decay' in df.columns and df['Decay'].sum() > 0.1:
        fig.add_trace(go.Scatter(
            x=df['time_hour'],
            y=df['Decay'],
            mode='lines',
            name='Decay',
            line=dict(color='#3498db', width=3),
            fill='tozeroy',
            fillcolor='rgba(52, 152, 219, 0.1)'
        ))

    fig.update_layout(
        title=title,
        xaxis_title='시간 (hours)',
        yaxis_title='비율 (%)',
        height=500,
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        xaxis=dict(
            range=[0, 72],
            dtick=12,
            gridcolor='rgba(128,128,128,0.2)'
        ),
        yaxis=dict(
            range=[0, 100],
            gridcolor='rgba(128,128,128,0.2)'
        )
    )

    return fig


def create_comparison_chart(original_df, predicted_df, data_source, case_name):
    """원본 시뮬레이션과 PINN 예측 비교 차트"""
    fig = go.Figure()

    # Decay가 유의미한지 확인
    has_decay_original = 'Decay' in original_df.columns and original_df['Decay'].sum() > 0.1
    has_decay_predicted = 'Decay' in predicted_df.columns and predicted_df['Decay'].sum() > 0.1

    if data_source == "원본 시뮬레이션" or data_source == "둘 다 비교":
        # 원본 데이터 - 실선
        fig.add_trace(go.Scatter(
            x=original_df['time_hour'],
            y=original_df['Blood'],
            mode='lines',
            name='Blood (원본)',
            line=dict(color='#e74c3c', width=3),  # 빨강
            legendgroup='original'
        ))
        fig.add_trace(go.Scatter(
            x=original_df['time_hour'],
            y=original_df['Lymph'],
            mode='lines',
            name='Lymph (원본)',
            line=dict(color='#27ae60', width=3),  # 초록
            legendgroup='original'
        ))
        fig.add_trace(go.Scatter(
            x=original_df['time_hour'],
            y=original_df['ECM'],
            mode='lines',
            name='ECM (원본)',
            line=dict(color='#f39c12', width=3),  # 주황
            legendgroup='original'
        ))
        if has_decay_original:
            fig.add_trace(go.Scatter(
                x=original_df['time_hour'],
                y=original_df['Decay'],
                mode='lines',
                name='Decay (원본)',
                line=dict(color='#3498db', width=3),  # 파랑
                legendgroup='original'
            ))

    if data_source == "PINN 예측" or data_source == "둘 다 비교":
        # 예측 데이터 - 점선
        dash_style = 'dash' if data_source == "둘 다 비교" else 'solid'
        opacity = 0.7 if data_source == "둘 다 비교" else 1.0

        fig.add_trace(go.Scatter(
            x=predicted_df['time_hour'],
            y=predicted_df['Blood'],
            mode='lines',
            name='Blood (PINN)',
            line=dict(color='#e74c3c', width=2, dash=dash_style),  # 빨강
            opacity=opacity,
            legendgroup='predicted'
        ))
        fig.add_trace(go.Scatter(
            x=predicted_df['time_hour'],
            y=predicted_df['Lymph'],
            mode='lines',
            name='Lymph (PINN)',
            line=dict(color='#27ae60', width=2, dash=dash_style),  # 초록
            opacity=opacity,
            legendgroup='predicted'
        ))
        fig.add_trace(go.Scatter(
            x=predicted_df['time_hour'],
            y=predicted_df['ECM'],
            mode='lines',
            name='ECM (PINN)',
            line=dict(color='#f39c12', width=2, dash=dash_style),  # 주황
            opacity=opacity,
            legendgroup='predicted'
        ))
        if has_decay_predicted:
            fig.add_trace(go.Scatter(
                x=predicted_df['time_hour'],
                y=predicted_df['Decay'],
                mode='lines',
                name='Decay (PINN)',
                line=dict(color='#3498db', width=2, dash=dash_style),  # 파랑
                opacity=opacity,
                legendgroup='predicted'
            ))

    title = f"Blood / Lymph / ECM / Decay 농도 변화 - {case_name}"
    if data_source == "둘 다 비교":
        title += " (실선: 원본, 점선: PINN)"

    fig.update_layout(
        title=title,
        xaxis_title='시간 (hours)',
        yaxis_title='비율 (%)',
        height=500,
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        xaxis=dict(
            range=[0, 72],
            dtick=12,
            gridcolor='rgba(128,128,128,0.2)'
        ),
        yaxis=dict(
            range=[0, 100],
            gridcolor='rgba(128,128,128,0.2)'
        )
    )

    return fig


def create_stacked_area_chart(df, title="누적 분포 변화"):
    """스택 영역 차트"""
    fig = go.Figure()

    # Blood - 빨강
    fig.add_trace(go.Scatter(
        x=df['time_hour'],
        y=df['Blood'],
        mode='lines',
        name='Blood',
        stackgroup='one',
        fillcolor='rgba(231, 76, 60, 0.7)',
        line=dict(color='#e74c3c', width=0.5)
    ))

    # Lymph - 초록
    fig.add_trace(go.Scatter(
        x=df['time_hour'],
        y=df['Lymph'],
        mode='lines',
        name='Lymph',
        stackgroup='one',
        fillcolor='rgba(39, 174, 96, 0.7)',
        line=dict(color='#27ae60', width=0.5)
    ))

    # ECM - 주황
    fig.add_trace(go.Scatter(
        x=df['time_hour'],
        y=df['ECM'],
        mode='lines',
        name='ECM',
        stackgroup='one',
        fillcolor='rgba(243, 156, 18, 0.7)',
        line=dict(color='#f39c12', width=0.5)
    ))

    fig.update_layout(
        title=title,
        xaxis_title='시간 (hours)',
        yaxis_title='비율 (%)',
        height=400,
        hovermode='x unified',
        xaxis=dict(range=[0, 72], dtick=12),
        yaxis=dict(range=[0, 100])
    )

    return fig


def create_final_ratio_chart(predictions):
    """최종 비율 도넛 차트"""
    # 색상: Blood=빨강, Lymph=초록, ECM=주황
    fig = go.Figure(data=[go.Pie(
        labels=['Blood', 'Lymph', 'ECM'],
        values=[predictions['Blood'], predictions['Lymph'], predictions['ECM']],
        hole=0.5,
        marker_colors=['#e74c3c', '#27ae60', '#f39c12'],
        textinfo='label+percent',
        textfont_size=14
    )])

    fig.update_layout(
        height=350,
        margin=dict(t=20, b=20, l=20, r=20),
        showlegend=False
    )

    return fig


def calculate_auc(df: pd.DataFrame) -> dict:
    """
    시계열 데이터의 AUC (Area Under Curve) 계산
    사다리꼴 적분법 사용

    Args:
        df: DataFrame with columns ['time_hour', 'Blood', 'Lymph', 'ECM', 'Decay']

    Returns:
        dict with AUC values for each component
    """
    time = df['time_hour'].values

    # 사다리꼴 적분 (numpy trapz 사용)
    auc_blood = np.trapz(df['Blood'].values, time)
    auc_lymph = np.trapz(df['Lymph'].values, time)
    auc_ecm = np.trapz(df['ECM'].values, time)
    auc_decay = np.trapz(df['Decay'].values, time) if 'Decay' in df.columns else 0

    return {
        'Blood': auc_blood,
        'Lymph': auc_lymph,
        'ECM': auc_ecm,
        'Decay': auc_decay
    }


def create_auc_chart(auc_values: dict, title: str = "AUC 분포"):
    """AUC 기반 도넛 차트"""
    auc_decay = auc_values.get('Decay', 0)
    has_decay = auc_decay > 0.1

    # 색상: Blood=빨강, Lymph=초록, ECM=주황, Decay=파랑
    if has_decay:
        total_auc = auc_values['Blood'] + auc_values['Lymph'] + auc_values['ECM'] + auc_decay
        pct_blood = auc_values['Blood'] / total_auc * 100
        pct_lymph = auc_values['Lymph'] / total_auc * 100
        pct_ecm = auc_values['ECM'] / total_auc * 100
        pct_decay = auc_decay / total_auc * 100

        fig = go.Figure(data=[go.Pie(
            labels=['Blood', 'Lymph', 'ECM', 'Decay'],
            values=[pct_blood, pct_lymph, pct_ecm, pct_decay],
            hole=0.5,
            marker_colors=['#e74c3c', '#27ae60', '#f39c12', '#3498db'],
            textinfo='label+percent',
            textfont_size=14,
            hovertemplate="<b>%{label}</b><br>AUC: %{customdata:.1f} %·h<br>비율: %{percent}<extra></extra>",
            customdata=[auc_values['Blood'], auc_values['Lymph'], auc_values['ECM'], auc_decay]
        )])
    else:
        total_auc = auc_values['Blood'] + auc_values['Lymph'] + auc_values['ECM']
        pct_blood = auc_values['Blood'] / total_auc * 100
        pct_lymph = auc_values['Lymph'] / total_auc * 100
        pct_ecm = auc_values['ECM'] / total_auc * 100

        fig = go.Figure(data=[go.Pie(
            labels=['Blood', 'Lymph', 'ECM'],
            values=[pct_blood, pct_lymph, pct_ecm],
            hole=0.5,
            marker_colors=['#e74c3c', '#27ae60', '#f39c12'],
            textinfo='label+percent',
            textfont_size=14,
            hovertemplate="<b>%{label}</b><br>AUC: %{customdata:.1f} %·h<br>비율: %{percent}<extra></extra>",
            customdata=[auc_values['Blood'], auc_values['Lymph'], auc_values['ECM']]
        )])

    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor='center'),
        height=350,
        margin=dict(t=50, b=20, l=20, r=20),
        showlegend=False
    )

    return fig


# ============== 메인 앱 ==============

def main():
    init_session_state()

    st.markdown('<h1 style="font-size: 4rem; font-weight: bold; color: #1E88E5; text-align: left; margin-bottom: 0;">🧬 림프칩 PINN 에이전트</h1>', unsafe_allow_html=True)
    st.markdown('<p style="font-size: 1.5rem; color: #666; text-align: left; margin-bottom: 2rem;">0~72시간 약물 분포 시뮬레이션</p>', unsafe_allow_html=True)

    # 사이드바
    st.sidebar.title("⚙️ 메뉴")
    page = st.sidebar.radio(
        "페이지 선택",
        ["📈 시간별 농도 변화", "🎯 파라미터 입력", "ℹ️ 도움말"]
    )

    # 캐시 초기화 버튼
    st.sidebar.markdown("---")
    if st.sidebar.button("🔄 데이터 새로고침", help="캐시된 데이터를 모두 새로고침합니다"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.session_state.interpolator = None
        st.session_state.ts_model = None
        st.session_state.model = None
        st.rerun()

    # 모델 로드 (UI 표시 없음)
    if st.session_state.model is None:
        model = load_pretrained_model()
        if model:
            st.session_state.model = model
            st.session_state.trained = True

    if st.session_state.interpolator is None:
        st.session_state.interpolator = load_interpolator()

    if st.session_state.ts_model is None:
        ts_model, ts_checkpoint = load_timeseries_model()
        if ts_model:
            st.session_state.ts_model = ts_model
            st.session_state.ts_checkpoint = ts_checkpoint
            st.session_state.ts_trained = True

    # 페이지 라우팅
    if page == "📈 시간별 농도 변화":
        page_time_series()
    elif page == "🎯 파라미터 입력":
        page_parameter_input()
    elif page == "ℹ️ 도움말":
        page_help()


def page_time_series():
    """시간별 농도 변화 페이지 - 메인 기능"""
    st.header("📈 시간별 농도 변화 (0~72시간)")

    st.markdown("**원본 시뮬레이션 데이터**와 **PINN 예측**을 비교합니다.")

    # 데이터 로드
    case_data = load_case_time_series()

    if case_data is None or len(case_data) == 0:
        st.error("시뮬레이션 데이터를 로드할 수 없습니다.")
        return

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📁 시뮬레이션 케이스 선택")

        # Case 선택
        case_names = list(case_data.keys())
        selected_case = st.selectbox(
            "케이스 선택",
            case_names,
            index=0,
            help="각 케이스는 다른 파라미터 조합을 가집니다"
        )

        # 선택된 케이스의 파라미터 표시
        if selected_case:
            params = case_data[selected_case]['params']
            st.markdown("**케이스 파라미터:**")

            param_display = []
            for key, value in params.items():
                param_display.append(f"- **{key}**: {format_scientific(value)}")
            st.markdown("\n".join(param_display))

        st.markdown("---")

        # 데이터 소스 선택
        st.subheader("📊 표시할 데이터")

        data_source = st.radio(
            "데이터 소스",
            ["원본 시뮬레이션", "PINN 예측", "둘 다 비교"],
            index=2,
            horizontal=False
        )

    with col2:
        # 선택된 케이스의 시계열 데이터
        original_data = case_data[selected_case]['time_series']
        params = case_data[selected_case]['params']

        # PINN 예측 생성 (선택된 케이스의 파라미터 기반)
        # 시계열 PINN 파라미터 순서: ['Lp_ve', 'K', 'P_oncotic', 'sigma_ve', 'D_gel', 'kdecay']

        params_normalized = []
        for param_name in TS_PARAM_ORDER:
            if param_name in params:
                normalized = normalize_param(params[param_name], param_name)
            elif param_name == 'kdecay':
                normalized = -1  # 기본값: decay 없음
            else:
                normalized = 0  # 중간값
            params_normalized.append(normalized)

        # 예측 모델 우선순위: 보간기 > PINN > 근사함수
        if st.session_state.interpolator is not None:
            predicted_data = predict_with_interpolator(
                st.session_state.interpolator,
                params_normalized,
                time_max=72.0,
                n_points=100
            )
        elif st.session_state.ts_model is not None:
            predicted_data = predict_timeseries_pinn(
                st.session_state.ts_model,
                params_normalized,
                time_max=72.0,
                n_points=100
            )
        else:
            # 폴백: 기존 근사 함수 사용
            predicted_data = generate_time_series_prediction(params_normalized)

        # 탭으로 차트 구성 (누적 영역 제거)
        tab1, tab2 = st.tabs(["📈 라인 차트", "📋 데이터 테이블"])

        with tab1:
            fig = create_comparison_chart(original_data, predicted_data, data_source, selected_case)
            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            st.markdown("**주요 시점별 분포 (원본 시뮬레이션)**")

            # 주요 시점 데이터 추출
            key_times = [0, 6, 12, 24, 48, 72]
            key_data = []

            for t in key_times:
                # 가장 가까운 시점 찾기
                closest_idx = (original_data['time_hour'] - t).abs().idxmin()
                row = original_data.iloc[closest_idx]
                key_data.append({
                    '시간': f"{t}h",
                    'Blood (%)': f"{row['Blood']:.2f}",
                    'Lymph (%)': f"{row['Lymph']:.2f}",
                    'ECM (%)': f"{row['ECM']:.2f}"
                })

            st.dataframe(pd.DataFrame(key_data), use_container_width=True, hide_index=True)

        # 원본 데이터 최종값
        final_original = original_data.iloc[-1]
        final_predicted = predicted_data.iloc[-1]

        # 72시간 후 최종 분포 (파라미터 입력과 동일한 형식)
        st.subheader("🎯 72시간 후 최종 분포")

        # Decay가 유의미한지 확인
        has_decay_orig = 'Decay' in final_original and final_original['Decay'] > 0.1
        has_decay_pred = 'Decay' in final_predicted and final_predicted['Decay'] > 0.1

        if data_source == "둘 다 비교":
            st.markdown("**원본 시뮬레이션:**")
            cols = st.columns(4 if has_decay_orig else 3)
            with cols[0]:
                st.markdown(f"""
                <div class="result-box" style="background: linear-gradient(135deg, #e74c3c, #c0392b);">
                    <h2 style="margin:0; font-size:2rem;">{final_original['Blood']:.1f}%</h2>
                    <p style="margin:0;">🔴 Blood</p>
                </div>
                """, unsafe_allow_html=True)
            with cols[1]:
                st.markdown(f"""
                <div class="result-box" style="background: linear-gradient(135deg, #27ae60, #1e8449);">
                    <h2 style="margin:0; font-size:2rem;">{final_original['Lymph']:.1f}%</h2>
                    <p style="margin:0;">🟢 Lymph</p>
                </div>
                """, unsafe_allow_html=True)
            with cols[2]:
                st.markdown(f"""
                <div class="result-box" style="background: linear-gradient(135deg, #f39c12, #e67e22);">
                    <h2 style="margin:0; font-size:2rem;">{final_original['ECM']:.1f}%</h2>
                    <p style="margin:0;">🟠 ECM</p>
                </div>
                """, unsafe_allow_html=True)
            if has_decay_orig:
                with cols[3]:
                    st.markdown(f"""
                    <div class="result-box" style="background: linear-gradient(135deg, #3498db, #2980b9);">
                        <h2 style="margin:0; font-size:2rem;">{final_original['Decay']:.1f}%</h2>
                        <p style="margin:0;">🔵 Decay</p>
                    </div>
                    """, unsafe_allow_html=True)

            st.markdown("**PINN 예측:**")
            cols = st.columns(4 if has_decay_pred else 3)
            with cols[0]:
                delta = final_predicted['Blood'] - final_original['Blood']
                st.markdown(f"""
                <div class="result-box" style="background: linear-gradient(135deg, #e74c3c, #c0392b); opacity: 0.85;">
                    <h2 style="margin:0; font-size:2rem;">{final_predicted['Blood']:.1f}%</h2>
                    <p style="margin:0;">🔴 Blood ({delta:+.1f}%)</p>
                </div>
                """, unsafe_allow_html=True)
            with cols[1]:
                delta = final_predicted['Lymph'] - final_original['Lymph']
                st.markdown(f"""
                <div class="result-box" style="background: linear-gradient(135deg, #27ae60, #1e8449); opacity: 0.85;">
                    <h2 style="margin:0; font-size:2rem;">{final_predicted['Lymph']:.1f}%</h2>
                    <p style="margin:0;">🟢 Lymph ({delta:+.1f}%)</p>
                </div>
                """, unsafe_allow_html=True)
            with cols[2]:
                delta = final_predicted['ECM'] - final_original['ECM']
                st.markdown(f"""
                <div class="result-box" style="background: linear-gradient(135deg, #f39c12, #e67e22); opacity: 0.85;">
                    <h2 style="margin:0; font-size:2rem;">{final_predicted['ECM']:.1f}%</h2>
                    <p style="margin:0;">🟠 ECM ({delta:+.1f}%)</p>
                </div>
                """, unsafe_allow_html=True)
            if has_decay_pred:
                with cols[3]:
                    delta = final_predicted['Decay'] - (final_original['Decay'] if has_decay_orig else 0)
                    st.markdown(f"""
                    <div class="result-box" style="background: linear-gradient(135deg, #3498db, #2980b9); opacity: 0.85;">
                        <h2 style="margin:0; font-size:2rem;">{final_predicted['Decay']:.1f}%</h2>
                        <p style="margin:0;">🔵 Decay ({delta:+.1f}%)</p>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            final_row = final_original if data_source == "원본 시뮬레이션" else final_predicted
            has_decay = 'Decay' in final_row and final_row['Decay'] > 0.1
            cols = st.columns(4 if has_decay else 3)
            with cols[0]:
                st.markdown(f"""
                <div class="result-box" style="background: linear-gradient(135deg, #e74c3c, #c0392b);">
                    <h2 style="margin:0; font-size:2rem;">{final_row['Blood']:.1f}%</h2>
                    <p style="margin:0;">🔴 Blood</p>
                </div>
                """, unsafe_allow_html=True)
            with cols[1]:
                st.markdown(f"""
                <div class="result-box" style="background: linear-gradient(135deg, #27ae60, #1e8449);">
                    <h2 style="margin:0; font-size:2rem;">{final_row['Lymph']:.1f}%</h2>
                    <p style="margin:0;">🟢 Lymph</p>
                </div>
                """, unsafe_allow_html=True)
            with cols[2]:
                st.markdown(f"""
                <div class="result-box" style="background: linear-gradient(135deg, #f39c12, #e67e22);">
                    <h2 style="margin:0; font-size:2rem;">{final_row['ECM']:.1f}%</h2>
                    <p style="margin:0;">🟠 ECM</p>
                </div>
                """, unsafe_allow_html=True)
            if has_decay:
                with cols[3]:
                    st.markdown(f"""
                    <div class="result-box" style="background: linear-gradient(135deg, #3498db, #2980b9);">
                        <h2 style="margin:0; font-size:2rem;">{final_row['Decay']:.1f}%</h2>
                        <p style="margin:0;">🔵 Decay</p>
                    </div>
                    """, unsafe_allow_html=True)

        # AUC 계산 및 파이 차트
        st.subheader("📊 AUC (Area Under Curve) 분포")

        if data_source == "둘 다 비교":
            auc_col1, auc_col2 = st.columns(2)
            with auc_col1:
                auc_original = calculate_auc(original_data)
                fig = create_auc_chart(auc_original, "원본 시뮬레이션 AUC")
                st.plotly_chart(fig, use_container_width=True)
            with auc_col2:
                auc_predicted = calculate_auc(predicted_data)
                fig = create_auc_chart(auc_predicted, "PINN 예측 AUC")
                st.plotly_chart(fig, use_container_width=True)
        else:
            auc_data = original_data if data_source == "원본 시뮬레이션" else predicted_data
            auc_values = calculate_auc(auc_data)
            title = "원본 시뮬레이션 AUC" if data_source == "원본 시뮬레이션" else "PINN 예측 AUC"
            fig = create_auc_chart(auc_values, title)
            st.plotly_chart(fig, use_container_width=True)


def page_parameter_input():
    """파라미터 직접 입력 및 즉시 예측"""
    st.header("🎯 파라미터 입력")

    # 보간기 또는 모델 필요
    if st.session_state.interpolator is None and st.session_state.model is None:
        st.warning("⚠️ 모델이 로드되지 않았습니다.")
        return

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📝 파라미터 입력")

        params_actual = {}
        params_normalized = []

        for param_name in PARAM_ORDER:
            config = PARAM_RANGES[param_name]

            st.markdown(f"**{config['name']}** ({config['unit']})")

            # kdecay는 0을 허용하는 특수한 경우
            if config['scale'] == 'log_zero':
                value = st.number_input(
                    f"{param_name}",
                    min_value=0.0,
                    max_value=float(config['high'] * 10),
                    value=float(config['mid']),
                    format="%.2e",
                    label_visibility="collapsed",
                    key=f"param_{param_name}"
                )
                st.caption(f"범위: 0 (분해 없음) ~ {format_scientific(config['high'])}")
            else:
                value = st.number_input(
                    f"{param_name}",
                    min_value=float(config['low'] * 0.1),
                    max_value=float(config['high'] * 10),
                    value=float(config['mid']),
                    format="%.2e" if config['scale'] == 'log' else "%.4f",
                    label_visibility="collapsed",
                    key=f"param_{param_name}"
                )
                st.caption(f"범위: {format_scientific(config['low'])} ~ {format_scientific(config['high'])}")

            params_actual[param_name] = value
            normalized = normalize_param(value, param_name)
            params_normalized.append(normalized)

    with col2:
        st.subheader("📊 예측 결과 (72시간)")

        # 시계열 파라미터 순서로 변환 (보간기용)
        ts_params = []
        for param_name in TS_PARAM_ORDER:
            if param_name in PARAM_ORDER:
                idx = PARAM_ORDER.index(param_name)
                ts_params.append(params_normalized[idx])
            else:
                ts_params.append(-1 if param_name == 'kdecay' else 0)

        # 보간 모델로 시계열 예측 (그래프와 동일한 모델 사용)
        if st.session_state.interpolator is not None:
            ts_data = predict_with_interpolator(
                st.session_state.interpolator,
                ts_params,
                time_max=72.0,
                n_points=100
            )
            # 72시간 최종값 추출
            final_row = ts_data.iloc[-1]
            predictions = {
                'Blood': final_row['Blood'],
                'Lymph': final_row['Lymph'],
                'ECM': final_row['ECM'],
                'Decay': final_row['Decay'] if 'Decay' in final_row else 0
            }
        elif st.session_state.model is not None:
            # 폴백: 예전 PINN 모델 사용 (kdecay 제외한 6개 파라미터)
            old_params = [p for i, p in enumerate(params_normalized) if PARAM_ORDER[i] != 'kdecay']
            predictions = predict_ratios(st.session_state.model, old_params)
            predictions['Decay'] = 0
            ts_data = None
        else:
            st.error("예측 모델을 로드할 수 없습니다.")
            return

        # 결과 표시 (항상 4개 컬럼으로 동일한 크기 유지)
        has_decay = predictions.get('Decay', 0) > 0.1
        r1, r2, r3, r4 = st.columns(4)

        with r1:
            st.markdown(f"""
            <div class="result-box" style="background: linear-gradient(135deg, #e74c3c, #c0392b);">
                <h2 style="margin:0; font-size:2rem;">{predictions['Blood']:.1f}%</h2>
                <p style="margin:0;">🔴 Blood</p>
            </div>
            """, unsafe_allow_html=True)

        with r2:
            st.markdown(f"""
            <div class="result-box" style="background: linear-gradient(135deg, #27ae60, #1e8449);">
                <h2 style="margin:0; font-size:2rem;">{predictions['Lymph']:.1f}%</h2>
                <p style="margin:0;">🟢 Lymph</p>
            </div>
            """, unsafe_allow_html=True)

        with r3:
            st.markdown(f"""
            <div class="result-box" style="background: linear-gradient(135deg, #f39c12, #e67e22);">
                <h2 style="margin:0; font-size:2rem;">{predictions['ECM']:.1f}%</h2>
                <p style="margin:0;">🟠 ECM</p>
            </div>
            """, unsafe_allow_html=True)

        with r4:
            if has_decay:
                st.markdown(f"""
                <div class="result-box" style="background: linear-gradient(135deg, #3498db, #2980b9);">
                    <h2 style="margin:0; font-size:2rem;">{predictions['Decay']:.1f}%</h2>
                    <p style="margin:0;">🔵 Decay</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-box" style="background: linear-gradient(135deg, #95a5a6, #7f8c8d);">
                    <h2 style="margin:0; font-size:2rem;">0.0%</h2>
                    <p style="margin:0;">🔵 Decay</p>
                </div>
                """, unsafe_allow_html=True)

        st.info(f"💉 순환계 (Blood + Lymph): **{predictions['Blood'] + predictions['Lymph']:.1f}%**")

        # 시계열 그래프
        st.subheader("📈 시간별 변화 예측")

        # 이미 ts_data가 있으면 재사용 (보간기에서 이미 생성됨)
        # ts_data가 없는 경우만 다른 방법으로 생성
        if ts_data is None:
            if st.session_state.ts_model is not None:
                ts_data = predict_timeseries_pinn(
                    st.session_state.ts_model,
                    ts_params,
                    time_max=72.0,
                    n_points=100
                )
            else:
                ts_data = generate_time_series_prediction(params_normalized)

        fig = create_time_series_chart(ts_data, "")
        st.plotly_chart(fig, use_container_width=True)

        # AUC 파이 차트
        st.subheader("📊 AUC (Area Under Curve) 분포")
        auc_values = calculate_auc(ts_data)
        fig = create_auc_chart(auc_values, "예측 AUC")
        st.plotly_chart(fig, use_container_width=True)


def page_condition_comparison():
    """조건별 비교 페이지"""
    st.header("📊 조건별 비교")

    # Summary 데이터 로드
    summary_df = load_simulation_summary()

    if summary_df is None or len(summary_df) == 0:
        st.error("데이터를 로드할 수 없습니다.")
        return

    st.subheader("시뮬레이션 결과 비교")

    # 조건 선택
    selected_conditions = st.multiselect(
        "비교할 조건 선택",
        summary_df['condition'].tolist(),
        default=summary_df['condition'].tolist()[:5]
    )

    if not selected_conditions:
        st.warning("조건을 선택하세요.")
        return

    filtered_df = summary_df[summary_df['condition'].isin(selected_conditions)]

    # 비교 차트
    fig = go.Figure()

    fig.add_trace(go.Bar(
        name='Blood',
        x=filtered_df['condition'],
        y=filtered_df['Blood'],
        marker_color='#e74c3c'
    ))

    fig.add_trace(go.Bar(
        name='Lymph',
        x=filtered_df['condition'],
        y=filtered_df['Lymph'],
        marker_color='#3498db'
    ))

    fig.add_trace(go.Bar(
        name='ECM',
        x=filtered_df['condition'],
        y=filtered_df['ECM'],
        marker_color='#2ecc71'
    ))

    fig.update_layout(
        barmode='group',
        height=500,
        xaxis_title='조건',
        yaxis_title='비율 (%)',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    st.plotly_chart(fig, use_container_width=True)

    # 데이터 테이블
    st.subheader("📋 상세 데이터")
    st.dataframe(
        filtered_df.style.format({
            'Blood': '{:.1f}%',
            'Lymph': '{:.1f}%',
            'ECM': '{:.1f}%'
        }),
        use_container_width=True,
        hide_index=True
    )


def page_help():
    """도움말 페이지"""
    st.header("ℹ️ 도움말")

    st.markdown("""
    ## 🧬 림프칩 PINN 에이전트

    이 애플리케이션은 림프칩 시뮬레이션 결과를 **시간에 따른 그래프**로 시각화합니다.

    ### 📌 주요 기능

    1. **📈 시간별 농도 변화**: 0~72시간 동안 Blood/Lymph/ECM 분포 변화 시뮬레이션
    2. **🎯 파라미터 입력**: 실제 물리 단위로 파라미터 입력 및 예측
    3. **📊 조건별 비교**: 다양한 조건에서의 시뮬레이션 결과 비교

    ### 📊 파라미터 설명
    """)

    param_data = []
    for param_name in PARAM_ORDER:
        config = PARAM_RANGES[param_name]
        param_data.append({
            '파라미터': config['name'],
            '단위': config['unit'],
            'Low': format_scientific(config['low']),
            'Mid': format_scientific(config['mid']),
            'High': format_scientific(config['high'])
        })

    st.dataframe(pd.DataFrame(param_data), use_container_width=True, hide_index=True)

    st.markdown("""
    ### 🎯 주요 영향 관계

    | 파라미터 | 증가 시 효과 |
    |----------|-------------|
    | **Lp ↑** | Lymph ↑ (림프로 더 많이 이동) |
    | **K ↑** | 전체 유체 이동 증가 |
    | **D ↑** | 평형 도달 속도 증가 |
    | **σ ↑** | 용질 투과 감소, Blood ↓ |
    | **kdecay ↑** | Decay ↑ (약물 분해 증가) |
    | **MW ↓** | Blood ↑ (작은 분자가 혈관 통과 용이) |
    | **MW ↑** | Lymph ↑ (큰 분자는 림프로 제거) |

    ### 💊 약물별 분자량 참고

    | 약물 | 분자량 (kDa) | 특성 |
    |------|-------------|------|
    | **INS (인슐린)** | 5.8 | 작은 분자 → Blood 80% |
    | **ALB (알부민)** | 66.5 | 중간 분자 → Lymph 70% |
    | **IgG (항체)** | 150 | 큰 분자 → Lymph 67% |
    """)


if __name__ == "__main__":
    main()
