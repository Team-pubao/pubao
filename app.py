"""Streamlit dashboard and new industrial complex simulator for the Pubao project."""

from __future__ import annotations

import json
import math
from io import BytesIO
from pathlib import Path

import folium
import matplotlib
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import statsmodels.api as sm
import streamlit as st
from branca.colormap import linear
from scipy import stats as scipy_stats
from statsmodels.stats.multitest import multipletests
from streamlit_folium import st_folium


ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
RESULTS_DIR = ROOT_DIR / "results"
TABLES_DIR = RESULTS_DIR / "tables"

SIDO_ORDER = [
    "서울",
    "부산",
    "대구",
    "인천",
    "광주",
    "대전",
    "울산",
    "세종",
    "경기",
    "강원",
    "충북",
    "충남",
    "전북",
    "전남",
    "경북",
    "경남",
    "제주",
]

SIDO_NAME_MAP = {
    "서울특별시": "서울",
    "부산광역시": "부산",
    "대구광역시": "대구",
    "인천광역시": "인천",
    "광주광역시": "광주",
    "대전광역시": "대전",
    "울산광역시": "울산",
    "세종특별자치시": "세종",
    "경기도": "경기",
    "강원도": "강원",
    "강원특별자치도": "강원",
    "충청북도": "충북",
    "충청남도": "충남",
    "전라북도": "전북",
    "전북특별자치도": "전북",
    "전라남도": "전남",
    "경상북도": "경북",
    "경상남도": "경남",
    "제주특별자치도": "제주",
}

ROLE_ORDER = ["heavy_export", "power_intensive", "logistics", "capital_intensive", "labor_intensive", "other"]
ROLE_LABELS = {
    "heavy_export": "중량·수출",
    "power_intensive": "전력집약",
    "logistics": "물류",
    "capital_intensive": "자본집약",
    "labor_intensive": "노동집약",
    "other": "기타",
}
ROLE_COLORS = {
    "heavy_export": "#c53d3d",
    "power_intensive": "#e29a35",
    "logistics": "#d9bf3f",
    "capital_intensive": "#2d62a3",
    "labor_intensive": "#5aa0bf",
    "other": "#9a9a9a",
}

FACTOR_COLUMNS = {
    "항만": "log_하역능력_합계",
    "IC": "IC밀도_개당1000km2",
    "전력": "log_산업용전력_2024",
    "임금": "평균임금_2024_백만원",
}
SIGUNGU_FACTOR_COLUMNS = {
    "항만 접근": "항만거리_km",
    "IC 밀도": "IC밀도_per1000km2",
    "산업용 전력": "산업용전력_2024_kWh",
    "평균급여": "평균급여_만원",
}

ROLE_HYPOTHESIS_ORDER = [
    "heavy_export",
    "power_intensive",
    "logistics",
    "capital_intensive",
    "labor_intensive",
    "other",
]

MAJOR_SITES = [
    {"기업": "삼성전자 평택캠퍼스", "라벨": "삼성전자 평택", "산업": "반도체·전자", "역할": "power_intensive", "가설": "H3", "lat": 37.0621, "lng": 127.0574},
    {"기업": "SK하이닉스 이천", "라벨": "SK하이닉스 이천", "산업": "반도체·전자", "역할": "power_intensive", "가설": "H3", "lat": 37.2524, "lng": 127.4890},
    {"기업": "현대자동차 울산공장", "라벨": "현대차 울산", "산업": "자동차", "역할": "heavy_export", "가설": "H1", "lat": 35.5384, "lng": 129.3718},
    {"기업": "포스코 포항제철소", "라벨": "포스코 포항", "산업": "1차금속", "역할": "heavy_export", "가설": "H1", "lat": 36.0030, "lng": 129.3887},
    {"기업": "HD현대중공업 울산", "라벨": "HD현대重 울산", "산업": "조선·기타운송장비", "역할": "heavy_export", "가설": "H1", "lat": 35.5146, "lng": 129.4386},
    {"기업": "삼성중공업 거제조선소", "라벨": "삼성重 거제", "산업": "조선·기타운송장비", "역할": "heavy_export", "가설": "H1", "lat": 34.8922, "lng": 128.6066},
    {"기업": "LG화학 여수공장", "라벨": "LG화학 여수", "산업": "석유화학", "역할": "heavy_export", "가설": "H1", "lat": 34.8377, "lng": 127.7314},
    {"기업": "롯데케미칼 대산공장", "라벨": "롯데케미칼 대산", "산업": "석유화학", "역할": "heavy_export", "가설": "H1", "lat": 36.9964, "lng": 126.3868},
]

CHART_FONT = "Apple SD Gothic Neo, Pretendard, Malgun Gothic, sans-serif"
CHART_PRIMARY = "#1d4ed8"
CHART_ACCENT = "#6366f1"
CHART_MUTED = "#9ca3af"
CHART_INK = "#111827"


def configure_korean_plot_font() -> str | None:
    """Configure Noto Sans KR with platform fallbacks for matplotlib/seaborn."""
    installed = {font.name for font in fm.fontManager.ttflist}
    selected = next(
        (name for name in ["Noto Sans KR", "Malgun Gothic", "AppleGothic", "NanumGothic"] if name in installed),
        None,
    )
    if selected:
        matplotlib.rcParams["font.family"] = selected
        matplotlib.rcParams["axes.unicode_minus"] = False
        sns.set_theme(font=selected, rc={"axes.unicode_minus": False})
    return selected


MATPLOTLIB_KOREAN_FONT = configure_korean_plot_font()


def apply_chart_theme(fig: go.Figure, *, height: int | None = None, show_legend: bool = True) -> go.Figure:
    """Apply unified theme to a plotly figure — larger fonts, clean axes."""
    fig.update_layout(
        font=dict(family=CHART_FONT, color=CHART_INK, size=14),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        margin=dict(l=16, r=16, t=60, b=16),
        title=dict(font=dict(size=17, color=CHART_INK, family=CHART_FONT), x=0.01, xanchor="left"),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
            font=dict(size=13, color=CHART_INK), bgcolor="rgba(255,255,255,0.8)",
        ),
        showlegend=show_legend,
        hoverlabel=dict(font=dict(family=CHART_FONT, size=13), bgcolor="#0f172a", font_color="#ffffff"),
    )
    fig.update_xaxes(
        showgrid=True, gridcolor="#f3f4f6", zeroline=False, linecolor="#e5e7eb",
        tickfont=dict(size=12, color="#4b5563"), title_font=dict(size=12, color="#6b7280"),
    )
    fig.update_yaxes(
        showgrid=True, gridcolor="#f3f4f6", zeroline=False, linecolor="#e5e7eb",
        tickfont=dict(size=12, color="#4b5563"), title_font=dict(size=12, color="#6b7280"),
    )
    if height is not None:
        fig.update_layout(height=height)
    return fig


st.set_page_config(page_title="푸바오 — 산업 입지 분석", layout="wide", page_icon="🏭")


def inject_css() -> None:
    """Apply refined minimal dashboard styling."""
    st.markdown(
        """
        <style>
        /* ── Base ─────────────────────────────────── */
        html, body, .stApp {
            font-family: "Apple SD Gothic Neo", "Pretendard", "Malgun Gothic", "Noto Sans KR", -apple-system, sans-serif;
            -webkit-font-smoothing: antialiased;
            -moz-osx-font-smoothing: grayscale;
            color: #171717;
        }
        .stApp { background: #f9fafb; }
        .block-container {
            padding-top: 1rem;
            padding-bottom: 1.5rem;
            max-width: 1400px;
        }

        /* ── Hero ─────────────────────────────────── */
        .hero {
            margin-bottom: 0.85rem;
            padding-bottom: 0.7rem;
            border-bottom: 1px solid #e5e7eb;
        }
        .hero .eyebrow {
            font-size: 0.7rem;
            letter-spacing: 0.06em;
            color: #1d4ed8;
            font-weight: 700;
            margin-bottom: 0.25rem;
        }
        .hero h1 {
            color: #111827;
            font-size: 1.3rem;
            font-weight: 700;
            margin: 0 0 0.2rem;
            letter-spacing: -0.02em;
            line-height: 1.25;
        }
        .hero p {
            color: #4b5563;
            margin: 0;
            font-size: 0.95rem;
            line-height: 1.55;
            max-width: 760px;
        }
        .hero .meta {
            margin-top: 0.4rem;
            font-size: 0.75rem;
            color: #9ca3af;
        }
        .hero .meta span:not(:last-child)::after {
            content: "·";
            margin: 0 0.55rem;
            color: #d1d5db;
        }

        /* ── Section titles ─────────────────────── */
        .section-title {
            color: #111827;
            font-size: 1.0rem;
            font-weight: 700;
            margin: 0.8rem 0 0.15rem;
            letter-spacing: -0.018em;
        }
        .section-sub {
            color: #6b7280;
            font-size: 0.8rem;
            margin: 0 0 0.45rem;
            line-height: 1.4;
        }

        /* ── Unified card system ─────────────────── */
        .factor-card, .rank-card, .recommend-card, .info-card {
            background: #ffffff;
            border: 1px solid #e5e7eb;
            border-radius: 12px;
            transition: border-color 0.15s ease, box-shadow 0.15s ease;
        }
        .factor-card:hover, .rank-card:hover, .recommend-card:hover {
            border-color: #c7d2fe;
            box-shadow: 0 4px 12px rgba(99, 102, 241, 0.08);
        }

        /* Factor card (탐색 모드) */
        .factor-card {
            padding: 0.65rem 0.9rem 0.6rem;
            margin-bottom: 0.5rem;
        }
        .factor-card-head {
            display: flex;
            justify-content: space-between;
            align-items: baseline;
            margin-bottom: 0.4rem;
        }
        .factor-card-title {
            font-size: 0.92rem;
            font-weight: 600;
            color: #111827;
            letter-spacing: -0.005em;
        }
        .factor-card-rank {
            font-size: 0.74rem;
            color: #6b7280;
            font-weight: 500;
            font-variant-numeric: tabular-nums;
        }
        .factor-card-rank strong {
            color: #1d4ed8;
            font-weight: 700;
        }
        .factor-card-value {
            font-size: 1.4rem;
            font-weight: 700;
            color: #111827;
            line-height: 1.05;
            letter-spacing: -0.03em;
            font-variant-numeric: tabular-nums;
            display: flex;
            align-items: baseline;
            gap: 0.35rem;
            margin-bottom: 0.15rem;
        }
        .factor-card-value-unit {
            font-size: 0.8rem;
            font-weight: 500;
            color: #9ca3af;
            letter-spacing: 0;
        }
        .factor-card-sub {
            font-size: 0.76rem;
            color: #6b7280;
            margin: 0 0 0.5rem;
            line-height: 1.4;
        }
        .factor-card-bar {
            background: #eef2f7;
            height: 5px;
            border-radius: 999px;
            overflow: hidden;
            margin-bottom: 0.45rem;
        }
        .factor-card-bar-fill {
            background: linear-gradient(90deg, #3b82f6, #6366f1);
            height: 100%;
            border-radius: 999px;
        }
        .factor-card-bar-foot {
            display: flex;
            justify-content: space-between;
            font-size: 0.7rem;
            color: #9ca3af;
            margin-bottom: 0.4rem;
            font-variant-numeric: tabular-nums;
        }
        .factor-card-tag {
            display: inline-flex;
            align-items: center;
            font-size: 0.76rem;
            color: #4b5563;
            font-weight: 500;
            font-variant-numeric: tabular-nums;
        }
        .factor-card-tag .dot {
            display: inline-block;
            width: 7px; height: 7px;
            border-radius: 50%;
            margin-right: 0.5rem;
        }

        /* Rank card (TOP 5) */
        .rank-card {
            padding: 1rem 1.15rem 0.95rem;
            min-height: 124px;
        }
        .rank-card .medal {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 22px; height: 22px;
            border-radius: 50%;
            font-size: 0.74rem;
            font-weight: 700;
            color: #ffffff;
            margin-right: 0.55rem;
            vertical-align: -3px;
        }
        .rank-card .medal.r1 { background: #d97706; }
        .rank-card .medal.r2 { background: #94a3b8; }
        .rank-card .medal.r3 { background: #b45309; }
        .rank-card .medal.r4, .rank-card .medal.r5 { background: #cbd5e1; color: #475569; }
        .rank-card .title {
            font-size: 0.95rem;
            font-weight: 600;
            color: #111827;
            margin-bottom: 0.5rem;
            letter-spacing: -0.005em;
        }
        .rank-card .score {
            font-size: 1.65rem;
            font-weight: 700;
            color: #1d4ed8;
            line-height: 1.05;
            letter-spacing: -0.03em;
            font-variant-numeric: tabular-nums;
        }
        .rank-card .score-unit {
            font-size: 0.8rem;
            color: #9ca3af;
            font-weight: 500;
            margin-left: 0.15rem;
        }
        .rank-card .sub {
            font-size: 0.76rem;
            color: #6b7280;
            margin-top: 0.5rem;
            line-height: 1.5;
        }

        /* Recommend card (탐색 TOP 3) */
        .recommend-card {
            padding: 0.95rem 1.1rem;
            margin-bottom: 0.5rem;
        }
        .recommend-card b {
            color: #111827;
            font-size: 0.95rem;
            font-weight: 600;
            letter-spacing: -0.005em;
        }
        .recommend-card .score-chip {
            display: inline-block;
            color: #6b7280;
            font-size: 0.76rem;
            font-weight: 500;
            margin-left: 0.55rem;
            font-variant-numeric: tabular-nums;
        }
        .recommend-card .score-chip strong {
            color: #1d4ed8;
            font-weight: 700;
        }

        /* Small note */
        .small-note {
            color: #6b7280;
            font-size: 0.82rem;
            line-height: 1.55;
        }

        /* ── Streamlit metric (STEP 2) ───────────── */
        div[data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid #e5e7eb;
            border-radius: 10px;
            padding: 1rem 1.15rem;
        }
        div[data-testid="stMetricLabel"] p {
            color: #6b7280;
            font-weight: 500;
            font-size: 0.82rem;
        }
        div[data-testid="stMetricValue"] {
            color: #111827;
            font-weight: 700;
            letter-spacing: -0.02em;
        }
        div[data-testid="stMetricDelta"] {
            font-size: 0.76rem !important;
            color: #6b7280 !important;
        }
        div[data-testid="stMetricDelta"] svg { display: none; }

        /* ── Status chip (가설) — dot pattern ────── */
        .status-chip {
            display: inline-flex;
            align-items: center;
            padding: 0.2rem 0.7rem;
            border-radius: 999px;
            font-size: 0.74rem;
            font-weight: 500;
            background: #ffffff;
            border: 1px solid #e5e5e5;
            color: #525252;
            margin-bottom: 0.6rem;
        }
        .status-chip::before {
            content: "";
            display: inline-block;
            width: 6px; height: 6px;
            border-radius: 50%;
            margin-right: 0.45rem;
        }
        .status-support::before { background: #16a34a; }
        .status-partial::before { background: #d97706; }
        .status-limited::before { background: #dc2626; }

        /* ── Sidebar ─────────────────────────────── */
        section[data-testid="stSidebar"] {
            background: #ffffff;
            border-right: 1px solid #e5e5e5;
        }
        section[data-testid="stSidebar"] .stRadio > label p,
        section[data-testid="stSidebar"] .stSelectbox > label p {
            font-weight: 600;
            color: #0a0a0a;
            font-size: 0.82rem;
        }
        .sidebar-title {
            font-size: 0.72rem;
            font-weight: 700;
            color: #737373;
            margin: 0.5rem 0 0.4rem;
            text-transform: uppercase;
            letter-spacing: 0.12em;
        }

        /* ── Expander ────────────────────────────── */
        div[data-testid="stExpander"] {
            border: 1px solid #e5e5e5;
            border-radius: 10px;
            background: #ffffff;
            margin-bottom: 0.5rem;
        }
        div[data-testid="stExpander"] summary {
            font-weight: 600;
            color: #0a0a0a;
            font-size: 0.93rem;
        }

        /* ── Alert (st.info, st.success) ─────────── */
        div[data-testid="stAlert"] {
            background: #fafafa;
            border: 1px solid #e5e5e5;
            border-radius: 10px;
        }
        div[data-testid="stAlert"] p { color: #0a0a0a; }
        .method-badges {
            display: flex;
            gap: 0.4rem;
            margin: 0.2rem 0 0.8rem;
            flex-wrap: wrap;
        }
        .method-badge {
            display: inline-flex;
            align-items: center;
            padding: 0.22rem 0.55rem;
            border-radius: 999px;
            background: #eff6ff;
            color: #1d4ed8;
            border: 1px solid #bfdbfe;
            font-size: 0.72rem;
            font-weight: 700;
            letter-spacing: 0.01em;
        }
        .unit-note {
            padding: 0.72rem 0.9rem;
            margin: 0.4rem 0 1rem;
            border-left: 3px solid #1d4ed8;
            background: #f8fafc;
            color: #475569;
            font-size: 0.82rem;
            line-height: 1.55;
        }

        /* ── Hypothesis selector cards (st.button) ── */
        div[data-testid="stButton"] > button {
            text-align: left;
            white-space: normal;
            height: 100%;
            min-height: 104px;
            padding: 0.7rem 0.9rem;
            border-radius: 12px;
            line-height: 1.45;
            font-size: 0.82rem;
            font-weight: 600;
            color: #111827;
            box-shadow: none;
            transition: border-color .15s ease, box-shadow .15s ease, background .15s ease;
        }
        div[data-testid="stButton"] > button p {
            text-align: left;
            white-space: normal;
            font-weight: 600;
        }
        div[data-testid="stButton"] > button[kind="secondary"] {
            background: #ffffff;
            border: 1px solid #e5e7eb;
        }
        div[data-testid="stButton"] > button[kind="secondary"]:hover {
            border-color: #c7d2fe;
            box-shadow: 0 4px 12px rgba(99,102,241,.10);
            color: #111827;
        }
        div[data-testid="stButton"] > button[kind="primary"],
        div[data-testid="stButton"] > button[kind="primary"]:hover,
        div[data-testid="stButton"] > button[kind="primary"]:active,
        div[data-testid="stButton"] > button[kind="primary"]:focus {
            background: #eff6ff;
            border: 2px solid #1d4ed8;
            color: #0f172a;
            box-shadow: 0 4px 14px rgba(29,78,216,.18);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def short_sido(name: object) -> object:
    """Normalize province names to short labels."""
    if not isinstance(name, str):
        return name
    return SIDO_NAME_MAP.get(name.strip(), name.strip())


def sorted_by_sido(df: pd.DataFrame) -> pd.DataFrame:
    """Sort dataframe by the fixed 시도 order."""
    order = {name: idx for idx, name in enumerate(SIDO_ORDER)}
    out = df.copy()
    if "시도" in out.columns:
        out["_order"] = out["시도"].map(order)
        out = out.sort_values("_order").drop(columns="_order")
    return out.reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_integrated_df() -> pd.DataFrame:
    """Load 2024 integrated data and dashboard helper columns."""
    df = pd.read_excel(DATA_DIR / "07_통합분석_2024.xlsx", sheet_name="통합DF")
    df["시도"] = df["시도"].map(short_sido)
    for col in ["하역능력_합계", "물동량_2023_톤", "무역항수", "IC수"]:
        if col in df.columns:
            df[col] = df[col].fillna(0)
    df["log_하역능력_합계"] = np.log1p(df["하역능력_합계"])
    df["평균임금_2024_백만원"] = df["평균임금_2024"] / 1_000_000
    df["평균임금_2024_만원"] = df["평균임금_2024"] / 10_000
    return sorted_by_sido(df)


@st.cache_data(show_spinner=False)
def load_y_panel() -> pd.DataFrame:
    """Load 2020-2024 long panel data."""
    df = pd.read_excel(DATA_DIR / "02_사업체수_패널.xlsx")
    df["시도"] = df["시도"].map(short_sido)
    return df


@st.cache_data(show_spinner=False)
def load_industry_mapping() -> pd.DataFrame:
    """Load industry roles and sorting metadata."""
    df = pd.read_excel(DATA_DIR / "01_산업매핑.xlsx")
    df = df[["분석그룹", "축_역할"]].drop_duplicates().copy()
    df["role_order"] = df["축_역할"].map({role: idx for idx, role in enumerate(ROLE_ORDER)}).fillna(99)
    return df.sort_values(["role_order", "분석그룹"]).reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_geojson_dict() -> dict:
    """Load province GeoJSON as a plain dict with normalized 시도 names."""
    with open(DATA_DIR / "지도_시도경계.json", encoding="utf-8") as f:
        geo = json.load(f)
    valid: list[dict] = []
    for feature in geo.get("features", []):
        props = feature.get("properties", {}) or {}
        raw_name = props.get("시도") or props.get("name") or props.get("SIDO_NM") or props.get("NAME")
        short = short_sido(raw_name) if isinstance(raw_name, str) else None
        if isinstance(short, str) and short in SIDO_ORDER:
            props["시도"] = short
            feature["properties"] = props
            valid.append(feature)
    valid.sort(key=lambda f: SIDO_ORDER.index(f["properties"]["시도"]))
    return {"type": geo.get("type", "FeatureCollection"), "features": valid}


@st.cache_data(show_spinner=False)
def load_sigungu_factors() -> pd.DataFrame:
    """Load 229-city/county location factors for maps and context only."""
    path = DATA_DIR / "시군구_입지요인.csv"
    df = pd.read_csv(path, encoding="utf-8-sig")
    df["시도"] = df["시도"].map(short_sido)
    df["지도키"] = df["시도"].astype(str) + "|" + df["시군구"].astype(str)
    return df


@st.cache_data(show_spinner=False)
def load_sigungu_geojson_dict() -> dict:
    """Load simplified city/county boundaries and attach a normalized map key."""
    path = DATA_DIR / "시군구_경계.geojson"
    with open(path, encoding="utf-8") as file:
        geo = json.load(file)
    for feature in geo.get("features", []):
        props = feature.get("properties", {}) or {}
        sido = short_sido(props.get("시도"))
        sigungu = props.get("시군구_표준") or props.get("시군구") or props.get("name")
        props["시도"] = sido
        props["시군구_표준"] = sigungu
        props["지도키"] = f"{sido}|{sigungu}"
        feature["properties"] = props
    return geo


@st.cache_data(show_spinner=False)
def industry_role_members() -> dict[str, list[str]]:
    """Return industries grouped by the six theoretical roles."""
    mapping = load_industry_mapping()
    return {
        role: mapping.loc[mapping["축_역할"].eq(role), "분석그룹"].drop_duplicates().tolist()
        for role in ROLE_HYPOTHESIS_ORDER
    }


@st.cache_data(show_spinner=False)
def load_analysis_df() -> pd.DataFrame:
    """Build the province-level regression frame with six role shares."""
    df = load_integrated_df().copy()
    for role, industries in industry_role_members().items():
        columns = [f"비중_{industry}" for industry in industries if f"비중_{industry}" in df.columns]
        df[f"역할비중_{role}"] = df[columns].sum(axis=1)
        count_columns = [industry for industry in industries if industry in df.columns]
        df[f"역할사업체수_{role}"] = df[count_columns].sum(axis=1)
    df["권역유형"] = np.where(df["시도"].isin(["서울", "부산", "대구", "인천", "광주", "대전", "울산", "세종"]), "광역시", "도")
    return df


def _clean_regression_data(df: pd.DataFrame, y_col: str, x_cols: list[str]) -> pd.DataFrame:
    """Return finite numeric rows for a regression."""
    columns = [y_col, *x_cols]
    work = df[columns].replace([np.inf, -np.inf], np.nan).copy()
    for col in columns:
        work[col] = pd.to_numeric(work[col], errors="coerce")
    return work.dropna()


def _standardize(series: pd.Series) -> pd.Series:
    """Population-standardize a numeric series."""
    std = float(series.std(ddof=0))
    if math.isclose(std, 0.0):
        return pd.Series(0.0, index=series.index)
    return (series - series.mean()) / std


def fit_simple_regression(df: pd.DataFrame, y_col: str, x_col: str) -> dict[str, float]:
    """Fit a simple OLS and report raw and standardized effects."""
    work = _clean_regression_data(df, y_col, [x_col])
    raw_x = sm.add_constant(work[[x_col]], has_constant="add")
    raw_model = sm.OLS(work[y_col], raw_x).fit()
    standardized = pd.DataFrame({"y": _standardize(work[y_col]), "x": _standardize(work[x_col])})
    std_model = sm.OLS(standardized["y"], sm.add_constant(standardized[["x"]], has_constant="add")).fit()
    return {
        "β": float(raw_model.params[x_col]),
        "표준화β": float(std_model.params["x"]),
        "SE": float(raw_model.bse[x_col]),
        "p_value": float(raw_model.pvalues[x_col]),
        "R²": float(raw_model.rsquared),
        "N": int(raw_model.nobs),
        "절편": float(raw_model.params["const"]),
    }


def fit_multifactor_regression(df: pd.DataFrame, y_col: str) -> tuple[object, pd.DataFrame]:
    """Fit a four-factor OLS using standardized X and return row predictions."""
    x_cols = list(FACTOR_COLUMNS.values())
    work = df[["시도", y_col, *x_cols]].replace([np.inf, -np.inf], np.nan).dropna().copy()
    x_std = pd.DataFrame({label: _standardize(work[col]) for label, col in FACTOR_COLUMNS.items()}, index=work.index)
    model = sm.OLS(work[y_col], sm.add_constant(x_std, has_constant="add")).fit()
    result = work[["시도", y_col]].rename(columns={y_col: "실제"}).copy()
    result["예측"] = model.predict(sm.add_constant(x_std, has_constant="add"))
    result["유치여지"] = result["예측"] - result["실제"]
    return model, result


@st.cache_data(show_spinner=False)
def build_beta_results(level: str = "산업") -> pd.DataFrame:
    """Build comparable standardized beta results for 26 industries or six roles."""
    df = load_analysis_df()
    if level == "역할":
        targets = {ROLE_LABELS[role]: f"역할비중_{role}" for role in ROLE_HYPOTHESIS_ORDER}
    else:
        targets = {industry: f"비중_{industry}" for industry in get_industries()}
    rows: list[dict[str, object]] = []
    for target, y_col in targets.items():
        for factor, x_col in FACTOR_COLUMNS.items():
            result = fit_simple_regression(df, y_col, x_col)
            rows.append({"대상": target, "요인": factor, **result})
    output = pd.DataFrame(rows)
    output["p_FDR"] = multipletests(output["p_value"].to_numpy(), method="fdr_bh")[1]
    output["유의_FDR"] = output["p_FDR"] < 0.05
    return output


@st.cache_data(show_spinner=False)
def build_factor_competition() -> pd.DataFrame:
    """Rank the four factors by explanatory strength across 26 industries."""
    results = build_beta_results("산업")
    summary = (
        results.groupby("요인", as_index=False)
        .agg(
            **{
                "평균_R²": ("R²", "mean"),
                "중앙_R²": ("R²", "median"),
                "평균_절대_표준화β": ("표준화β", lambda values: float(np.mean(np.abs(values)))),
                "유의산업수": ("유의_FDR", "sum"),
            }
        )
        .sort_values(["평균_R²", "평균_절대_표준화β"], ascending=False)
        .reset_index(drop=True)
    )
    summary["순위"] = np.arange(1, len(summary) + 1)
    return summary


@st.cache_data(show_spinner=False)
def build_hypothesis_summary_live() -> pd.DataFrame:
    """Build the replacement H1-H8 summary directly from current data."""
    df = load_analysis_df()
    h1 = fit_simple_regression(df, "역할비중_heavy_export", FACTOR_COLUMNS["항만"])
    h2 = fit_simple_regression(df, "역할비중_logistics", FACTOR_COLUMNS["IC"])
    h3 = fit_simple_regression(df, "비중_반도체·전자", FACTOR_COLUMNS["전력"])
    h4_capital = fit_simple_regression(df, "역할비중_capital_intensive", FACTOR_COLUMNS["임금"])
    h4_labor = fit_simple_regression(df, "비중_섬유·의복·가죽", FACTOR_COLUMNS["임금"])
    h5 = build_factor_competition().iloc[0]
    h6 = build_beta_results("산업")
    h6_sig = h6[h6["유의_FDR"]]
    role_cols = [f"역할사업체수_{role}" for role in ROLE_HYPOTHESIS_ORDER]
    concentration = df[role_cols].apply(lambda col: col.nlargest(3).sum() / col.sum())
    h8_model, h8 = fit_multifactor_regression(df, "역할비중_power_intensive")

    def support(result: dict[str, float], expected_sign: int = 1) -> str:
        correct_sign = result["표준화β"] * expected_sign > 0
        if correct_sign and result["p_value"] < 0.05:
            return "지지"
        if correct_sign:
            return "부분지지"
        return "기각"

    h4_conclusion = (
        "지지"
        if h4_capital["표준화β"] > 0 and h4_labor["표준화β"] < 0
        and h4_capital["p_value"] < 0.05 and h4_labor["p_value"] < 0.05
        else "부분지지" if h4_capital["표준화β"] > 0 or h4_labor["표준화β"] < 0 else "기각"
    )
    return pd.DataFrame(
        [
            {
                "가설": "H1",
                "한줄": "시도 항만 인프라가 클수록 중량수출 산업 비중이 높다",
                "통계": f"표준화β={h1['표준화β']:.2f}, R²={h1['R²']:.2f}, p={h1['p_value']:.3f}",
                "결론": support(h1),
                "결론문": "시도 단위 항만 인프라와 중량수출 산업 비중의 연관성 패턴을 확인했다.",
            },
            {
                "가설": "H2",
                "한줄": "IC 밀도가 높을수록 물류 산업 비중이 높다",
                "통계": f"표준화β={h2['표준화β']:.2f}, R²={h2['R²']:.2f}, p={h2['p_value']:.3f}",
                "결론": support(h2),
                "결론문": "IC 밀도와 도매·소매·운송 산업 비중 사이의 양의 연관성이 나타났다.",
            },
            {
                "가설": "H3",
                "한줄": "산업용 전력이 많을수록 반도체·전자 비중이 높다",
                "통계": f"표준화β={h3['표준화β']:.2f}, R²={h3['R²']:.2f}, p={h3['p_value']:.3f}",
                "결론": support(h3),
                "결론문": "전력과 반도체·전자 비중은 양의 패턴이나 통계적 강도는 제한적이다.",
            },
            {
                "가설": "H4",
                "한줄": "임금이 높으면 자본집약 산업은 늘고 노동집약 산업은 줄어든다",
                "통계": f"자본 β={h4_capital['표준화β']:.2f}, 섬유 β={h4_labor['표준화β']:.2f}",
                "결론": h4_conclusion,
                "결론문": "고임금 지역의 자본집약 산업 집중은 보이지만 노동집약 산업의 음의 패턴은 약하다.",
            },
            {
                "가설": "H5",
                "한줄": "항만·IC·전력·임금 중 산업 입지를 가장 잘 설명하는 요인은 다르다",
                "통계": f"1위 {h5['요인']} · 평균 R²={h5['평균_R²']:.2f} · 유의산업 {int(h5['유의산업수'])}개",
                "결론": "지지",
                "결론문": "4요인의 설명력 순위가 뚜렷하게 갈리는 연관성 패턴이 나타났다.",
            },
            {
                "가설": "H6",
                "한줄": "산업마다 잘 맞는 입지 조건이 다르다",
                "통계": f"유의 셀 {len(h6_sig)}/{len(h6)}개",
                "결론": "지지" if h6_sig["요인"].nunique() >= 3 else "부분지지",
                "결론문": "산업별 유의한 입지요인이 서로 달라 맞춤형 입지 전략이 필요하다.",
            },
            {
                "가설": "H7",
                "한줄": "산업 비중은 일부 시도와 권역에 편중된다",
                "통계": f"역할별 상위 3개 시도 집중도 최대 {concentration.max():.1%}",
                "결론": "지지",
                "결론문": "산업 역할별 비중이 특정 시도에 집중되는 지역 편중 패턴이 확인된다.",
            },
            {
                "가설": "H8",
                "한줄": "입지 조건은 좋지만 산업 비중이 낮은 지역을 유치 후보로 찾을 수 있다",
                "통계": f"전력집약 모형 R²={h8_model.rsquared:.2f} · 최대 유치여지 {h8['유치여지'].max():.3f}",
                "결론": "탐색적",
                "결론문": "예측 비중이 실제보다 높은 지역을 정책 검토용 유치 후보로 제시한다.",
            },
        ]
    )


def get_industries() -> list[str]:
    """Return 26 industries in role order."""
    return load_industry_mapping()["분석그룹"].tolist()


def factor_stats() -> pd.DataFrame:
    """Return min, max, mean, and std for simulator factors."""
    df = load_integrated_df()
    rows = []
    for label, col in FACTOR_COLUMNS.items():
        rows.append({"요인": label, "컬럼": col, "min": df[col].min(), "max": df[col].max(), "mean": df[col].mean(), "std": df[col].std(ddof=0)})
    return pd.DataFrame(rows).set_index("요인")


def normalized_factor_vector(raw: dict[str, float]) -> pd.Series:
    """Convert raw factor inputs to z-scores using the 17 provinces."""
    stats = factor_stats()
    z_scores = {}
    for factor in FACTOR_COLUMNS:
        std = stats.loc[factor, "std"]
        z_scores[factor] = 0.0 if std == 0 else (raw[factor] - stats.loc[factor, "mean"]) / std
    return pd.Series(z_scores)


def standardized_beta() -> pd.DataFrame:
    """Return comparable standardized beta values aligned to factor labels."""
    results = build_beta_results("산업")
    return results.pivot(index="대상", columns="요인", values="표준화β").reindex(get_industries())


def recommend_industries(user_factors_raw: dict[str, float], top_n: int = 5) -> tuple[pd.Series, pd.DataFrame, pd.Series]:
    """Recommend industries from factor inputs and decompose contributions."""
    beta = standardized_beta()
    x_user_z = normalized_factor_vector(user_factors_raw)
    contributions = beta.mul(x_user_z, axis=1)
    scores = contributions.sum(axis=1)
    if math.isclose(scores.max(), scores.min()):
        normalized = pd.Series(50.0, index=scores.index)
    else:
        normalized = (scores - scores.min()) / (scores.max() - scores.min()) * 100
    top = normalized.nlargest(top_n)
    return top, contributions.loc[top.index], x_user_z


def render_badges(*labels: str) -> None:
    """Render visualization and method badges above a chart."""
    badges = "".join(f'<span class="method-badge">{label}</span>' for label in labels)
    st.markdown(f'<div class="method-badges">{badges}</div>', unsafe_allow_html=True)


def add_vertical_legend(
    fmap: folium.Map,
    *,
    caption: str,
    colors: list[str],
    minimum: float,
    maximum: float,
    reverse: bool = False,
) -> None:
    """Add a compact vertical color legend to a Folium map."""
    gradient = ", ".join(reversed(colors) if reverse else colors)
    legend = f"""
    <div style="position:fixed;right:18px;bottom:32px;z-index:9999;background:white;
                border:1px solid #cbd5e1;border-radius:8px;padding:10px 12px;
                box-shadow:0 2px 8px rgba(15,23,42,.14);font-size:11px;color:#334155;">
      <div style="font-weight:700;margin-bottom:7px;max-width:110px;">{caption}</div>
      <div style="display:flex;gap:8px;align-items:stretch;">
        <div style="width:14px;height:120px;border-radius:4px;
                    background:linear-gradient(to top,{gradient});"></div>
        <div style="display:flex;flex-direction:column;justify-content:space-between;">
          <span>{maximum:,.2f}</span><span>{(minimum + maximum) / 2:,.2f}</span><span>{minimum:,.2f}</span>
        </div>
      </div>
    </div>
    """
    fmap.get_root().html.add_child(folium.Element(legend))


def add_major_site_markers(fmap: folium.Map) -> None:
    """Add major industrial sites with an always-on, styled name label (no hover needed)."""
    layer = folium.FeatureGroup(name="주요 대기업 사업장", show=True)
    for site in MAJOR_SITES:
        color = ROLE_COLORS.get(site["역할"], "#1d4ed8")
        folium.CircleMarker(
            location=[site["lat"], site["lng"]],
            radius=5,
            color="#ffffff",
            weight=1.5,
            fill=True,
            fill_color=color,
            fill_opacity=0.95,
            popup=(
                f"<b>{site['기업']}</b><br>{site['산업']}<br>"
                f"연결 가설: {site['가설']} · {ROLE_LABELS.get(site['역할'], '기타')}"
            ),
        ).add_to(layer)
        label_html = (
            f'<div style="transform:translate(9px,-11px);display:inline-flex;align-items:center;gap:5px;'
            f'white-space:nowrap;background:rgba(255,255,255,0.96);border:1px solid {color};'
            f'border-left:4px solid {color};border-radius:7px;padding:2px 8px;'
            f'box-shadow:0 1px 5px rgba(15,23,42,.18);'
            f'font-family:Pretendard,\'Apple SD Gothic Neo\',\'Malgun Gothic\',sans-serif;'
            f'font-size:11px;font-weight:700;color:#0f172a;letter-spacing:-0.01em;">'
            f'<span style="width:7px;height:7px;border-radius:50%;background:{color};flex:none;"></span>'
            f'{site["라벨"]}</div>'
        )
        folium.Marker(
            location=[site["lat"], site["lng"]],
            icon=folium.DivIcon(icon_size=(0, 0), icon_anchor=(0, 0), html=label_html),
            tooltip=f"{site['기업']} · {site['산업']}",
        ).add_to(layer)
    layer.add_to(fmap)


def find_similar_sido(user_factors_raw: dict[str, float]) -> tuple[str, float]:
    """Find the closest real province to a simulator factor profile."""
    df = load_integrated_df()
    user_z = normalized_factor_vector(user_factors_raw)
    stats = factor_stats()
    distances = []
    for _, row in df.iterrows():
        z = []
        for factor, col in FACTOR_COLUMNS.items():
            std = stats.loc[factor, "std"]
            z.append(0.0 if std == 0 else (row[col] - stats.loc[factor, "mean"]) / std)
        distance = float(np.linalg.norm(np.array(z) - user_z.values))
        distances.append((row["시도"], distance))
    return min(distances, key=lambda item: item[1])


def industry_comment(industry: str, contributions: pd.Series) -> str:
    """Create a short theory-based comment for a recommended industry."""
    mapping = load_industry_mapping().set_index("분석그룹")
    role = mapping.loc[industry, "축_역할"] if industry in mapping.index else "other"
    top_factor = contributions.abs().sort_values(ascending=False).index[0]
    role_text = {
        "heavy_export": "항만·전력 기반 제조 클러스터와 잘 맞는 패턴",
        "power_intensive": "고전력 입지가 핵심인 첨단 제조 패턴",
        "logistics": "도로 접근성과 시장 인접성이 중요한 물류 패턴",
        "capital_intensive": "고임금·도시형 서비스 및 지식산업 패턴",
        "labor_intensive": "비용 민감도가 높은 전통 제조 패턴",
        "other": "복합 입지요인이 작동하는 일반 산업 패턴",
    }
    return f"{ROLE_LABELS.get(role, '기타')} 산업군 — {top_factor} 기여가 가장 크고, {role_text.get(role, role_text['other'])}입니다."


def rank_of_value(df: pd.DataFrame, col: str, sido: str, ascending: bool = False) -> int:
    """Return selected province rank among 17."""
    ranked = df[["시도", col]].sort_values(col, ascending=ascending).reset_index(drop=True)
    match = ranked.index[ranked["시도"] == sido]
    return int(match[0]) + 1 if len(match) else 0


def render_header() -> None:
    """Render dashboard header."""
    st.markdown(
        """
        <div class="hero">
          <div class="eyebrow">한국 산업 입지 분석 대시보드</div>
          <h1>푸바오 · 산업별 입지 결정 요인 탐색기</h1>
          <div class="meta">
            <span>팀 푸바오</span>
            <span>아주대 융합시스템공학과</span>
            <span>최종 발표 2026.06.22</span>
            <span>N = 17 시도 · 5년 패널</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def make_industry_map(selected_sido: str, selected_target: str, target_level: str = "6역할") -> folium.Map:
    """Build the fixed-2024 province industry or role choropleth."""
    df = load_analysis_df()
    if target_level == "6역할":
        role = next(key for key, label in ROLE_LABELS.items() if label == selected_target)
        share_col = f"역할비중_{role}"
        count_col = None
    else:
        share_col = f"비중_{selected_target}"
        count_col = selected_target
    share_map = dict(zip(df["시도"], df[share_col]))
    count_map = dict(zip(df["시도"], df[count_col])) if count_col else {}

    geo = json.loads(json.dumps(load_geojson_dict()))
    for feature in geo["features"]:
        sido = feature["properties"]["시도"]
        feature["properties"][share_col] = float(share_map.get(sido, 0.0))
        feature["properties"]["대상"] = selected_target
        feature["properties"]["사업체수"] = float(count_map.get(sido, 0.0)) if count_col else None

    shares = [f["properties"][share_col] for f in geo["features"]]
    value_min = float(min(shares))
    value_max = float(max(shares))
    cmap = linear.YlGnBu_09.scale(value_min, value_max)
    fmap = folium.Map(location=[36.4, 127.8], zoom_start=7, tiles="OpenStreetMap", control_scale=True)

    def style(feature: dict) -> dict:
        props = feature["properties"]
        share = props.get(share_col)
        sido = props.get("시도")
        return {
            "fillColor": cmap(share) if share is not None else "#eef2f7",
            "color": "#64748b",
            "weight": 1.1,
            "fillOpacity": 0.98 if sido == selected_sido else 0.66,
        }

    tooltip_fields = ["시도", "대상", share_col]
    tooltip_aliases = ["시도", "산업", "사업체 비중"]
    if count_col:
        tooltip_fields.insert(2, "사업체수")
        tooltip_aliases.insert(2, "사업체수")
    folium.GeoJson(
        geo,
        name=f"2024년 {selected_target} 비중",
        style_function=style,
        tooltip=folium.GeoJsonTooltip(
            fields=tooltip_fields,
            aliases=tooltip_aliases,
            localize=True,
        ),
    ).add_to(fmap)
    add_major_site_markers(fmap)
    add_vertical_legend(
        fmap,
        caption=f"2024 {selected_target} 비중",
        colors=["#ffffd9", "#c7e9b4", "#41b6c4", "#225ea8"],
        minimum=value_min,
        maximum=value_max,
    )
    folium.LayerControl(collapsed=True).add_to(fmap)
    return fmap


def make_sigungu_factor_map(factor_label: str) -> folium.Map:
    """Build a city/county map for one contextual location factor."""
    df = load_sigungu_factors()
    geo = json.loads(json.dumps(load_sigungu_geojson_dict()))
    source_col = SIGUNGU_FACTOR_COLUMNS[factor_label]
    map_col = "지도값"
    values = df[source_col].astype(float)
    if factor_label == "산업용 전력":
        display_values = np.log10(values.clip(lower=1))
        legend_caption = "산업용 전력 log10(kWh)"
    else:
        display_values = values
        legend_caption = factor_label
    value_map = dict(zip(df["지도키"], display_values))
    raw_map = dict(zip(df["지도키"], values))
    port_map = dict(zip(df["지도키"], df["최근접무역항"]))

    valid_values: list[float] = []
    for feature in geo.get("features", []):
        props = feature["properties"]
        key = props["지도키"]
        value = value_map.get(key)
        props[map_col] = None if value is None or pd.isna(value) else float(value)
        props["원자료"] = None if key not in raw_map or pd.isna(raw_map[key]) else float(raw_map[key])
        props["최근접무역항"] = port_map.get(key, "")
        if props[map_col] is not None:
            valid_values.append(props[map_col])

    value_min = float(np.nanmin(valid_values))
    value_max = float(np.nanmax(valid_values))
    cmap = linear.YlGnBu_09.scale(value_min, value_max)
    reverse = factor_label == "항만 접근"

    def style(feature: dict) -> dict:
        value = feature["properties"].get(map_col)
        if value is None:
            fill = "#e5e7eb"
        else:
            color_value = value_max - (value - value_min) if reverse else value
            fill = cmap(color_value)
        return {"fillColor": fill, "color": "#cbd5e1", "weight": 0.55, "fillOpacity": 0.78}

    fmap = folium.Map(location=[36.4, 127.8], zoom_start=7, tiles="OpenStreetMap", control_scale=True)
    fields = ["시도", "시군구_표준", "원자료"]
    aliases = ["시도", "시군구", factor_label]
    if factor_label == "항만 접근":
        fields.append("최근접무역항")
        aliases.append("최근접 무역항")
    folium.GeoJson(
        geo,
        name=f"시군구 {factor_label}",
        style_function=style,
        tooltip=folium.GeoJsonTooltip(fields=fields, aliases=aliases, localize=True),
    ).add_to(fmap)
    add_major_site_markers(fmap)
    add_vertical_legend(
        fmap,
        caption=legend_caption + (" · 가까울수록 진함" if reverse else ""),
        colors=["#ffffd9", "#c7e9b4", "#41b6c4", "#225ea8"],
        minimum=value_min,
        maximum=value_max,
        reverse=reverse,
    )
    folium.LayerControl(collapsed=True).add_to(fmap)
    return fmap


def render_factor_cards(selected_sido: str) -> None:
    """Render four refined location-factor cards for a selected province."""
    df = load_integrated_df()
    row = df[df["시도"] == selected_sido].iloc[0]
    cards = [
        {
            "title": "항만 접근",
            "value": f"{row['하역능력_합계']:,.0f}",
            "unit": "톤",
            "sub": f"log값 {row['log_하역능력_합계']:.2f} · 무역항 {row['무역항수']:.0f}개",
            "rank_col": "log_하역능력_합계",
        },
        {
            "title": "도로 인프라",
            "value": f"{row['IC밀도_개당1000km2']:.1f}",
            "unit": "개 / 1,000㎢",
            "sub": f"고속도로 IC 총 {row['IC수']:.0f}개",
            "rank_col": "IC밀도_개당1000km2",
        },
        {
            "title": "산업용 전력",
            "value": f"{row['log_산업용전력_2024']:.2f}",
            "unit": "log(GWh)",
            "sub": "2024년 산업용 전력 사용량",
            "rank_col": "log_산업용전력_2024",
        },
        {
            "title": "평균임금",
            "value": f"{row['평균임금_2024_만원']:.0f}",
            "unit": "만원 / 월",
            "sub": f"백만원 환산 {row['평균임금_2024_백만원']:.2f}",
            "rank_col": "평균임금_2024_백만원",
        },
    ]

    cols = st.columns(2)
    for idx, card in enumerate(cards):
        col_values = df[card["rank_col"]]
        mean = float(col_values.mean())
        std = float(col_values.std(ddof=0))
        raw_value = float(row[card["rank_col"]])
        z = (raw_value - mean) / std if std > 0 else 0.0
        rank = rank_of_value(df, card["rank_col"], selected_sido, ascending=False)
        rank_pct = (17 - rank + 1) / 17 * 100

        if z >= 1.0:
            tag, dot = "매우 높음", "#dc2626"
        elif z >= 0.3:
            tag, dot = "약간 높음", "#d97706"
        elif z >= -0.3:
            tag, dot = "평균 수준", "#a3a3a3"
        elif z >= -1.0:
            tag, dot = "약간 낮음", "#2563eb"
        else:
            tag, dot = "매우 낮음", "#1e3a8a"

        with cols[idx % 2]:
            st.markdown(
                f"""
                <div class="factor-card">
                  <div class="factor-card-head">
                    <div class="factor-card-title">{card['title']}</div>
                    <div class="factor-card-rank"><strong>{rank}</strong> / 17위</div>
                  </div>
                  <div class="factor-card-value">{card['value']}<span class="factor-card-value-unit">{card['unit']}</span></div>
                  <div class="factor-card-sub">{card['sub']}</div>
                  <div class="factor-card-bar"><div class="factor-card-bar-fill" style="width:{rank_pct:.0f}%"></div></div>
                  <div class="factor-card-bar-foot">
                    <span>17위</span>
                    <span>1위</span>
                  </div>
                  <div class="factor-card-tag">
                    <span class="dot" style="background:{dot}"></span>
                    평균 대비 {z:+.2f}σ · {tag}
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_matching_recommendation(selected_sido: str) -> None:
    """Render top 3 matched industries for a selected province."""
    df = load_integrated_df()
    row = df[df["시도"] == selected_sido].iloc[0]
    raw = {factor: float(row[col]) for factor, col in FACTOR_COLUMNS.items()}
    top, contributions, _ = recommend_industries(raw, top_n=3)
    st.markdown('<div class="section-title">어울리는 산업 TOP 3</div>', unsafe_allow_html=True)
    st.markdown('<p class="section-sub">이 시도의 입지 조건과 가장 잘 맞는 산업입니다.</p>', unsafe_allow_html=True)
    for idx, (industry, score) in enumerate(top.items()):
        dominant = contributions.loc[industry].abs().sort_values(ascending=False).index[0]
        st.markdown(
            f"""
            <div class="recommend-card">
              <b>{idx + 1}. {industry}</b><span class="score-chip">적합도 <strong>{score:.0f}</strong> / 100</span><br>
              <span class="small-note">주요 기여: <b>{dominant}</b> · {industry_comment(industry, contributions.loc[industry])}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_time_series(selected_sido: str, selected_target: str, target_level: str) -> None:
    """Render selected province and national mean time-series."""
    panel = load_y_panel()
    if target_level == "6역할":
        role = next(key for key, label in ROLE_LABELS.items() if label == selected_target)
        industries = industry_role_members()[role]
        industry_df = (
            panel[panel["분석그룹"].isin(industries)]
            .groupby(["시도", "연도"], as_index=False)["사업체수"]
            .sum()
        )
    else:
        industry_df = panel[panel["분석그룹"] == selected_target].copy()
    selected = industry_df[industry_df["시도"] == selected_sido][["연도", "사업체수"]].copy()
    selected["구분"] = selected_sido
    mean_df = industry_df.groupby("연도", as_index=False)["사업체수"].mean()
    mean_df["구분"] = "17개 시도 평균"
    chart_df = pd.concat([selected, mean_df], ignore_index=True)
    fig = px.line(chart_df, x="연도", y="사업체수", color="구분", markers=True, title=f"{selected_sido} · {selected_target} 사업체수 추이")
    fig.update_traces(line=dict(width=4), marker=dict(size=12, line=dict(width=2, color="#ffffff")))
    for trace in fig.data:
        if trace.name == "17개 시도 평균":
            trace.line.dash = "dash"
            trace.line.color = CHART_MUTED
            trace.marker.color = CHART_MUTED
        else:
            trace.line.color = CHART_PRIMARY
            trace.marker.color = CHART_PRIMARY
            trace.mode = "lines+markers+text"
            trace.text = [f"{int(y):,}" for y in trace.y]
            trace.textposition = "top center"
            trace.textfont = dict(size=12, color=CHART_INK, family=CHART_FONT)
    apply_chart_theme(fig, height=250)
    fig.update_layout(legend_title_text="", hovermode="x unified", margin=dict(l=16, r=16, t=44, b=8))
    y_min = float(chart_df["사업체수"].min())
    y_max = float(chart_df["사업체수"].max())
    if y_max > y_min:
        span = y_max - y_min
        fig.update_yaxes(range=[max(0, y_min - span * 0.12), y_max + span * 0.30])
    else:
        fig.update_yaxes(range=[0, y_max * 1.3 if y_max > 0 else 1])
    st.plotly_chart(fig, width="stretch")


def correlation_variables() -> dict[str, str]:
    """Return province-level variables available to the correlation explorer."""
    variables = {
        "항만 인프라(log 하역능력)": FACTOR_COLUMNS["항만"],
        "IC 밀도": FACTOR_COLUMNS["IC"],
        "산업용 전력(log)": FACTOR_COLUMNS["전력"],
        "평균임금(백만원)": FACTOR_COLUMNS["임금"],
    }
    for role in ROLE_HYPOTHESIS_ORDER:
        variables[f"{ROLE_LABELS[role]} 산업 비중"] = f"역할비중_{role}"
    return variables


def _correlation_interpretation(value: float, method: str) -> str:
    """Create a plain-language interpretation for a correlation coefficient."""
    strength = "매우 강한" if abs(value) >= 0.8 else "강한" if abs(value) >= 0.6 else "중간" if abs(value) >= 0.4 else "약한" if abs(value) >= 0.2 else "거의 없는"
    direction = "양의" if value > 0 else "음의" if value < 0 else "방향 없는"
    return f"{method} 기준 {strength} {direction} 연관성 패턴입니다."


def render_correlation_explorer() -> None:
    """Render Pearson, Spearman, and Kendall correlation exploration."""
    st.markdown('<div class="section-title">상관 탐색기</div>', unsafe_allow_html=True)
    st.markdown('<p class="section-sub">시도 17개 안에서 두 변수를 선택해 선형·순위 연관성을 함께 비교합니다.</p>', unsafe_allow_html=True)
    variables = correlation_variables()
    labels = list(variables)
    col1, col2 = st.columns(2)
    with col1:
        x_label = st.selectbox("X 변수", labels, index=labels.index("IC 밀도"), key="corr_x")
    with col2:
        y_label = st.selectbox("Y 변수", labels, index=labels.index("물류 산업 비중"), key="corr_y")
    df = load_analysis_df()[["시도", variables[x_label], variables[y_label]]].dropna().copy()
    x = df[variables[x_label]]
    y = df[variables[y_label]]
    pearson = scipy_stats.pearsonr(x, y)
    spearman = scipy_stats.spearmanr(x, y)
    kendall = scipy_stats.kendalltau(x, y)
    methods = [
        ("Pearson", pearson.statistic, pearson.pvalue),
        ("Spearman", spearman.statistic, spearman.pvalue),
        ("Kendall", kendall.statistic, kendall.pvalue),
    ]
    metric_cols = st.columns(3)
    for col, (method, value, p_value) in zip(metric_cols, methods):
        with col:
            st.metric(method, f"{value:+.3f}", delta=f"p={p_value:.3f}", delta_color="off")
            st.caption(_correlation_interpretation(float(value), method))
    render_badges("산점도", "상관")
    fig = px.scatter(df, x=variables[x_label], y=variables[y_label], text="시도", trendline="ols", title=f"{x_label} × {y_label}")
    fig.update_traces(textposition="top center", marker=dict(size=11, color=CHART_PRIMARY, line=dict(color="white", width=1.5)))
    apply_chart_theme(fig, height=440, show_legend=False)
    fig.update_xaxes(title=x_label)
    fig.update_yaxes(title=y_label)
    st.plotly_chart(fig, width="stretch")


@st.cache_data(show_spinner=False)
def build_seaborn_distribution_images() -> tuple[bytes, bytes]:
    """Build role boxplot and factor pairplot images with seaborn."""
    df = load_analysis_df()
    role_columns = {f"역할비중_{role}": ROLE_LABELS[role] for role in ROLE_HYPOTHESIS_ORDER}
    long = df[["시도", *role_columns]].melt(id_vars="시도", var_name="역할", value_name="사업체 비중")
    long["역할"] = long["역할"].map(role_columns)
    fig, ax = plt.subplots(figsize=(11, 5.5), constrained_layout=True)
    sns.boxplot(data=long, x="역할", y="사업체 비중", color="#93c5fd", ax=ax)
    sns.stripplot(data=long, x="역할", y="사업체 비중", color="#1e3a8a", alpha=0.6, size=3.5, ax=ax)
    ax.set_title("6개 산업 역할별 시도 분포")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=20)
    box_buffer = BytesIO()
    fig.savefig(box_buffer, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    pair_df = df[[*FACTOR_COLUMNS.values(), "권역유형"]].rename(
        columns={column: label for label, column in FACTOR_COLUMNS.items()}
    )
    grid = sns.pairplot(
        pair_df,
        vars=list(FACTOR_COLUMNS),
        hue="권역유형",
        corner=True,
        diag_kind="hist",
        plot_kws={"s": 42, "alpha": 0.75},
    )
    grid.fig.suptitle("4대 입지요인 pairplot", y=1.02)
    pair_buffer = BytesIO()
    grid.fig.savefig(pair_buffer, format="png", dpi=140, bbox_inches="tight")
    plt.close(grid.fig)
    return box_buffer.getvalue(), pair_buffer.getvalue()


def render_distribution_section() -> None:
    """Render seaborn role distributions and factor pairplot."""
    st.markdown('<div class="section-title">분포 시각화</div>', unsafe_allow_html=True)
    st.markdown('<p class="section-sub">산업 역할별 편차와 4대 입지요인의 동시 분포를 확인합니다.</p>', unsafe_allow_html=True)
    box_image, pair_image = build_seaborn_distribution_images()
    tab1, tab2 = st.tabs(["역할별 boxplot", "4요인 pairplot"])
    with tab1:
        render_badges("박스플롯", "분포")
        st.image(box_image, width="stretch")
    with tab2:
        render_badges("페어플롯", "상관")
        st.image(pair_image, width="stretch")


def render_explore_mode(selected_sido: str, selected_target: str) -> None:
    """Render a compact exploration view — map and factor summary, with a time-series below."""
    left, right = st.columns([3, 2])
    with left:
        st.markdown('<div class="section-title">시도별 산업 비중 지도</div>', unsafe_allow_html=True)
        st.markdown(
            f'<p class="section-sub">2024년 고정 · {selected_target} 사업체 비중 · 진할수록 집중도 높음</p>',
            unsafe_allow_html=True,
        )
        st_folium(make_industry_map(selected_sido, selected_target, "26산업"), height=370, width=780, returned_objects=[])
    with right:
        st.markdown('<div class="section-title">입지요인 요약</div>', unsafe_allow_html=True)
        st.markdown(f'<p class="section-sub">{selected_sido} · 항만 · 도로 · 전력 · 임금</p>', unsafe_allow_html=True)
        render_factor_cards(selected_sido)

    st.markdown('<div class="section-title">2020~2024 시계열 변화</div>', unsafe_allow_html=True)
    st.markdown(
        f'<p class="section-sub">{selected_sido} · {selected_target} vs 17개 시도 평균</p>',
        unsafe_allow_html=True,
    )
    render_time_series(selected_sido, selected_target, "26산업")


def support_status(conclusion: str) -> tuple[str, str]:
    """Return (label, css-class) for a hypothesis conclusion."""
    if "지지" in conclusion and "부분" not in conclusion:
        return "지지", "status-support"
    if "부분" in conclusion:
        return "부분지지", "status-partial"
    return "제한적", "status-limited"


HYP_STATUS_DOT = {"지지": "🟢", "부분지지": "🟡", "제한적": "🔴", "탐색적": "🔵"}


def render_hypothesis_cards() -> str:
    """Render 8 clickable H1-H8 cards and return the selected hypothesis number."""
    summary = build_hypothesis_summary_live()
    if "selected_hyp" not in st.session_state:
        st.session_state["selected_hyp"] = "H1"
    records = summary.to_dict("records")
    for r in range(2):
        cols = st.columns(4)
        for c in range(4):
            row = records[r * 4 + c]
            h = row["가설"]
            label, _ = support_status(str(row["결론"]))
            if row["결론"] == "탐색적":
                label = "탐색적"
            dot = HYP_STATUS_DOT.get(label, "⚪")
            is_selected = st.session_state["selected_hyp"] == h
            if cols[c].button(
                f"{h} · {dot} {label}\n\n{row['한줄']}",
                key=f"hyp_btn_{h}",
                use_container_width=True,
                type="primary" if is_selected else "secondary",
            ):
                st.session_state["selected_hyp"] = h
                st.rerun()
    return st.session_state["selected_hyp"]


def _significance_stars(p_value: float) -> str:
    """Return conventional significance stars."""
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return ""


def render_regression_scatter(
    *,
    y_col: str,
    x_col: str,
    title: str,
    x_label: str,
    y_label: str,
) -> None:
    """Render a labeled simple-regression scatter plot."""
    df = load_analysis_df()[["시도", x_col, y_col]].dropna()
    result = fit_simple_regression(df, y_col, x_col)
    render_badges("산점도", "회귀")
    fig = px.scatter(df, x=x_col, y=y_col, text="시도", trendline="ols", title=title)
    fig.update_traces(textposition="top center", marker=dict(size=12, color=CHART_PRIMARY, line=dict(color="white", width=1.4)))
    apply_chart_theme(fig, height=500, show_legend=False)
    fig.update_xaxes(title=x_label)
    fig.update_yaxes(title=y_label)
    fig.add_annotation(
        x=0.02,
        y=0.98,
        xref="paper",
        yref="paper",
        text=f"표준화β={result['표준화β']:.2f} · R²={result['R²']:.2f} · p={result['p_value']:.3f}",
        showarrow=False,
        align="left",
        bgcolor="rgba(255,255,255,.9)",
        bordercolor="#bfdbfe",
        borderpad=7,
    )
    st.plotly_chart(fig, width="stretch")


def render_h4_wage_chart() -> None:
    """Render H4 wage betas with significance and sign encoding."""
    results = build_beta_results("산업")
    targets = ["정보통신·IT", "금융·보험", "전문·과학·기술", "의료정밀·전기·기계", "섬유·의복·가죽", "목재·종이·인쇄"]
    plot_df = results[(results["요인"] == "임금") & results["대상"].isin(targets)].copy()
    plot_df["유의도"] = np.where(plot_df["p_value"] < 0.05, "유의(p<0.05)", "비유의")
    plot_df["표시"] = plot_df.apply(lambda row: f"{row['표준화β']:+.2f}", axis=1)
    plot_df = plot_df.sort_values("표준화β")
    render_badges("막대그래프", "회귀")
    fig = px.bar(
        plot_df,
        x="표준화β",
        y="대상",
        color="유의도",
        orientation="h",
        text="표시",
        color_discrete_map={"유의(p<0.05)": "#1d4ed8", "비유의": "#cbd5e1"},
        title="H4 임금과 자본·노동집약 산업 비중의 연관성",
    )
    fig.add_vline(x=0, line_color="#475569", line_width=1)
    apply_chart_theme(fig, height=470)
    fig.update_layout(legend_title_text="통계적 유의도")
    fig.update_xaxes(title="표준화 β · 음수는 임금이 높을수록 비중이 낮은 패턴")
    fig.update_yaxes(title="")
    st.plotly_chart(fig, width="stretch")


def render_factor_competition_chart() -> None:
    """Render H5 four-factor explanatory-strength ranking."""
    competition = build_factor_competition().sort_values("평균_R²")
    render_badges("순위 막대", "회귀")
    fig = px.bar(
        competition,
        x="평균_R²",
        y="요인",
        orientation="h",
        color="평균_절대_표준화β",
        text=competition.apply(lambda row: f"{int(row['순위'])}위 · 유의산업 {int(row['유의산업수'])}개", axis=1),
        color_continuous_scale="Blues",
        title="H5 4대 요인 영향력 순위 · 26개 산업 평균",
    )
    apply_chart_theme(fig, height=430, show_legend=False)
    fig.update_xaxes(title="26개 산업 단순회귀 평균 R²")
    fig.update_yaxes(title="")
    st.plotly_chart(fig, width="stretch")


def render_beta_heatmap() -> None:
    """Render the H6 standardized-beta heatmap for all 26 industries with every cell labeled."""
    results = build_beta_results("산업")
    order = get_industries()
    beta = results.pivot(index="대상", columns="요인", values="표준화β").reindex(order)[list(FACTOR_COLUMNS)]
    pvals = results.pivot(index="대상", columns="요인", values="p_value").reindex(order)[list(FACTOR_COLUMNS)]
    fdr = results.pivot(index="대상", columns="요인", values="p_FDR").reindex(order)[list(FACTOR_COLUMNS)]
    text = np.empty(beta.shape, dtype=object)
    hover = np.empty(beta.shape, dtype=object)
    for row_idx, target in enumerate(beta.index):
        for col_idx, factor in enumerate(beta.columns):
            value = beta.loc[target, factor]
            p_value = pvals.loc[target, factor]
            q_value = fdr.loc[target, factor]
            text[row_idx, col_idx] = "" if pd.isna(value) else f"{value:+.2f}"
            hover[row_idx, col_idx] = f"{target}<br>{factor}<br>표준화β={value:+.3f}<br>p={p_value:.4f}<br>FDR q={q_value:.4f}"
    render_badges("히트맵", "회귀")
    fig = go.Figure(
        go.Heatmap(
            z=beta.to_numpy(),
            x=beta.columns,
            y=beta.index,
            zmin=-1,
            zmax=1,
            zmid=0,
            colorscale="RdBu",
            reversescale=True,
            text=text,
            texttemplate="%{text}",
            textfont=dict(size=12),
            customdata=hover,
            hovertemplate="%{customdata}<extra></extra>",
            colorbar=dict(title="표준화 β"),
            xgap=2,
            ygap=2,
        )
    )
    fig.update_layout(title="H6 산업 × 입지요인 표준화 β · 26개 산업 전체 셀 표시", height=920)
    apply_chart_theme(fig, height=920, show_legend=False)
    fig.update_xaxes(side="top", title="")
    fig.update_yaxes(title="", autorange="reversed")
    st.plotly_chart(fig, width="stretch")
    significant = int((fdr < 0.05).sum().sum())
    st.info(
        f"결론: 26개 산업 × 4개 요인 {beta.size}개 셀의 표준화 β를 모두 표시했습니다. "
        f"이 중 통계적으로 유의한(FDR q<0.05) 셀은 {significant}개이며, "
        "산업별로 대표 입지요인이 서로 갈리는 연관성 패턴이 나타납니다."
    )


def _target_column(target_level: str, target: str) -> str:
    """Resolve a role or industry UI target to its share column."""
    if target_level == "6역할":
        role = next(key for key, label in ROLE_LABELS.items() if label == target)
        return f"역할비중_{role}"
    return f"비중_{target}"


def _target_count_column(target_level: str, target: str) -> str:
    """Resolve a role or industry UI target to its establishment-count column."""
    if target_level == "6역할":
        role = next(key for key, label in ROLE_LABELS.items() if label == target)
        return f"역할사업체수_{role}"
    return target


def make_concentration_map(target: str, target_level: str) -> folium.Map:
    """Build an H7 province map from each province's share of national establishments."""
    df = load_analysis_df()
    count_col = _target_count_column(target_level, target)
    counts = pd.to_numeric(df[count_col], errors="coerce").fillna(0.0)
    total = float(counts.sum())
    national_share = counts / total if total > 0 else pd.Series(0.0, index=df.index)
    count_map = dict(zip(df["시도"], counts))
    share_map = dict(zip(df["시도"], national_share))
    top_sido = df.loc[national_share.idxmax(), "시도"]

    geo = json.loads(json.dumps(load_geojson_dict()))
    for feature in geo["features"]:
        sido = feature["properties"]["시도"]
        feature["properties"]["대상"] = target
        feature["properties"]["사업체수"] = float(count_map.get(sido, 0.0))
        feature["properties"]["전국비중"] = float(share_map.get(sido, 0.0))

    values = [feature["properties"]["전국비중"] for feature in geo["features"]]
    value_min = float(min(values))
    value_max = float(max(values))
    cmap = linear.YlOrRd_09.scale(value_min, value_max)

    def style(feature: dict) -> dict:
        props = feature["properties"]
        return {
            "fillColor": cmap(props["전국비중"]),
            "color": "#64748b",
            "weight": 1.1,
            "fillOpacity": 0.98 if props["시도"] == top_sido else 0.7,
        }

    fmap = folium.Map(location=[36.4, 127.8], zoom_start=7, tiles="OpenStreetMap", control_scale=True)
    folium.GeoJson(
        geo,
        name=f"{target} 전국 사업체 분포",
        style_function=style,
        tooltip=folium.GeoJsonTooltip(
            fields=["시도", "대상", "사업체수", "전국비중"],
            aliases=["시도", "산업", "사업체수", "전국 사업체 비중"],
            localize=True,
        ),
    ).add_to(fmap)
    add_major_site_markers(fmap)
    add_vertical_legend(
        fmap,
        caption=f"{target} 전국 사업체 비중",
        colors=["#ffffcc", "#feb24c", "#f03b20", "#800026"],
        minimum=value_min,
        maximum=value_max,
    )
    folium.LayerControl(collapsed=True).add_to(fmap)
    return fmap


def render_h7_concentration() -> None:
    """Render province industry concentration and contextual city/county factors."""
    level = "26산업"
    c1, c2 = st.columns([2, 1.5])
    with c1:
        target = st.selectbox("산업", get_industries(), key="h7_target")
    with c2:
        factor = st.selectbox("시군구 보완 요인", list(SIGUNGU_FACTOR_COLUMNS), key="h7_factor")
    df = load_analysis_df()
    count_col = _target_count_column(level, target)
    ranking = df[["시도", count_col]].rename(columns={count_col: "사업체수"})
    ranking["전국 사업체 비중"] = ranking["사업체수"] / ranking["사업체수"].sum()
    ranking = ranking.sort_values("전국 사업체 비중", ascending=False)
    top3_share = ranking.head(3)["전국 사업체 비중"].sum()
    left, right = st.columns([1, 1])
    with left:
        render_badges("단계구분도", "공간")
        st_folium(make_concentration_map(target, level), height=560, width=700, returned_objects=[])
    with right:
        render_badges("단계구분도", "공간")
        st_folium(make_sigungu_factor_map(factor), height=560, width=700, returned_objects=[])
    st.dataframe(
        ranking.head(5).style.format({"사업체수": "{:,.0f}", "전국 사업체 비중": "{:.2%}"}),
        width="stretch",
        hide_index=True,
    )
    st.info(f"결론: {target} 사업체가 많은 상위 3개 시도가 전국 해당 사업체의 {top3_share:.1%}를 차지하는 지역 편중 패턴입니다.")


def make_residual_map(residuals: pd.DataFrame, target: str) -> folium.Map:
    """Build an H8 candidate map from predicted-minus-actual share gaps."""
    geo = json.loads(json.dumps(load_geojson_dict()))
    gap_map = residuals.set_index("시도")["유치여지"].to_dict()
    actual_map = residuals.set_index("시도")["실제"].to_dict()
    predicted_map = residuals.set_index("시도")["예측"].to_dict()
    max_abs = max(abs(residuals["유치여지"].min()), abs(residuals["유치여지"].max()))
    cmap = linear.RdBu_11.scale(-max_abs, max_abs)
    for feature in geo["features"]:
        sido = feature["properties"]["시도"]
        feature["properties"]["유치여지"] = float(gap_map[sido])
        feature["properties"]["실제"] = float(actual_map[sido])
        feature["properties"]["예측"] = float(predicted_map[sido])

    fmap = folium.Map(location=[36.4, 127.8], zoom_start=7, tiles="OpenStreetMap")
    folium.GeoJson(
        geo,
        name=f"{target} 유치 후보",
        style_function=lambda feature: {
            "fillColor": cmap(feature["properties"]["유치여지"]),
            "color": "#64748b",
            "weight": 1,
            "fillOpacity": 0.78,
        },
        tooltip=folium.GeoJsonTooltip(
            fields=["시도", "예측", "실제", "유치여지"],
            aliases=["시도", "예측 비중", "실제 비중", "예측-실제"],
            localize=True,
        ),
    ).add_to(fmap)
    add_vertical_legend(
        fmap,
        caption="예측-실제 · 양수=유치여지",
        colors=["#2166ac", "#f7f7f7", "#b2182b"],
        minimum=-max_abs,
        maximum=max_abs,
    )
    return fmap


def render_h8_candidates() -> None:
    """Render predicted-minus-actual attraction candidates."""
    level = "26산업"
    options = get_industries()
    default = options.index("반도체·전자") if "반도체·전자" in options else 0
    target = st.selectbox("산업 선택", options, index=default, key="h8_target")
    y_col = _target_column(level, target)
    model, residuals = fit_multifactor_regression(load_analysis_df(), y_col)
    candidates = residuals.sort_values("유치여지", ascending=False).reset_index(drop=True)
    left, right = st.columns([3, 2])
    with left:
        render_badges("잔차 지도", "회귀")
        st_folium(make_residual_map(residuals, target), height=570, width=780, returned_objects=[])
    with right:
        render_badges("후보 표", "회귀")
        display = candidates.head(7).copy()
        display.insert(0, "순위", np.arange(1, len(display) + 1))
        st.dataframe(
            display.style.format({"실제": "{:.3%}", "예측": "{:.3%}", "유치여지": "{:+.3%}"}),
            width="stretch",
            hide_index=True,
        )
        st.caption("유치여지 = 입지요인으로 예측한 비중 - 실제 비중. 양수일수록 조건 대비 산업 비중이 낮은 지역입니다.")
    st.info(f"결론: {target} 모형 R²={model.rsquared:.2f}. {candidates.iloc[0]['시도']}가 가장 큰 양의 잔차를 보여 탐색적 유치 후보로 제시됩니다.")


def render_selected_hypothesis(h_num: str) -> None:
    """Render exactly one visualization and one conclusion for a hypothesis."""
    summary = build_hypothesis_summary_live().set_index("가설")
    row = summary.loc[h_num]
    st.markdown(f"### {h_num}. {row['한줄']}")
    st.caption(row["통계"])
    if h_num == "H1":
        st.caption("회귀는 시도 항만 인프라를 사용하고, 시군구 항만거리는 지도 설명에만 사용합니다.")
        render_regression_scatter(
            y_col="역할비중_heavy_export",
            x_col=FACTOR_COLUMNS["항만"],
            title="H1 항만 인프라와 중량수출 산업 비중",
            x_label="log(시도 하역능력+1)",
            y_label="중량수출 산업 비중",
        )
    elif h_num == "H2":
        render_regression_scatter(
            y_col="역할비중_logistics",
            x_col=FACTOR_COLUMNS["IC"],
            title="H2 IC 밀도와 물류 산업 비중",
            x_label="IC 밀도(개/1,000km²)",
            y_label="물류 역할 비중 · 도매+소매+운송",
        )
    elif h_num == "H3":
        render_regression_scatter(
            y_col="비중_반도체·전자",
            x_col=FACTOR_COLUMNS["전력"],
            title="H3 산업용 전력과 반도체·전자 비중",
            x_label="log 산업용 전력",
            y_label="반도체·전자 사업체 비중",
        )
    elif h_num == "H4":
        render_h4_wage_chart()
    elif h_num == "H5":
        render_factor_competition_chart()
    elif h_num == "H6":
        render_beta_heatmap()
    elif h_num == "H7":
        render_h7_concentration()
    elif h_num == "H8":
        render_h8_candidates()
    st.success(f"한 줄 결론: {row['결론문']} 인과가 아니라 연관성·패턴으로 해석합니다.")


DATA_SOURCES = [
    ("종속변수 Y · 사업체수", "통계청 KOSIS 전국사업체조사 (DT_1K52F01, 산업 중분류)", "2020~2024"),
    ("H1 항만 · 하역능력/무역항수", "해양수산부 항만정보 (data.go.kr/15088273)", "2023"),
    ("H1 항만 · 물동량", "해양수산부 항만별 물동량 통계", "2023"),
    ("H2 고속도로 IC", "한국도로공사 IC 위치 (data.go.kr/15112762)", "2024"),
    ("H3 산업용 전력", "한국전력거래소 EPSIS (data.go.kr/15054416)", "2020~2024"),
    ("H4 평균임금", "고용노동부 사업체노동력조사 (data.go.kr/3069922)", "2020~2024"),
    ("산업분류 매핑", "통계청 한국표준산업분류(KSIC) · 71 중분류→26 분석그룹", "—"),
    ("시도 경계 지도", "southkorea-maps 행정구역 GeoJSON (KOSTAT 2018)", "2018"),
]

SIGUNGU_SOURCES = [
    ("시군구 산업용 전력", "한국전력공사(KEPCO) 시군구별 전력판매량(산업용)", "2023~2024"),
    ("시군구 평균급여", "국세청 시·군·구별 근로소득 연말정산(주소지 기준)", "2023"),
    ("시군구 IC·항만거리·면적", "한국도로공사 IC 좌표 + 해수부 무역항 + southkorea-maps 2018 경계", "2018~2024"),
]

DATASET_FILES = [
    ("07_통합분석_2024.xlsx", "회귀 메인 입력 · 17 시도 × 68 변수", "시도"),
    ("02_사업체수_패널.xlsx", "5년 패널 사업체수 (2,210행)", "시도×연도"),
    ("02_사업체수_2024.xlsx", "2024 단년 사업체수 (17×28)", "시도"),
    ("01_산업매핑.xlsx", "KSIC 71 중분류 → 26 분석그룹 매핑", "산업분류"),
    ("03_H1_항만.xlsx", "항만수·하역능력·물동량", "시도"),
    ("04_H2_IC.xlsx", "면적·IC수·IC밀도", "시도"),
    ("05_H3_전력.xlsx", "5년치 산업용 전력 + log 변환", "시도"),
    ("06_H4_임금.xlsx", "5년치 평균임금", "시도"),
    ("시군구_입지요인.csv", "229 시군구 × 4요인 (지도 전용)", "시군구"),
    ("시군구_경계.geojson", "시군구 경계 (지도 전용)", "시군구"),
    ("지도_시도경계.json", "시도 경계 GeoJSON", "시도"),
]


def render_data_sources() -> None:
    """Render data sources and dataset list at the bottom of the hypothesis tab."""
    st.markdown('<div class="section-title">사용된 자료 출처 · 데이터셋 목록</div>', unsafe_allow_html=True)
    st.markdown(
        '<p class="section-sub">회귀 본체는 시도 N=17 자료, 시군구 자료는 지도·맥락 설명 전용입니다. '
        '모든 출처는 공개 통계(KOSIS·data.go.kr 등)입니다.</p>',
        unsafe_allow_html=True,
    )
    tab1, tab2, tab3 = st.tabs(["원본 자료 출처", "시군구 보조 자료", "데이터셋 파일"])
    with tab1:
        st.dataframe(
            pd.DataFrame(DATA_SOURCES, columns=["항목", "출처", "기준연도"]),
            width="stretch",
            hide_index=True,
        )
    with tab2:
        st.dataframe(
            pd.DataFrame(SIGUNGU_SOURCES, columns=["항목", "출처", "기준연도"]),
            width="stretch",
            hide_index=True,
        )
        st.caption("Y(사업체수)가 시군구×산업 중분류로 공개되지 않아, 시군구 4요인은 회귀가 아닌 지도 시각화에만 사용합니다.")
    with tab3:
        st.dataframe(
            pd.DataFrame(DATASET_FILES, columns=["파일", "내용", "분석단위"]),
            width="stretch",
            hide_index=True,
        )
    st.caption(
        "강건성 참고: 5년 패널 고정효과(FE) 보조검증은 results/tables/h7_panel.xlsx에 포함되며, "
        "항만·IC는 시점불변이라 시도 FE에서 제외됩니다. 모든 결과는 인과가 아니라 연관성·패턴으로 해석합니다."
    )


def render_hypothesis_mode() -> None:
    """Render the replacement H1-H8 flow."""
    st.markdown('<div class="section-title">새 H1~H8 가설 검증</div>', unsafe_allow_html=True)
    st.markdown(
        '<p class="section-sub">아래 8개 카드를 클릭하면 선택한 가설의 상세 결과가 표시됩니다. 회귀는 시도 N=17, 시군구는 지도 전용입니다.</p>',
        unsafe_allow_html=True,
    )
    selected = render_hypothesis_cards()
    render_selected_hypothesis(selected)
    render_data_sources()


def top_real_sidos_for_recommendations(top_industries: list[str]) -> str:
    """Find provinces where the recommended industries are most concentrated."""
    df = load_integrated_df()
    share_cols = [f"비중_{industry}" for industry in top_industries if f"비중_{industry}" in df.columns]
    ranking = df.assign(추천산업_합산비중=df[share_cols].sum(axis=1)).sort_values("추천산업_합산비중", ascending=False)
    return ", ".join(ranking.head(5)["시도"].tolist())


def render_zscore_chart(x_user_z: pd.Series) -> None:
    """Render an intuitive horizontal z-score chart with reference bands."""
    z_df = x_user_z.reset_index()
    z_df.columns = ["요인", "z-score"]
    z_df = z_df.iloc[::-1].reset_index(drop=True)
    bar_colors = ["#b91c1c" if v >= 0 else "#1e40af" for v in z_df["z-score"]]
    text_labels = []
    for v in z_df["z-score"]:
        if v >= 1.0:
            tag = "매우 높음"
        elif v >= 0.3:
            tag = "약간 높음"
        elif v >= -0.3:
            tag = "평균 수준"
        elif v >= -1.0:
            tag = "약간 낮음"
        else:
            tag = "매우 낮음"
        text_labels.append(f"{v:+.2f}σ  ·  {tag}")
    fig = go.Figure()
    fig.add_vrect(x0=-0.3, x1=0.3, fillcolor="#f1f5f9", line_width=0, layer="below")
    fig.add_trace(go.Bar(
        x=z_df["z-score"], y=z_df["요인"], orientation="h",
        marker=dict(color=bar_colors, line=dict(width=0)),
        text=text_labels,
        textposition="outside",
        textfont=dict(size=14, family=CHART_FONT, color=CHART_INK),
        cliponaxis=False,
        hovertemplate="<b>%{y}</b><br>z-score: %{x:+.3f}σ<extra></extra>",
    ))
    fig.update_layout(title="입력 조건 표준화 점수 — 17개 시도 평균 대비 강도")
    apply_chart_theme(fig, height=320, show_legend=False)
    fig.add_vline(x=0, line_color="#64748b", line_width=2)
    max_abs = max(2.5, float(np.abs(z_df["z-score"]).max()) * 1.4)
    fig.update_xaxes(
        range=[-max_abs, max_abs],
        title_text="z-score  (음수 = 평균 미만,  양수 = 평균 초과,  단위: 표준편차 σ)",
        zeroline=False,
    )
    fig.update_yaxes(title_text="", tickfont=dict(size=14, color=CHART_INK))
    st.plotly_chart(fig, width="stretch")


def render_factor_distribution(user_raw: dict[str, float]) -> None:
    """Render box plot comparing user inputs to 17-province distributions — split per factor for clarity."""
    df = load_integrated_df()
    factor_labels = {"항만": "항만 (log)", "IC": "IC 밀도", "전력": "전력 (log)", "임금": "임금 (백만원)"}
    cols = st.columns(4)
    for idx, (factor, col_name) in enumerate(FACTOR_COLUMNS.items()):
        with cols[idx]:
            values = df[col_name].values
            user_val = user_raw[factor]
            mean_val = float(np.mean(values))
            fig = go.Figure()
            fig.add_trace(go.Box(
                y=values, name="17개 시도",
                marker=dict(color="#94a3b8", size=8),
                line=dict(color="#64748b", width=2),
                fillcolor="rgba(148,163,184,0.25)",
                boxpoints="all", jitter=0.4, pointpos=0,
                hovertemplate="%{y:.2f}<extra></extra>",
                showlegend=False,
            ))
            fig.add_trace(go.Scatter(
                y=[user_val], x=["17개 시도"], mode="markers+text",
                marker=dict(size=20, color="#b91c1c", symbol="diamond", line=dict(width=2.5, color="#ffffff")),
                text=[f"입력 {user_val:.2f}"],
                textposition="middle right",
                textfont=dict(size=12, color="#b91c1c", family=CHART_FONT),
                name="입력값",
                hovertemplate=f"<b>입력값</b><br>{user_val:.2f}<extra></extra>",
                showlegend=False,
            ))
            fig.update_layout(title=factor_labels[factor])
            apply_chart_theme(fig, height=340, show_legend=False)
            fig.add_hline(y=mean_val, line_color="#cbd5e1", line_dash="dot", line_width=1)
            fig.update_xaxes(showticklabels=False, title_text="")
            fig.update_yaxes(title_text="")
            st.plotly_chart(fig, width="stretch", key=f"factor_box_{factor}")


SIM_FACTOR_RANGES = {"항만": (0.0, 14.0, 0.1), "IC": (0.0, 32.0, 0.1), "전력": (13.0, 19.0, 0.1), "임금": (250, 450, 5)}


def _region_factor_values(region: str) -> dict[str, float]:
    """Return raw factor values for a region or the national mean."""
    df = load_integrated_df()
    if region == "전국 평균":
        return {
            "항만": float(df["log_하역능력_합계"].mean()),
            "IC": float(df["IC밀도_개당1000km2"].mean()),
            "전력": float(df["log_산업용전력_2024"].mean()),
            "임금": float(df["평균임금_2024_만원"].mean()),
        }
    row = df[df["시도"] == region].iloc[0]
    return {
        "항만": float(row["log_하역능력_합계"]),
        "IC": float(row["IC밀도_개당1000km2"]),
        "전력": float(row["log_산업용전력_2024"]),
        "임금": float(row["평균임금_2024_만원"]),
    }


def _apply_region_preset() -> None:
    """Streamlit callback — sync slider values to the chosen preset."""
    preset = st.session_state.get("sim_preset", "전국 평균")
    if preset == "직접 입력":
        return
    values = _region_factor_values(preset)
    st.session_state["sim_port"] = round(values["항만"], 2)
    st.session_state["sim_ic"] = round(values["IC"], 1)
    st.session_state["sim_power"] = round(values["전력"], 2)
    st.session_state["sim_wage"] = int(round(values["임금"] / 5) * 5)


def _ensure_sim_defaults() -> None:
    """Initialize session state for simulator sliders."""
    if "sim_preset" not in st.session_state:
        st.session_state["sim_preset"] = "전국 평균"
        defaults = _region_factor_values("전국 평균")
        st.session_state["sim_port"] = round(defaults["항만"], 2)
        st.session_state["sim_ic"] = round(defaults["IC"], 1)
        st.session_state["sim_power"] = round(defaults["전력"], 2)
        st.session_state["sim_wage"] = int(round(defaults["임금"] / 5) * 5)


def render_top5_ranking_chart(top: pd.Series, contributions: pd.DataFrame) -> None:
    """Render a clean horizontal bar chart for TOP 5 ranking with dominant factor labels."""
    plot_df = top.reset_index()
    plot_df.columns = ["산업", "적합도"]
    plot_df = plot_df.iloc[::-1].reset_index(drop=True)
    dominant_factors = [
        contributions.loc[industry].abs().sort_values(ascending=False).index[0]
        for industry in plot_df["산업"]
    ]
    n = len(plot_df)
    rank_colors = {1: "#d97706", 2: "#94a3b8", 3: "#b45309"}
    colors = []
    for i in range(n):
        rank = n - i
        colors.append(rank_colors.get(rank, "#cbd5e1"))
    labels = [f"{v:.0f}점  ·  {f}" for v, f in zip(plot_df["적합도"], dominant_factors)]
    fig = go.Figure(go.Bar(
        x=plot_df["적합도"],
        y=plot_df["산업"],
        orientation="h",
        marker=dict(color=colors, line=dict(width=0)),
        text=labels,
        textposition="outside",
        textfont=dict(size=15, family=CHART_FONT, color=CHART_INK),
        cliponaxis=False,
        hovertemplate="<b>%{y}</b><br>적합도: %{x:.1f}점<extra></extra>",
    ))
    fig.update_layout(title="추천 산업 TOP 5 — 적합도 + 핵심 기여 요인")
    apply_chart_theme(fig, height=380, show_legend=False)
    fig.update_xaxes(range=[0, 130], title_text="적합도 (0~100점)")
    fig.update_yaxes(title_text="", tickfont=dict(size=14, color=CHART_INK))
    st.plotly_chart(fig, width="stretch")


def render_simulator_mode() -> None:
    """Render a compact two-section industrial complex simulator."""
    _ensure_sim_defaults()

    st.markdown('<div class="section-title">신규 산업단지 조건 설정</div>', unsafe_allow_html=True)
    st.markdown(
        '<p class="section-sub">실제 시도를 불러오거나 4개 입지요인을 직접 조정합니다. 점수는 표준화 β 기반 상대 적합도입니다.</p>',
        unsafe_allow_html=True,
    )

    preset_options = ["전국 평균", "직접 입력"] + SIDO_ORDER
    st.selectbox(
        "지역 프리셋",
        preset_options,
        key="sim_preset",
        on_change=_apply_region_preset,
        help="실제 시도를 선택하면 그 지역의 4개 요인 값으로 슬라이더가 자동 설정됩니다.",
    )

    stats = factor_stats()
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        port_min, port_max, port_step = SIM_FACTOR_RANGES["항만"]
        st.slider(
            "항만 접근 — log(하역능력+1)",
            min_value=port_min, max_value=port_max, step=port_step,
            key="sim_port",
            help="0 = 내륙 · 8~10 = 일반 항만 · 12+ = 부산·인천급",
        )
        st.caption(f"17개 시도 분포: 최저 {stats.loc['항만','min']:.2f} · 평균 {stats.loc['항만','mean']:.2f} · 최고 {stats.loc['항만','max']:.2f}")
    with c2:
        power_min, power_max, power_step = SIM_FACTOR_RANGES["전력"]
        st.slider(
            "산업용 전력 — log(GWh)",
            min_value=power_min, max_value=power_max, step=power_step,
            key="sim_power",
            help="15 = 서울·제주 · 17 = 일반 광역도 · 18+ = 경기·충남급",
        )
        st.caption(f"17개 시도 분포: 최저 {stats.loc['전력','min']:.2f} · 평균 {stats.loc['전력','mean']:.2f} · 최고 {stats.loc['전력','max']:.2f}")
    with c3:
        ic_min, ic_max, ic_step = SIM_FACTOR_RANGES["IC"]
        st.slider(
            "고속도로 IC 밀도 — 개/천㎢",
            min_value=ic_min, max_value=ic_max, step=ic_step,
            key="sim_ic",
            help="5 미만 = 강원·경북 · 10~15 = 일반도 · 25+ = 광역시",
        )
        st.caption(f"17개 시도 분포: 최저 {stats.loc['IC','min']:.2f} · 평균 {stats.loc['IC','mean']:.2f} · 최고 {stats.loc['IC','max']:.2f}")
    with c4:
        wage_min, wage_max, wage_step = SIM_FACTOR_RANGES["임금"]
        st.slider(
            "평균임금 — 만원/월",
            min_value=wage_min, max_value=wage_max, step=wage_step,
            key="sim_wage",
            help="300 미만 = 일반도 · 350~390 = 광역도 · 400+ = 서울·울산",
        )
        wage_actual = load_integrated_df()["평균임금_2024_만원"]
        st.caption(f"17개 시도 분포: 최저 {wage_actual.min():.0f} · 평균 {wage_actual.mean():.0f} · 최고 {wage_actual.max():.0f}만원")

    raw = {
        "항만": st.session_state["sim_port"],
        "IC": st.session_state["sim_ic"],
        "전력": st.session_state["sim_power"],
        "임금": st.session_state["sim_wage"] / 100,
    }
    similar_sido, distance = find_similar_sido(raw)
    top, contributions, x_user_z = recommend_industries(raw, top_n=5)
    flat_scores = bool(float(x_user_z.abs().max()) < 0.05 or np.allclose(top.values, top.iloc[0]))

    st.markdown('<div class="section-title">추천 결과</div>', unsafe_allow_html=True)
    st.markdown(
        '<p class="section-sub">입력 조건과 가장 가까운 실제 시도, 그리고 평균 대비 입지우위가 큰 산업을 함께 보여줍니다.</p>',
        unsafe_allow_html=True,
    )
    m1, m2, m3 = st.columns(3)
    m1.metric("가장 유사한 시도", similar_sido, delta=f"표준화 거리 {distance:.2f}", delta_color="off")
    m2.metric("선택 프리셋", st.session_state["sim_preset"])
    m3.metric("TOP 1 추천 산업", "차이 없음" if flat_scores else top.index[0], delta=None if flat_scores else f"적합도 {top.iloc[0]:.0f}점", delta_color="off")

    if flat_scores:
        st.info("전국 평균 입력에서는 모든 요인의 z-score가 0이라 산업별 상대 입지우위가 갈리지 않습니다. 프리셋을 고르거나 슬라이더를 조정해 주세요.")
    else:
        render_badges("순위 막대", "회귀")
        render_top5_ranking_chart(top, contributions)
        top_sidos = top_real_sidos_for_recommendations(top.index.tolist())
        st.success(f"추천 산업들이 실제로 많이 모인 시도 TOP 5: {top_sidos}")

    with st.expander("입력 조건 진단 보기"):
        render_badges("표준화 막대", "분포")
        render_zscore_chart(x_user_z)
        render_factor_distribution(raw)

    st.caption("시뮬레이터는 17개 시도 단위 회귀계수 기반 정책 데모입니다. 세부 부지 선정에는 토지·규제·인력·공급망 자료가 추가로 필요합니다.")


def render_sidebar() -> tuple[str, str, str]:
    """Render mode control and expose filters only in Explore mode."""
    st.sidebar.markdown('<div class="sidebar-title">Mode</div>', unsafe_allow_html=True)
    mode = st.sidebar.radio(
        "모드",
        ["탐색", "가설 결과", "신규 산단 시뮬레이터"],
        index=0,
        label_visibility="collapsed",
    )

    selected_sido = "경기"
    selected_target = "반도체·전자"
    if mode == "탐색":
        st.sidebar.markdown('<div class="sidebar-title" style="margin-top:1.3rem">Filter</div>', unsafe_allow_html=True)
        selected_sido = st.sidebar.selectbox("시도", SIDO_ORDER, index=SIDO_ORDER.index("경기"))
        industries = get_industries()
        default_idx = industries.index("반도체·전자") if "반도체·전자" in industries else 0
        selected_target = st.sidebar.selectbox("산업", industries, index=default_idx)
        st.sidebar.caption("산업 지도 기준연도: 2024년 고정")

    st.sidebar.markdown('<div class="sidebar-title" style="margin-top:1.5rem">Team</div>', unsafe_allow_html=True)
    st.sidebar.caption("팀 푸바오 · 아주대 융합시스템공학과")
    st.sidebar.caption("이동혁 · 서찬 · 박현민 · 정현문")
    return mode, selected_sido, selected_target


def main() -> None:
    """Run the Streamlit dashboard."""
    inject_css()
    render_header()
    mode, selected_sido, selected_target = render_sidebar()

    if mode == "탐색":
        render_explore_mode(selected_sido, selected_target)
    elif mode == "가설 결과":
        render_hypothesis_mode()
    else:
        render_simulator_mode()

    if mode == "탐색":
        st.caption(f"현재 선택값: {selected_sido} · {selected_target} · 2024년")


if __name__ == "__main__":
    main()
