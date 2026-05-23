"""Streamlit dashboard and new industrial complex simulator for the Pubao project."""

from __future__ import annotations

import json
import math
from pathlib import Path

import folium
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from branca.colormap import linear
from streamlit_folium import st_folium


ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
RESULTS_DIR = ROOT_DIR / "results"
TABLES_DIR = RESULTS_DIR / "tables"
FIGURES_DIR = RESULTS_DIR / "figures"

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
FACTOR_BETA_COLUMNS = {"항만": "H1_항만", "IC": "H2_IC", "전력": "H3_전력", "임금": "H4_임금"}
HEATMAP_LABELS = {"H1_항만": "H1 항만", "H2_IC": "H2 IC", "H3_전력": "H3 전력", "H4_임금": "H4 임금"}

HYPOTHESIS_IMAGES = {
    "H1": ["h1_port_scatter_grid.png"],
    "H2": ["h2_road_scatter.png"],
    "H3": ["h3_power_scatter.png", "h3_power_industry_sensitivity.png"],
    "H4": ["h4_wage_scatter.png", "h4_wage_industry_signs.png"],
    "H5": ["h5_moran_scatter_grid.png"],
    "H6": ["h6_industry_factor_beta_heatmap.png"],
    "H7": ["h7_panel_vs_single.png"],
    "H8": ["h8_group_beta_compare.png"],
}

MEDALS = ["1위", "2위", "3위", "4위", "5위"]

CHART_FONT = "Apple SD Gothic Neo, Pretendard, Malgun Gothic, sans-serif"
CHART_PRIMARY = "#1d4ed8"
CHART_ACCENT = "#6366f1"
CHART_MUTED = "#9ca3af"
CHART_INK = "#111827"


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
            padding-top: 2rem;
            padding-bottom: 4rem;
            max-width: 1400px;
        }

        /* ── Hero ─────────────────────────────────── */
        .hero {
            margin-bottom: 2rem;
            padding-bottom: 1.5rem;
            border-bottom: 1px solid #e5e7eb;
        }
        .hero .eyebrow {
            font-size: 0.72rem;
            text-transform: uppercase;
            letter-spacing: 0.14em;
            color: #1d4ed8;
            font-weight: 700;
            margin-bottom: 0.65rem;
        }
        .hero h1 {
            color: #111827;
            font-size: 1.9rem;
            font-weight: 700;
            margin: 0 0 0.55rem;
            letter-spacing: -0.025em;
            line-height: 1.2;
        }
        .hero p {
            color: #4b5563;
            margin: 0;
            font-size: 0.95rem;
            line-height: 1.55;
            max-width: 760px;
        }
        .hero .meta {
            margin-top: 1.1rem;
            font-size: 0.78rem;
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
            font-size: 1.12rem;
            font-weight: 700;
            margin: 2rem 0 0.3rem;
            letter-spacing: -0.018em;
        }
        .section-sub {
            color: #6b7280;
            font-size: 0.86rem;
            margin: 0 0 1.2rem;
            line-height: 1.55;
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
            padding: 1.1rem 1.2rem 1rem;
            margin-bottom: 0.75rem;
        }
        .factor-card-head {
            display: flex;
            justify-content: space-between;
            align-items: baseline;
            margin-bottom: 0.7rem;
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
            font-size: 1.85rem;
            font-weight: 700;
            color: #111827;
            line-height: 1.05;
            letter-spacing: -0.03em;
            font-variant-numeric: tabular-nums;
            display: flex;
            align-items: baseline;
            gap: 0.35rem;
            margin-bottom: 0.25rem;
        }
        .factor-card-value-unit {
            font-size: 0.8rem;
            font-weight: 500;
            color: #9ca3af;
            letter-spacing: 0;
        }
        .factor-card-sub {
            font-size: 0.78rem;
            color: #6b7280;
            margin: 0 0 0.9rem;
            line-height: 1.5;
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
            margin-bottom: 0.7rem;
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
    df = pd.read_excel(DATA_DIR / "07_통합_분석DF_2024.xlsx", sheet_name="통합DF")
    df["시도"] = df["시도"].map(short_sido)
    df["log_하역능력_합계"] = np.log1p(df["하역능력_합계"])
    df["평균임금_2024_백만원"] = df["평균임금_2024"] / 1_000_000
    df["평균임금_2024_만원"] = df["평균임금_2024"] / 10_000
    return sorted_by_sido(df)


@st.cache_data(show_spinner=False)
def load_y_panel() -> pd.DataFrame:
    """Load 2020-2024 long panel data."""
    path = next(DATA_DIR.glob("02_Y_*long.xlsx"))
    df = pd.read_excel(path)
    df["시도"] = df["시도"].map(short_sido)
    return df


@st.cache_data(show_spinner=False)
def load_year_industry_df(year: int) -> pd.DataFrame:
    """Build a wide 시도×산업 dataframe with shares for the selected year."""
    panel = load_y_panel()
    year_df = panel[panel["연도"] == year]
    wide = year_df.pivot_table(index="시도", columns="분석그룹", values="사업체수", aggfunc="sum").reset_index()
    industries = [col for col in wide.columns if col != "시도"]
    wide["_전체사업체수"] = wide[industries].sum(axis=1)
    for industry in industries:
        wide[f"비중_{industry}"] = wide[industry] / wide["_전체사업체수"]
    return sorted_by_sido(wide)


@st.cache_data(show_spinner=False)
def load_industry_mapping() -> pd.DataFrame:
    """Load industry roles and sorting metadata."""
    df = pd.read_excel(DATA_DIR / "01_산업매핑_71중분류_to_26그룹.xlsx")
    df = df[["분석그룹", "축_역할"]].drop_duplicates().copy()
    df["role_order"] = df["축_역할"].map({role: idx for idx, role in enumerate(ROLE_ORDER)}).fillna(99)
    return df.sort_values(["role_order", "분석그룹"]).reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_h6_beta_matrix() -> pd.DataFrame:
    """Load H6 beta matrix with industry index."""
    beta = pd.read_excel(TABLES_DIR / "h6_heterogeneity.xlsx", sheet_name="β매트릭스")
    return beta.set_index("산업")


@st.cache_data(show_spinner=False)
def load_h6_p_matrix() -> pd.DataFrame:
    """Load H6 p-value matrix with industry index."""
    pvals = pd.read_excel(TABLES_DIR / "h6_heterogeneity.xlsx", sheet_name="p매트릭스")
    return pvals.set_index("산업")


@st.cache_data(show_spinner=False)
def load_final_summary() -> pd.DataFrame:
    """Load final H1-H8 summary table."""
    return pd.read_excel(RESULTS_DIR / "final_summary.xlsx", sheet_name="가설_결론표")


@st.cache_data(show_spinner=False)
def load_geojson_dict() -> dict:
    """Load province GeoJSON as a plain dict with normalized 시도 names."""
    with open(DATA_DIR / "지도_시도.json", encoding="utf-8") as f:
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
def load_hypothesis_image_paths() -> dict[str, list[str]]:
    """Return available figure paths by hypothesis."""
    available: dict[str, list[str]] = {}
    for h_num, names in HYPOTHESIS_IMAGES.items():
        paths = [str(FIGURES_DIR / name) for name in names if (FIGURES_DIR / name).exists()]
        available[h_num] = paths
    return available


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
    """Return H6 beta matrix aligned to dashboard factor labels."""
    beta = load_h6_beta_matrix()[list(FACTOR_BETA_COLUMNS.values())].rename(columns={v: k for k, v in FACTOR_BETA_COLUMNS.items()})
    return beta


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
          <p>17개 시도 × 26개 산업그룹 × 8개 가설을 시각적으로 탐색하고, 신규 산업단지의 입지 조건을 시뮬레이션합니다.</p>
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


def make_industry_map(selected_sido: str, selected_industry: str, selected_year: int) -> folium.Map:
    """Build a choropleth map for selected industry share."""
    df = load_year_industry_df(selected_year)
    share_col = f"비중_{selected_industry}"
    share_map = dict(zip(df["시도"], df[share_col]))
    count_map = dict(zip(df["시도"], df[selected_industry]))

    geo = json.loads(json.dumps(load_geojson_dict()))
    for feature in geo["features"]:
        sido = feature["properties"]["시도"]
        feature["properties"][share_col] = float(share_map.get(sido, 0.0))
        feature["properties"][selected_industry] = float(count_map.get(sido, 0.0))

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
            "color": "#1d4ed8" if sido == selected_sido else "#94a3b8",
            "weight": 3 if sido == selected_sido else 1,
            "fillOpacity": 0.78,
        }

    folium.GeoJson(
        geo,
        name=f"{selected_year}년 {selected_industry} 비중",
        style_function=style,
        tooltip=folium.GeoJsonTooltip(
            fields=["시도", selected_industry, share_col],
            aliases=["시도", "사업체수", "사업체 비중"],
            localize=True,
        ),
    ).add_to(fmap)
    cmap.caption = f"{selected_year}년 {selected_industry} 사업체 비중"
    cmap.add_to(fmap)
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
        dominant = contributions.loc[industry].sort_values(ascending=False).index[0]
        st.markdown(
            f"""
            <div class="recommend-card">
              <b>{idx + 1}. {industry}</b><span class="score-chip">적합도 <strong>{score:.0f}</strong> / 100</span><br>
              <span class="small-note">주요 기여: <b>{dominant}</b> · {industry_comment(industry, contributions.loc[industry])}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_time_series(selected_sido: str, selected_industry: str) -> None:
    """Render selected province and national mean time-series."""
    panel = load_y_panel()
    industry_df = panel[panel["분석그룹"] == selected_industry].copy()
    selected = industry_df[industry_df["시도"] == selected_sido][["연도", "사업체수"]].copy()
    selected["구분"] = selected_sido
    mean_df = industry_df.groupby("연도", as_index=False)["사업체수"].mean()
    mean_df["구분"] = "17개 시도 평균"
    chart_df = pd.concat([selected, mean_df], ignore_index=True)
    fig = px.line(chart_df, x="연도", y="사업체수", color="구분", markers=True, title=f"{selected_sido} · {selected_industry} 사업체수 추이")
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
    apply_chart_theme(fig, height=420)
    fig.update_layout(legend_title_text="", hovermode="x unified")
    st.plotly_chart(fig, width="stretch")


def render_explore_mode(selected_sido: str, selected_industry: str, selected_year: int) -> None:
    """Render exploration mode."""
    left, right = st.columns([3, 2])
    with left:
        st.markdown('<div class="section-title">시도별 산업 비중 지도</div>', unsafe_allow_html=True)
        st.markdown(
            f'<p class="section-sub">{selected_year}년 · {selected_industry} 사업체 비중 · 진할수록 집중도 높음</p>',
            unsafe_allow_html=True,
        )
        st_folium(make_industry_map(selected_sido, selected_industry, selected_year), height=540, width=900, returned_objects=[])
    with right:
        st.markdown('<div class="section-title">입지요인 요약</div>', unsafe_allow_html=True)
        st.markdown(f'<p class="section-sub">{selected_sido} · 항만 · 도로 · 전력 · 임금</p>', unsafe_allow_html=True)
        render_factor_cards(selected_sido)
        render_matching_recommendation(selected_sido)

    st.markdown('<div class="section-title">2020~2024 시계열 변화</div>', unsafe_allow_html=True)
    st.markdown(
        f'<p class="section-sub">{selected_sido} vs 17개 시도 평균</p>',
        unsafe_allow_html=True,
    )
    render_time_series(selected_sido, selected_industry)


def support_status(conclusion: str) -> tuple[str, str]:
    """Return (label, css-class) for a hypothesis conclusion."""
    if "지지" in conclusion and "부분" not in conclusion:
        return "지지", "status-support"
    if "부분" in conclusion:
        return "부분지지", "status-partial"
    return "제한적", "status-limited"


HYPOTHESIS_INTERPRETATION = {
    "H1": {
        "question": "바다와 가까운 지역(항만 인프라가 좋은 곳)에 중공업이 더 많이 모일까?",
        "method": "17개 시도의 항만 하역능력 vs 중량·수출 산업(석유화학·1차금속·자동차·조선) 비중을 단순 회귀.",
        "finding": "조선·기타운송장비에서 약한 양의 관계 확인. 다른 중공업은 항만보다 다른 요인에 더 민감.",
        "meaning": "항만은 일부 중공업의 입지 결정에 영향을 주지만, 전력·임금·역사적 클러스터 등 다른 요인과 함께 작용한다. 신규 산단을 항만 옆에 짓는다고 자동으로 중공업이 따라오지는 않는다.",
    },
    "H2": {
        "question": "고속도로 IC가 많은 곳에 물류·창고·도소매 산업이 더 모일까?",
        "method": "시도별 IC 밀도(개/1000㎢) vs 물류 산업 비중을 단순 회귀.",
        "finding": "R² = 0.62로 4개 요인 중 가장 명확한 양의 관계. p-value도 매우 유의함.",
        "meaning": "물류 산업단지를 계획한다면 IC 접근성이 가장 중요한 요인이다. 4개 요인 중 정책적 시사점이 가장 분명한 결과.",
    },
    "H3": {
        "question": "산업용 전력 사용량이 큰 지역에 반도체·전자 산업이 더 모일까?",
        "method": "log 산업용 전력 vs 반도체·전자 비중 회귀 + 산업별 민감도 비교.",
        "finding": "약한 양의 관계 (R² ≈ 0.13). 반도체·전자는 전력에 민감하지만, 전력 하나만으로는 입지가 결정되지 않음.",
        "meaning": "고전력 인프라는 반도체 산업의 필수 조건이지만 충분 조건은 아니다. 기존 클러스터(경기·충남)와의 연계, 인력 풀, 용수 등이 함께 작용한다.",
    },
    "H4": {
        "question": "임금이 높은 지역에는 자본집약 산업(IT·금융)이, 낮은 지역에는 노동집약 산업(섬유)이 더 많을까?",
        "method": "시도별 평균임금 vs 산업 비중. 자본집약은 양(+), 노동집약은 음(−)의 부호를 기대.",
        "finding": "이론과 부호 일치율 66.7%. IT 산업이 임금 높은 지역에 모이는 패턴은 약하게 관찰.",
        "meaning": "임금이 입지 결정에 미치는 영향은 산업마다 방향과 강도가 다르다. '비용만 보면 노동집약은 지방으로'라는 단순 논리는 한국 데이터에서 부분적으로만 성립한다.",
    },
    "H5": {
        "question": "같은 산업이 지리적으로 인접한 시도끼리 뭉치는 현상이 있을까? (예: 충남-경기 반도체 벨트)",
        "method": "Moran's I 공간 자기상관 통계량. 양수면 유사 지역이 인접, 음수면 분산.",
        "finding": "26개 산업 중 1개만 통계적으로 유의한 양의 군집을 보임. 대부분 산업은 명확한 공간 군집 없음.",
        "meaning": "산업의 지리적 클러스터링은 일부 산업(예: 반도체)에만 강하게 나타나며, 대부분은 17개 시도 단위에서 군집 패턴이 약하다. 더 작은 단위(시군구)에서 다시 봐야 할 수 있다.",
    },
    "H6": {
        "question": "산업마다 같은 입지요인(항만·IC·전력·임금)에 대해 다르게 반응할까?",
        "method": "26개 산업 × 4개 요인 회귀계수 β 매트릭스. Spearman 순위상관으로 이론적 순위와 비교.",
        "finding": "산업별 β의 방향과 크기가 명확히 다름. 3개 검정 중 1개에서 이론 순위와 통계적으로 일치.",
        "meaning": "“모든 산업에 동일한 입지 정책”은 비효율적이다. 산업별로 어떤 요인에 민감한지 매트릭스로 확인 후 맞춤형 입지 전략을 짜야 한다. (히트맵 참고)",
    },
    "H7": {
        "question": "단년(2024) 분석 결과가 5년 패널 + 시도 고정효과 모형에서도 유지될까?",
        "method": "2020~2024 long 패널 + 시도 고정효과(FE) 회귀로 시계열 + 시도 차이를 모두 통제.",
        "finding": "강건 효과 0/16개. 단년 결과가 패널·고정효과 통제 시 대부분 약화됨.",
        "meaning": "단년 N=17 결과는 시도 간 고유한 특성(역사·정책·문화)에 상당 부분 흡수된다. 결과는 인과가 아니라 연관성으로 해석해야 하며, 보수적 해석이 필요하다.",
    },
    "H8": {
        "question": "광역시(8개)와 도(9개)는 같은 입지요인에 다르게 반응할까? — 도시형 vs 지방형 구분",
        "method": "두 그룹 분할 후 각각 회귀, β 계수 차이의 통계적 유의성 검정.",
        "finding": "24개 조합 중 5개에서 통계적으로 유의한 그룹 차이. 일부 산업은 도시·지방 입지 패턴이 명확히 다름.",
        "meaning": "신규 산단 입지를 결정할 때 “광역시급 도시 vs 지방 도” 구분을 같이 고려해야 한다. 모든 산업에 동일 잣대를 들이대면 안 된다.",
    },
}


def render_hypothesis_cards() -> None:
    """Render H1-H8 summary expanders with figures and plain-language interpretation."""
    summary = load_final_summary()
    image_paths = load_hypothesis_image_paths()
    for _, row in summary.iterrows():
        h_num = row["가설번호"]
        label, css = support_status(str(row["결론"]))
        title = f"{h_num}  ·  {row['가설_요약']}    [{label}]"
        with st.expander(title, expanded=h_num in {"H2", "H6"}):
            st.markdown(
                f'<span class="status-chip {css}">{label}</span>',
                unsafe_allow_html=True,
            )
            interp = HYPOTHESIS_INTERPRETATION.get(h_num)
            if interp is not None:
                st.markdown(
                    f"""
                    <div style="background:#f8fafc;border-left:4px solid #1d4ed8;border-radius:8px;padding:0.95rem 1.15rem;margin:0.4rem 0 0.9rem;">
                      <p style="margin:0 0 0.5rem;font-size:0.95rem;color:#0f172a;"><b>무엇을 확인했나</b> · {interp['question']}</p>
                      <p style="margin:0 0 0.5rem;font-size:0.92rem;color:#475569;"><b>어떻게 검정했나</b> · {interp['method']}</p>
                      <p style="margin:0 0 0.5rem;font-size:0.92rem;color:#0f172a;"><b>무엇이 나왔나</b> · {interp['finding']}</p>
                      <p style="margin:0;font-size:0.95rem;color:#1d4ed8;"><b>이게 무슨 뜻인가</b> · {interp['meaning']}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            c1, c2 = st.columns([2, 3])
            with c1:
                st.markdown(f"**핵심 통계량**  ·  {row['핵심_통계량']}")
                st.markdown(f"**p-value**  ·  {row['p-value']:.4f}" if pd.notna(row["p-value"]) else "**p-value**  ·  -")
                st.markdown(f"**결론**  ·  {row['결론']}")
                st.caption("N=17 기반 분석이므로 인과가 아니라 연관성으로 해석합니다.")
            with c2:
                for path in image_paths.get(h_num, [])[:2]:
                    st.image(path, width="stretch")


def render_beta_heatmap() -> None:
    """Render interactive H6 beta heatmap."""
    beta = load_h6_beta_matrix()
    pvals = load_h6_p_matrix()
    order = get_industries()
    cols = list(HEATMAP_LABELS.keys())
    z = beta.reindex(order)[cols]
    p = pvals.reindex(order)[cols]
    hover = np.array(
        [
            [f"산업: {industry}<br>요인: {HEATMAP_LABELS[col]}<br>β: {z.loc[industry, col]:.6f}<br>p-value: {p.loc[industry, col]:.4f}" for col in cols]
            for industry in z.index
        ]
    )
    fig = px.imshow(
        z,
        x=[HEATMAP_LABELS[col] for col in cols],
        y=z.index,
        color_continuous_scale="RdBu_r",
        color_continuous_midpoint=0,
        aspect="auto",
        title="산업 × 입지요인 β 인터랙티브 히트맵 (빨강 = 양의 영향, 파랑 = 음의 영향)",
    )
    fig.update_traces(customdata=hover, hovertemplate="%{customdata}<extra></extra>")
    apply_chart_theme(fig, height=860, show_legend=False)
    fig.update_xaxes(side="top", showgrid=False, tickfont=dict(size=14, color=CHART_INK), title_text="")
    fig.update_yaxes(showgrid=False, tickfont=dict(size=12, color=CHART_INK), title_text="")
    fig.update_layout(coloraxis_colorbar=dict(title="β", tickfont=dict(size=12), thickness=18))
    st.plotly_chart(fig, width="stretch")


def render_hypothesis_mode() -> None:
    """Render hypothesis-result mode."""
    st.markdown('<div class="section-title">H1~H8 결과 카드</div>', unsafe_allow_html=True)
    st.markdown(
        '<p class="section-sub">기본 4개(직접효과 H1~H4) + 심화 4개(공간·이질성·패널·집단 H5~H8)</p>',
        unsafe_allow_html=True,
    )
    render_hypothesis_cards()
    st.markdown('<div class="section-title">산업별 입지요인 민감도</div>', unsafe_allow_html=True)
    st.markdown(
        '<p class="section-sub">26개 산업 × 4개 입지요인 β 매트릭스 · 진한 빨강(+)/파랑(−)일수록 민감도 큼</p>',
        unsafe_allow_html=True,
    )
    render_beta_heatmap()


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


SIM_FACTOR_KEYS = {"항만": "sim_port", "IC": "sim_ic", "전력": "sim_power", "임금": "sim_wage"}
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
        contributions.loc[industry].sort_values(ascending=False).index[0]
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
    """Render new industrial complex simulator."""
    _ensure_sim_defaults()

    st.markdown('<div class="section-title">STEP 1 · 산업단지 입지 조건 설정</div>', unsafe_allow_html=True)
    st.markdown(
        '<p class="section-sub">실제 시도를 골라 자동 입력하거나, 직접 슬라이더로 가상 조건을 만들어 보세요.</p>',
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
    c1, c2 = st.columns(2)
    with c1:
        port_min, port_max, port_step = SIM_FACTOR_RANGES["항만"]
        st.slider(
            "항만 접근 — log(하역능력+1)",
            min_value=port_min, max_value=port_max, step=port_step,
            key="sim_port",
            help="0 = 내륙 · 8~10 = 일반 항만 · 12+ = 부산·인천급",
        )
        st.caption(f"17개 시도 분포: 최저 {stats.loc['항만','min']:.2f} · 평균 {stats.loc['항만','mean']:.2f} · 최고 {stats.loc['항만','max']:.2f}")

        power_min, power_max, power_step = SIM_FACTOR_RANGES["전력"]
        st.slider(
            "산업용 전력 — log(GWh)",
            min_value=power_min, max_value=power_max, step=power_step,
            key="sim_power",
            help="15 = 서울·제주 · 17 = 일반 광역도 · 18+ = 경기·충남급",
        )
        st.caption(f"17개 시도 분포: 최저 {stats.loc['전력','min']:.2f} · 평균 {stats.loc['전력','mean']:.2f} · 최고 {stats.loc['전력','max']:.2f}")
    with c2:
        ic_min, ic_max, ic_step = SIM_FACTOR_RANGES["IC"]
        st.slider(
            "고속도로 IC 밀도 — 개/천㎢",
            min_value=ic_min, max_value=ic_max, step=ic_step,
            key="sim_ic",
            help="5 미만 = 강원·경북 · 10~15 = 일반도 · 25+ = 광역시",
        )
        st.caption(f"17개 시도 분포: 최저 {stats.loc['IC','min']:.2f} · 평균 {stats.loc['IC','mean']:.2f} · 최고 {stats.loc['IC','max']:.2f}")

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

    st.markdown('<div class="section-title">STEP 2 · 유사 지역 매칭</div>', unsafe_allow_html=True)
    st.markdown(
        f'<p class="section-sub">입력한 4개 요인을 표준화(z-score)한 뒤, 17개 시도 중 가장 가까운 지역을 찾았습니다.</p>',
        unsafe_allow_html=True,
    )
    m1, m2, m3 = st.columns(3)
    m1.metric("가장 유사한 시도", similar_sido, delta=f"표준화 거리 {distance:.2f}", delta_color="off")
    m2.metric("선택 프리셋", st.session_state["sim_preset"])
    m3.metric("TOP 1 추천 산업", top.index[0], delta=f"적합도 {top.iloc[0]:.0f}점", delta_color="off")

    st.markdown('<div class="section-title">STEP 3 · 추천 산업 TOP 5</div>', unsafe_allow_html=True)
    st.markdown(
        '<p class="section-sub">H6 회귀계수와 입력 조건의 z-score를 곱해 산출한 상대 적합도. 100점에 가까울수록 적합.</p>',
        unsafe_allow_html=True,
    )
    render_top5_ranking_chart(top, contributions)

    top_sidos = top_real_sidos_for_recommendations(top.index.tolist())
    st.success(f"추천 산업들이 실제로 많이 모인 시도 TOP 5 : {top_sidos}")

    st.markdown('<div class="section-title">STEP 4 · 입력 조건 진단</div>', unsafe_allow_html=True)
    st.markdown(
        '<p class="section-sub">상단: 4개 요인의 평균 대비 강도(z-score) · 하단: 실제 17개 시도 분포 위 입력값 위치</p>',
        unsafe_allow_html=True,
    )

    render_zscore_chart(x_user_z)
    render_factor_distribution(raw)

    st.caption("시뮬레이터는 17개 시도 단위 회귀계수 기반 정책 데모입니다. 세부 부지 선정에는 토지·규제·인력·공급망 자료가 추가로 필요합니다.")


def render_sidebar() -> tuple[str, str, str, int]:
    """Render sidebar controls and return selected values."""
    st.sidebar.markdown('<div class="sidebar-title">Mode</div>', unsafe_allow_html=True)
    mode = st.sidebar.radio(
        "모드",
        ["탐색", "가설 결과", "신규 산단 시뮬레이터"],
        index=0,
        label_visibility="collapsed",
    )

    st.sidebar.markdown('<div class="sidebar-title" style="margin-top:1.3rem">Filter</div>', unsafe_allow_html=True)
    selected_sido = st.sidebar.selectbox("시도", SIDO_ORDER, index=SIDO_ORDER.index("경기"))
    industries = get_industries()
    default_idx = industries.index("반도체·전자") if "반도체·전자" in industries else 0
    selected_industry = st.sidebar.selectbox("산업", industries, index=default_idx)
    selected_year = st.sidebar.selectbox("연도", [2024, 2023, 2022, 2021, 2020], index=0)

    st.sidebar.markdown('<div class="sidebar-title" style="margin-top:1.5rem">Team</div>', unsafe_allow_html=True)
    st.sidebar.caption("팀 푸바오 · 아주대 융합시스템공학과")
    st.sidebar.caption("이동혁 · 서찬 · 박현민 · 정현문")
    return mode, selected_sido, selected_industry, int(selected_year)


def main() -> None:
    """Run the Streamlit dashboard."""
    inject_css()
    render_header()
    mode, selected_sido, selected_industry, selected_year = render_sidebar()

    if mode == "탐색":
        render_explore_mode(selected_sido, selected_industry, selected_year)
    elif mode == "가설 결과":
        render_hypothesis_mode()
    else:
        render_simulator_mode()

    st.caption(f"현재 선택값: {selected_sido} · {selected_industry} · {selected_year}년")


if __name__ == "__main__":
    main()
