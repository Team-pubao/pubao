"""Build version-2 analysis tables and presentation figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import app


ROOT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = ROOT_DIR / "results"
TABLES_DIR = RESULTS_DIR / "tables"
FIGURES_DIR = RESULTS_DIR / "figures"


def build_h8_candidates() -> pd.DataFrame:
    """Build predicted-minus-actual candidate rows for all six roles."""
    df = app.load_analysis_df()
    blocks = []
    for role in app.ROLE_HYPOTHESIS_ORDER:
        label = app.ROLE_LABELS[role]
        model, residuals = app.fit_multifactor_regression(df, f"역할비중_{role}")
        residuals.insert(0, "역할", label)
        residuals["모형_R²"] = float(model.rsquared)
        blocks.append(residuals)
    return pd.concat(blocks, ignore_index=True)


def save_workbook() -> Path:
    """Save the replacement H1-H8 calculation tables."""
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    path = TABLES_DIR / "pubao_v2_analysis.xlsx"
    summary = app.build_hypothesis_summary_live()
    competition = app.build_factor_competition()
    industry_beta = app.build_beta_results("산업")
    role_beta = app.build_beta_results("역할")
    candidates = build_h8_candidates()
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="H1-H8_요약", index=False)
        competition.to_excel(writer, sheet_name="H5_요인경합", index=False)
        industry_beta.to_excel(writer, sheet_name="H6_26산업", index=False)
        role_beta.to_excel(writer, sheet_name="H6_6역할", index=False)
        candidates.to_excel(writer, sheet_name="H8_유치후보", index=False)
    return path


def save_h1_h4_scatter() -> Path:
    """Save the four direct-effect regression charts."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    df = app.load_analysis_df()
    specs = [
        ("H1 항만", app.FACTOR_COLUMNS["항만"], "역할비중_heavy_export", "중량수출 비중"),
        ("H2 IC", app.FACTOR_COLUMNS["IC"], "역할비중_logistics", "물류 비중"),
        ("H3 전력", app.FACTOR_COLUMNS["전력"], "비중_반도체·전자", "반도체·전자 비중"),
        ("H4 임금", app.FACTOR_COLUMNS["임금"], "역할비중_capital_intensive", "자본집약 비중"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), constrained_layout=True)
    for ax, (title, x_col, y_col, y_label) in zip(axes.ravel(), specs):
        result = app.fit_simple_regression(df, y_col, x_col)
        sns.regplot(
            data=df,
            x=x_col,
            y=y_col,
            ax=ax,
            scatter_kws={"s": 48, "alpha": 0.82, "color": "#1d4ed8"},
            line_kws={"color": "#000080"},
        )
        for _, row in df.iterrows():
            ax.annotate(row["시도"], (row[x_col], row[y_col]), fontsize=7, xytext=(3, 3), textcoords="offset points")
        ax.set_title(f"{title} · 표준화β={result['표준화β']:.2f}, R²={result['R²']:.2f}, p={result['p_value']:.3f}")
        ax.set_ylabel(y_label)
    path = FIGURES_DIR / "pubao_v2_h1_h4_scatter.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def save_h5_ranking() -> Path:
    """Save the H5 factor competition ranking."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    competition = app.build_factor_competition().sort_values("평균_R²")
    fig, ax = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
    bars = ax.barh(competition["요인"], competition["평균_R²"], color="#1d4ed8")
    for bar, count in zip(bars, competition["유의산업수"]):
        ax.text(bar.get_width() + 0.004, bar.get_y() + bar.get_height() / 2, f"유의산업 {int(count)}개", va="center")
    ax.set_title("H5 4대 요인 영향력 순위")
    ax.set_xlabel("26개 산업 평균 R²")
    ax.set_ylabel("")
    path = FIGURES_DIR / "pubao_v2_h5_factor_ranking.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def save_h6_heatmap() -> Path:
    """Save a significance-masked 26-industry standardized beta heatmap."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    results = app.build_beta_results("산업")
    order = app.get_industries()
    beta = results.pivot(index="대상", columns="요인", values="표준화β").reindex(order)[list(app.FACTOR_COLUMNS)]
    fdr = results.pivot(index="대상", columns="요인", values="p_FDR").reindex(order)[list(app.FACTOR_COLUMNS)]
    annotations = beta.copy().astype(object)
    for target in beta.index:
        for factor in beta.columns:
            q_value = fdr.loc[target, factor]
            annotations.loc[target, factor] = (
                f"{beta.loc[target, factor]:+.2f}{app._significance_stars(q_value)}"
                if q_value < 0.05
                else ""
            )
    fig, ax = plt.subplots(figsize=(9, 12), constrained_layout=True)
    ax.set_facecolor("#d1d5db")
    sns.heatmap(
        beta,
        mask=fdr >= 0.05,
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        annot=annotations,
        fmt="",
        linewidths=0.8,
        linecolor="white",
        cbar_kws={"label": "표준화 β"},
        ax=ax,
    )
    ax.set_title("H6 산업 × 입지요인 표준화 β · FDR 비유의 셀 회색")
    ax.set_xlabel("")
    ax.set_ylabel("")
    path = FIGURES_DIR / "pubao_v2_h6_masked_heatmap.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def save_h8_candidates() -> Path:
    """Save the leading attraction candidates for six roles."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    candidates = build_h8_candidates()
    top = (
        candidates.sort_values(["역할", "유치여지"], ascending=[True, False])
        .groupby("역할")
        .head(3)
        .copy()
    )
    top["후보"] = top["역할"] + " · " + top["시도"]
    role_colors = {
        "중량·수출": "#c53d3d",
        "전력집약": "#e29a35",
        "물류": "#d9bf3f",
        "자본집약": "#2d62a3",
        "노동집약": "#5aa0bf",
        "기타": "#9a9a9a",
    }
    top = top.sort_values("유치여지")
    fig, ax = plt.subplots(figsize=(10, 9), constrained_layout=True)
    ax.barh(top["후보"], top["유치여지"], color=top["역할"].map(role_colors))
    ax.axvline(0, color="#475569", linewidth=1)
    ax.set_title("H8 역할별 유치 후보 TOP 3 · 예측 비중 - 실제 비중")
    ax.set_xlabel("유치여지")
    ax.set_ylabel("")
    path = FIGURES_DIR / "pubao_v2_h8_candidates.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    """Build all version-2 outputs and print the live conclusions."""
    workbook = save_workbook()
    figures = [
        save_h1_h4_scatter(),
        save_h5_ranking(),
        save_h6_heatmap(),
        save_h8_candidates(),
    ]
    summary = app.build_hypothesis_summary_live()
    print("푸바오 V2 H1-H8 결과")
    print(summary[["가설", "통계", "결론", "결론문"]].to_string(index=False))
    print(f"\nSaved: {workbook}")
    for figure in figures:
        print(f"Saved: {figure}")


if __name__ == "__main__":
    main()
