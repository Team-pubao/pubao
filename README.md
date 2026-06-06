# 푸바오 프로젝트

한국 17개 시도의 산업별 사업체 비중과 항만·IC·산업용 전력·임금의 연관성을 탐색하는 Streamlit 대시보드입니다.

## 실행

```powershell
uv venv --python 3.11
uv pip install -r requirements.txt
.venv\Scripts\streamlit.exe run app.py
```

접속 주소는 `http://localhost:8501`입니다.

## 정적 산출물 생성

```powershell
.venv\Scripts\python.exe build_v2_outputs.py
```

생성 파일:

- `results/tables/pubao_v2_analysis.xlsx`
- `results/figures/pubao_v2_h1_h4_scatter.png`
- `results/figures/pubao_v2_h5_factor_ranking.png`
- `results/figures/pubao_v2_h6_masked_heatmap.png`
- `results/figures/pubao_v2_h8_candidates.png`

## 현재 구조

```text
app.py                  Streamlit 대시보드
build_v2_outputs.py     V2 Excel·PNG 생성
data/                   분석 입력과 출처 보관 자료
results/figures/        V2 발표용 PNG
results/tables/         V2 분석 결과 Excel
```

회귀는 시도 17개 자료만 사용합니다. `data/시군구_입지요인.csv`와
`data/시군구_경계.geojson`은 지도와 지역 맥락 시각화 전용입니다.

모든 결과는 인과 효과가 아니라 관찰된 연관성·패턴으로 해석합니다.
