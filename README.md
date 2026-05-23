# 푸바오 프로젝트: 한국 산업별 입지 결정 요인 분석

17개 시도, 26개 산업그룹, 2020~2024년 사업체수 데이터를 바탕으로 산업별 입지 결정 요인을 검정하는 Python 분석 프로젝트입니다.

## 실행 환경

```powershell
uv venv --python 3.11
uv pip install -r requirements.txt
```

시스템 Python이 없다면 `uv run --python 3.11 python src/00_validate_data.py`처럼 실행해도 됩니다.

## 주요 실행 순서

```powershell
uv run --python 3.11 python src/00_validate_data.py
uv run --python 3.11 python src/h1_port.py
uv run --python 3.11 python src/h2_road.py
uv run --python 3.11 python src/h3_power.py
uv run --python 3.11 python src/h4_wage.py
uv run --python 3.11 python src/h5_spatial.py
uv run --python 3.11 python src/h6_heterogeneity.py
uv run --python 3.11 python src/h7_panel.py
uv run --python 3.11 python src/h8_groups.py
uv run --python 3.11 python src/final_report.py
```

## 산출물

- `results/tables/`: 가설별 Excel 결과표
- `results/figures/`: 발표용 PNG 그림
- `results/maps/`: Folium HTML 지도
- `final_report/`: 최종 통합표, 발표 핵심 그림, 발표 노트

## Streamlit 대시보드

발표용 인터랙티브 대시보드는 다음 명령으로 실행합니다.

```powershell
streamlit run app.py
```

uv 가상환경을 직접 사용할 때는 다음처럼 실행해도 됩니다.

```powershell
.venv\Scripts\streamlit.exe run app.py
```

접속 주소는 `http://localhost:8501` 입니다. 첫 실행은 데이터와 지도를 캐싱하느라 몇 초 걸릴 수 있습니다.

모든 분석 결과는 인과 효과가 아니라 시도 단위 횡단면 및 패널 자료에서 관찰되는 연관성으로 해석합니다.
