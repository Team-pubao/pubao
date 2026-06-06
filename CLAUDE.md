# 푸바오 프로젝트 개발 규약

## 프로젝트

- 한국 17개 시도의 산업 입지와 4대 요인(항만·IC·전력·임금)의 연관성을 탐색한다.
- 산업은 26개 분석그룹과 6개 역할로 제공한다.
- Streamlit 앱은 탐색, H1~H8 결과, 신규 산단 시뮬레이터의 3개 모드로 구성한다.

## 분석 단위

- 회귀 본체: 시도 17개
- 시군구 229개 자료: 지도와 지역 맥락 시각화 전용
- 시도 산업 Y와 시군구 입지요인 X를 하나의 회귀에 혼합하지 않는다.
- 모든 결론은 인과가 아닌 연관성·패턴으로 표현한다.

## 현재 H1~H8

- H1: 항만 인프라와 중량·수출 산업
- H2: IC 밀도와 물류 산업
- H3: 산업용 전력과 반도체·전자
- H4: 임금과 자본·노동집약 산업
- H5: 4대 요인의 산업별 설명력 경합
- H6: 산업별 대표 입지 조건
- H7: 산업의 지역 편중
- H8: 예측 비중이 실제 비중보다 높은 유치 후보

## 핵심 파일

```text
app.py
build_v2_outputs.py
requirements.txt
data/
results/figures/pubao_v2_*.png
results/tables/pubao_v2_analysis.xlsx
```

## 데이터

앱이 직접 읽는 파일:

- `data/01_산업매핑.xlsx`
- `data/02_사업체수_패널.xlsx`
- `data/07_통합분석_2024.xlsx`
- `data/지도_시도경계.json`
- `data/시군구_입지요인.csv`
- `data/시군구_경계.geojson`

그 밖의 `data/` 파일은 출처와 재현성을 위한 원본·중간 정제 자료이므로 임의로 삭제하거나 수정하지 않는다.

## 구현 규칙

- 한국어 컬럼명을 유지한다.
- 시도 순서는 서울부터 제주까지 고정한다.
- 회귀 결과는 β, 표준화 β, p, R², N을 보고한다.
- H6의 다중 비교는 Benjamini-Hochberg FDR을 사용한다.
- 한글 그래프 폰트는 Noto Sans KR, Malgun Gothic 등의 순서로 탐색한다.
- 코드 변경 후 `py_compile`과 Streamlit AppTest를 실행한다.

## 실행

```powershell
.venv\Scripts\streamlit.exe run app.py
.venv\Scripts\python.exe build_v2_outputs.py
```
