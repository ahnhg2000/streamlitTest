import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import os
import subprocess
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, roc_curve
import lightgbm as lgb
from imblearn.over_sampling import SMOTE

st.set_page_config(page_title="자동화 분석 파이프라인 대시보드", layout="wide")

st.sidebar.title("데이터 분석 파이프라인")

# 1. 동적 데이터셋 목록 로드
def get_csv_files(folder_path="dataset"):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path, exist_ok=True)
    files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    return files

csv_files = get_csv_files()
if not csv_files:
    st.sidebar.warning("dataset 폴더에 CSV 파일이 없습니다.")
    st.stop()

selected_csv = st.sidebar.selectbox("분석할 데이터셋 선택", csv_files)
dataset_name = os.path.splitext(selected_csv)[0]
file_path = os.path.join("dataset", selected_csv)

st.sidebar.markdown("---")
menu = st.sidebar.radio("파이프라인 단계 선택", [
    "1. 데이터 통찰 (EDA)", 
    "2. 대화형 ML 시뮬레이터", 
    "3. 자동화 보고서 (PDF) 배포"
])

@st.cache_data
def load_and_preprocess_data(path):
    df = pd.read_csv(path)
    
    # 동적 자동 전처리 (간이 버전)
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())
    for col in cat_cols:
         df[col] = df[col].fillna(df[col].mode()[0])
    return df

@st.cache_resource
def train_model(df):
    target = df.columns[-1]
    
    # 타겟 정수형 인코딩
    if df[target].dtype == 'object' or len(df[target].unique()) > 2:
        df[target] = pd.factorize(df[target])[0]

    X = df.drop(target, axis=1)
    X = pd.get_dummies(X, drop_first=True)
    y = df[target]
    
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
    
    if y_train.value_counts().min() > 5:
        smote = SMOTE(random_state=42)
        X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
    else:
        X_train_sm, y_train_sm = X_train, y_train
    
    model = lgb.LGBMClassifier(random_state=42, verbose=-1)
    model.fit(X_train_sm, y_train_sm)
    
    return model, scaler, X_test, y_test, X.columns, target

df = load_and_preprocess_data(file_path)
model, scaler, X_test, y_test, feature_cols, target_col = train_model(df)

st.title(f"📊 지능형 데이터 분석 대시보드 - `{selected_csv}`")

if menu == "1. 데이터 통찰 (EDA)":
    st.header("1장. 탐색적 데이터 분석 (EDA)")
    
    col1, col2 = st.columns(2)
    col1.metric("데이터 샘플 수", df.shape[0])
    col2.metric("변수(Feature) 개수", df.shape[1] - 1)

    st.subheader("데이터 미리보기")
    st.dataframe(df.head())
    
    st.subheader("수치형 변수 상관관계")
    num_df = df.select_dtypes(include=[np.number])
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(num_df.corr(), annot=False, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

elif menu == "2. 대화형 ML 시뮬레이터":
    st.header("2장. LightGBM 모델 성능 및 다각도 분석")
    
    y_pred = model.predict(X_test)
    try:
        y_proba = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_proba)
    except:
        y_proba = np.zeros(len(y_test))
        auc = 0.0

    acc = accuracy_score(y_test, y_pred)
    
    col1, col2 = st.columns(2)
    col1.metric("최적 모델 엔진", "LightGBM")
    col2.metric(f"AUC Score (Target: {target_col})", f"{auc:.4f}")
    
    st.markdown("---")
    
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("피처 중요도 (Feature Importance)")
        importances = model.feature_importances_
        feature_df = pd.DataFrame({'Feature': feature_cols, 'Importance': importances}).sort_values('Importance', ascending=False).head(15)
        fig_bar = px.bar(feature_df, x='Importance', y='Feature', orientation='h', width=500, height=400)
        fig_bar.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig_bar)
        
    with col_b:
        st.subheader("다이내믹 테스트 (임의 레코드 평가)")
        st.info("실제 데이터의 첫 번째 행 정보를 바탕으로 모델 예측을 시뮬레이션 합니다.")
        sample_data = df.iloc[[0]].drop(target_col, axis=1)
        sample_encoded = pd.get_dummies(sample_data, drop_first=True)
        # 피처 컬럼 맞추기 (누락 0 처리)
        for c in feature_cols:
            if c not in sample_encoded.columns:
                sample_encoded[c] = 0
        sample_encoded = sample_encoded[feature_cols]

        scaled_sample = scaler.transform(sample_encoded)
        try:
            prob = model.predict_proba(scaled_sample)[0, 1]
            st.warning(f"예측된 Class 1 발생 확률: **{prob*100:.2f}%**")
        except:
            st.warning("이진분류 확률 예측을 지원하지 않는 다중 클래스 혹은 회귀 포맷입니다.")
        st.write("샘플 입력 피처값:")
        st.dataframe(sample_data)

elif menu == "3. 자동화 보고서 (PDF) 배포":
    st.header("3장. E2E 파이프라인: 리포트 자동 생성 릴리즈")
    st.markdown(f"`{selected_csv}` 데이터셋을 엔진으로 전달하여, 백그라운드 분석 스크립트를 수행하고 **가로형 Marp 마크다운 및 PDF 보고서**를 일괄 생성합니다.")
    
    if st.button("🚀 전체 파이프라인 실행 및 Marp 보고서 릴리즈"):
        st.info("백그라운드에서 분석 스크립트(`analyze.py`)가 실행됩니다... (약 10~20초 소요)")
        
        # 1. analyze.py 실행 (윈도우 환경 호환성을 위해 shell=True 적용)
        result = subprocess.run(f"uv run analyze.py --file {selected_csv}", shell=True, capture_output=True, text=True)
        
        if result.returncode == 0:
            st.success("✅ 백그라운드 파이썬 스크립트 실행 완료 및 `.md` 마크다운 보고서 조립 성공!")
            
            # 2. Marp CLI로 PDF 렌더링
            md_file = f"report/{dataset_name}_Report.md"
            pdf_file = f"report/{dataset_name}_Report.pdf"
            
            if os.path.exists(md_file):
                st.info("Marp 렌더링 엔진 호출 중 (마크다운 → 가로 PDF 변환)...")
                # 설치된 marp-cli 이용 (윈도우 환경 호환성을 위해 shell=True 적용 및 로컬 이미지 접근 권한 추가)
                marp_res = subprocess.run(f"npx @marp-team/marp-cli@latest {md_file} --html --allow-local-files --pdf -o {pdf_file}", shell=True, capture_output=True, text=True)
                
                if marp_res.returncode == 0:
                    st.success(f"🎉 PDF 문서 변환 성공! 경로: `{pdf_file}`")
                    with open(pdf_file, "rb") as f:
                        st.download_button("📥 생성된 PDF 보고서 다운로드", f, file_name=f"{dataset_name}_AI_Report.pdf", mime="application/pdf")
                else:
                    st.error("Marp 변환 중 오류 발생 (Node.js/Marp CLI 설치 확인 필요)")
                    st.code(marp_res.stderr)
            else:
                st.error("생성된 마크다운 파일을 찾을 수 없습니다.")
        else:
            st.error("분석 스크립트 실행 중 치명적 오류가 발생했습니다.")
            st.code(result.stderr)

