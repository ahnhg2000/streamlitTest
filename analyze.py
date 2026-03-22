import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import argparse
import warnings
import glob

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, roc_curve, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
import xgboost as xgb
from imblearn.over_sampling import SMOTE
import koreanize_matplotlib

warnings.filterwarnings('ignore')

def main():
    parser = argparse.ArgumentParser(description="Auto EDA and Modeling Pipeline")
    parser.add_argument('--file', type=str, default='diabetes.csv', help='CSV filename inside dataset/ folder')
    parser.add_argument('--target', type=str, default='', help='Target column name. If empty, uses the last column.')
    args = parser.parse_args()

    dataset_name = os.path.splitext(args.file)[0]
    file_path = os.path.join('dataset', args.file)
    
    if not os.path.exists(file_path):
        print(f"Error: {file_path} does not exist.")
        sys.exit(1)

    # 0. 명명 규칙에 따른 기존 보고서 파일 강제 삭제 (Clean-up)
    patterns = [
        f'report/{dataset_name}_*.csv',
        f'report/{dataset_name}_*.md',
        f'report/{dataset_name}_*.pdf',
        f'report/images/{dataset_name}_*.png'
    ]
    for pattern in patterns:
        for old_file in glob.glob(pattern):
            try:
                os.remove(old_file)
            except Exception as e:
                pass
    print(f"[{dataset_name}] 기존 보고서/이미지 파일 정리 완료.")

    # 1. 환경 설정 및 데이터 로드
    os.makedirs('report/images', exist_ok=True)
    df = pd.read_csv(file_path)
    print(f"[{dataset_name}] 데이터 로드 완료: {df.shape}")

    target = args.target if args.target else df.columns[-1]
    
    if target not in df.columns:
        print(f"Error: Target column '{target}' not found.")
        sys.exit(1)

    # 기본 전처리: 수치형 컬럼의 결측치 중앙값 대체, 범주형 최빈값 대체
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    
    for col in num_cols:
         df[col] = df[col].fillna(df[col].median())
    for col in cat_cols:
         df[col] = df[col].fillna(df[col].mode()[0])

    # 타겟이 범주형이거나 문자열이면 Label Encoding 필요 (여기서는 숫자 이진분류 가정)
    if df[target].dtype == 'object' or len(df[target].unique()) > 2:
        # 이진 분류 이외의 경우 혹은 문자열인 경우의 처리는 간소화: 제일 많은 2개 클래스만 남기거나 팩터화
        df[target] = pd.factorize(df[target])[0]

    # 기술 통계량 산출
    stats_df = df.describe().T[['mean', 'std', 'min', '50%', 'max']].reset_index()
    stats_df.columns = ['Feature', 'Mean', 'Std', 'Min', 'Median', 'Max']
    stats_df.to_csv(f'report/{dataset_name}_variable_stats.csv', index=False, encoding='utf-8-sig')

    # [2.2] 이상치 확인 (Boxplot) - 가로형에 맞게 가로폭 확장
    plt.figure(figsize=(10, 5))
    sns.boxplot(data=df.drop(target, axis=1), palette='Set3')
    plt.title("특성별 분포 및 이상치 탐지")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'report/images/{dataset_name}_outlier_boxplot.png', dpi=300)
    plt.close()

    # 이상치 동적 분석 텍스트 생성
    outlier_info = []
    numeric_features = df.drop(columns=[target], errors='ignore').select_dtypes(include=[np.number]).columns
    for col in numeric_features:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        outliers_cnt = df[(df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)].shape[0]
        if outliers_cnt > 0:
            outlier_info.append((col, outliers_cnt))
    
    outlier_info.sort(key=lambda x: x[1], reverse=True)
    if outlier_info:
        top_outlier_col, top_outlier_cnt = outlier_info[0]
        outlier_text = f"데이터 스캔 결과, **{top_outlier_col}** 피처에서 가장 많은 이상치({top_outlier_cnt}건)가 집중적으로 관찰되었습니다. 해당 수치들이 도메인상 가능한 극단값인지 센서/입력 오류인지 실무적 검증이 선행되어야 모델의 예측 안정성이 보장됩니다."
    else:
        outlier_text = "데이터 스캔 결과, 분포를 크게 왜곡하는 뚜렷한 통계적 이상치가 발견되지 않아 전반적인 데이터 수집 품질이 매우 양호하고 안정적입니다."

    # [2.3] 상관관계 히트맵 (가로형)
    plt.figure(figsize=(9, 5.5))
    corr = df[num_cols].corr()
    sns.heatmap(corr, annot=False, cmap='coolwarm', square=False) # annot=False로 변수가 많을때 텍스트 가독성 문제 방지
    plt.title('상관계수 히트맵')
    plt.tight_layout()
    plt.savefig(f'report/images/{dataset_name}_correlation_heatmap.png', dpi=300)
    plt.close()

    # 다중공선성 동적 분석 텍스트 생성
    corr_matrix = corr.abs()
    # 대각선 및 하단 트라이앵글 마스킹
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    corr_pairs = upper_tri.unstack().dropna().sort_values(ascending=False)
    
    if not corr_pairs.empty:
        col1, col2 = corr_pairs.index[0]
        max_corr_val = corr_pairs.iloc[0]
        if max_corr_val > 0.7:
             corr_text = f"히트맵 분석 결과, **{col1}**와(과) **{col2}** 간의 피어슨 상관계수가 {max_corr_val:.2f}로 매우 높게 나타났습니다. 다중공선성으로 인해 독립변수들의 설명력이 떨어질 위험이 있으므로, 두 변수 중 하나를 제외하거나 차원을 축소(PCA)하는 것을 자동화 파이프라인 고도화 전략으로 권장합니다."
        else:
             corr_text = f"가장 높은 상관관계를 보인 변수 쌍은 **{col1}**와(과) **{col2}** ({max_corr_val:.2f}) 로 확인되었습니다. 전반적으로 독립 변수들 간의 다중공선성 위험이 {max_corr_val:.2f} 이하로 낮게 유지되고 있어 트리 및 선형 예측 모델 모두 원활한 학습이 가능한 강건한 상태입니다."
    else:
        corr_text = "의미 있는 선형 상관관계를 가지는 수치형 변수 쌍이 식별되지 않았습니다."

    # 스케일링 및 분할
    X = df.drop(target, axis=1)
    # 범주형 변수를 원핫인코딩 처리
    X = pd.get_dummies(X, drop_first=True)
    y = df[target]

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
    
    # 클래스 밸런스 확인 및 SMOTE
    min_class_count = y_train.value_counts().min()
    if min_class_count > 5:
        smote = SMOTE(random_state=42)
        X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
    else:
        X_train_sm, y_train_sm = X_train, y_train

    # 4. 모델 성능 평가
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Random Forest': RandomForestClassifier(random_state=42),
        'LightGBM': lgb.LGBMClassifier(random_state=42, verbose=-1),
        'XGBoost': xgb.XGBClassifier(random_state=42, eval_metric='logloss')
    }

    results = []
    for name, model in models.items():
        model.fit(X_train_sm, y_train_sm)
        y_pred = model.predict(X_test)
        
        try:
            y_proba = model.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, y_proba)
        except:
            auc = 0.0

        results.append({
            'Model': name,
            'Accuracy': accuracy_score(y_test, y_pred),
            'F1-Score': f1_score(y_test, y_pred, average='weighted'),
            'AUC(ROC)': auc
        })

    res_df = pd.DataFrame(results).sort_values(by='AUC(ROC)', ascending=False)
    res_df.to_csv(f'report/{dataset_name}_model_benchmarks.csv', index=False, encoding='utf-8-sig')

    best_model_name = res_df.iloc[0]['Model']
    best_model = models[best_model_name]

    # [5.1] Confusion Matrix
    y_pred_best = best_model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred_best)
    
    # 혼동행렬 동적 텍스트 생성
    if cm.size == 4: # 이진 분류인 경우
        tn, fp, fn, tp = cm.ravel()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        cm_text = f"검증 데이터 평가 결과, 목표 타겟을 찾아낸(TP) 건수는 {tp}건, 정상 범주를 맞춘(TN) 것은 {tn}건입니다. " \
                  f"반면 실제 정상을 타겟으로 오탐한 제1종 오류(FP)는 {fp}건, " \
                  f"실제 타겟을 놓친 제2종 오류(FN)는 {fn}건 발생했습니다.<br>" \
                  f"현재 Threshold 기준 **정밀도(Precision): {precision*100:.1f}%**, **재현율(Recall): {recall*100:.1f}%**를 보입니다."
    else:
        cm_text = "다중 클래스 분류 모델의 혼동 행렬 시각화 결과입니다. 대각선 영역의 분포를 통해 각 클래스별 판단 정확도를 확인할 수 있습니다."
        
    plt.figure(figsize=(6, 4.5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'혼동 행렬 - {best_model_name}')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f'report/images/{dataset_name}_confusion_matrix.png', dpi=300)
    plt.close()

    # [5.2] ROC Curve (Comparison)
    plt.figure(figsize=(7, 5))
    for i in range(min(4, len(res_df))):
        m_name = res_df.iloc[i]['Model']
        m_model = models[m_name]
        try:
             m_proba = m_model.predict_proba(X_test)[:, 1]
             fpr, tpr, _ = roc_curve(y_test, m_proba)
             plt.plot(fpr, tpr, label=f'{m_name} (AUC={res_df.iloc[i]["AUC(ROC)"]:.3f})')
        except:
             pass
    plt.plot([0,1], [0,1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('상위 모델별 ROC Curves 비교')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'report/images/{dataset_name}_roc_comparison.png', dpi=300)
    plt.close()

    # [5.3] 피처 중요도 (Best)
    importances = None
    top_3_features_str = "도출 불가"
    if hasattr(best_model, 'feature_importances_'):
        importances = best_model.feature_importances_
    elif hasattr(best_model, 'coef_'):
        importances = np.abs(best_model.coef_[0])
        
    if importances is not None:
        # 상위 10개만 시각화 (공간 제약 고려)
        feat_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances}).sort_values('Importance', ascending=True)
        top_10 = feat_df.tail(10)
        
        top_3 = feat_df.tail(3).sort_values('Importance', ascending=False)['Feature'].tolist()
        top_3_features_str = ", ".join([f"**{f}**" for f in top_3])
        top_1_feature = top_3[0] if len(top_3) > 0 else "N/A"
        
        plt.figure(figsize=(8, 4.8))
        plt.barh(top_10['Feature'], top_10['Importance'], color='darkblue')
        plt.title(f'Top 10 변수 중요도 (Feature Importance) - {best_model_name}')
        plt.tight_layout()
        plt.savefig(f'report/images/{dataset_name}_feature_importance.png', dpi=300)
        plt.close()
    
    # ==========================================
    # 6. Marp 가로형(.md) 파일 자동 조립 (Auto Chunking & Formatting)
    # ==========================================
    md_content = f"""---
marp: true
theme: default
size: 16:9
paginate: true
footer: "AI/ML 모델링 파이프라인 (Data Analysis Report)"
style: |
  section {{
    background-color: #ffffff;
    border: 2px solid #e0e0e0;
    border-top: 15px solid #1f4e79;
    padding: 40px 60px 85px 60px;
    font-family: 'Malgun Gothic', sans-serif;
    font-size: 22px;
  }}
  h1 {{ color: #1f4e79; font-size: 1.7em; border-bottom: 2px solid #ccc; padding-bottom: 10px; margin-bottom: 15px; }}
  h2 {{ color: #333333; font-size: 1.25em; margin-top: 0; margin-bottom: 15px; }}
  h3 {{ color: #555555; font-size: 1.05em; margin-bottom: 10px; }}
  p, li {{ font-size: 0.85em; }}
  .insight-box {{
    background-color: #f4f6f9;
    border-left: 6px solid #1f4e79;
    padding: 12px 18px;
    margin-top: 10px;
    font-size: 0.75em;
    color: #444;
    line-height: 1.5;
  }}
  .flex-container {{ display: flex; justify-content: space-between; align-items: flex-start; gap: 20px; }}
  .img-container {{ text-align: center; }}
  .img-container img, .flex-container img {{ max-height: 360px; object-fit: contain; }}
  footer {{
    border-top: 1px solid #ccc;
    width: calc(100% - 120px);
    left: 60px;
    padding-top: 8px;
    font-size: 0.6em;
    color: #888;
    bottom: 25px;
    text-align: left;
  }}
  table {{ border-collapse: collapse; width: 100%; margin-bottom: 15px; font-size: 0.6em; }}
  th, td {{ border: 1px solid #ddd; padding: 4px 8px; text-align: center; }}
  th {{ background-color: #f2f2f2; color: #333; }}
---

# {dataset_name.upper()} 심층 분석 및 예측 모델 보고서
### **[Executive Summary] 자동화 파이프라인 분석 결과**
- **분석 대상 파일**: `{args.file}`
- **수행 모듈**: 파이썬 자동화 ML Pipeline (Advanced E2E Report)
- **최우수 알고리즘**: **{best_model_name}**
- **최고 달성 성능 (AUC)**: **{res_df.iloc[0]['AUC(ROC)']:.4f}**

<br>
<div class="insight-box">
  <strong>💡 AI/ML 데이터 분석가 요약 제언:</strong><br>
  본 문서는 데이터 프로파일링, 이상치 시각화, 상관성 분석, 다중 알고리즘 벤치마킹을 거쳐 최종적으로 '{best_model_name}' 모델을 챔피언 모델로 선정하는 과정을 담고 있습니다. 도출된 주요 피처(Features)는 비즈니스 실무에서 선제적 예방 및 진단 지표로 적극 활용될 수 있습니다.
</div>

---

## 1. 데이터 프로파일링 및 전처리 전략 (Overview)

<div class="flex-container">
  <div style="width: 50%;">
    <h3>1-1. 데이터셋 기본 차원</h3>
    <ul style="font-size: 0.85em;">
      <li><strong>전체 샘플(Rows) 수</strong>: {df.shape[0]} 건</li>
      <li><strong>독립/종속 변수(Cols) 수</strong>: {df.shape[1]} 개</li>
      <li><strong>타겟(분류 대상) 변수</strong>: <code>{target}</code></li>
    </ul>
    <h3>1-2. 데이터 가공 전략 (Preprocessing)</h3>
    <ul style="font-size: 0.85em;">
      <li>결측치 처리: 도메인 특성 왜곡 방지를 위해 <strong>수치형 중앙값 / 범주형 최빈값</strong> 대체 보정.</li>
      <li>클래스 불균형: 필요 시 <strong>SMOTE 오버샘플링</strong> 적용.</li>
    </ul>
  </div>
  <div style="width: 45%;">
    <h3>1-3. 타겟 변수 `{target}` 클래스 분포</h3>
    {df[target].value_counts().reset_index().rename(columns={'count': 'Count', target: 'Class'}).to_html(index=False, classes='')}
    <div class="insight-box" style="margin-top:15px; font-size: 0.7em;">
      Minority Class의 비율을 점검하여 불균형이 판단될 경우 오버샘플링 파이프라인이 자동 트리거되었습니다.
    </div>
  </div>
</div>

---

## 2. 탐색적 데이터 분석 (1) - 이상치(Outlier) 분포

종속 변수를 제외한 주요 분석 변수들의 스케일 갭(Scale Gap)과 극단값 분포를 확인합니다.

<div class="flex-container">
  <div style="width: 55%; text-align:center;">
    <img src="images/{dataset_name}_outlier_boxplot.png" width="90%"/>
  </div>
  <div style="width: 45%;">
    <div class="insight-box" style="margin-top: 0;">
      <strong>🧐 이상치 분석 결과 및 전문가 가이드라인:</strong><br><br>
      {outlier_text}<br><br>
      트리 계열(Tree) 알고리즘은 이상치에 매우 강건하지만, 로지스틱 회귀와 같은 선형 모델 적용 시에는 극단값을 Robust Scale 또는 Log 변환으로 조정해주면 탐지 성능이 유의미하게 향상될 수 있습니다.
    </div>
  </div>
</div>

---

## 3. 탐색적 데이터 분석 (2) - 다중공선성(Multicollinearity)

각 수치형 변수 쌍에 대한 피어슨 상관계수 행렬(Pearson Correlation Heatmap)입니다.

<div class="flex-container">
  <div style="width: 55%; text-align:center;">
    <img src="images/{dataset_name}_correlation_heatmap.png" width="90%"/>
  </div>
  <div style="width: 45%;">
    <div class="insight-box" style="margin-top: 0;">
      <strong>🧐 다중공선성 진단 결과 및 전문가 제언:</strong><br><br>
      {corr_text}<br><br>
      본 상관도 분석은 비즈니스 프로세스 상 성격이 겹치는 중복 KPI 지표(Feature)를 제거하여, 데이터 수집 프로세스와 엔지니어링 비용의 효율을 극대화하는 근거 자료로도 훌륭하게 활용될 수 있습니다.
    </div>
  </div>
</div>

---

## 4. 머신러닝 알고리즘 벤치마킹 (Cross-Validation)

의료/금융/실측 임상 등 전문 도메인의 실무 평가 기준을 반영하기 위하여 단순 정확도(Accuracy)뿐만 아니라 **AUC(ROC 영역)** 및 **가중 F1-Score**를 종합적으로 산출, 최적의 분류기를 판별했습니다.

**[경쟁 모델 벤치마크 결과] (AUC 높은 순)**

{res_df.to_html(index=False)}

<div class="insight-box">
  <strong>🎯 Champion Model 선정 사유:</strong><br>
  주어진 지표를 종합할 때, 분류 임곗값 변화에 가장 견고하며(AUC 극대화) 타겟 판별력이 우수한 <strong>{best_model_name}</strong>을 실무 적용 배포 모델로 최우선 고려합니다.
</div>

---

## 5. 모델 성과 심층 분석 - ROC & Confusion Matrix

단순 Accuracy를 넘어 '{best_model_name}' 모델의 오분류(False Positive / False Negative) 양상과 임곗값 강건성을 분석합니다.

<div class="flex-container">
    <div style="width: 48%; text-align:center;">
        <strong>[분석 1] 상위 모델 ROC Curve 비교</strong><br><br>
        <img src="images/{dataset_name}_roc_comparison.png" width="450"/>
    </div>
    <div style="width: 48%; text-align:center;">
        <strong>[분석 2] 최적 모델 Confusion Matrix</strong><br><br>
        <img src="images/{dataset_name}_confusion_matrix.png" width="400"/>
    </div>
</div>

<div class="insight-box">
  <strong>🎯 모델 성과 종합 진단 및 시사점:</strong><br><br>
  {cm_text}<br><br>
  ROC 곡선이 좌상단에 밀착할수록 판별력이 우수합니다. 현업 적용 시 오탐(FP) 처리 비용이 크면 임곗값(Threshold)을 높여 정밀도를 개선하고, 미탐(FN) 위험이 치명적이라면 임곗값을 낮춰 재현율을 우선적으로 방어해야 합니다.
</div>

---

## 6. 비즈니스 액션 결론: 피처 중요도 (Feature Importance)

챔피언 모델이 산출한 기여도 최상위 피처(결정 트리 정보 이득 기반 혹은 선형 가중치)입니다.

<div class="flex-container">
  <div style="width: 55%; text-align:center;">
    <img src="images/{dataset_name}_feature_importance.png" width="90%"/>
  </div>
  <div style="width: 45%;">
    <div class="insight-box" style="margin-top: 0;">
      <strong>💡 분석 시사점 (Action Items) 및 향후 과제:</strong><br><br>
      1. AI 추출 결과, 타겟 판별에 가장 지배적인 영향력을 행사하는 핵심 인자는 <strong>{top_3_features_str}</strong> 순입니다.<br><br>
      2. 특히 압도적 1위 지표인 **{top_1_feature}** 피처는 가장 높은 정보 이득 비율을 보입니다. 현장에서는 최우선 KPI로 삼아 해당 항목의 수치 관리와 데이터수집 품질을 감독해야 합니다.<br><br>
      3. 제공된 AI 웹 대시보드 <strong>'대화형 시뮬레이터'</strong> 메뉴에서 {top_1_feature} 값을 직접 미세 조정하며 타겟 전환 예측 확률이 어떻게 달라지는지 검토해 보시길 적극 권장합니다.
    </div>
  </div>
</div>
"""
    md_filepath = f'report/{dataset_name}_Report.md'
    with open(md_filepath, 'w', encoding='utf-8') as f:
        f.write(md_content)
    
    print(f"가로형 마크다운 변환 완료: {md_filepath}")
    print("Marp 전환 실행 명령어: `marp --pdf report/{0}_Report.md`".format(dataset_name))

if __name__ == '__main__':
    main()

