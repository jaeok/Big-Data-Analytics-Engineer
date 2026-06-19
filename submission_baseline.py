import pandas as pd
import numpy as np
import sklearn.metrics as sm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import xgboost as xgb
import lightgbm as lgb

# ==============================================================================
# 📂 CSV 파일에서 데이터 읽기
# ==============================================================================
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
print('train.csv / test.csv 읽기 완료')

# ==============================================================================
# 🏆 분류(Classification) 베이스라인
# ==============================================================================
print("\n================== [분류 문제 분석 시작] ==================")

# [1단계] 기본 신상명세 확보 (ID 챙기기, 타겟 설정)
TARGET = 'target'
test_id = test['id'] if 'id' in test.columns else pd.Series(range(len(test)), name='id')

# [2단계] 결측치 일괄 대치 (평균 및 UNKNOWN 채우기)
# - 수치형 변수 채우기
train = train.fillna(train.mean(numeric_only=True))
test = test.fillna(test.mean(numeric_only=True))

# - 범주형 변수 채우기
cat_cols = [c for c in train.columns if train[c].dtype.name in ['object', 'category'] and c != TARGET]
for col in cat_cols:
    train[col] = train[col].fillna('UNKNOWN')
    test[col] = test[col].fillna('UNKNOWN')

# [3단계] 범주형 변수 라벨 인코딩
for col in cat_cols:
    le = LabelEncoder()
    train[col] = le.fit_transform(train[col])
    test[col] = le.transform(test[col])

# [4단계] 문제지(Features)와 정답지(Target) 분해
X_train = train.drop(columns=[TARGET, 'id'], errors='ignore') # ID 컬럼은 학습에 불필요하므로 제거
X_test = test.drop(columns=['id'], errors='ignore')
y_train = train[TARGET]

# [5단계] 모의고사 쪼개기
# 분류일 때만 stratify를 사용하고, 회귀일 때는 제거
is_classification = (y_train.nunique() <= 2 and set(y_train.dropna().astype(int).unique()).issubset({0, 1}))

if is_classification:
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
else:
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )

# [6단계] 암기하기 쉬운 초경량 3대장 분류 모델 정의
models = {
    'RandomForest': RandomForestClassifier(random_state=42, n_jobs=-1),
    'XGBoost': xgb.XGBClassifier(random_state=42, n_jobs=-1, eval_metric='logloss'),
    'LightGBM': lgb.LGBMClassifier(random_state=42, n_jobs=-1, verbose=-1)
}

# [7단계] 모델 학습 및 모의고사 채점
best_score = 0
best_model_name = ''
best_model = None

for name, model in models.items():
    model.fit(X_tr, y_tr)
    val_pred_proba = model.predict_proba(X_val)[:, 1]
    val_pred_cls = model.predict(X_val)

    # ROC-AUC는 확률값이 필요하고, F1은 클래스 라벨이 필요함
    score_auc = sm.roc_auc_score(y_val, val_pred_proba)
    score_f1 = sm.f1_score(y_val, val_pred_cls)

    print(f"🌲 {name}의 ROC-AUC: {score_auc:.4f} / F1: {score_f1:.4f}")

    if score_auc > best_score:
        best_score = score_auc
        best_model_name = name
        best_model = model

print(f"🥇 최종 1등 선정 모델: {best_model_name} (검증 점수: {best_score:.4f})")

# [8단계] 1등 한 놈 납치해서 최종 예측 및 제출 파일 저장
final_pred_proba = best_model.predict_proba(X_test)[:, 1]
submit = pd.DataFrame({'id': test_id, 'target': final_pred_proba})
submit.to_csv('result_classification.csv', index=False)
print("💾 분류 예측 제출 파일 저장 완료! (result_classification.csv)")

# ==============================================================================
# 📈 회귀(Regression) 베이스라인
# ==============================================================================
print("\n================== [회귀 문제 분석 시작] ==================")

# 회귀용 데이터는 train/test를 다시 읽기 위해 복사본 사용
train_reg = pd.read_csv('train.csv')
test_reg = pd.read_csv('test.csv')

# target이 숫자형인지 확인
if train_reg['target'].dtype.kind not in 'biufc':
    raise ValueError("회귀용 데이터는 target이 연속형 숫자여야 합니다.")

# 회귀용 전처리
train_reg = train_reg.fillna(train_reg.mean(numeric_only=True))
test_reg = test_reg.fillna(test_reg.mean(numeric_only=True))

cat_cols_reg = [c for c in train_reg.columns if train_reg[c].dtype.name in ['object', 'category'] and c != 'target']
for col in cat_cols_reg:
    train_reg[col] = train_reg[col].fillna('UNKNOWN')
    test_reg[col] = test_reg[col].fillna('UNKNOWN')

for col in cat_cols_reg:
    le = LabelEncoder()
    train_reg[col] = le.fit_transform(train_reg[col])
    test_reg[col] = le.transform(test_reg[col])

X_train_reg = train_reg.drop(columns=['target', 'id'], errors='ignore')
X_test_reg = test_reg.drop(columns=['id'], errors='ignore')
y_train_reg = train_reg['target']

# 회귀는 연속형 target 이므로 stratify 없이 분할
X_tr_reg, X_val_reg, y_tr_reg, y_val_reg = train_test_split(
    X_train_reg, y_train_reg, test_size=0.2, random_state=42
)

models_reg = {
    'RandomForest': RandomForestRegressor(random_state=42, n_jobs=-1),
    'XGBoost': xgb.XGBRegressor(random_state=42, n_jobs=-1),
    'LightGBM': lgb.LGBMRegressor(random_state=42, n_jobs=-1, verbose=-1)
}

best_rmse = float('inf')
best_reg_name = ''
best_reg_model = None

for name, model in models_reg.items():
    model.fit(X_tr_reg, y_tr_reg)
    val_pred = model.predict(X_val_reg)

    score_rmse = np.sqrt(sm.mean_squared_error(y_val_reg, val_pred))
    score_r2 = sm.r2_score(y_val_reg, val_pred)

    print(f"📈 {name}의 RMSE: {score_rmse:.4f} / R2: {score_r2:.4f}")
    if score_rmse < best_rmse:
        best_rmse = score_rmse
        best_reg_name = name
        best_reg_model = model

print(f"🥇 최종 1등 선정 모델: {best_reg_name} (검증 오차: {best_rmse:.4f})")

final_pred_reg = best_reg_model.predict(X_test_reg)
submit_reg = pd.DataFrame({'id': test_reg['id'] if 'id' in test_reg.columns else range(len(test_reg)), 'target': final_pred_reg})
submit_reg.to_csv('result_regression.csv', index=False)
print("💾 회귀 예측 제출 파일 저장 완료! (result_regression.csv)")


