import pandas as pd
import numpy as np
import sklearn.metrics as sm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import lightgbm as lgb

# ==========================================================
# Load data
# ==========================================================
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# ==========================================================
# Config
# ==========================================================
TARGET = 'target'
METRIC = 'ROC-AUC'
ID_COL = 'id'

# ==========================================================
# Helper functions
# ==========================================================
def fill_missing(df):
    df = df.copy()
    for col in df.columns:
        if df[col].dtype.kind in 'iufc':
            df[col] = df[col].fillna(df[col].mean())
        else:
            df[col] = df[col].fillna('UNKNOWN')
    return df


def encode_categoricals(train_df, test_df):
    train_df = train_df.copy()
    test_df = test_df.copy()
    for col in train_df.columns:
        if train_df[col].dtype == 'object' or train_df[col].dtype.name == 'category':
            le = LabelEncoder()
            train_df[col] = le.fit_transform(train_df[col])
            test_df[col] = le.transform(test_df[col])
    return train_df, test_df


# ==========================================================
# Preprocess
# ==========================================================
train = fill_missing(train)
test = fill_missing(test)

train, test = encode_categoricals(train, test)

# ID handling
if ID_COL in train.columns:
    train_id = train[ID_COL]
else:
    train_id = pd.Series(range(len(train)), name='id')

if ID_COL in test.columns:
    test_id = test[ID_COL]
else:
    test_id = pd.Series(range(len(test)), name='id')

X_train = train.drop(columns=[TARGET, ID_COL], errors='ignore')
X_test = test.drop(columns=[ID_COL], errors='ignore')
y_train = train[TARGET]

# ==========================================================
# Split
# ==========================================================
# Classification if target is binary (0/1)
if y_train.nunique() <= 2 and set(y_train.dropna().astype(int).unique()).issubset({0, 1}):
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    task = 'classification'
else:
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )
    task = 'regression'

# ==========================================================
# Model definitions
# ==========================================================
if task == 'classification':
    models = {
        'RandomForest': RandomForestClassifier(random_state=42, n_jobs=-1),
        'LightGBM': lgb.LGBMClassifier(random_state=42, n_jobs=-1, verbose=-1)
    }
else:
    models = {
        'RandomForest': RandomForestRegressor(random_state=42, n_jobs=-1),
        'LightGBM': lgb.LGBMRegressor(random_state=42, n_jobs=-1, verbose=-1)
    }

# ==========================================================
# Evaluate and select best model
# ==========================================================
if task == 'classification':
    best_score = -np.inf
    best_model = None
    best_name = ''

    for name, model in models.items():
        model.fit(X_tr, y_tr)
        val_pred_proba = model.predict_proba(X_val)[:, 1]
        val_pred = model.predict(X_val)

        score_auc = sm.roc_auc_score(y_val, val_pred_proba)
        score_f1 = sm.f1_score(y_val, val_pred)

        print(f"{name}: ROC-AUC={score_auc:.6f} | F1={score_f1:.6f}")

        if score_auc > best_score:
            best_score = score_auc
            best_model = model
            best_name = name

    print(f"Best model: {best_name} ({METRIC}: {best_score:.6f})")

    final_pred = best_model.predict_proba(X_test)[:, 1]
else:
    best_score = np.inf
    best_model = None
    best_name = ''

    for name, model in models.items():
        model.fit(X_tr, y_tr)
        val_pred = model.predict(X_val)

        score_rmse = np.sqrt(sm.mean_squared_error(y_val, val_pred))
        score_r2 = sm.r2_score(y_val, val_pred)

        print(f"{name}: RMSE={score_rmse:.6f} | R2={score_r2:.6f}")

        if score_rmse < best_score:
            best_score = score_rmse
            best_model = model
            best_name = name

    print(f"Best model: {best_name} ({METRIC}: {best_score:.6f})")

    final_pred = best_model.predict(X_test)

# ==========================================================
# Save submission
# ==========================================================
result = pd.DataFrame({
    'id': test_id,
    'target': final_pred
})
result.to_csv('result.csv', index=False)

print('result.csv saved')
