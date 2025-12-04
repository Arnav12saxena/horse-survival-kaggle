import os
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
import optuna

# Set base path to current script directory
base_path = os.path.abspath(os.path.dirname(__file__))

# Paths to files
train_path = os.path.join(base_path, "train.csv")
test_path = os.path.join(base_path, "test.csv")
origin_path = os.path.join(base_path, "horse.csv")

# Verify dataset files exist
for path in [train_path, test_path, origin_path]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset file not found at: {path}")

# Load data
print("Loading data...")
train = pd.read_csv(train_path)
origin = pd.read_csv(origin_path)
test = pd.read_csv(test_path)

# Combine train and original dataset
train_total = pd.concat([train, origin], ignore_index=True)
train_total.drop_duplicates(inplace=True)

# Encode outcome labels
le = LabelEncoder()
y = le.fit_transform(train_total["outcome"])
X = train_total.drop(columns=["outcome"])

# Drop 'id' if it exists
if 'id' in X.columns:
    X = X.drop(columns=['id'])

test_ids = test["id"]
test = test.drop(columns=["id"])

# Handle missing values
numerical_columns = X.select_dtypes(include=['float64', 'int64']).columns
categorical_columns = X.select_dtypes(include=['object']).columns

# Impute numerical columns with median
num_imputer = SimpleImputer(strategy='median')
X[numerical_columns] = num_imputer.fit_transform(X[numerical_columns])
test[numerical_columns] = num_imputer.transform(test[numerical_columns])

# Impute categorical columns with mode
cat_imputer = SimpleImputer(strategy='most_frequent')
X[categorical_columns] = cat_imputer.fit_transform(X[categorical_columns])
test[categorical_columns] = cat_imputer.transform(test[categorical_columns])

# =====================
# Feature Engineering
# =====================
print("Performing feature engineering...")

# Deviation from normal rectal temperature
X['deviation_from_normal_temp'] = X['rectal_temp'].apply(lambda x: abs(x - 37.8))
test['deviation_from_normal_temp'] = test['rectal_temp'].apply(lambda x: abs(x - 37.8))

# Pulse categories
X['pulse_category'] = pd.cut(X['pulse'], bins=[0, 40, 80, 120, float('inf')],
                             labels=['critical', 'normal', 'elevated', 'high'])
test['pulse_category'] = pd.cut(test['pulse'], bins=[0, 40, 80, 120, float('inf')],
                                labels=['critical', 'normal', 'elevated', 'high'])

# Total protein categories
X['total_protein_category'] = pd.cut(X['total_protein'], bins=[0, 5, 7, 9, float('inf')],
                                     labels=['low', 'normal', 'elevated', 'high'])
test['total_protein_category'] = pd.cut(test['total_protein'], bins=[0, 5, 7, 9, float('inf')],
                                        labels=['low', 'normal', 'elevated', 'high'])

# Packed cell volume categories
X['packed_cell_volume_category'] = pd.cut(X['packed_cell_volume'], bins=[0, 30, 50, float('inf')],
                                          labels=['low', 'normal', 'high'])
test['packed_cell_volume_category'] = pd.cut(test['packed_cell_volume'], bins=[0, 30, 50, float('inf')],
                                             labels=['low', 'normal', 'high'])

# Interaction: pulse * respiratory_rate
X['pulse_respiratory_interaction'] = X['pulse'] * X['respiratory_rate']
test['pulse_respiratory_interaction'] = test['pulse'] * test['respiratory_rate']

# Drop lesion_3 if it exists
if 'lesion_3' in X.columns:
    X = X.drop(columns=['lesion_3'])
if 'lesion_3' in test.columns:
    test = test.drop(columns=['lesion_3'])

# Convert categorical features to string
categorical_features = ['hospital_number', 'pulse_category',
                        'total_protein_category', 'packed_cell_volume_category']
for col in categorical_features:
    X[col] = X[col].astype(str)
    test[col] = test[col].astype(str)

# =====================
# Encode categorical features
# =====================
categorical_columns = X.select_dtypes(include=['object']).columns
oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
all_values = pd.concat([X[categorical_columns], test[categorical_columns]], axis=0)
oe.fit(all_values)
X[categorical_columns] = oe.transform(X[categorical_columns])
test[categorical_columns] = oe.transform(test[categorical_columns])

# =====================
# Optuna Hyperparameter Tuning
# =====================
def objective(trial):
    params = {
        'objective': 'multiclass',
        'num_class': len(np.unique(y)),
        'metric': 'multi_logloss',
        'boosting_type': 'gbdt',
        'verbosity': -1,
        'learning_rate': trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        'num_leaves': trial.suggest_int("num_leaves", 20, 60),
        'max_depth': trial.suggest_int("max_depth", 3, 8),
        'min_child_samples': trial.suggest_int("min_child_samples", 10, 50),
        'min_child_weight': trial.suggest_float("min_child_weight", 1e-5, 0.1, log=True),
        'subsample': trial.suggest_float("subsample", 0.6, 1.0),
        'colsample_bytree': trial.suggest_float("colsample_bytree", 0.6, 1.0),
        'lambda_l1': trial.suggest_float("lambda_l1", 1e-5, 5.0, log=True),
        'lambda_l2': trial.suggest_float("lambda_l2", 1e-5, 5.0, log=True)
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = []

    for train_idx, val_idx in cv.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = lgb.LGBMClassifier(**params, n_estimators=300)
        model.fit(X_train, y_train,
                  eval_set=[(X_val, y_val)],
                  eval_metric='multi_logloss',
                  callbacks=[lgb.early_stopping(20, verbose=False)])

        preds = model.predict(X_val)
        score = f1_score(y_val, preds, average="micro")
        scores.append(score)

    return np.mean(scores)

print("Tuning LightGBM hyperparameters...")
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=25)
print("Best LightGBM parameters:", study.best_params)

# =====================
# Final Model Training with Best Params
# =====================
best_params = study.best_params
best_params.update({
    'objective': 'multiclass',
    'num_class': len(np.unique(y)),
    'metric': 'multi_logloss',
    'boosting_type': 'gbdt',
    'verbosity': -1,
    'n_estimators': 300
})

print("Performing cross-validation...")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = []
feature_importance = pd.DataFrame(index=X.columns)

for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    model = lgb.LGBMClassifier(**best_params)
    model.fit(X_train, y_train,
              eval_set=[(X_val, y_val)],
              eval_metric='multi_logloss',
              callbacks=[lgb.early_stopping(20, verbose=False)])

    preds = model.predict(X_val)
    score = f1_score(y_val, preds, average="micro")
    scores.append(score)

    print(f"Fold {fold + 1} Micro-F1 Score: {score:.4f}")

    feature_importance[f"fold_{fold+1}"] = model.feature_importances_

print(f"\n✅ Mean CV Micro-F1 Score: {np.mean(scores):.4f}")

# Feature importance summary
feature_importance['avg'] = feature_importance.mean(axis=1)
print("\nTop 5 features by importance:")
print(feature_importance['avg'].sort_values(ascending=False).head())

# =====================
# Train final model on all data
# =====================
print("Training final model...")
model = lgb.LGBMClassifier(**best_params)
model.fit(X, y)

# Ensure test data columns match
test = test[X.columns]

# Predict on test data
print("Generating predictions...")
test_preds = model.predict(test)
test_preds = le.inverse_transform(test_preds)

# Prepare submission
submission = pd.DataFrame({
    'id': test_ids,
    'outcome': test_preds
})

# Save submission
submission_path = os.path.join(base_path, "submission.csv")
submission.to_csv(submission_path, index=False)
print(f"✅ Submission file 'submission.csv' created successfully at: {submission_path}")
# =====================
# Save Feature Importance Plot
# =====================
import matplotlib.pyplot as plt
import numpy as np

importance = model.feature_importances_
feature_names = X.columns

plt.figure(figsize=(10, 8))
indices = np.argsort(importance)[::-1]
plt.barh(np.array(feature_names)[indices], importance[indices])
plt.xlabel("Feature Importance")
plt.ylabel("Feature Name")
plt.title("LightGBM Feature Importance")
plt.gca().invert_yaxis()
plt.tight_layout()

output_path = os.path.join(base_path, "feature_importance.png")
plt.savefig(output_path, dpi=300)
print(f"Feature importance saved at: {output_path}")

'''
Fold 1 Micro-F1 Score: 0.7818
Fold 2 Micro-F1 Score: 0.7068
Fold 3 Micro-F1 Score: 0.7394
Fold 4 Micro-F1 Score: 0.7752
Fold 5 Micro-F1 Score: 0.7647

✅ Mean CV Micro-F1 Score: 0.7536

Top 5 features by importance:
lesion_1              701.4
hospital_number       669.8
total_protein         459.8
packed_cell_volume    457.4
pulse                 413.4
Name: avg, dtype: float64
'''