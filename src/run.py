import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from config import (
    X_CSV_FILE, Y_CSV_FILE, COLS_TO_ENCODE, MIN_CLASS_COUNT, 
    TEST_SIZE, RANDOM_STATE, LOGISTIC_MAX_ITER, DT_MAX_DEPTH, 
    DT_MIN_SAMPLES_SPLIT, MODEL_RESULTS_CSV
)

X = pd.read_csv(X_CSV_FILE)
y = pd.read_csv(Y_CSV_FILE)


# Using COLS_TO_ENCODE from config

cols_to_encode = [col for col in COLS_TO_ENCODE if col in X.columns]
X_encoded = pd.get_dummies(X, columns=cols_to_encode)
if isinstance(y, pd.DataFrame):
    y = y.iloc[:, 0]


counts = y.value_counts()
valid_classes = counts[counts > MIN_CLASS_COUNT].index
valid_mask = y.isin(valid_classes)
X_encoded = X_encoded[valid_mask]
y = y[valid_mask]


le = LabelEncoder()
y_encoded = le.fit_transform(y)
num_classes = len(set(y_encoded))
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y_encoded, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_encoded
)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

accuracies = {}

### model configuration using AI
objective = 'binary:logistic' if num_classes == 2 else 'multi:softmax'
eval_metric = 'logloss' if num_classes == 2 else 'mlogloss'
xgb_model = xgb.XGBClassifier(objective=objective, eval_metric=eval_metric, random_state=RANDOM_STATE)
xgb_model.fit(X_train, y_train)
accuracies['XGBoost'] = accuracy_score(y_test, xgb_model.predict(X_test))


lr_model = LogisticRegression(max_iter=LOGISTIC_MAX_ITER, random_state=RANDOM_STATE, n_jobs=-1)
lr_model.fit(X_train_scaled, y_train)
accuracies['Logistic Regression'] = accuracy_score(y_test, lr_model.predict(X_test_scaled))

dt_model = DecisionTreeClassifier(max_depth=DT_MAX_DEPTH, min_samples_split=DT_MIN_SAMPLES_SPLIT, random_state=RANDOM_STATE)
dt_model.fit(X_train, y_train)
accuracies['Decision Tree'] = accuracy_score(y_test, dt_model.predict(X_test))
###

results_df = pd.DataFrame([accuracies], index=['Accuracy'])

results_df = results_df.applymap(lambda x: f"{x:.4f}")
print(results_df.to_string())

output_csv = MODEL_RESULTS_CSV
results_df.to_csv(output_csv)
print(f"Results successfully saved to {output_csv}")
