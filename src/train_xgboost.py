import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

from config import (
    X_CSV_FILE, Y_CSV_FILE, COLS_TO_ENCODE, TEST_SIZE, RANDOM_STATE
)

def train_xgboost_model():

    try:
        X = pd.read_csv(X_CSV_FILE)
        y = pd.read_csv(Y_CSV_FILE)
    except FileNotFoundError:
        print("Error:")
        return

    # Using COLS_TO_ENCODE from config
    cols_to_encode = [col for col in COLS_TO_ENCODE if col in X.columns]
    
    X_encoded = pd.get_dummies(X, columns=cols_to_encode)

    if isinstance(y, pd.DataFrame):
        y = y.iloc[:, 0]  # Assuming the target is the first/only column
    

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    num_classes = len(set(y_encoded))
    objective = 'binary:logistic' if num_classes == 2 else 'multi:softmax'
    eval_metric = 'logloss' if num_classes == 2 else 'mlogloss'
   ###
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y_encoded, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    model = xgb.XGBClassifier(
        objective=objective,
        eval_metric=eval_metric,
        random_state=RANDOM_STATE
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")

    m_target_names = [str(cls) for cls in le.classes_]
    print(classification_report(y_test, y_pred, target_names=m_target_names))

if __name__ == "__main__":
    train_xgboost_model()
