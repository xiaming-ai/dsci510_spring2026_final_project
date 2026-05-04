import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression

from config import (
    X_CSV_FILE, Y_CSV_FILE, COLS_TO_ENCODE, MIN_CLASS_COUNT, 
    TEST_SIZE, RANDOM_STATE, LOGISTIC_MAX_ITER
)

def train_logistic_regression():
    try:
        X = pd.read_csv(X_CSV_FILE)
        y = pd.read_csv(Y_CSV_FILE)
    except FileNotFoundError:
        print("Error")
        return

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
    ###

    model = LogisticRegression(max_iter=LOGISTIC_MAX_ITER, random_state=RANDOM_STATE, n_jobs=-1)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    
    target_names = [str(cls) for cls in le.classes_]
    print(classification_report(y_test, y_pred, target_names=target_names))

if __name__ == "__main__":
    train_logistic_regression()
