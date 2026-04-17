import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

def train_decision_tree_model():
    try:
        X = pd.read_csv('x.csv')
        y = pd.read_csv('y.csv')
    except FileNotFoundError:
        print("Error")
        return

    cols_to_encode = [
        'WHYTO', 'WHYFROM', 'URBRUR', 'URBAN', 'HBHUR', 'CENSUS_R', 
        'R_SEX', 'EDUC', 'WORKER', 'LIF_CYC', 'MEDCOND', 'CONDRIVE', 
        'W_CHAIR', 'W_NONE', 'DRIVER', 'TRPHHVEH', 'TDWKND', 'LOOP_TRIP'
    ]

    cols_to_encode = [col for col in cols_to_encode if col in X.columns]
    X_encoded = pd.get_dummies(X, columns=cols_to_encode)

    if isinstance(y, pd.DataFrame):
        y = y.iloc[:, 0]
    
    # AI generated with data preprocessing part
    counts = y.value_counts()
    valid_classes = counts[counts > 10].index
    valid_mask = y.isin(valid_classes)
    X_encoded = X_encoded[valid_mask]
    y = y[valid_mask]

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    num_classes = len(set(y_encoded))
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    ###

    model = DecisionTreeClassifier(
        max_depth=20, 
        min_samples_split=50, 
        random_state=42
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    
    target_names = [str(cls) for cls in le.classes_]
    print(classification_report(y_test, y_pred, target_names=target_names))

if __name__ == "__main__":
    train_decision_tree_model()

    model = DecisionTreeClassifier(
        max_depth=20, 
        min_samples_split=50, 
        random_state=42
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    
    target_names = [str(cls) for cls in le.classes_]
    print(classification_report(y_test, y_pred, target_names=target_names))

if __name__ == "__main__":
    train_decision_tree_model()
