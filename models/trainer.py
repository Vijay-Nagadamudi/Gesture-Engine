import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

DATASET_PATH = "data/gesture_dataset.csv"
MODEL_DIR = "models/trained"

def load_dataset():
    data = pd.read_csv(DATASET_PATH)
    X = data.drop("label", axis = 1)
    y = data["label"]
    
    return X, y

def train_models(X_train, y_train):
    svm_model = SVC(
        kernel = "rbf",
        probability = True,
        random_state = 42
    )
    
    decision_tree_model = DecisionTreeClassifier(random_state = 42)
    
    svm_model.fit(X_train, y_train)
    decision_tree_model.fit(X_train, y_train)
    
    return svm_model, decision_tree_model

def evaluate_model(model, X_test, y_test, model_name):
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"\n {'=' * 50}")
    print(f"{model_name}")
    print(f"{'=' * 50}")
    print(f"Accuracy : {accuracy:.4f}]\n")
    print(
        classification_report(
        y_test,
        predictions
        )
    )
    return accuracy

def save_model(model, filename):
    os.makedirs(MODEL_DIR, exist_ok = True)
    path = os.path.join(MODEL_DIR, filename)
    joblib.dump(model,path)
    print(f"Saved Model : {path}")
    
def main():
    print("Loading Dataset....")
    
    X, y = load_dataset()
    
    print(f"Total Samples : {len(X)}")
    print(f"Total features : {X.shape[1]}")
    print(f"Total classes : {y.nunique()}")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size = 0.2,
        random_state = 42,
        stratify = y
    )
    
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")

    print("\nTraining models...")
    
    svm_model, decision_tree_model = train_models(
        X_train,
        y_train
    )
    
    svm_accuracy = evaluate_model(
        svm_model,
        X_test,
        y_test,
        "SVM"
    )
    
    tree_accuracy = evaluate_model(
        decision_tree_model,
        X_test,
        y_test,
        "Decision Tree"
    )
    
    save_model(
        svm_model,
        "svm_model.joblib"
    )
    
    save_model(
        decision_tree_model,
        "decision_tree_model.joblib"
    )
    
    print("\nTraining completed!")
    
    if svm_accuracy > tree_accuracy:
        print("Better model: SVM")
    elif tree_accuracy > svm_accuracy:
        print("Better model: Decision Tree")
    else:
        print("Both models have the same accuracy.")


if __name__ == "__main__":
    main()

