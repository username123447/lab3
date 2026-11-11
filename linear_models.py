from grader_contracts.linear_models import (
    BaselineComparisonResult,
    ClassificationInput,
    ClassificationResult,
    RegressionInput,
    RegressionResult,
    ClassificationMetrics
)
import pandas as pd  # pyright: ignore[reportMissingModuleSource]
from sklearn.model_selection import train_test_split  # pyright: ignore[reportMissingModuleSource]
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import *
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
import json


def train_linear_regression(data: RegressionInput) -> RegressionResult:
    df = pd.read_csv('Student_Performance.csv')
    df['Extracurricular Activities'] = df['Extracurricular Activities'].map({'Yes': 1, 'No': 0})
    y = df['Performance Index']
    X = df.drop(columns=['Performance Index'])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2, 
        random_state=42
        )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    N, D = X_train_scaled.shape
    w = np.zeros(D)
    b = 0.0
    mse_history = []

    for epoch in range(data.epochs):
        y_pred = np.dot(X_train_scaled, w) + b
        mse = np.mean((y_pred - y_train)**2)
        mse_history.append(mse)

        dw = (2 / N) * np.dot(X_train_scaled.T, y_pred - y_train)
        db = (2 / N) * np.sum(y_pred - y_train)

        w = w - data.learning_rate * dw
        b = b - data.learning_rate * db

    test_predictions = np.dot(X_test_scaled, w) + b

    return RegressionResult(
        weights = w,
        bias = b,
        loss_history = mse_history,
        predictions = test_predictions
    )

def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))


def train_logistic_classifier(data: ClassificationInput) -> ClassificationResult:
    df = pd.read_csv('german.csv', sep=';')
    y = df['Creditability']
    X = df.drop(columns=['Creditability'])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    N, D = X_train_scaled.shape
    w = np.zeros(D)
    b = 0.0
    loss_history = []

    for epoch in range(data.epochs):
        z = np.dot(X_train_scaled, w) + b
        y_pred = sigmoid(z)

        epsilon = 1e-15
        y_pred_clipped = np.clip(y_pred, epsilon, 1 - epsilon)
        loss = -np.mean(y_train * np.log(y_pred_clipped) + (1 - y_train) * np.log(1 - y_pred_clipped))
        loss_history.append(loss)

        error = y_pred - y_train
        dw = (1 / N) * np.dot(X_train_scaled.T, error)
        db = (1 / N) * np.sum(error)

        w = w - data.learning_rate * dw
        b = b - data.learning_rate * db

    test_z = np.dot(X_test_scaled, w) + b
    test_probabilities = sigmoid(test_z)
    test_predictions = (test_probabilities >= 0.5).astype(int)

    roc_auc = roc_auc_score(y_test, test_probabilities)
    accuracy = accuracy_score(y_test, test_predictions)
    precision = precision_score(y_test, test_predictions)
    recall = recall_score(y_test, test_predictions)
    f1 = f1_score(y_test, test_predictions)

    print("ROC-AUC:", roc_auc)
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1:", f1)

    return ClassificationResult(
        weights = w,
        bias = b,
        loss_history = loss_history,
        probabilities = test_probabilities,
        predictions = test_predictions
    )

def fit_logistic_regression(X_train, y_train, X_test, data):
    N, D = X_train.shape

    w = np.zeros(D)
    b = 0.0
    loss_history = []

    for epoch in range(data.epochs):
        z = np.dot(X_train, w) + b
        y_pred = sigmoid(z)

        epsilon = 1e-15
        y_pred_clipped = np.clip(y_pred, epsilon, 1 - epsilon)

        loss = -np.mean(
            y_train * np.log(y_pred_clipped)
            + (1 - y_train) * np.log(1 - y_pred_clipped)
        )
        loss_history.append(loss)

        error = y_pred - y_train

        dw = (1 / N) * np.dot(X_train.T, error)
        db = (1 / N) * np.sum(error)

        w = w - data.learning_rate * dw
        b = b - data.learning_rate * db

    test_z = np.dot(X_test, w) + b
    probabilities = sigmoid(test_z)
    predictions = (probabilities >= 0.5).astype(int)

    return ClassificationResult(
        weights=w,
        bias=b,
        loss_history=loss_history,
        probabilities=probabilities,
        predictions=predictions
    )

def compare_classical_models(data: ClassificationInput) -> BaselineComparisonResult:
    df = pd.read_csv('german.csv', sep=';')
    y = df['Creditability']
    X = df.drop(columns=['Creditability'])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    models = {
        "logistic_regression": None,
        "decision_tree": DecisionTreeClassifier(random_state=42),
        "knn": KNeighborsClassifier()
    }

    results = {}

    for name, model in models.items():

        if name == "logistic_regression":
            result = fit_logistic_regression(
                X_train, y_train, X_test, data
            )
            predictions = result.predictions
            probabilities = result.probabilities

        else:
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            probabilities = model.predict_proba(X_test)[:, 1]

        results[name] = ClassificationMetrics(
            roc_auc=roc_auc_score(y_test, probabilities),
            accuracy=accuracy_score(y_test, predictions),
            precision=precision_score(y_test, predictions, zero_division=0),
            recall=recall_score(y_test, predictions, zero_division=0),
            f1=f1_score(y_test, predictions, zero_division=0)
        )

    return BaselineComparisonResult(
        logistic_regression=results["logistic_regression"],
        decision_tree=results["decision_tree"],
        knn=results["knn"]
    )
    
# data = ClassificationInput(
#     train_features=None,
#     train_target=None,
#     test_features=None,
#     test_target=None,
#     learning_rate=0.1,
#     epochs=1000
# )

# result = compare_classical_models(data)

# print("Логистическая регрессия:", result.logistic_regression)
# print("Дерево решений:", result.decision_tree)
# print("KNN:", result.knn)