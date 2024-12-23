import optuna
import psycopg2
from sklearn.datasets import fetch_california_housing, load_breast_cancer
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, accuracy_score
import matplotlib.pyplot as plt
import numpy as np

# Подключение к PostgreSQL
def connect_to_db():
    return psycopg2.connect(
        dbname="optuna",
        user="postgres",  # Убедитесь, что нет лишнего символа ":"
        password="248321",
        host="localhost",
        port="5432"
    )

# Загрузка данных
def load_data(dataset_name="california"):
    if dataset_name == "california":
        data = fetch_california_housing()  # Замена на California Housing Dataset
        X, y = data.data, data.target
        model = RandomForestRegressor(random_state=42)
        metric = mean_squared_error
    elif dataset_name == "cancer":
        data = load_breast_cancer()
        X, y = data.data, data.target
        model = RandomForestClassifier(random_state=42)
        metric = accuracy_score
    else:
        raise ValueError("Unsupported dataset")
    return X, y, model, metric

# Целевая функция для оптимизации
def objective(trial):
    n_estimators = trial.suggest_int("n_estimators", 50, 500)
    max_depth = trial.suggest_int("max_depth", 3, 15)
    min_samples_split = trial.suggest_int("min_samples_split", 2, 10)
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 10)

    model.set_params(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf
    )

    scores = cross_val_score(model, X, y, cv=5, scoring="neg_mean_squared_error" if metric == mean_squared_error else "accuracy")
    return scores.mean()

# Визуализация результатов
def plot_optimization_history(study):
    plt.figure(figsize=(10, 6))
    plt.plot(study.trials_dataframe()["value"], label="Objective Value")
    plt.xlabel("Trial")
    plt.ylabel("Objective Value")
    plt.title("Optimization History")
    plt.legend()
    plt.show()

def plot_param_importances(study):
    optuna.visualization.plot_param_importances(study).show()

# Основной код
if __name__ == "__main__":
    # Подключение к базе данных
    conn = connect_to_db()
    optuna.storages.RDBStorage(url="postgresql://postgres:248321@localhost:5432/optuna")

    # Загрузка данных
    X, y, model, metric = load_data(dataset_name="california")  # Замена на California Housing Dataset

    # Создание объекта Study
    study = optuna.create_study(
        storage="postgresql://postgres:248321@localhost:5432/optuna",
        study_name="random_forest_optimization",
        direction="minimize" if metric == mean_squared_error else "maximize"
    )
    # Оптимизация
    study.optimize(objective, n_trials=50)

    # Визуализация
    plot_optimization_history(study)
    plot_param_importances(study)

    # Закрытие соединения с базой данных
    conn.close()