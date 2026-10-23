from pathlib import Path
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "student_data.csv"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)
MODEL_PATH = MODEL_DIR / "student_performance_model.joblib"

FEATURES = [
    "study_hours",
    "attendance",
    "previous_marks",
    "assignment_completion",
    "sleep_hours",
    "extracurricular",
]
TARGET = "final_marks"

def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}. Make sure data/student_data.csv exists."
        )

    data = pd.read_csv(DATA_PATH)
    required = FEATURES + [TARGET]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    data = data[required].dropna()
    X = data[FEATURES]
    y = data[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=250,
        max_depth=12,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    rmse = mean_squared_error(y_test, predictions) ** 0.5
    r2 = r2_score(y_test, predictions)

    joblib.dump({"model": model, "features": FEATURES}, MODEL_PATH)

    print("\nModel training completed!")
    print(f"Rows used: {len(data)}")
    print(f"Model saved to: {MODEL_PATH}")
    print("\nEvaluation on held-out test data:")
    print(f"MAE : {mae:.2f} marks")
    print(f"RMSE: {rmse:.2f} marks")
    print(f"R²  : {r2:.3f}")
    print("\nNote: this project uses synthetic sample data for demonstration.")

if __name__ == "__main__":
    main()
