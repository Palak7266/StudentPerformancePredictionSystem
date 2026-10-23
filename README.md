# Student Performance Prediction System

An intermediate-level AIML mini project that estimates student exam marks from study habits and learning indicators using a Random Forest regression model and a Streamlit web app.

## Features

- Interactive web form for student learning indicators
- Random Forest regression model
- Train/test split and evaluation using MAE, RMSE and R²
- Estimated marks and a simple performance category
- Easy-to-run Windows/VS Code setup

## Tech stack

- Python
- Pandas
- Scikit-learn
- Streamlit
- Joblib

## Project structure

```text
StudentPerformanceAI/
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
├── data/
│   └── student_data.csv
└── models/
    └── (model file generated after training)
```

## Run on Windows

Use Python 3.11 or 3.12 for a straightforward setup.

1. Open this folder in VS Code.
2. Open **Terminal → New Terminal**.
3. Create and activate a virtual environment:

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

   If PowerShell blocks activation, use Command Prompt terminal and run:

   ```bat
   .venv\Scripts\activate.bat
   ```

4. Install packages:

   ```bash
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

5. Train the model:

   ```bash
   python train_model.py
   ```

6. Start the dashboard:

   ```bash
   streamlit run app.py
   ```

7. Open the local URL printed in the terminal (usually `http://localhost:8501`).

## Dataset

`data/student_data.csv` contains 500 **synthetic sample records** created for demonstration and practice. Columns:

| Column | Meaning |
|---|---|
| `study_hours` | Study hours per day |
| `attendance` | Attendance percentage |
| `previous_marks` | Previous exam marks |
| `assignment_completion` | Percentage of assignments completed |
| `sleep_hours` | Sleep hours per night |
| `extracurricular` | 0 = No, 1 = Yes |
| `final_marks` | Target exam marks from 0 to 100 |

This is not a real institutional dataset. Model scores measured on this generated data do not establish real-world accuracy. For a serious project, replace it with a reliable, permissioned dataset and document its source.

## How it works

1. Load the CSV with Pandas.
2. Use six input features and `final_marks` as the target.
3. Split records into training (80%) and test (20%) sets.
4. Train a `RandomForestRegressor`.
5. Evaluate held-out predictions with MAE, RMSE and R².
6. Save the trained model to `models/student_performance_model.joblib`.
7. Streamlit loads the model, accepts form inputs and displays an estimated score.

## Model metrics

When you run `python train_model.py`, the terminal prints MAE, RMSE and R² for the test set. Record the actual output in your project report; do not claim a fixed accuracy before running it.

- **MAE**: average absolute difference between predicted and actual marks.
- **RMSE**: penalizes larger prediction errors more strongly.
- **R²**: compares the model with a baseline that predicts the mean target value.

## Limitations and responsible use

- The included dataset is synthetic and uses simplified relationships.
- Predictions can be biased or wrong and should not be used for admissions, grading, discipline, or other high-stakes decisions.
- Sleep, attendance, and study habits are sensitive personal context; collect only what is needed and get permission when using real student data.
- Performance category thresholds are simple demo rules, not official academic standards.

## Suggested future improvements

- Add data visualizations and feature importance
- Compare Linear Regression, Decision Tree and Random Forest
- Add input validation and automated tests
- Use a real, permissioned dataset and evaluate on a representative test set

## Resume description

**Student Performance Prediction System | Python, Scikit-learn, Pandas, Streamlit**  
Built an interactive ML application that estimates student exam marks from learning indicators. Implemented a Random Forest regression pipeline, train/test evaluation using MAE/RMSE/R², model serialization with Joblib, and a Streamlit interface for predictions.

## License

For learning and portfolio use. Verify dataset rights before redistributing any replacement data.
