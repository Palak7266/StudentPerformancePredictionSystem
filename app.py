from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "student_performance_model.joblib"

st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="centered",
)

st.title("🎓 Student Performance Prediction System")
st.write(
    "Enter a student's study and learning habits to estimate their exam marks "
    "using a machine-learning model."
)

st.info(
    "Educational demo only: the included dataset is synthetic. Predictions are "
    "estimates, not guaranteed exam results or a substitute for teacher guidance."
)

if not MODEL_PATH.exists():
    st.warning("The trained model was not found. First run: python train_model.py")
    st.stop()

bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
features = bundle["features"]

with st.form("student_form"):
    st.subheader("Student details")
    col1, col2 = st.columns(2)

    with col1:
        study_hours = st.slider("Study hours per day", 0.0, 12.0, 3.0, 0.5)
        attendance = st.slider("Attendance (%)", 0, 100, 80)
        previous_marks = st.slider("Previous exam marks (%)", 0, 100, 65)

    with col2:
        assignment_completion = st.slider("Assignments completed (%)", 0, 100, 75)
        sleep_hours = st.slider("Sleep hours per night", 0.0, 12.0, 7.0, 0.5)
        extracurricular = st.selectbox(
            "Participates in extracurricular activities?",
            options=[("No", 0), ("Yes", 1)],
            format_func=lambda item: item[0],
        )[1]

    submitted = st.form_submit_button("Predict performance", type="primary")

if submitted:
    input_data = pd.DataFrame([{
        "study_hours": study_hours,
        "attendance": attendance,
        "previous_marks": previous_marks,
        "assignment_completion": assignment_completion,
        "sleep_hours": sleep_hours,
        "extracurricular": extracurricular,
    }])[features]

    prediction = float(model.predict(input_data)[0])
    prediction = max(0.0, min(100.0, prediction))

    if prediction >= 75:
        category = "High"
        message = "Keep up consistent study habits and revise regularly."
    elif prediction >= 50:
        category = "Moderate"
        message = "A steady revision plan and completed assignments may help."
    else:
        category = "Needs support"
        message = "Consider a study timetable and ask a teacher for targeted support."

    st.subheader("Prediction result")
    col1, col2 = st.columns(2)
    col1.metric("Estimated exam marks", f"{prediction:.1f}/100")
    col2.metric("Performance category", category)

    st.progress(int(round(prediction)))
    st.write(message)

    with st.expander("See the values sent to the model"):
        st.dataframe(input_data, use_container_width=True)

st.divider()
st.caption("Built with Python, Pandas, Scikit-learn, Joblib and Streamlit.")
