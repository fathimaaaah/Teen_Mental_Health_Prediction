import streamlit as st
import joblib
import numpy as np

model = joblib.load("teen_mental_health_model (1).joblib")

st.set_page_config(
    page_title="Teen Mental Health Prediction",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 Teen Mental Health Prediction System")
st.markdown("Predict the risk of depression based on lifestyle and mental health factors.")

age = st.number_input("Age", 10, 20, 16)

gender = st.selectbox("Gender", ["Male", "Female"])
gender = 1 if gender == "Male" else 0

daily_social_media_hours = st.slider(
    "Daily Social Media Hours", 0.0, 15.0, 4.0
)

platform_usage = st.selectbox(
    "Platform Usage Level", [0, 1, 2, 3, 4, 5]
)

sleep_hours = st.slider(
    "Sleep Hours", 0.0, 12.0, 7.0
)

screen_time_before_sleep = st.slider(
    "Screen Time Before Sleep", 0.0, 5.0, 1.0
)

academic_performance = st.slider(
    "Academic Performance", 0.0, 10.0, 5.0
)

physical_activity = st.slider(
    "Physical Activity", 0.0, 10.0, 5.0
)

social_interaction_level = st.slider(
    "Social Interaction Level", 0.0, 10.0, 5.0
)

stress_level = st.slider(
    "Stress Level", 0, 10, 5
)

anxiety_level = st.slider(
    "Anxiety Level", 0, 10, 5
)

addiction_level = st.slider(
    "Addiction Level", 0, 10, 5
)

mental_health_risk_score = st.slider(
    "Mental Health Risk Score", 0, 100, 50
)

sleep_quality = st.selectbox(
    "Sleep Quality", [0, 1, 2]
)

digital_wellbeing_flag = st.selectbox(
    "Digital Wellbeing", [0, 1]
)

if st.button("🔍 Predict"):

    features = np.array([[
        age,
        gender,
        daily_social_media_hours,
        platform_usage,
        sleep_hours,
        screen_time_before_sleep,
        academic_performance,
        physical_activity,
        social_interaction_level,
        stress_level,
        anxiety_level,
        addiction_level,
        mental_health_risk_score,
        sleep_quality,
        digital_wellbeing_flag
    ]])

    prediction = model.predict(features)[0]

probability = model.predict_proba(features)[0][1] * 100

if probability >= 70:
    st.error(f"🔴 High Risk of Depression ({probability:.2f}%)")
elif probability >= 40:
    st.warning(f"🟠 Moderate Risk of Depression ({probability:.2f}%)")
else:
    st.success(f"🟢 Low Risk of Depression ({probability:.2f}%)")