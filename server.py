from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware  
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI(
    title="Teen Mental Health Prediction API",
    description="AI-powered depression risk prediction system for teenagers",
    version="1.0.0"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = joblib.load("teen_mental_health_model (1).joblib")


class TeenInput(BaseModel):
    age: int
    gender: int
    daily_social_media_hours: float
    platform_usage: int
    sleep_hours: float
    screen_time_before_sleep: float
    academic_performance: float
    physical_activity: float
    social_interaction_level: float
    stress_level: int
    anxiety_level: int
    addiction_level: int
    mental_health_risk_score: float
    sleep_quality: int
    digital_wellbeing_flag: int


@app.get("/")
def home():
    return {
        "project": "Teen Mental Health Prediction",
        "status": "Running",
        "model": "Random Forest Classifier"
    }


@app.post("/predict")
def predict(data: TeenInput):

    features = pd.DataFrame([{
        "age": data.age,
        "gender": data.gender,
        "daily_social_media_hours": data.daily_social_media_hours,
        "platform_usage": data.platform_usage,
        "sleep_hours": data.sleep_hours,
        "screen_time_before_sleep": data.screen_time_before_sleep,
        "academic_performance": data.academic_performance,
        "physical_activity": data.physical_activity,
        "social_interaction_level": data.social_interaction_level,
        "stress_level": data.stress_level,
        "anxiety_level": data.anxiety_level,
        "addiction_level": data.addiction_level,
        "mental_health_risk_score": data.mental_health_risk_score,
        "sleep_quality": data.sleep_quality,
        "digital_wellbeing_flag": data.digital_wellbeing_flag
    }])

    prediction = model.predict(features)[0]

    probability = 0

    if hasattr(model, "predict_proba"):
        probability = float(model.predict_proba(features)[0][1]) * 100

    risk_level = (
        "Low Risk"
        if probability < 30
        else "Moderate Risk"
        if probability < 70
        else "High Risk"
    )
    if probability < 30:

        recommendation = "Maintain healthy habits and regular sleep."

    elif probability < 70:

        recommendation = "Monitor stress levels and improve lifestyle balance."

    else:

        recommendation = "Consider speaking with a counselor or mental health professional."

    return {

        "prediction": int(prediction),

        "depression_probability": f"{probability:.2f}%",

        "risk_level": risk_level,

        "recommendation": recommendation

    }
