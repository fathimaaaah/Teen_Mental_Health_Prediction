import requests

url = "http://127.0.0.1:8000/predict"

payload = {
    "age": 16,
    "gender": 1,
    "daily_social_media_hours": 5,
    "platform_usage": 3,
    "sleep_hours": 6,
    "screen_time_before_sleep": 2,
    "academic_performance": 7,
    "physical_activity": 5,
    "social_interaction_level": 6,
    "stress_level": 8,
    "anxiety_level": 7,
    "addiction_level": 6,
    "mental_health_risk_score": 70,
    "sleep_quality": 3,
    "digital_wellbeing_flag": 0
}

response = requests.post(url, json=payload)

print("Status Code:", response.status_code)
print("Response:", response.json())