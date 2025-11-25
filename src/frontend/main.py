import streamlit as st
import requests
import json

st.write("""
# Depression Classification App  
This app can be used as an auxiliary tool to see if a students has Depression symptoms
""")

st.sidebar.header('User Input Parameters')

def user_input_features():
    Gender = st.sidebar.selectbox("Gender (0 = Female, 1 = Male)", [0, 1])
    Age = st.sidebar.number_input("Age", min_value=10, max_value=100, value=20)
    AcademicPressure = st.sidebar.slider("Academic Pressure (0-5)", 0.0, 5.0, 3.0)
    StudySatisfaction = st.sidebar.slider("Study Satisfaction (0-5)", 0.0, 5.0, 3.0)
    SleepDuration = st.sidebar.number_input("Sleep Duration (<5 hours = 0, 5-6 hours = 1, 7-8 hours = 2, >8 hours = 3)", [0, 1, 2, 3])
    DietaryHabits = st.sidebar.selectbox("Dietary Habits (Healthy = 0, Unhealthy = 1, Moderate = 2)", [0, 1, 2])
    Degree = st.sidebar.selectbox("Degree (Secondary = 0, Undergraduate = 1, Postgraduate = 2, Doctorate = 3)", [0, 1, 2, 3])
    City = st.sidebar.text_input("City", "Toronto")
    FamilyHistory = st.sidebar.selectbox("Family History of Mental Illness (0 = No, 1 = Yes)", [0, 1])
    SuicidalThoughts = st.sidebar.selectbox("Suicidal Thoughts (0 = No, 1 = Yes)", [0, 1])

    input_dict = {
        "Gender": Gender,
        "Age": Age,
        "AcademicPressure": AcademicPressure,
        "StudySatisfaction": StudySatisfaction,
        "SleepDuration": SleepDuration,
        "DietaryHabits": DietaryHabits,
        "Degree": Degree,
        "City": City,
        "FamilyHistory": FamilyHistory,
        "SuicidalThoughts": SuicidalThoughts
    }

    return input_dict


input_dict = user_input_features()

API_URL = "http://127.0.0.1:8000/predict" 

if st.button("Predict"):
    response = requests.post(
        url=API_URL,
        data=json.dumps(input_dict)
    )

    if response.status_code == 200:
        pred = response.json()["prediction"]
        label = " ¡Yay! No Depression" if pred == 0 else "... Depression Likely"
        st.success(f"Prediction: **{label}**")

    else:
        st.error("Error calling the API. Check the server logs.")
