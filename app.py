import streamlit as st
import pandas as pd
import joblib

# Load the model
model = joblib.load("models/salary_model.pkl")

st.set_page_config(page_title="Employee Salary Predictor", layout="centered")

st.title("💼 Employee Salary Prediction")
st.write("Fill in the details below to predict the estimated salary.")

# User input fields
experience = st.slider("Years of Experience", 0, 30, 1)
education = st.selectbox("Education Level", ["High School", "Bachelors", "Masters", "PhD"])
job_role = st.selectbox("Job Role", ["Software Engineer", "Data Analyst", "Manager", "HR Specialist", "System Administrator", "UX Designer"])
company_size = st.radio("Company Size", ["S", "M", "L"])
location = st.selectbox("Location", ["US", "IN", "CA", "UK", "DE", "AU"])

# Prepare input as DataFrame
input_df = pd.DataFrame({
    "Experience": [experience],
    "Education Level": [education],
    "Job Role": [job_role],
    "Company Size": [company_size],
    "Location": [location]
})

# Predict salary
if st.button("Predict Salary"):
    salary = model.predict(input_df)[0]
    st.success(f"💰 Estimated Salary: **${int(salary):,} USD**")
