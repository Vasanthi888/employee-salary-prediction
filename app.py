import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load("models/salary_model.pkl")

st.set_page_config(page_title="PayPredict", layout="centered")
st.title("💼 PayPredict: Employee Salary Estimator")
st.write("Fill in the details to predict the employee's estimated salary.")

# Input fields
experience = st.slider("Years of Experience", 0, 30, 1)
education = st.selectbox("Education Level", ["High School", "Bachelors", "Masters", "PhD"])
job_role = st.selectbox("Job Role", ["Software Engineer", "Data Analyst", "Manager", "HR Specialist", "System Administrator", "UX Designer"])
company_size = st.radio("Company Size", ["S", "M", "L"])
location = st.selectbox("Location", ["US", "IN", "CA", "UK", "DE", "AU"])

# Create input DataFrame
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
