import streamlit as st
import joblib

# Load saved components

model = joblib.load("../models/fake_job_rf_smote_model.pkl")
vectorizer = joblib.load("../models/tfidf_vectorizer.pkl")
selector = joblib.load("../models/feature_selector.pkl")

st.title("Fake Job Detection System")

st.write("Fill the job details below to check if the job posting is real or fake.")

# Form fields
title = st.text_input("Job Title")

company_profile = st.text_area("Company Profile")

description = st.text_area("Job Description")

requirements = st.text_area("Job Requirements")

benefits = st.text_area("Benefits")

# Button
if st.button("Check Job Authenticity"):

    text = title + " " + company_profile + " " + description + " " + requirements + " " + benefits

    X = vectorizer.transform([text])
    X = selector.transform(X)

    prediction = model.predict(X)

    if prediction[0] == 1:
        st.error("⚠️ This job posting is likely FAKE.")
    else:
        st.success("✅ This job posting appears to be REAL.")