import streamlit as st
import joblib
import boto3
import os
import tempfile
import pandas as pd
import tarfile

# S3 info
s3_bucket = 'sagemaker-eu-north-1-902376761557'
s3_key = 'logreg-sklearn-2025-05-05-16-03-22-929/output/model.tar.gz'

@st.cache_resource
def load_model_from_s3():
    # Create temp directory
    with tempfile.TemporaryDirectory() as tmpdir:
        local_tar_path = os.path.join(tmpdir, 'model.tar.gz')
        extract_dir = os.path.join(tmpdir, 'model')
        
        os.makedirs(extract_dir, exist_ok=True)
        
        # Create S3 client and download the tar.gz
        s3 = boto3.client('s3')
        s3.download_file(s3_bucket, s3_key, local_tar_path)

        # Extract the tar.gz file
        with tarfile.open(local_tar_path, 'r:gz') as tar:
            tar.extractall(path=extract_dir)

        # Find model file (assumes 'model.joblib' inside)
        model_path = os.path.join(extract_dir, 'model.joblib')
        model = joblib.load(model_path)
    
    return model

# Load model
model = load_model_from_s3()

# Streamlit UI
st.title("Heart Disease Prediction App")
st.write("Enter the patient's data to predict the likelihood of heart disease.")

# Input fields
age = st.number_input("Age", 20, 100, 50)
sex = st.selectbox("Sex", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")

cp = st.selectbox("Chest Pain Type", [1, 2, 3, 4], format_func=lambda x: {
    1: "Typical Angina", 
    2: "Atypical Angina", 
    3: "Non-anginal Pain", 
    4: "Asymptomatic"
}[x])

trestbps = st.number_input("Resting Blood Pressure", 80, 200, 120)
chol = st.number_input("Cholesterol (mg/dl)", 100, 400, 200)

fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")

restecg = st.selectbox("Resting ECG", [0, 1, 2], format_func=lambda x: {
    0: "Normal", 
    1: "ST-T Wave Abnormality", 
    2: "Left Ventricular Hypertrophy"
}[x])

thalach = st.number_input("Max Heart Rate Achieved", 60, 250, 150)
exang = st.selectbox("Exercise Induced Angina", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")

oldpeak = st.slider("ST Depression (Oldpeak)", 0.0, 6.0, 1.0)

slope = st.selectbox("Slope of Peak Exercise ST Segment", [1, 2, 3], format_func=lambda x: {
    1: "Upsloping", 
    2: "Flat", 
    3: "Downsloping"
}[x])

ca = st.selectbox("Number of Major Vessels (0–3) Colored by Fluoroscopy", [0, 1, 2, 3])

thal = st.selectbox("Thalassemia", [3, 6, 7], format_func=lambda x: {
    3: "Normal", 
    6: "Fixed Defect", 
    7: "Reversible Defect"
}[x])



# Predict
if st.button("Predict"):
    input_data = pd.DataFrame([[age, sex, cp, trestbps, chol, fbs, restecg,
                                thalach, exang, oldpeak, slope, ca, thal]],
                              columns=["age", "sex", "cp", "trestbps", "chol", "fbs",
                                       "restecg", "thalach", "exang", "oldpeak", "slope",
                                       "ca", "thal"])
    prediction = model.predict(input_data)[0]
    st.success(f"Prediction: {'Heart Disease Detected' if prediction == 1 else 'No Heart Disease'}")

