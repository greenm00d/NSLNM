import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

# Load the model
model = joblib.load('XGBoost-NSLNM.pkl')

# Define feature options
MLR = {
    0: '≤0.164',
    1: '>0.164'
}

# Define feature options
SII = {
    0: '≤526.3',
    1: '>526.3'
}

# Define feature options
PNI = {
    0: '≤50.7',
    1: '>50.7'
}

# Define feature options
ALP = {
    0: '≤114',
    1: '>114'
}

# Define feature options
BMI = {
    0: '<18.5 (Kg/m²)',
    1: '18.5~23.9 (Kg/m²)',
    2: '24~27.9 (Kg/m²)',
    3: '≥28 (Kg/m²)'
}

# Define feature options
# biochemical = {
# 0: 'no (1)',
# 1: 'yes (2)'
# }

# Define feature names
feature_names = [
    "Age", "PNI", "MLR", "BMI", "SII", "ALP"
]

# Streamlit user interface
st.title("Non-sentinel lymph node metastasis Predictor")

# age: numerical input
Age = st.number_input("Age:", min_value=1, max_value=120, value=30)

# age: numerical input
# infertility_time = st.number_input("infertility_time:", min_value=1, max_value=120, value=50)

# age: numerical input
# menarche_age = st.number_input("menarche_age:", min_value=1, max_value=120, value=50)

# age: numerical input
# AMH = st.number_input("AMH:", min_value=1, max_value=120, value=50)

# age: numerical input
# gn_dose = st.number_input("gn_dose:", min_value=1, max_value=120, value=50)

# age: numerical input
# gn_days = st.number_input("gn_days:", min_value=1, max_value=120, value=50)

# age: numerical input
# oocyte = st.number_input("oocyte:", min_value=1, max_value=120, value=50)

# cp: categorical selection
PNI = st.selectbox("PNI:", options=list(PNI.keys()), format_func=lambda x: PNI[x])

# cp: categorical selection
MLR = st.selectbox("MLR:", options=list(MLR.keys()), format_func=lambda x: MLR[x])

# cp: categorical selection
BMI = st.selectbox("BMI:", options=list(BMI.keys()), format_func=lambda x: BMI[x])

# cp: categorical selection
SII = st.selectbox("SII:", options=list(SII.keys()), format_func=lambda x: SII[x])

# cp: categorical selection
ALP = st.selectbox("ALP:", options=list(ALP.keys()), format_func=lambda x: ALP[x])

# Process inputs and make predictions
feature_values = [Age, PNI, MLR, BMI, SII, ALP]
features = np.array([feature_values])

if st.button("Predict"):
    # Predict class and probabilities
    predicted_class = model.predict(features)[0]
    predicted_proba = model.predict_proba(features)[0]

    # Display prediction results
    st.write(f"**Predicted Class:** {predicted_class}")
    st.write(f"**Prediction Probabilities:** {predicted_proba}")

    # Generate advice based on prediction results
    probability = predicted_proba[predicted_class] * 100

    if predicted_class == 1:
        advice = (
            f"According to our model, you have a high risk of Non-sentinel lymph node metastasis. "
            f"The model predicts that your probability of having NSLNM is {probability:.1f}%. "
            "While this is just an estimate, it suggests that you may be at significant risk. "
            "I recommend that you consult a breast surgeon as soon as possible for further evaluation and "
            "the axillary lymph node dissection should be considered."
        )
    else:
        advice = (
            f"According to our model, you have a low risk of Non-sentinel lymph node metastasis. "
            f"The model predicts that your probability of not having NSLNM is {probability:.1f}%. "
            "I recommend that a breast conservation surgery is more suitable if the conditions are met"
        )
    st.write(advice)

    # Calculate SHAP values and display force plot
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(pd.DataFrame([feature_values], columns=feature_names))

    shap.force_plot(explainer.expected_value, shap_values[0], pd.DataFrame([feature_values], columns=feature_names),
                    matplotlib=True)
    plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=1200)

    st.image("shap_force_plot.png")
