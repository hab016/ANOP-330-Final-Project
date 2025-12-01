import pickle
import streamlit as st
import numpy as np

st.markdown(
    """
    <style>
    body {
        background-color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)


with open("model.pkl", "rb") as file:
    model = pickle.load(file)

st.title("Bucknell University Reunion Prediction App")


parent_current = st.radio("Are you a current parent?", ["No", "Yes"])
greek = st.radio("Were you involved in greek life?", ["No", "Yes"])
bucknell_staff = st.radio("Are you a Bucknell staff member?", ["No", "Yes"])


parent_current = 1 if parent_current == "Yes" else 0
greek = 1 if greek == "Yes" else 0
bucknell_staff = 1 if bucknell_staff == "Yes" else 0

counts_student_activities = st.slider(
    "How many student activities did you participate in?",
    min_value=0,
    max_value=10,       
    step=1
)

reunion_years_out = st.slider(
    "How many years ago did you graduate?",
    min_value=0,
    max_value=75,      
    step=1
)

input_data = np.array([[
    parent_current,
    greek,
    bucknell_staff,
    counts_student_activities,
    reunion_years_out
]])

if st.button("Predict"):

    # Get probability of attending
    prob_attend = model.predict_proba(input_data)[0][1]

# Set custom threshold
    threshold = 0.3

# Decide prediction based on your threshold
    prediction = 1 if prob_attend >= threshold else 0
    prob_not_attend = 1 - prob_attend

# Convert to percentages
    prob_attend_pct = round(prob_attend * 100, 2)
    prob_not_attend_pct = round(prob_not_attend * 100, 2)

# Interpretation
    if prediction == 1:
        interpretation = (
            f" **This person is likely to accept the reunion invitation.**\n"
            f"There is a **{prob_attend_pct}%** chance they will accept."
        )
    else:
        interpretation = (
            f" **This person is unlikely to accept the reunion invitation.**\n"
            f"There is a **{prob_not_attend_pct}%** chance they will *not* accept."
        )

    st.write(interpretation)
