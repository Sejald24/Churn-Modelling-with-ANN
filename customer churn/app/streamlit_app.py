import streamlit as st
import pandas as pd

# Page title
st.title("Bank Churn Prediction")
st.write('Entre Customer details')
# Input fields
credit_score = st.number_input("Credit Score", min_value=0, step=1)
geography = st.selectbox("Geography", ["France", "Spain", "Germany", "Other"])
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.number_input("Age", min_value=0, step=1)
tenure = st.number_input("Tenure (years)", min_value=0, step=1)
balance = st.number_input("Balance", min_value=0.0, step=100.0, format="%.2f")
num_products = st.number_input("Number of Products", min_value=0, step=1)
has_cr_card = st.selectbox("Has Credit Card", [0, 1])
is_active_member = st.selectbox("Is Active Member", [0, 1])
estimated_salary = st.number_input("Estimated Salary", min_value=0.0, step=100.0, format="%.2f")

# Submit button
if st.button("Submit"):
    # Create DataFrame for the input
    input_data = pd.DataFrame({
        "CreditScore": credit_score,
        "Geography": geography,
        "Gender": gender,
        "Age": age,
        "Tenure": tenure,
        "Balance": balance,
        "NumOfProducts": num_products,
        "HasCrCard": has_cr_card,
        "IsActiveMember": is_active_member,
        "EstimatedSalary": estimated_salary
    })
    response = requests.post("http://localhost:8000/predict", json=input_data.to_dict(orient="records")[0])
    if response.status_code == 200:
        result = response.json()
        st.subheader("Prediction Result")
        st.write(f"Churn Probability: {result['Churn_probability']:.2f}")
        st.write(f"Churned: {'Yes' if result['churned'] == 1 else 'No'}")
    else:
        st.error("Error in prediction. Please try again.")
    
