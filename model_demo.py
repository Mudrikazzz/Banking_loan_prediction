import streamlit as st
import pandas as pd
import pickle


st.title("Loan Approval Prediction")
st.write("Fill the bellow details to predict you Loan approval")


person_age = st.number_input("Age",min_value=18,max_value=100)
person_income = st.number_input("Income",min_value=1000,max_value=100000)
person_home_ownership = st.selectbox("Home Ownership",['RENT', 'OWN', 'MORTGAGE', 'OTHER'])
person_emp_length = st.number_input("employee length",min_value=0,max_value=30)
loan_intent = st.selectbox("Loan Intent",['PERSONAL', 'EDUCATION', 'MEDICAL', 'VENTURE', 'HOMEIMPROVEMENT','DEBTCONSOLIDATION'])
loan_grade = st.selectbox("Loan Grade",['A','B','C','D','E','F','G'])
loan_amnt =  st.number_input("Loan Amount",min_value=1000,max_value=1000000)
loan_int_rate = st.number_input("Loan Interest",min_value=1,max_value=30)
loan_percent_income = st.number_input("Loan precent income",min_value=1,max_value=100)
cb_person_default_on_file = st.selectbox("Default History",["Y","N"])
cb_person_cred_hist_length = st.number_input("credit history length",min_value=1,max_value=100)


if st.button("Predict Loan Status"):
    input_data = {
        "person_age":person_age,"person_income":person_income,
        "person_home_ownership":person_home_ownership,
        "person_emp_length":person_emp_length,
        "loan_intent":loan_intent,
        "loan_grade":{"A":1,"B":2,"C":3,"D":4,"E":5,"F":6,"G":7}[loan_grade],
        "loan_amnt":loan_amnt,
        "loan_int_rate":loan_int_rate,
        "loan_percent_income":loan_percent_income,
        "cb_person_default_on_file":cb_person_default_on_file,
        "cb_person_cred_hist_length":cb_person_cred_hist_length
    }
    
    df = pd.DataFrame([input_data])
    try:
        with open(r"C:\Users\hp\Downloads\classes\Demo\loan_predict.pkl", "rb") as file:
            model = pickle.load(file)

        prediction = model.predict(df)[0]
        st.success(f"✅ Your loan is: {'Approved' if prediction == 0 else 'Denied'}")
    except Exception as e:
        st.error(f"❌ Something went wrong: {e}")