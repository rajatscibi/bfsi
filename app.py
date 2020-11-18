from pycaret.regression import load_model, predict_model
import streamlit as st
import pandas as pd
import numpy as np

model = load_model('Final XGB Model A112020')

def predict(model, input_df):
    predictions_df = predict_model(estimator=model, data=input_df)
    predictions = predictions_df['Label'][0]
    return predictions

def run():

    from PIL import Image
    image = Image.open('logo.png')
    image_hospital = Image.open('loan.jpg')

    st.image(image,use_column_width=False)

    add_selectbox = st.sidebar.selectbox(
    "How would you like to predict?",
    ("Online", "Batch"))

    st.sidebar.info('This app is created to predict loan eligibility')
    st.sidebar.success('https://www.sci-bi.com')
    
    st.sidebar.image(image_hospital)

    st.title("Loan Eligibility Prediction App")

    if add_selectbox == 'Online':

        Gender = st.selectbox('Gender', ['Male', 'Female'])
        Dependents = st.selectbox('Dependents', ['0','1','2','3+'])
        ApplicantIncome = st.number_input('ApplicantIncome', min_value=0, max_value=10000, value=1000)
        CoapplicantIncome = st.number_input('CoapplicantIncome', min_value=0, max_value=10000, value=1000)
        LoanAmount = st.number_input('LoanAmount', min_value=0, max_value=10000, value=1000)
        Loan_Amount_Term = st.number_input('Loan_Amount_Term', min_value=0, max_value=10000, value=1000)
        Credit_History = st.number_input('Credit_History', min_value=0, max_value=1, value=1)
        CibilScore = st.number_input('Cibil Score', min_value=0, max_value=10000, value=1000)
        
        if st.checkbox('Married'):
            Married = 'Yes'
        else:
            Married = 'No'
        if st.checkbox('Graduate'):
            Education = 'Graduate'
        else:
            Education = 'Not Graduate'
        if st.checkbox('Self_Employed'):
            Self_Employed = 'Yes'
        else:
            Self_Employed = 'No'
        Property_Area = st.selectbox('Property_Area', ['Urban', 'Semiurban', 'Rural'])

        output=""

        input_dict = {'Gender' : Gender, 'Dependents' : Dependents, 'ApplicantIncome' : ApplicantIncome, 'CoapplicantIncome' : CoapplicantIncome, 'LoanAmount' : LoanAmount, 'Loan_Amount_Term': Loan_Amount_Term, 'Credit_History' : Credit_History, 'Cibil Score' : CibilScore, 'Married' : Married, 'Education' : Education, 'Self_Employed' : Self_Employed, 'Property_Area' : Property_Area }
        input_df = pd.DataFrame([input_dict])

        if st.button("Predict"):
            output = predict(model=model, input_df=input_df)
            output = str(output)

        st.success('The output is {}'.format(output))

    if add_selectbox == 'Batch':

        file_upload = st.file_uploader("Upload csv file for predictions", type=["csv"])

        if file_upload is not None:
            data = pd.read_csv(file_upload)
            predictions = predict_model(estimator=model,data=data)
            st.write(predictions)

if __name__ == '__main__':
    run()
