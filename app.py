from pycaret.regression import load_model, predict_model
import streamlit as st
import pandas as pd
import numpy as np

model = load_model('Final RF Model')

def predict(model, input_df):
    predictions_df = predict_model(estimator=model, data=input_df)
    predictions = predictions_df['Label'][0]
    return predictions

def run():

    from PIL import Image
    image = Image.open('logobiat.png')
    

    st.image(image,use_column_width=False)

    add_selectbox = st.sidebar.selectbox(
    "How would you like to predict?",
    ("Online", "Batch"))

    st.sidebar.info('This app is created to predict default credit card ')
    st.sidebar.success('https://www.pycaret.org')
    
    st.sidebar.image(image_hospital)

    st.title("Default Credit Card Prediction Application")

    if add_selectbox == 'Online':


        ID = st.number_input('ID', min_value=1, max_value=100000000, value=25)
        AMT = st.number_input('AMT', min_value=1, max_value=10000000, value=25)
        SEX= st.selectbox('Sex', ['1', '2'])
        EDUCATION = st.selectbox('EDUCATION', [0,1,2,3,4,5,6])
        AGE = st.number_input('AGE', min_value=1, max_value=100, value=25)
        RS_1 = st.selectbox('RS_1', [-2,-1,1,2,3,4,5,6,7,8,9])
        RS_2 = st.selectbox('RS_2', [-2,-1,1,2,3,4,5,6,7,8,9])
        RS_3 = st.selectbox('RS_3', [-2,-1,1,2,3,4,5,6,7,8,9])
        RS_4 = st.selectbox('RS_4', [-2,-1,1,2,3,4,5,6,7,8,9])
        RS_5 = st.selectbox('RS_5', [-2,-1,1,2,3,4,5,6,7,8,9])
        RS_6 = st.selectbox('RS_6', [-2,-1,1,2,3,4,5,6,7,8,9])  
        F_AMT1 = st.number_input('F_AMT1', min_value=0, max_value=10000000, value=25)
        F_AMT2 = st.number_input('F_AMT2', min_value=0, max_value=10000000, value=25)
        F_AMT3 = st.number_input('F_AMT3', min_value=0, max_value=10000000, value=25)
        F_AMT4 = st.number_input('F_AMT4', min_value=0, max_value=10000000, value=25)
        F_AMT5 = st.number_input('F_AMT5', min_value=0, max_value=10000000, value=25)
        F_AMT6 = st.number_input('F_AMT6', min_value=0, max_value=10000000, value=25)
        PAMT1 = st.number_input('PAMT1', min_value=0, max_value=10000000, value=25)
        PAMT2 = st.number_input('PAMT2', min_value=0, max_value=10000000, value=25)
        PAMT3 = st.number_input('PAMT3', min_value=0, max_value=10000000, value=25)
        PAMT4 = st.number_input('PAMT4', min_value=0, max_value=10000000, value=25)
        PAMT5 = st.number_input('PAMT5', min_value=0, max_value=10000000, value=25)
        PAMT6 = st.number_input('PAMT5', min_value=0, max_value=10000000, value=25)
        

    

        output=""

        input_dict = {'ID' : ID, 'AMT' : AMT, 'SEX' : SEX, 'EDUCATION' : EDUCATION, 'AGE' : AGE, 'RS_1' : RS_1, 'RS_2' : RS_2, 'RS_3' : RS_3, 'RS_4' : RS_4, 'RS_5' : RS_5, 'RS_6' : RS_6, 'F_AMT1' : F_AMT1, 'F_AMT2' : F_AMT2, 'F_AMT3' : F_AMT3, 'F_AMT4' : F_AMT4, 'F_AMT5' : F_AMT5, 'F_AMT6' : F_AMT6, 'PAMT1' : PAMT1, 'PAMT2' : PAMT2, 'PAMT3' : PAMT3, 'PAMT4' : PAMT4, 'PAMT5' : PAMT5, 'PAMT6' : PAMT6}
        input_df = pd.DataFrame([input_dict])

        if st.button("Predict"):
            output = predict(model=model, input_df=input_df)
            output = '$' + str(output)

        st.success('The output is {}'.format(output))

    if add_selectbox == 'Batch':

        file_upload = st.file_uploader("Upload csv file for predictions", type=["csv"])

        if file_upload is not None:
            data = pd.read_csv(file_upload)
            predictions = predict_model(estimator=model,data=data)
            st.write(predictions)

if __name__ == '__main__':
    run()