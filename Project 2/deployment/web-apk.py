import streamlit as st
import pandas as pd
import pickle

st.title("Credit Card Default Prediction")
st.sidebar.header('User Input Features:-')


def user_input():
    limit_balance = st.sidebar.number_input('BALANCE LIMIT OF ACCOUNT',10000,value=20000)
    pay1_input = st.sidebar.slider('PAYMENT STATUS IN SEPTEMBER', -2, 8, 0)
    bill_amount1 = st.sidebar.number_input('BILL AMOUNT IN SEPTEMBER',value = 100000 )
    pay_amount1 = st.sidebar.number_input('AMOUNT PAID IN SEPTEMBER',0,value= 20000)

    data = {'PAYMENT1_STATUS':pay1_input, 'PAYMENT_AMOUNT1':pay_amount1, 'LIMIT_BALANCE':limit_balance, 'BILL_AMOUNT':bill_amount1}
    features = pd.DataFrame(data, index=[0])
    return features


input_data = user_input()
st.subheader('USER INPUT VALUES')
st.write(input_data)

load_model = pickle.load(open('model.pkl','rb'))

prediction = load_model.predict(input_data)
prediction_probability = load_model.predict_proba(input_data)

st.subheader('prediction')
# if prediction
if prediction:
    st.error('Card Defaulted')
else:
    st.success('Card Not Defaulted')

st.write('Prediction Probabilities')
st.write(prediction_probability)


