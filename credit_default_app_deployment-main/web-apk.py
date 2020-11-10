import streamlit as st
import pandas as pd
import pickle

st.title("Credit Card Default Prediction")
st.sidebar.header('User Input Features:-')

def mapping(x):
    dict = {-1:'on time', 0:'no credit taken',1:'one month delay',2:'two month delay',3:'three month delay',4:'four month delay',5:'five month delay',6:'six month delay',7:'seven month delay',8:'eight month delay',9:'nine month delay'}
    if x in dict.keys():
        return dict[x]



def user_input():
    limit_balance = st.sidebar.number_input('BALANCE LIMIT OF ACCOUNT',10000, value=20000)
    pay1_input = st.sidebar.selectbox('PAYMENT STATUS ', (-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,), format_func=mapping)
    bill_amount1 = st.sidebar.number_input('BILL AMOUNT ',value = 0 )
    pay_amount1 = st.sidebar.number_input('AMOUNT PAID ',0,value= 0)

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
    st.error('Card will Default next month')
else:
    st.success('Card will Not Default next month')
    st.balloons()
print('\n')
st.write('Prediction Probabilities')
button = st.button('Show Prediction Probabilities')
if button:
    st.write(prediction_probability)



