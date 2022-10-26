import pandas as pd
import numpy as np
import pickle
import streamlit as st
import requests
import json
import sklearn



# load pipe
pipe = pickle.load(open("preprocess_churn.pkl", "rb"))
# widget input
with st.form(key='form_parameters'):
    st.title("Model Prediction Churn")
    SeniorCitizen= st.selectbox("Apakah pelanggan lanjut usia (1) atau tidak (0)?", ['0','1'])
    Partner= st.selectbox("Apakah pelanggan memiliki pasangan?", ['Yes','No'])
    Dependents= st.selectbox("Apakah pelanggan memiliki tanggunan?", ['Yes','No'])
    tenure=st.number_input("Total Berapa bulan berlangganan",min_value=0,help='1=sebulan, 12=setahun')
    PhoneService= st.selectbox("Apakah pelanggan menggunakan layanan telephone?", ['Yes','No'])
    MultipleLines= st.selectbox("Apakah pelanggan menggunakan layanan telephone dan internet?", ['Yes', 'No', 'No phone service'])
    InternetService= st.selectbox("Tipe Layanan Internet", ['Dsl','Fiber optic', 'No'],index=1)
    OnlineSecurity= st.selectbox("Langganan Produk Keamanan Online",['Yes','No internet service', 'No'],help="seperti firewall dll")
    OnlineBackup= st.selectbox("Langganan Produk Online Backup",['Yes','No internet service', 'No'])
    DeviceProtection= st.selectbox("Langganan Produk Device Protection",['Yes','No internet service', 'No'])
    TechSupport= st.selectbox("Langganan Produk Dukungan Teknis",['Yes','No internet service', 'No'],help="Jasa IT support")
    StreamingTV= st.selectbox("Langganan Produk Dukungan Streaming TV",['Yes','No internet service', 'No'],help="Layanan TV Kabel")
    StreamingMovies= st.selectbox("Langganan Produk Dukungan Streaming Movie",['Yes','No internet service', 'No'],help="Layanan Movie seperti HBO Dll")
    Contract= st.selectbox("Jangka Waktu Contract",['Month-to-month','One year', 'Two year'],help="Lama Contract")
    PaperlessBilling= st.selectbox("Apakah pelanggan menggunakan layanan Paperless Billang",['Yes','No',])
    PaymentMethod= st.selectbox("Metode Pembayaran Pelanggan",['Mailed check', 'Electronic check', 'Bank transfer (automatic)', 'Credit card (automatic)'],help="otomatis = auto debit, cek elektronik= kirim email,cek pos= kirim kartu pos")
    MonthlyCharges= st.number_input("Jumlah Tagihan Bulanan",min_value=0)
    TotalCharges= st.number_input("Total Biaya selama berlangganan",min_value=0,help='Total uang yang dikeluarkan pelanggan selama contract')
    submitted= st.form_submit_button('Predict')

# input to dataframe
new_data = {
        'SeniorCitizen':SeniorCitizen,
        'Partner':Partner,
        'Dependents':Dependents,
        'tenure':tenure,
        'PhoneService':PhoneService,
        'MultipleLines':MultipleLines,
        'InternetService':InternetService,
        'OnlineSecurity':OnlineSecurity,
        'OnlineBackup':OnlineBackup,
        'DeviceProtection':DeviceProtection,
        'TechSupport':TechSupport,
        'StreamingTV':StreamingTV,
        'StreamingMovies':StreamingMovies,
        'Contract':Contract,
        'PaperlessBilling':PaperlessBilling,
        'PaymentMethod':PaymentMethod,
        'MonthlyCharges':MonthlyCharges,
        'TotalCharges':TotalCharges
        }
new_data = pd.DataFrame([new_data])

#preprocessing
new_data = pipe.transform(new_data)
new_data = new_data.tolist()

#input ke model
input_data_json = json.dumps({
    "signature_name": "serving_default",
    "instances": new_data
})


# # inference
URL = "http://chur-model-ululazmi.herokuapp.com/v1/models/churn_model:predict"
r = requests.post(URL, data=input_data_json)

if submitted:
    if r.status_code == 200:
         res = r.json()
         if res['predictions'][0][0] >= 0.5:
                st.title('Churn')
         else:
            st.title('Tidak Churn')
    else:
        st.write('Error')