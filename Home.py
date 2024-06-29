import streamlit as st
import pandas as pd
import numpy as np
from process import tampilData
from clasification import klasifikasiC45,klasifikasi_KNN
from sklearn.datasets import data
from sklearn.metrics import (accuracy_score,    
                             precision_score,
                             recall_score,
                             f1_score,
                             confusion_matrix,
                             classification_report)

import streamlit as st
from joblib import load

#Membuat layout dengan kolom
col1, col2, col3 = st.columns([2, 1, 1])

with col1:
    st.sidebar.image('logo.png', width=230)

text = """Rio Daud Fernando Nainggolan
        09021382025117"""
st.sidebar.markdown(text, unsafe_allow_html=True)

st.title("""PERBANDINGAN METODE C4.5 DAN K-NEAREST NEIGHBOR UNTUK KLASIFIKASI KELULUSAN MAHASISWA""")

st.sidebar.header("Test Data")
metode = st.sidebar.selectbox("Pilih Algoritma", [" ", "C45", "KNN"], key="algo_selectbox")
klasifikasi_button = st.sidebar.button('Klasifikasi')

data_placeholder = st.empty()

#data = upload_dataset()
def upload_dataset2():
    upload_file = st.file_uploader("Upload Dataset dalam bentuk xlsx", type=["xlsx"], key="dataset_test")
    if upload_file is not None:
        #dataset = pd.read_excel(upload_file, sep=',')
        dataset = pd.read_excel(upload_file)
        return dataset
    return None
data = upload_dataset2()

if data is not None:
    if klasifikasi_button and metode != '':
        from joblib import load

        if metode == 'C45' and klasifikasi_button:
            model_file = 'model.pkl'
            # st.write('Load Model C45')
            #Melakukan prediksi pada data mining
            model = load(model_file)

            print(data.info())

            x = data[['STATUS MAHASISWA','STATUS NIKAH','IPS 1','IPS 2','IPS 3','IPS 4','IPS 5','IPS 6','IPS 7','IPS 8','IPK ']]
            #st.write(x)
        
            y=data['STATUS KELULUSAN']
            #st.write(y)

            y_predict = model.predict(x)
            #st.write(y_predict)

            data['PREDIKSI'] = y_predict
            data['STATUS'] = data['PREDIKSI'].map({0: 'Terlambat', 1: 'Tepat'})

            st.write(data)

            st.write("Akurasi Data Testing C45")
            st.text(f'Accuracy              : {accuracy_score(y,y_predict)}')
            st.text(f'Precision Accuracy    : {precision_score(y,y_predict)}')
            st.text(f'Recall Accuracy       : {recall_score(y,y_predict)}')
            st.text(f'F1-Score Accuracy     : {f1_score(y,y_predict)}')

            st.text('Confusion Matrix')
            st.write(confusion_matrix(y,y_predict))
            #st.text(f'Confusion Matrix : {confusion_matrix(y,y_predict)}')

            st.write('Classification Report')
            #st.text(f'Classification Report : {classification_report(y,y_predict)}')
            st.code(classification_report(y,y_predict), language='markdown')

        elif metode == 'KNN' and klasifikasi_button:
            model_file = 'knn.pkl'
            # st.write('Load Model KNN')
            #Melakukan prediksi pada data mining
            model = load(model_file)

            print(data.info())

            x = data[['STATUS MAHASISWA','STATUS NIKAH','IPS 1','IPS 2','IPS 3','IPS 4','IPS 5','IPS 6','IPS 7','IPS 8','IPK ']]
            #st.write(x)

            x.fillna(0, inplace=True)  # Untuk mengganti NaN menjadi 0
            x.replace("NaN", '0', inplace=True)
            y=data['STATUS KELULUSAN']
            #st.write(y)

            y_predict = model.predict(x)
            #st.write(y_predict)

            data['PREDIKSI'] = y_predict
            data['STATUS'] = data['PREDIKSI'].map({0: 'Terlambat', 1: 'Tepat'})

            st.write(data)

            st.write("Akurasi Data Testing KNN")
            st.text(f'Accuracy              : {accuracy_score(y,y_predict)}')
            st.text(f'Precision Accuracy    : {precision_score(y,y_predict)}')
            st.text(f'Recall Accuracy       : {recall_score(y,y_predict)}')
            st.text(f'F1-Score Accuracy     : {f1_score(y,y_predict)}')

            st.text('Confusion Matrix')
            st.write(confusion_matrix(y,y_predict))
            #st.text(f'Confusion Matrix : {confusion_matrix(y,y_predict)}')

            st.write('Classification Report')
            #st.text(f'Classification Report : {classification_report(y,y_predict)}')
            st.code(classification_report(y,y_predict), language='markdown')

        else :
            print("error")
        
        model = load(model_file)
        #st.write('Load Model')

    st.write("Masukkan nilai-nilai mahasiswa untuk memprediksi kelulusan mereka.")

        # Fungsi validasi input
    def validate_input(value):
        try:
            return float(value)
        except ValueError:
            return None
    
    # Memuat model saat aplikasi dijalankan
    model_c45 = load('model.pkl')
    model_knn = load('knn.pkl')

    # Input nilai dari pengguna
    status_mahasiswa = st.selectbox('Status Mahasiswa', options=[None, 0, 1], format_func=lambda x: 'Pilih Status' if x is None else ('Mahasiswa' if x == 0 else 'Bekerja'))
    status_nikah = st.selectbox('Status Nikah', options=[None, 0, 1], format_func=lambda x: 'Pilih Status' if x is None else ('Belum Menikah' if x == 0 else 'Menikah'))
    ips1 = st.number_input('Nilai IPS 1', step=0.01, format="%.2f", value=None)
    ips2 = st.number_input('Nilai IPS 2', min_value=None, max_value=4.0, step=0.01, format="%.2f", value=None)
    ips3 = st.number_input('Nilai IPS 3', min_value=None, max_value=4.0, step=0.01, format="%.2f", value=None)
    ips4 = st.number_input('Nilai IPS 4', min_value=None, max_value=4.0, step=0.01, format="%.2f", value=None)
    ips5 = st.number_input('Nilai IPS 5', min_value=None, max_value=4.0, step=0.01, format="%.2f", value=None)
    ips6 = st.number_input('Nilai IPS 6', min_value=None, max_value=4.0, step=0.01, format="%.2f", value=None)
    ips7 = st.number_input('Nilai IPS 7', min_value=None, max_value=4.0, step=0.01, format="%.2f", value=None)
    ips8 = st.number_input('Nilai IPS 8', min_value=None, max_value=4.0, step=0.01, format="%.2f", value=None)
    ipk = st.number_input('Nilai IPK', min_value=None, max_value=4.0, step=0.01, format="%.2f", value=None)

    # Pilih metode prediksi
    metode = st.selectbox("Pilih Metode Algoritma", ["C45", "KNN"], key='metode')

    if st.button('Prediksi', key='predict'):
        # Memeriksa apakah ada nilai yang kosong
        input_values = [status_nikah, status_mahasiswa, ips1, ips2, ips3, ips4, ips5, ips6, ips7, ips8, ipk]
        if any(val is None for val in input_values):
            st.warning("Pastikan semua nilai telah dimasukkan dengan benar.")
        elif any(val == 0 for val in input_values[2:]):
            st.write('Hasil Prediksi: Terlambat')
        else:
            input_data = np.array([input_values])

            if metode == "C45":
               # st.write('Menggunakan Model C45')
                prediction = model_c45.predict(input_data)
            elif metode == "KNN":
               # st.write('Menggunakan Model KNN')
                prediction = model_knn.predict(input_data)

            # Menampilkan hasil prediksi
            status = 'Tepat' if prediction[0] == 1 else 'Terlambat'
            st.write(f'Hasil Prediksi: {status}')