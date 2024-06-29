import streamlit as st
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score, 
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

def preprocessing_data(data):
    x = data[['STATUS MAHASISWA','STATUS NIKAH','IPS 1','IPS 2','IPS 3','IPS 4','IPS 5','IPS 6','IPS 7','IPS 8','IPK ']]
    y = data['STATUS KELULUSAN']

def tampilData(data):
    # Tampilkan subheader
    st.subheader("Data Mahasiswa")
    
    # Tampilkan tabel menggunakan st.write()
    st.write(data)

    return data