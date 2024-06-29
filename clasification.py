from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
import streamlit as st
from sklearn.metrics import (accuracy_score,
                             precision_score,
                             recall_score,
                             f1_score,
                             confusion_matrix,
                             classification_report)

def klasifikasiC45(data):     
    st.text("METODE C45")

    data.fillna(0, inplace=True)  # Untuk mengganti NaN menjadi 0
    data.replace("NaN", '0', inplace=True)
    
    x = data[['STATUS MAHASISWA','STATUS NIKAH','IPS 1','IPS 2','IPS 3','IPS 4','IPS 5','IPS 6','IPS 7','IPS 8','IPK ']]
    y = data['STATUS KELULUSAN']

    #c45 model
    model = DecisionTreeClassifier()

    #memasukkan data training pada fungsi c45   
    data_training = model.fit(x,y)

    #Melakukan prediksi pada data mining
    y_predict = data_training.predict(x)

    st.write('Data Train')
    #st.write(y_predict) 

    data['PREDIKSI'] = y_predict
    data['STATUS'] = data['PREDIKSI'].map({0: 'Terlambat', 1: 'Tepat'})

    st.write(data)

    st.write("Akurasi Data Training")
    st.text(f'SVM Accuracy          : {accuracy_score(y,y_predict)}')
    st.text(f'Precision Accuracy    : {precision_score(y,y_predict)}')
    st.text(f'Recall Accuracy       : {recall_score(y,y_predict)}')
    st.text(f'F1-Score Accuracy     : {f1_score(y,y_predict)}')

    # st.text('Confusion Matrix')
    # st.write(confusion_matrix(y,y_predict))
    #st.text(f'Confusion Matrix : {confusion_matrix(y,y_predict)}')

    st.write('Classification Report')
    #st.text(f'Classification Report : {classification_report(y,y_predict)}')
    st.code(classification_report(y,y_predict), language='markdown')
    
    from joblib import dump

    import pickle
    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    return y_predict    

def klasifikasi_KNN(data):     
    st.text("METODE KNN")

    data.fillna(0, inplace=True)  # Untuk mengganti NaN menjadi 0
    data.replace("NaN", '0', inplace=True)

    x = data[['STATUS MAHASISWA','STATUS NIKAH','IPS 1','IPS 2','IPS 3','IPS 4','IPS 5','IPS 6','IPS 7','IPS 8','IPK ']]
    y = data['STATUS KELULUSAN']

    #knn model
    model = KNeighborsClassifier(n_neighbors=5)

    #memasukkan data training pada fungsi knn   
    data_training = model.fit(x,y)

    #Melakukan prediksi pada data mining
    y_predict = data_training.predict(x)

    st.write('Data Train')
    #st.write(y_predict) 

    data['PREDIKSI'] = y_predict
    data['STATUS'] = data['PREDIKSI'].map({0: 'Terlambat', 1: 'Tepat'})

    st.write(data)

    st.write("Akurasi Data Training")
    st.text(f'Accuracy              : {accuracy_score(y,y_predict)}')
    st.text(f'Precision Accuracy    : {precision_score(y,y_predict)}')
    st.text(f'Recall Accuracy       : {recall_score(y,y_predict)}')
    st.text(f'F1-Score Accuracy     : {f1_score(y,y_predict)}')

    # st.text('Confusion Matrix')
    # st.write(confusion_matrix(y,y_predict))
    #st.text(f'Confusion Matrix : {confusion_matrix(y,y_predict)}')

    st.write('Classification Report')
    #st.text(f'Classification Report : {classification_report(y,y_predict)}')
    st.code(classification_report(y,y_predict), language='markdown')

    from joblib import dump

    import pickle
    with open('knn.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    return y_predict    #Mengembalikan nilai prediksi dari model untuk data uji.