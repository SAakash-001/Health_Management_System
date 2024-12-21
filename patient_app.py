# patient_app.py

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import pickle as pkl
import time
import pymongo
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from sklearn.preprocessing import LabelEncoder

def retrieve_data(uri, db_name, collection_name='patient_records', save=False):
    client = MongoClient(uri, server_api=ServerApi('1'))
    db = client[db_name]
    collection = db[collection_name]
    docs = collection.find({})
    all_docs = []
    for doc in docs:
        doc.pop('_id', None)
        all_docs.append(doc)
    data = pd.DataFrame(all_docs)
    if save:
        data.to_csv(f'all_{collection_name}_data.csv', index=False)
    client.close()
    return data

def preprocess_data(df, 
                    recent_seconds=None,
                    recent_minutes=None,
                    recent_hours=None,
                    recent_days=None):
    if 'timestamp' not in df.columns:
        st.error('The data does not contain a `timestamp` column.')
        return pd.DataFrame()
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df = df.dropna(subset=['timestamp'])
    if recent_seconds is not None:
        cutoff = pd.Timestamp.now() - pd.Timedelta(seconds=recent_seconds)
        df = df[df['timestamp'] >= cutoff]
    elif recent_minutes is not None:
        cutoff = pd.Timestamp.now() - pd.Timedelta(minutes=recent_minutes)
        df = df[df['timestamp'] >= cutoff]
    elif recent_hours is not None:
        cutoff = pd.Timestamp.now() - pd.Timedelta(hours=recent_hours)
        df = df[df['timestamp'] >= cutoff]
    elif recent_days is not None:
        cutoff = pd.Timestamp.now() - pd.Timedelta(days=recent_days)
        df = df[df['timestamp'] >= cutoff]
    return df

@st.cache_resource
def load_model(model_name):
    return pkl.load(open(
        f'ml_models/{model_name.lower().replace(" ", "_")}_model.pkl',
        'rb'
    ))

def app():
    if not st.session_state.get('logged_in') or st.session_state.get('role') != 'patient':
        st.error("Access denied. Please log in as a patient to view this page.")
        return
    
    patient_id = st.session_state.get('patient_id')
    if not patient_id:
        st.error("Patient ID not found. Please log in again.")
        return
    
    # st.image('Assets/HYTRA logo.png', caption='Be Healthy Always', width=220)
    # st.write('\n')

    uri = "mongodb+srv://admin:root@mtec-cluster.subjmhs.mongodb.net/?retryWrites=true&w=majority&appName=Mtec-Cluster"

    col1, col2 = st.columns([1, 1])
    with col1:
        st.write('## Patient Data Monitor 📈')
        mean_placeholder = st.empty()
    with col2:
        c1, c2 = st.columns(2)
        with c1:
            time_period = st.selectbox("Select Time Period", ['Seconds', 'Minutes', 'Hours', 'Days'])
        with c2:
            ml_models = ['Logistic Regression', 'Random Forest', 'Decision Tree', 'Gradient Boosting', 'Gaussian Process'
                        #  , 'GAM'
                         ]
            selected_ml_model = st.selectbox('Select Predictor', ml_models)
        ml_model = load_model(selected_ml_model)
        individual_placeholder = st.empty()
    individual_placeholder_2 = st.empty()

    while True:
        data = retrieve_data(uri, 'health_management')
        # print('____'*10)
        # print(data.columns)
        with mean_placeholder.container():
            plot1, plot2 = st.columns(2)
            with plot1:
                readmit_counts = data['readmitted'].value_counts().reset_index()
                readmit_counts.columns = ['Readmission Status', 'Count']
                fig = px.pie(readmit_counts, names='Readmission Status', values='Count', title='Readmission Status Distribution')
                st.write(fig)
            with plot2:
                avg_time = data.groupby('medical_specialty')['time_in_hospital'].mean().reset_index()
                fig = px.bar(avg_time, x='medical_specialty', y='time_in_hospital', title='Average Time in Hospital by Medical Speciality')
                fig.update_layout(xaxis_tickangle=-45)
                st.write(fig)
        patient_data = data[data['patient_id'] == patient_id]
        if patient_data.empty:
            st.warning("No records found for your patient ID.")
            return
        if time_period == 'Seconds':
            df_preprocessed = preprocess_data(patient_data, recent_seconds=10)
        if time_period == 'Seconds':
            df_preprocessed = preprocess_data(patient_data, recent_seconds=10)
        elif time_period == 'Minutes':
            df_preprocessed = preprocess_data(patient_data, recent_minutes=1)
        elif time_period == 'Hours':
            df_preprocessed = preprocess_data(patient_data, recent_hours=1)
        elif time_period == 'Days':
            df_preprocessed = preprocess_data(patient_data, recent_days=1)
        else:
            df_preprocessed = preprocess_data(patient_data)
        df_preprocessed = preprocess_data(patient_data, recent_days=1)
        if df_preprocessed.empty:
            st.write("No data available for the selected time period.")
            time.sleep(1)
            continue
        
        with individual_placeholder.container():
            features = ['race', 'gender', 'age', 'admission_type', 'time_in_hospital',
                    'medical_specialty', 'num_lab_procedures', 'num_procedures',
                    'num_medications', 'number_outpatient', 'number_emergency',
                    'number_inpatient', 'diag_1', 'diag_2', 'diag_3', 'HbA1c_result']
            df_for_prediction = df_preprocessed[features]
            df_encoded = df_for_prediction.copy()
            for col in df_encoded.columns:
                if df_encoded[col].dtype == 'object':
                    le = LabelEncoder()
                    df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
            probabilities = ml_model.predict_proba(df_encoded)[:,1]
            df_preprocessed['readmission_probability'] = probabilities
            fig = px.line(df_preprocessed, x='timestamp', y='readmission_probability', title='Readmission Probability Over Time', labels={'readmission_probability': 'Probability'})
            st.write(fig)
        
        with individual_placeholder_2.container():
            fig_col1, fig_col2, fig_col3, fig_col4 = st.columns(4)
            with fig_col1:
                fig = px.line(df_preprocessed, x='timestamp', y='num_lab_procedures', title='Number of Lab Procedures Over Time', labels={'num_lab_procedures': 'Number of Lab Procedures'})
                st.write(fig)
            with fig_col2:
                fig = px.line(df_preprocessed, x='timestamp', y='num_medications', title='Number of Medications Over Time', labels={'num_medications': 'Number of Medications'})
                st.write(fig)
            with fig_col3:
                fig = px.line(df_preprocessed, x='timestamp', y='time_in_hospital', title='Time in Hospital Over Time', labels={'time_in_hospital': 'Time in Hospital (Days)'})
                st.write(fig)
            with fig_col4:
                hba1c_map = {'Normal': 0, '>7': 1, '>8': 2, 'None': -1}
                df_preprocessed['HbA1c_result_numeric'] = df_preprocessed['HbA1c_result'].map(hba1c_map)
                fig = px.line(df_preprocessed, x='timestamp', y='HbA1c_result_numeric', title='HbA1c Result Over Time', labels={'HbA1c_result_numeric': 'HbA1c Result'})
                st.write(fig)
        time.sleep(1)