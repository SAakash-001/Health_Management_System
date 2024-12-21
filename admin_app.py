# admin_app.py

import streamlit as st
import os
import plotly.express as px
import pandas as pd
import numpy as np
import pickle as pkl
import pymongo

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import (
    LogisticRegression, 
    LinearRegression
)
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_curve, 
    auc
)
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from pygam import LinearGAM
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

import time
import warnings
warnings.filterwarnings("ignore")

@st.cache_data(ttl=60)
def retrieve_data(uri,
                 db_name='health_management',
                 collection_name='patient_records',
                 save=False):
    client = MongoClient(uri, server_api=ServerApi('1'))
    db = client[db_name]
    collection = db[collection_name]
    docs = collection.find({})
    all_docs = []
    for doc in docs:
        doc.pop('_id', None)
        all_docs.append(doc)
    data = pd.DataFrame(all_docs)
    return data

def preprocess_data(data,
                    impute_strategy='Mean',
                    num_records=100):
    data['timestamp'] = pd.to_datetime(
        data['timestamp'], errors='coerce'
    )
    data = data.dropna(subset=['timestamp'])
    data = data.sort_values(
        by='timestamp'
    ).tail(num_records)
    features = ['race', 'gender', 'age', 'admission_type', 'time_in_hospital',
                'medical_specialty', 'num_lab_procedures', 'num_procedures',
                'num_medications', 'number_outpatient', 'number_emergency',
                'number_inpatient', 'diag_1', 'diag_2', 'diag_3', 'HbA1c_result']
    target = 'readmitted'
    data[target] = data[target].map({
        'NO': 0,
        '>30': 1,
        '<30': 2
    })
    data.dropna(
        subset=[target],
        inplace=True
    )
    data[target] = data[target].astype(int)
    data = data[features + [target]].copy()
    categorical_cols = ['race', 'gender', 'age', 'admission_type', 'medical_specialty', 'HbA1c_result']
    le_dict = {}
    for col in categorical_cols:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col].astype(str))
        le_dict[col] = le
    if impute_strategy != 'None':
        if impute_strategy == 'Mean':
            imputer = SimpleImputer(strategy='mean')
        elif impute_strategy == 'Median':
            imputer = SimpleImputer(strategy='median')
        elif impute_strategy == 'Mode':
            imputer = SimpleImputer(strategy='most_frequent')
        data[features] = imputer.fit_transform(data[features])
    else:
        data.dropna(inplace=True)
    X = data[features]
    y = data[target]
    return train_test_split(X, y, test_size=0.2, random_state=42), le_dict

def train_model(X_train, y_train, model_type):
    model_file = f"ml_models/{model_type.lower().replace(' ', '_')}_model.pkl"
    if os.path.exists(model_file):
        model = pkl.load(open(model_file, 'rb'))
    else:
        if model_type == 'Logistic Regression':
            model = LogisticRegression(max_iter=1000)
            model.fit(X_train, y_train)
        elif model_type == 'Random Forest':
            model = RandomForestClassifier(max_depth=20,
                                           random_state=0)
            model.fit(X_train, y_train)
        elif model_type == 'Decision Tree':
            model = DecisionTreeClassifier()
            model.fit(X_train, y_train)
        elif model_type == 'Gradient Boosting':
            model = GradientBoostingClassifier()
            model.fit(X_train, y_train)
        elif model_type == 'Gaussian Process':
            model = GaussianProcessClassifier(kernel=RBF())
            model.fit(X_train, y_train)
        # elif model_type == 'GAM':
        #     # st.write(X_train)
        #     print("X_train shape:", X_train.shape)
        #     print("y_train shape:", y_train.shape)
        #     unique_classes = np.unique(y_train)
        #     classifiers = {}
        #     for class_label in unique_classes:
        #         y_train_binary = (y_train == class_label).astype(int)
        #         gam = LinearGAM().gridsearch(X_train, y_train_binary)
        #         print('h')
        #         classifiers[class_label] = gam
        #         print('h')
        #     print('h')
        #     model = classifiers
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    with open(model_file, 'wb') as file:
        pkl.dump(model, file)

    return model

def app():
    if not st.session_state.get('logged_in') or st.session_state.get('role') != 'admin':
        st.error("Access denied. Please log in as an admin to view this page.")
        return
    
    if 'fetch_counter' not in st.session_state:
        st.session_state['fetch_counter'] = 0
    
    st.markdown("""
    <style>
        #ReportStatus { display: none; }
    </style>
    """, unsafe_allow_html=True)

    st.image('Assets/HYTRA logo.png', caption='Be Healthy Always', width=250)

    st.write('\n')

    inputs_placeholder = st.empty()

    uri = "mongodb+srv://admin:root@mtec-cluster.subjmhs.mongodb.net/?retryWrites=true&w=majority&appName=Mtec-Cluster"

    model = None
    model_data = None
    submit_button = None
    num_records = 100

    with st.form(key='columns_in_form'):
        st.write("### Model Training Options")
        st.write("Configure the parameters for training the machine learning model.")
        c1, c2, c3 = st.columns(3)
        with c1:
            imputation_methods = ['None', 'Mean', 'Median', 'Mode']
            selected_imputation = st.selectbox('Imputation Method', imputation_methods, help='Choose a method to handle missing values in the data.')
        with c2:
            num_records = st.number_input("Number of Records", value=100, min_value=10, step=10, help="Number of recent records to use for training.")
        with c3:
            ml_models = ['Logistic Regression', 'Random Forest', 'Decision Tree', 'Gradient Boosting', 'Gaussian Process'
                        #  , 'GAM'
                         ]
            selected_ml_model = st.selectbox('Machine Learning Model', ml_models, help='Choose the machine learning algorithm to train the model.')
        submit_button = st.form_submit_button(label='Train the Machine Learning Model')
        data = retrieve_data(uri)
        model_data = data

    with inputs_placeholder.container():
        st.write("### Data Overview")
        st.write("Explore the dataset and understand the distributions of key features.")
        fig1, fig2, fig3, fig4, fig5 = st.columns(5)
        with fig1:
            # st.write("**Race Distribution**")
            race_counts = model_data['race'].value_counts().reset_index()
            race_counts.columns = ['Race', 'Count']
            fig_race = px.pie(race_counts, names='Race', values='Count', title='Race Distribution')
            st.plotly_chart(fig_race, use_container_width=True)
        with fig2:
            # st.write("**Gender Distribution**")
            gender_counts = model_data['gender'].value_counts().reset_index()
            gender_counts.columns = ['Gender', 'Count']
            fig_gender = px.pie(gender_counts, names='Gender', values='Count', title='Gender Distribution')
            st.plotly_chart(fig_gender, use_container_width=True)
        with fig3:
            # st.write("**Age Distribution**")
            age_counts = model_data['age'].value_counts().sort_index().reset_index()
            age_counts.columns = ['Age', 'Count']
            fig_age = px.pie(age_counts, names='Age', values='Count', title='Age Distribution')
            st.plotly_chart(fig_age, use_container_width=True)
        with fig4:
            # st.write("**Readmission Status Distribution**")
            readmit_counts = model_data['readmitted'].value_counts().reset_index()
            readmit_counts.columns = ['Readmission Status', 'Count']
            fig_readmit = px.pie(readmit_counts, names='Readmission Status', values='Count', title='Readmission Status Distribution')
            st.plotly_chart(fig_readmit, use_container_width=True)
        with fig5:
            # st.write("**Medical Specialty Distribution**")
            specialty_counts = model_data['medical_specialty'].value_counts().reset_index()
            specialty_counts.columns = ['Medical Specialty', 'Count']
            fig_specialty = px.pie(specialty_counts, names='Medical Specialty', values='Count', title='Medical Specialty Distribution')
            st.plotly_chart(fig_specialty, use_container_width=True)
        st.write("### Data Preview and Filtering")
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Sample Data**")
            st.dataframe(model_data.tail(num_records), hide_index=True)
        with col2:
            st.write("**Filter Data**")
            user_query = st.text_input('Enter your query (e.g., time_in_hospital > 5 and age == "[70-80)"):', '', help='Filter the data using pandas query syntax.')
            if user_query:
                try:
                    filtered_df = model_data.tail(num_records).query(user_query)
                    if not filtered_df.empty:
                        st.dataframe(filtered_df, hide_index=True)
                    else:
                        st.write("No data matches the query.")
                except Exception as e:
                    st.error(f"Error with query: {str(e)}")

    if submit_button:
        st.session_state['fetch_counter'] += 1
        with st.status("Model Build Started...", expanded=True) as status:
            try:
                st.write('Setting up Environment...')
                time.sleep(1)
                st.write('Finalizing Training and Testing Data...')
                (X_train, X_test, y_train, y_test), le_dict = preprocess_data(model_data, selected_imputation, num_records)
                st.toast('Model Setup Complete!', icon='✅')
                time.sleep(1)
                st.write("Training Model...")
                model = train_model(X_train, y_train, selected_ml_model)
                st.toast('Model Training Complete!', icon='✅')
                time.sleep(1)
                st.write("Testing Model...")
                y_pred = model.predict(X_test)
                y_pred = y_pred.astype(int)
                y_test = y_test.astype(int)
                acc = accuracy_score(y_test, y_pred)
                st.toast('Model Testing Complete!', icon='✅')
                time.sleep(1)
                st.write("Preparing Results...")
                time.sleep(1)
                status.update(label="Model Build Complete!", state="complete", expanded=False)
                st.toast('Model Build Complete!', icon='✅')
                # Save the model and label encoders
                model_file = f"ml_models/{selected_ml_model.lower().replace(' ', '_')}_model.pkl"
                with open(model_file, 'wb') as file:
                    pkl.dump(model, file)
                le_file = f"ml_models/label_encoders.pkl"
                with open(le_file, 'wb') as file:
                    pkl.dump(le_dict, file)
                st.success(f'Model trained! Accuracy: {acc*100:.2f}%')
            except Exception as e:
                st.toast('Model Build Failed!', icon='❗️')
                st.error(f"Model Build Failed: {str(e)}")