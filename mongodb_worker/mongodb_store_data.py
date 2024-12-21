import pymongo
import random
import datetime
import time

from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

def generate_patient_documents(num_docs, patient_id):
    documents = []
    races = ['Caucasian', 'AfricanAmerican', 'Asian', 'Hispanic', 'Other']
    genders = ['Male', 'Female']
    age_categories = ['[0-10)', '[10-20)', '[20-30)', '[30-40)', '[40-50)', '[50-60)', '[60-70)', '[70-80)', '[80-90)', '[90-100)']
    admission_types = ['Emergency', 'Urgent', 'Elective', 'Newborn', 'Trauma Center', 'Not Mapped', 'Unknown']
    medical_specialties = ['Cardiology', 'InternalMedicine', 'Family/GeneralPractice', 'Surgery-General', 'Orthopedics', 'Gastroenterology', 'Nephrology', 'Other']
    hba1c_results = ['>7', '>8', 'Normal', 'None']
    readmission_status = ['NO', '>30', '<30']

    for _ in range(num_docs):
        # patient_id = random.randint(100000, 999999)
        doc = {
            "patient_id": patient_id,
            "race": random.choice(races),
            "gender": random.choice(genders),
            "age": random.choice(age_categories),
            "admission_type": random.choice(admission_types),
            "time_in_hospital": random.randint(1, 14),
            "medical_specialty": random.choice(medical_specialties),
            "num_lab_procedures": random.randint(1, 100),
            "num_procedures": random.randint(0, 6),
            "num_medications": random.randint(1, 81),
            "number_outpatient": random.randint(0, 42),
            "number_emergency": random.randint(0, 76),
            "number_inpatient": random.randint(0, 21),
            "diag_1": random.randint(1, 999),
            "diag_2": random.randint(1, 999),
            "diag_3": random.randint(1, 999),
            "HbA1c_result": random.choice(hba1c_results),
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "readmitted": random.choice(readmission_status)
        }
        documents.append(doc)
    return documents

def save_documents(uri, db_name, collection_name, documents):
    client = MongoClient(uri, server_api=ServerApi('1'))
    db = client[db_name]
    collection = db[collection_name]
    result = collection.insert_many(documents)
    print(f"{len(result.inserted_ids)} documents saved successfully.")
    client.close()

if __name__ == "__main__":
    uri = "mongodb+srv://admin:root@mtec-cluster.subjmhs.mongodb.net/?retryWrites=true&w=majority&appName=Mtec-Cluster"
    num_docs = 1 
    interval = int(input("Enter the time interval in seconds between data generations: "))

    patient_id = input("Enter the patient id: ")
    while True:
        patient_documents = generate_patient_documents(num_docs, patient_id)
        save_documents(uri, 'health_management', 'patient_records', patient_documents)
        time.sleep(interval)