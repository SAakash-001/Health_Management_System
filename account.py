# account.py

import streamlit as st
import json
import os
import bcrypt

# Define the path to the credentials directory
CREDENTIALS_DIR = 'credentials'
PATIENT_CREDENTIALS_FILE = os.path.join(CREDENTIALS_DIR, 'patient_credentials.json')
ADMIN_CREDENTIALS_FILE = os.path.join(CREDENTIALS_DIR, 'admin_credentials.json')

def load_credentials(file_path):
    """Load credentials from a JSON file."""
    if not os.path.exists(file_path):
        return []
    with open(file_path, 'r') as file:
        try:
            return json.load(file)
        except json.JSONDecodeError:
            return []

def save_credentials(file_path, credentials):
    """Save credentials to a JSON file."""
    with open(file_path, 'w') as file:
        json.dump(credentials, file, indent=4)

def hash_password(plain_password):
    """Hash a plaintext password."""
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(plain_password.encode(), salt)
    return hashed.decode()

def verify_password(plain_password, hashed_password):
    """Verify a plaintext password against a hashed password."""
    return bcrypt.checkpw(plain_password.encode(), hashed_password.encode())

def app():
    st.title("Account")

    # Select between Login, Register, and Admin Login
    choice = st.selectbox('Select Option', ['Login', 'Register', 'Admin Login'])

    if choice == 'Login':
        email = st.text_input("Email Address:", key="login_email")
        password = st.text_input("Password:", type='password', key="login_password")

        if st.button("Login"):
            if not email or not password:
                st.error("Please enter both email and password.")
            else:
                # Load patient credentials
                patients = load_credentials(PATIENT_CREDENTIALS_FILE)
                user = next((user for user in patients if user['email'] == email), None)
                if user and verify_password(password, user['password']):
                    st.success("Logged in successfully as Patient!")
                    # Update session state
                    st.session_state['logged_in'] = True
                    st.session_state['role'] = 'patient'
                    st.session_state['selected'] = 'Dashboard'
                    st.session_state['patient_id'] = email  # Store email as patient_id
                    st.rerun()  # Corrected from st.rerun() to st.experimental_rerun()
                else:
                    st.error("Invalid email or password.")

    elif choice == 'Register':
        email = st.text_input("Email Address:", key="register_email")
        confirm_email = st.text_input("Confirm Email Address:", key="register_confirm_email")
        password = st.text_input("Password:", type='password', key="register_password")
        confirm_password = st.text_input("Confirm Password:", type='password', key="register_confirm_password")

        if st.button("Register"):
            if not email or not confirm_email or not password or not confirm_password:
                st.error("Please fill out all fields.")
            elif email != confirm_email:
                st.error("Email addresses do not match.")
            elif password != confirm_password:
                st.error("Passwords do not match.")
            else:
                # Load existing patient credentials
                patients = load_credentials(PATIENT_CREDENTIALS_FILE)
                if any(user['email'] == email for user in patients):
                    st.error("Email already registered. Please login.")
                else:
                    # Hash the password and save the new user
                    hashed_password = hash_password(password)
                    new_user = {
                        "email": email,
                        "password": hashed_password
                    }
                    patients.append(new_user)
                    save_credentials(PATIENT_CREDENTIALS_FILE, patients)
                    st.success("Registration successful! You are now logged in.")
                    # Update session state
                    st.session_state['logged_in'] = True
                    st.session_state['role'] = 'patient'
                    st.session_state['selected'] = 'Dashboard'
                    st.session_state['patient_id'] = email  # Store email as patient_id
                    st.rerun()  # Corrected from st.rerun() to st.experimental_rerun()

    else:  # Admin Login
        admin_id = st.text_input("Admin ID:", key="admin_id")
        password = st.text_input("Password:", type='password', key="admin_password")

        if st.button("Admin Login"):
            if not admin_id or not password:
                st.error("Please enter both Admin ID and password.")
            else:
                # Load admin credentials
                admins = load_credentials(ADMIN_CREDENTIALS_FILE)
                admin = next((admin for admin in admins if admin['admin_id'] == admin_id), None)
                if admin and verify_password(password, admin['password']):
                    st.success("Admin logged in successfully!")
                    # Update session state
                    st.session_state['logged_in'] = True
                    st.session_state['role'] = 'admin'
                    st.session_state['selected'] = 'Admin Dashboard'
                    # st.session_state['admin_id'] = admin_id  # Optional: Store admin_id if needed
                    st.rerun()  # Corrected from st.rerun() to st.experimental_rerun()
                else:
                    st.error("Invalid Admin ID or Password.")
