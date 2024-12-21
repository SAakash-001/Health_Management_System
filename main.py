# main.py

import streamlit as st
from streamlit_option_menu import option_menu

import home
import about
import account
import patient_app
import admin_app

# Set page configuration.\
st.set_page_config(
    page_title="HYTRA",
    layout="wide",
)

# Initialize session state variables if they don't exist
if 'selected' not in st.session_state:
    st.session_state['selected'] = 'Home'

if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False

if 'role' not in st.session_state:
    st.session_state['role'] = None  # Possible values: 'patient', 'admin'

class HytraApp:
    @staticmethod
    def run():
        # Define menu options based on login state and role
        if not st.session_state['logged_in']:
            menu_options = ['Home', 'About Us', 'Account']
            menu_icons = ['house-fill', 'info-circle', 'person-circle']
        else:
            if st.session_state['role'] == 'patient':
                menu_options = ['Home', 'About Us', 'Dashboard', 'Logout']
                menu_icons = ['house-fill', 'info-circle', 'speedometer2', 'box-arrow-right']
            elif st.session_state['role'] == 'admin':
                menu_options = ['Home', 'About Us', 'Admin Dashboard', 'Logout']
                menu_icons = ['house-fill', 'info-circle', 'shield-lock-fill', 'box-arrow-right']
            else:
                # Fallback in case of undefined role
                menu_options = ['Home', 'About Us', 'Logout']
                menu_icons = ['house-fill', 'info-circle', 'box-arrow-right']

        # Create the sidebar menu
        with st.sidebar:
            selected = option_menu(
                menu_title="HYTRA",
                options=menu_options,
                icons=menu_icons,
                menu_icon='chat-text-fill',
                default_index=menu_options.index(st.session_state['selected']) if st.session_state['selected'] in menu_options else 0,
            )
            # Update the selected page in session state
            if selected != st.session_state['selected']:
                st.session_state['selected'] = selected

        # Render the selected page
        if st.session_state['selected'] == 'Home':
            home.app()
        elif st.session_state['selected'] == 'About Us':
            about.app()
        elif st.session_state['selected'] == 'Account':
            account.app()
        elif st.session_state['selected'] == 'Dashboard' and st.session_state['role'] == 'patient':
            patient_app.app()
        elif st.session_state['selected'] == 'Admin Dashboard' and st.session_state['role'] == 'admin':
            admin_app.app()
        elif st.session_state['selected'] == 'Logout':
            # Perform logout
            st.session_state['logged_in'] = False
            st.session_state['role'] = None
            st.session_state['selected'] = 'Home'
            st.rerun()
        else:
            # Default to Home if no matching condition
            home.app()

HytraApp.run()
