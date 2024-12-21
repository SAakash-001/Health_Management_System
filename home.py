import streamlit as st
import pandas as pd  # for data processing
from PIL import Image  # for image loading

def app():
    # st.markdown("# Home")
    
    # loading images from the directory
    logo = Image.open('Assets/HYTRA logo.png')
    charts = Image.open('Assets/Landing Plots.png')

    pad1, col1, col2, pad2 = st.columns([0.001, 2, 1, 0.2])
    with col1:
        st.image(logo, width=250)
        st.markdown("## WELCOME TO HYTRA: ONE HEALTH DASHBOARD")
        st.markdown("### We are here for all your diagnostics")
        st.markdown(
            "Stay on top of your health with HYTRA, offering comprehensive insights that most other dashboards can't provide. Join us now to experience a new era of health management that is proactive, precise, and perfectly tailored to your needs."
        )
        if st.button("About Us", key="about_us_button", help="Learn more about HYTRA"):
            st.session_state['selected'] = 'About Us'
            st.rerun()
    with col2:
        st.image(charts, width=450)
        # st.markdown("One stop for your all key health metrics.")