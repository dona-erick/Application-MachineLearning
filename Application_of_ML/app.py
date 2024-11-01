import streamlit as st
from navigue import navigate

# Configuration de la page avec un titre et une disposition large
st.set_page_config(page_title="TrainingApp", layout="wide", initial_sidebar_state="collapsed")

# Appeler la fonction de navigation pour démarrer l'application
if "page" not in st.session_state:
    st.session_state["page"] = "Home"

navigate()
