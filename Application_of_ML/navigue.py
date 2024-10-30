# navigue.py
import streamlit as st
from home import home_page
from link import profile
from train import train

def navigate():

    # Définir la page par défaut si elle n'est pas encore définie


    # Menu de navigation dans la barre latérale
    page = st.sidebar.radio("Navigation", ["Home", "Profil", "Train"])
    
    if page == "Home":
        
        #### code integree
        home_page()
        if st.button('Faire un essai'):
            st.session_state["page"] = "Train"
            train()
    elif page == "Profil":
        st.session_state["page"] = "Profil"

    # Afficher la page appropriée
    if st.session_state["page"] == "Home":
        home_page()
    elif st.session_state["page"] == "Profil":
        profile()
    elif st.session_state["page"] == "Train":
        train()
