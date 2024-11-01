import streamlit as st
from home import home_page
from link import profile
from train import train

def navigate():
    # Initialiser la page par défaut si elle n'est pas encore définie
    if "page" not in st.session_state:
        st.session_state["page"] = "Home"

    # Menu de navigation dans la barre latérale
    page = st.sidebar.radio("Navigation", ["Home", "Profil", "Train"])

    # Mettre à jour la page actuelle selon la sélection
    st.session_state["page"] = page

    # Afficher la page appropriée
    if st.session_state["page"] == "Home":
        home_page()
        if st.button('Faire un essai'):
            st.session_state["page"] = "Train"
            train()  # Rediriger vers la page de modélisation
    elif st.session_state["page"] == "Profil":
        profile()
    elif st.session_state["page"] == "Train":
        train()
