import streamlit as st
from train import train
### fonction vpour la page d'accuei

def home_page():
    st.title('Bienvenue sur TrainingApp')
    st.write('''Cette application vous permet de rendre utile vos données 
             en choissisant un modèle d'apprentissage automatique approprié et l'entrainer.''')