import streamlit as st
from train import train
### fonction vpour la page d'accuei

def home_page():
    st.title('Bienvenue sur TrainingApp')
    st.markdown(
    """
    <style>
    .main {
        background: linear-gradient(to bottom right, #84fab0, #8fd3f4);
        color: #333333;
        padding: 1em;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.1);
    }
    </style>
        Bienvenue sur TrainingApp
            Transformez vos données en insights puissants grâce à TrainingApp,
            votre plateforme d'apprentissage automatique intuitive !

TrainingApp vous guide pas à pas dans le choix et l'entraînement de modèles d'apprentissage automatique adaptés à vos besoins,
que vous soyez novice en machine learning ou expert souhaitant optimiser vos processus. En quelques clics, vous pouvez explorer, 
prétraiter et modéliser vos données en utilisant des techniques avancées, le tout dans une interface conviviale et accessible.

🔍 Ce que vous pouvez faire avec TrainingApp :

Chargement et prétraitement des données : Importez facilement vos jeux de données, avec des outils pour gérer les valeurs manquantes, encoder les variables catégorielles et équilibrer les classes si nécessaire.
Choix des modèles : Sélectionnez le modèle d'apprentissage machine optimal parmi les options disponibles, ajustez ses hyperparamètres et améliorez la précision des prédictions.
Visualisation des résultats : Obtenez des graphiques clairs comme la courbe ROC, la matrice de confusion, et visualisez les prédictions pour mieux comprendre les performances de votre modèle.
Suivi et export des résultats : Accédez aux métriques clés pour chaque modèle et exportez facilement vos résultats.
Que vous travailliez sur des données de finance, de santé, ou de marketing, TrainingApp est conçu pour vous aider à obtenir rapidement des modèles performants et à prendre des décisions éclairées.

👉 Prêt à transformer vos données en décisions ? Avec TrainingApp, exploitez tout le potentiel de vos données en quelques clics et faites le premier pas vers des analyses puissantes et précises.

</div>
    """,
    unsafe_allow_html=True
)
