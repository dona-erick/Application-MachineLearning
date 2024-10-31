import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
import streamlit as st
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, precision_score, confusion_matrix, classification_report, auc
from sklearn.model_selection import train_test_split, GridSearchCV, GroupKFold
from sklearn.linear_model import LogisticRegression,LinearRegression, Ridge, Lasso
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier, 
                              HistGradientBoostingClassifier, BaggingClassifier, VotingClassifier, ExtraTreesClassifier, HistGradientBoostingRegressor, BaggingRegressor,
                              GradientBoostingRegressor, RandomForestRegressor, VotingRegressor, ExtraTreesRegressor)
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import (DecisionTreeClassifier,ExtraTreeClassifier, DecisionTreeRegressor, ExtraTreeRegressor)
from xgboost import XGBClassifier, XGBRegressor, XGBRFClassifier, XGBRFRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from imblearn.over_sampling import SMOTE



def train():
### title of application

    st.title('Application de Machine Learning Interactif')

    st.markdown("""Cette application est construite dans le but de vous aider à 
                réaliser vos modèles d'apprentissage automatiques plus 
                facilement sans écrire des lignes de codes. """)

    ### chargement des données
    def loading_data(files):
        if files:
            data = pd.read_csv(files)
            return data
        elif files:
            data = pd.read_xlsx(files)
            return data
    @st.cache_data(persist=True)  

    ### definir une fonction de prétraitement
    def preprocess_data(data, target, features, balance_data):
        
        ### gerer les valeurs manquantes
        if data.isnull().sum().sum() > 0:
            # st.write("Il y a des valeurs manquantes. Remplissage en cours...")
            # Remplir les valeurs manquantes pour les colonnes numériques avec la médiane
            numerical_features = data.select_dtypes(include=['int64', 'float64']).columns
            data[numerical_features] = data[numerical_features].fillna(data[numerical_features].median())
            
            # Remplir les valeurs manquantes pour les colonnes catégorielles avec le mode
            categorical_features = data.select_dtypes(include=['object']).columns
            data[categorical_features] = data[categorical_features].apply(lambda x: x.fillna(x.mode()[0]))
            
            st.write(data.head())
        else:
            st.write("Pas de valeurs manquantes.")
            
        cat_features = [col for col in data[features] if data[col].dtype == 'object']
        if cat_features:
            st.write("Encodage des variables catégorielles en cours...")
            data = pd.get_dummies(data, columns=cat_features, drop_first=True)
        
        ###  encoder la  colonne cible si elle est catégorielle
        if data[target].dtype == 'object':
            label = LabelEncoder()
            data[target] = label.fit_transform(data[target])
            
        ###  Vérifier et traiter les colonnes de type datetime ###
        date_features = [col for col in data.columns if pd.api.types.is_datetime64_any_dtype(data[col])]
        if date_features:
            st.write("Traitement des colonnes de date en cours...")
            # Extraire les caractéristiques de date pertinentes (année, mois, jour)
            for col in date_features:
                data[col + '_year'] = data[col].dt.year
                data[col + '_month'] = data[col].dt.month
                data[col + '_day'] = data[col].dt.day
            # Supprimer les colonnes d'origine de type datetime
            data = data.drop(columns=date_features)
            features = [col for col in features if col not in date_features]
        
        #if data[features].dtype == 'datetime':
        #if data is None or target is None or not features:
         #   st.error("Les données, la cible, ou les caractéristiques ne sont pas définies.")
          #  return None, None
        
        ### definir les features et les targets
        X = data[features]
        y = data[target]
        
         # Vérifier que X n’est pas vide avant le scaling
        if X.empty or y.empty:
            st.error("Les données de caractéristiques ou la cible sont vides après le prétraitement.")
            return None, None
        ### normaliser les données
        X_scaler = StandardScaler().fit_transform(X)
        
        ### verifier si les classes sont équilibrées
        equilibre = y.value_counts(normalize= True)
        if equilibre.min() <0.4 and balance_data:
            smote  = SMOTE()
            X, y = smote.fit_resample(X, y)
        return X, y

    ### modeles
    def get_model(model_name, params):
        if model_name == 'BaggingClassifier':
            model = BaggingClassifier(**params)
        elif model_name == "KNeighborsClassifier":
            model = KNeighborsClassifier(**params)
        elif model_name == "DecisionTreeClassifier":
            model = DecisionTreeClassifier(**params)
        elif model_name == "LogisticRegression":
            model = LogisticRegression(**params)
        elif model_name == "RandomForestClassifier":
            model = RandomForestClassifier(**params)
        elif model_name == 'Linear regression':
            model = LinearRegression(**params)
        if model_name == 'BaggingRegressor':
            model = BaggingRegressor(**params)
        elif model_name == "Voting":
            model = VotingClassifier(**params)
        elif model_name == "Votingregessor":
            model = VotingRegressor(**params)
        elif model_name == "DecisionTreeregressor":
            model = DecisionTreeRegressor(**params)
        elif model_name == "Lasso":
            model = Lasso(**params)
        elif model_name == "RandomForestRegressor":
            model = RandomForestRegressor(**params)
        elif model_name == 'ridge':
            model = Ridge(**params)
        return model
            
    ### chargement de fichier 
    files = st.sidebar.file_uploader('Download your files', type=["csv", "xlsx"], key="csv", accept_multiple_files=True)
    if files is not None:
        data = loading_data(files)
        st.write('Aperçu des données:', data.sample(5))
        st.write('Statistique des données:', data.describe())
        
        
        ### selectionner la colonne cible
        target = st.sidebar.selectbox("Choose the target columns of your dataset:", data.columns)
        ### selectionner les caractéristiques 
        features = st.sidebar.multiselect("Choose the features of dataframe:", data.columns.drop(target))
        ### appliquer le surechantillonage
        balance_data = st.sidebar.checkbox('Appliquer le suréchantillonage si la classe est déséquilibrée')
        

        
        ### selection des modèles de machine learning
        model_name = st.sidebar.selectbox("Define your model machine learning ",(
            "LogisticRegression",
            "RandomForestClassifier",
            'DecisionTreeClassifier',
            'BaggingClassifier',
            'KNeighborsClassifier',
            "Linear regression"
        ) )
        ### stockers les paramètres du modèle
        params = {}
        if model_name == "LogisticRegression":
            params["C"] = st.sidebar.slider("C", 0.01, 1.0)
            #params["fit_intercept"] = st.sidebar.selectbox("fit_intercept", ('True', 'False'))
            params["penalty"] = st.sidebar.selectbox('penalty', ("l2", "elasticnet"))
            params["solver"] = st.sidebar.selectbox('solver', ('lbfgs', "liblinear", "saga"))
            
        elif model_name == "RandomForestClassifier":
            params['n_estimators'] = st.sidebar.slider('n_estimators', 100, 500)
            params['max_depth'] = st.sidebar.slider('max_depth', 2, 10)
            params["criterion"] = st.sidebar.selectbox('criterion', ('gini', 'log_loss'))
        elif model_name == "BaggingClassifier":
            params["n_estimators"] = st.sidebar.slider('n_estimators', 1, 10)
            params['max_features'] =st.sidebar.slider('max_features', 0.0, 1.0)
        elif model_name == "DecisionTreeClassifier":
            params['criterion'] = st.sidebar.selectbox('critère', ('gini', 'log_loss'))
            params['max_depth'] = st.sidebar.slider("max_depth", 1, 10)
            params['max_features'] = st.sidebar.slider('max_features', 0.0, 1.0)
            
        elif model_name == 'KNeighborsClassifier':
            params["n_neighbors"] = st.sidebar.slider('Nombre de voisins', 1, 10)
            params['algorithm'] = st.sidebar.selectbox('Algorithm', ('auto', 'ball_tree', 'kde_tree', "brute"))
        elif model_name == 'Linear regression':
            params['n_jobs'] = st.sidebar.slider('n_jobs', -1, 1)
    ### division les données
        X,y = preprocess_data(data, target, features, balance_data)
        Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.3, random_state=42)
        
        
        ### entrainer le modele
        model = get_model(model_name, params)
        model.fit(Xtrain, ytrain)
        ypred = model.predict(Xtest)
        if model_name in ["LogisticRegression","RandomForestClassifier",'DecisionTreeClassifier','BaggingClassifier','KNeighborsClassifier']:
            Accuracy = accuracy_score(ypred, ytest, average = 'micro')
            Precision = precision_score(ypred, ytest)
        
            st.write(f"\n Les performances du modele sont: \n {Accuracy} et le score de precison : {Precision}")
        
        # Courbe ROC
            st.subheader("Courbe ROC")
            y_proba = model.predict_proba(Xtest)[:, 1]
            fpr, tpr, _ = roc_curve(ytest, y_proba)
            roc_auc = auc(fpr, tpr)
            plt.figure()
            plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlabel('Taux de faux positifs')
            plt.ylabel('Taux de vrais positifs')
            plt.title('Courbe ROC')
            plt.legend(loc="lower right")
            st.pyplot(plt)

        # Matrice de confusion
            st.subheader("Matrice de confusion")
            conf_matrix = confusion_matrix(ytest, ypred)
            sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
            plt.xlabel("Classe prédite")
            plt.ylabel("Classe réelle")
            st.pyplot(plt)
        elif model_name in ['Linear regression', "Ridge"]:
            RMAE= mean_squared_error(ypred, ytest)
            MAE  = mean_absolute_error(ypred, ytest)
            st.write(f"La valeurs carre moyenne absolue est : {RMAE} et la MAE est : {MAE}")

    # """faire la prediction"""

    #st.markdown(' **** Faire la prediction ****')
        
    #files_2 = st.sidebar.file_uploader('Download your files for the prediction', type=['csv', "xlsx"], key='uploader_csv')
    #if files_2 is not None:
    # df = loading_data(files)
        #st.write('Aperçu des données:', df.head(5))
    #   
    #  target = data[target]
        
        #features = [col for col in df.columns if col != target]
    #  features = st.sidebar.multiselect("Choose the features of dataframe:", df.columns.drop(target))
    # balance_data = st.sidebar.checkbox("Equilibrer les classes", value = False)
        #if st.button('Predire'):
            """prediction"""
        #   data_test, _ = preprocess_data(df, target, features, balance_data)
        #  prediction = model.predict(data_test)
            
        #  st.write("La prediction est:", prediction[0])


    st.markdown('Réalisé par :')
    st.subheader("Eric KOULODJI")