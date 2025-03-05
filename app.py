# Importation des bibliothèques nécessaires
import streamlit as st  # Pour créer une interface utilisateur interactive
import numpy as np  # Pour manipuler des tableaux de données numériques
import joblib  # Pour charger le modèle de Machine Learning pré-entraîné

# Titre de l'application
st.title("Prédiction du prix d'une voiture en fonction de ses caractéristiques")

# Sous-titre et description
st.subheader("Application réalisée par Oumar")
st.markdown("*Cette application utilise un modèle de Machine Learning pour prédire le prix d'une voiture*")


# Charger le modele de ML

model = joblib.load("final_model.joblib")


# Définition d'une fonction d'inférence pour effectuer la prédiction
def inference(wheel_base, length, width, curb_weight, engine_size, horsepower, city_mpg, highway_mpg, peak_rpm):
    """
    Cette fonction prend les caractéristiques d'une voiture en entrée 
    et utilise le modèle de Machine Learning pour prédire son prix.
    
    :param wheel_base: Empattement de la voiture
    :param length: Longueur de la voiture
    :param width: Largeur de la voiture
    :param curb_weight: Poids à vide de la voiture
    :param engine_size: Taille du moteur
    :param horsepower: Puissance du moteur en chevaux
    :param city_mpg: Consommation de carburant en ville (miles par gallon)
    :param highway_mpg: Consommation de carburant sur autoroute (miles par gallon)
    :param peak_rpm: Régime maximal du moteur (tours par minute)
    
    :return: Prédiction du prix de la voiture
    """
    
    # Création d'un tableau numpy contenant les caractéristiques fournies par l'utilisateur
    new_data = np.array([wheel_base, length, width, curb_weight, engine_size, 
                         horsepower, city_mpg, highway_mpg, peak_rpm])
    
    # Réorganisation du tableau sous la forme attendue par le modèle (1 ligne, N colonnes)
    pred = model.predict(new_data.reshape(1, -1))
    
    return pred



# Interface utilisateur : saisie des caractéristiques du véhicule
# L'utilisateur peut entrer les valeurs via des champs interactifs (number_input)

wheel_base = st.number_input("wheel_base:", value=90)  # Empattement
length = st.number_input('length:', value=150)  # Longueur de la voiture
width = st.number_input('width:', value=65)  # Largeur de la voiture
curb_weight = st.number_input('curb-weight:', value=200)  # Poids à vide de la voiture
engine_size = st.number_input('engine-size:', value=120)  # Taille du moteur
horsepower = st.number_input('horsepower:', value=110)  # Puissance du moteur
city_mpg = st.number_input('city-mpg:', value=20)  # Consommation en ville (mpg)
highway_mpg = st.number_input('highway-mpg:', value=30)  # Consommation sur autoroute (mpg)
peak_rpm = st.number_input('peak-rpm:', value=5000)  # Régime moteur maximal en tours/minute



# Boutton de prédiction 

if st.button("Predict"):
    # Exécution de la fonction d'inférence avec les valeurs saisies par l'utilisateur
    prediction = inference(wheel_base, length, width, curb_weight, engine_size, 
                           horsepower, city_mpg, highway_mpg, peak_rpm)
    


    resultat = f"Le prix estimé d'une voiture est : {prediction[0]} dollars"

    st.success(resultat)



