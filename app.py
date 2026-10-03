

import numpy as np
import pandas as pd
import joblib
import streamlit as st

# ----------------------------------------------------------------------
# Configuration de la page
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Prédiction de l'indice de performance d'un étudiant",
    page_icon="🚗",
    layout="centered",
)

DESCRIPTION = (
    "Ce modèle de machine learning (Lasso) permet de prédire "
    "l'indice de performance d'un étudiant à partir du nombre d'heures "
    "étudiées, des notes précédentes, de la pratique d'activités "
    "extrascolaires, du nombre d'heures de sommeil et du nombre de "
    "sujets d'examen pratiqués."
)

# ----------------------------------------------------------------------
# Chargement des artefacts (mis en cache : chargés une seule fois)
# ----------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    encoder = joblib.load("encoder.joblib")                # encodeurs
    pipe_from_grid_lasso = joblib.load("pipe_from_grid_lasso.joblib")    # pipeline du best modèle
    return encoder, pipe_from_grid_lasso


encoder, pipe_from_grid_lasso = load_artifacts()

# Fonction de prédiction
def pred_func(hours_studied, previous_scores, extracurricular, sleep_hours, papers_practiced):
    extracurricular_encode = encoder.transform([extracurricular])[0]

    observation = pd.DataFrame([{
        'Hours Studied': hours_studied,
        'Previous Scores': previous_scores,
        'Extracurricular Activities': extracurricular_encode,
        'Sleep Hours': sleep_hours,
        'Sample Question Papers Practiced': papers_practiced
    }])

    prediction = pipe_from_grid_lasso.predict(observation)[0]
    return round(float(prediction), 2)

def pred_func_csv(fichier_csv):
    df_multiple = pd.read_csv(fichier_csv)  # retirer .name

    predictions = []
    for _, ligne in df_multiple.iterrows():
        indice_predit = pred_func(
            ligne['Hours Studied'],
            ligne['Previous Scores'],
            ligne['Extracurricular Activities'],
            ligne['Sleep Hours'],
            ligne['Sample Question Papers Practiced']
        )
        predictions.append(indice_predit)

    df_multiple['Performance_Index_Predit'] = predictions

    return df_multiple

# ----------------------------------------------------------------------
# Interface
# ----------------------------------------------------------------------
st.title("Prédiction de l'indice de performance d'un étudiant")
st.markdown(DESCRIPTION)

onglet_simple, onglet_multiple = st.tabs(["Prédiction simple", "Prédiction multiple (CSV)"])

# --- Onglet 1 : prédiction simple ---
with onglet_simple:
    col1, col2 = st.columns(2)
    with col1:
        hours_studied = st.number_input("Heures étudiées (Hours Studied)", min_value=0.0, step=1.0)
        previous_scores = st.number_input("Notes précédentes (Previous Scores)", min_value=0.0, step=1.0)
        sleep_hours = st.number_input("Heures de sommeil (Sleep Hours)", min_value=0.0, step=0.5)

    with col2:
        extracurricular = st.selectbox(
            "Activités extrascolaires (Extracurricular Activities)",
            options=list(encoder.classes_)
        )
        papers_practiced = st.number_input(
            "Sujets d'examen pratiqués (Sample Question Papers Practiced)",
            min_value=0.0, step=1.0
        )

    if st.button("Prédire", key="bouton_simple"):
        resultat = pred_func(
            hours_studied, previous_scores, extracurricular, sleep_hours, papers_practiced
        )
        st.success(f"Indice de performance prédit : {resultat}")

# --- Onglet 2 : prédiction multiple via CSV ---
with onglet_multiple:
    fichier_entree = st.file_uploader("Déposer un fichier CSV", type=["csv"])

    if fichier_entree is not None:
        if st.button("Prédire pour tout le fichier", key="bouton_multiple"):
            df_resultat = pred_func_csv(fichier_entree)
            st.success(f"{len(df_resultat)} prédiction(s) effectuée(s).")
            st.dataframe(df_resultat)

            csv_telechargement = df_resultat.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Télécharger un fichier CSV",
                data=csv_telechargement,
                file_name="predictions.csv",
                mime="text/csv"
            )