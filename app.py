import streamlit as st
import joblib
import numpy as np
import pandas as pd
import json

st.title("Predição da Nota do Aluno")

# Carregar modelo
model = joblib.load("model.pkl")

# Carregar colunas usadas no treino
with open("colunas_modelo.json", "r") as f:
    colunas_modelo = json.load(f)

# Inputs do usuário
study_hours = st.number_input("Horas de estudo por dia", min_value=0.0, max_value=24.0, value=2.0)
social_media_hours = st.number_input("Horas em redes sociais por dia", min_value=0.0, max_value=24.0, value=3.0)
netflix_hours = st.number_input("Horas assistindo Netflix por dia", min_value=0.0, max_value=24.0, value=2.0)
sleep_hours = st.number_input("Horas de sono por dia", min_value=0.0, max_value=24.0, value=7.0)
exercise_frequency = st.slider("Frequência de exercícios físicos (0-10)", 0, 10, 5)
mental_health_rating = st.slider("Nota de saúde mental (0-10)", 0, 10, 6)
parental_edu = st.selectbox("Nível educacional dos pais", [
    "High School", "Bachelor's", "Master's", "PhD", "No Formal Education"
])

# Variáveis derivadas
tempo_tela = social_media_hours + netflix_hours
razao_tempotela_estudos = study_hours / tempo_tela if tempo_tela != 0 else 0
life_style = exercise_frequency + mental_health_rating

# Início do DataFrame com os dados inseridos
dados = {
    "study_hours_per_day": study_hours,
    "social_media_hours": social_media_hours,
    "netflix_hours": netflix_hours,
    "sleep_hours": sleep_hours,
    "exercise_frequency": exercise_frequency,
    "mental_health_rating": mental_health_rating,
    "tempo_tela": tempo_tela,
    "razao_tempotela_estudos": razao_tempotela_estudos,
    "life_style": life_style,
    f"parental_education_level_{parental_edu}": 1  # one-hot encoding manual
}

# Converte para DataFrame
df_input = pd.DataFrame([dados])

# Adiciona as colunas faltantes com 0
for col in colunas_modelo:
    if col not in df_input.columns:
        df_input[col] = 0

# Garante ordem correta
df_input = df_input[colunas_modelo]

# Previsão
if st.button("Prever Nota"):
    pred = model.predict(df_input)
    st.success(f"Nota prevista no exame: {pred[0]:.2f}")