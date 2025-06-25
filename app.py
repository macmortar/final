import streamlit as st
import joblib
import numpy as np

st.title("Predição da Nota do Aluno")
st.write("Preencha os dados abaixo para prever a nota final do aluno (exam_score).")

# Carregar modelo
model = joblib.load("model.pkl")

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

# One-hot encoding manual para parental_education_level
parental_levels = {
    "High School": [0, 0, 0, 0],
    "Bachelor's": [1, 0, 0, 0],
    "Master's": [0, 1, 0, 0],
    "PhD": [0, 0, 1, 0],
    "No Formal Education": [0, 0, 0, 1]
}

parental_encoded = parental_levels[parental_edu]

# Variáveis derivadas
tempo_tela = social_media_hours + netflix_hours
razao_tempotela_estudos = study_hours / tempo_tela if tempo_tela != 0 else 0
life_style = exercise_frequency + mental_health_rating

# Montar vetor de entrada (ordem deve bater com a do modelo treinado)
input_data = np.array([[
    study_hours,
    social_media_hours,
    netflix_hours,
    sleep_hours,
    exercise_frequency,
    mental_health_rating,
    tempo_tela,
    razao_tempotela_estudos,
    life_style,
    *parental_encoded  # descompacta os valores
]])

# Predição
if st.button("Prever Nota"):
    pred = model.predict(input_data)
    st.success(f"Nota prevista no exame: {pred[0]:.2f}")