import streamlit as st
import joblib
import numpy as np
import pandas as pd

st.title("Predição da Nota do Aluno")

# Carregar modelo
model = joblib.load("model.pkl")

st.subheader("Preencha os dados do aluno:")

# Entradas numéricas
age = st.slider("Idade", 10, 25, 17)
study_hours = st.slider("Horas de estudo por dia", 0.0, 12.0, 2.0)
social_media = st.slider("Horas de redes sociais por dia", 0.0, 12.0, 3.0)
netflix = st.slider("Horas de Netflix por dia", 0.0, 12.0, 2.0)
attendance = st.slider("Frequência de presença (%)", 0, 100, 90)
sleep = st.slider("Horas de sono por dia", 0.0, 12.0, 7.0)
exercise = st.slider("Frequência de exercícios (0-10)", 0, 10, 5)
mental = st.slider("Nota de saúde mental (0-10)", 0, 10, 6)

# Cálculos derivados
tempo_tela = social_media + netflix
razao_estudo_tela = study_hours / tempo_tela if tempo_tela != 0 else 0
life_style = exercise + mental

# Variáveis categóricas
gender = st.selectbox("Gênero", ["Feminino", "Masculino", "Outro"])
job = st.selectbox("Tem trabalho de meio período?", ["Não", "Sim"])
diet = st.selectbox("Qualidade da dieta", ["Ruim", "Normal", "Boa"])
internet = st.selectbox("Qualidade da internet", ["Ruim", "Normal", "Boa"])
extra = st.selectbox("Participa de atividades extracurriculares?", ["Não", "Sim"])

# One-hot encoding manual
gender_male = 1 if gender == "Masculino" else 0
gender_other = 1 if gender == "Outro" else 0
part_time_job_yes = 1 if job == "Sim" else 0
diet_good = 1 if diet == "Boa" else 0
diet_poor = 1 if diet == "Ruim" else 0
internet_good = 1 if internet == "Boa" else 0
internet_poor = 1 if internet == "Ruim" else 0
extra_yes = 1 if extra == "Sim" else 0

# Monta o vetor de entrada
input_data = pd.DataFrame([[
    age,
    study_hours,
    social_media,
    netflix,
    attendance,
    sleep,
    exercise,
    mental,
    tempo_tela,
    razao_estudo_tela,
    life_style,
    gender_male,
    gender_other,
    part_time_job_yes,
    diet_good,
    diet_poor,
    internet_good,
    internet_poor,
    extra_yes
]], columns=[
    "age", "study_hours_per_day", "social_media_hours", "netflix_hours",
    "attendance_percentage", "sleep_hours", "exercise_frequency",
    "mental_health_rating", "tempo_tela", "razao_tempotela_estudos", "life_style",
    "gender_Male", "gender_Other", "part_time_job_Yes",
    "diet_quality_Good", "diet_quality_Poor",
    "internet_quality_Good", "internet_quality_Poor",
    "extracurricular_participation_Yes"
])

# Adiciona colunas extras com 0 se existirem no modelo
# (útil se tiverem 24 variáveis no treino por exemplo)
faltando = set(model.n_features_in_) - set(input_data.columns)
for col in faltando:
    input_data[col] = 0

# Reordena
input_data = input_data[model.feature_names_in_]

# Previsão
if st.button("Prever Nota"):
    pred = model.predict(input_data)
    st.success(f"Nota prevista no exame: {pred[0]:.2f}")
