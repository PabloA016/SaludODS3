# Importar librerías necearias
import numpy as np
import streamlit as st
import pandas as pd

# Insertamos título
st.write(''' # ODS 3: Salud y Bienestar ''')
# Insertamos texto con formato
st.markdown("""
Esta aplicación utiliza **Machine Learning** para predecir la **esperanza de vida**
a partir del **gasto en salud per cápita**, alineado con el **ODS 3: Salud y Bienestar**.
""")

st.image("salud.jpg", caption="Mejorando la salud global.")


# Definimos cómo ingresará los datos el usuario
# Usaremos un deslizador
st.sidebar.header("Parámetros de Entrada")
# Definimos los parámetros de nuestro deslizador:
  # Límite inferior y superior basado en el rango de gasto_salud_usd del dataset
  # Valor inicial: un valor representativo, por ejemplo, la media o un valor central.
health_expenditure_input = st.sidebar.slider("Gasto en salud per cápita (USD)", 0.0, 15000.0, 2500.0)

# Cargamos el archivo con los datos (.csv)
df =  pd.read_csv('datos_ods3_salud.csv')
# Seleccionamos las variables
X = df[['gasto_salud_usd']]
y = df['esperanza_vida_años']

# Creamos y entrenamos el modelo
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=1613492)
LR = LinearRegression()
LR.fit(X_train,y_train)

# Hacemos la predicción con el modelo y el gasto en salud seleccionado por el usuario
b1 = LR.coef_
b0 = LR.intercept_
prediccion = b0 + b1[0]*health_expenditure_input

# Presentamos los resultados
st.subheader('Esperanza de vida predicha')
st.write(f'La esperanza de vida es: {prediccion:.2f} años')

if prediccion < 65:
        st.error("Clasificación: Baja")
elif prediccion < 75:
        st.warning("Clasificación: Moderada")
else:
        st.success("Clasificación: Alta")
