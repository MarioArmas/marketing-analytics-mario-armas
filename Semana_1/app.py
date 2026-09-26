import streamlit as st
from sklearn.linear_model import LinearRegression
import numpy as np

st.title("Configuracion inicial")
st.write("Primera prueba de uso de streamlit y ambiente de MA2026")

gasto=st.slider("Seleccine nivel de gasto en publicicdad", 10,200,50)

variable_x = np.array([[10], [20], [30], [40],[50]])
variable_y = np.array([15,25,35,45,55])

modelo_lr = LinearRegression()
modelo_lr.fit(variable_x,variable_y)

if st.button("Predecir"):
  resultado = modelo_lr.predict([[gasto]])
  st.success(f"Las ventas proyectadas para una inversion de ${gasto} son: ${resultado[0]}")