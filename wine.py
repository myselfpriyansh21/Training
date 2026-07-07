import streamlit as st
st.title("Wine Quality Checker")
st.write('''This app is designed to check the quality of a given wine. The user can input any number of wines, and this will be used in order to calculate the average score for each wine.''')
fixed_acidity = st.number_input("Enter the fixed acidity of the wine:", min_value=0.0, max_value=20.0, step=0.1)
volatile_acidity = st.number_input("Enter the volatile acidity of the wine:", min_value=0.0, max_value=5.0, step=0.1)   
citric_acidity = st.number_input("Enter the citric acidity of the wine:", min_value=0.0, max_value=1.0, step=0.1)
pH = st.number_input('Enter the pH value for this wine:', min_value=-5.0, max_value=10.0, step=0.01)
residual_sugar = st.number_input('Enter the residual sugar value for this wine:', min_value=0.0, max_value=15.0, step=0.1)
chlorides = st.number_input('Enter the chlorides value for this wine:', min_value=0.0, max_value=1.0, step=0.01)
free_sulfur_dioxide = st.number_input('Enter the free sulfur dioxide value for this wine:', min_value=0.0, max_value=72.0, step=1.0)
total_sulfur_dioxide = st.number_input('Enter the total sulfur dioxide value for this wine:', min_value=0.0, max_value=289.0, step=1.0)
density = st.number_input('Enter the density value for this wine:', min_value=0.0, max_value=1.5, step=0.001)
sulphates = st.number_input('Enter the sulphates value for this wine:', min_value=0.0, max_value=2.0, step=0.01)
alcohol = st.number_input('Enter the alcohol value for this wine:', min_value=0.0, max_value=15.0, step=0.1)

import tensorflow as tf
import numpy as np
model = tf.keras.models.load_model('wine_quality_model.h5')
input=[[fixed_acidity, volatile_acidity, citric_acidity, pH, residual_sugar, chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density, sulphates, alcohol]]
prediction = model.predict(np.array(input).reshape(1, -1))
if st.button("Predict Quality"):
    pred=np.argmax(prediction+3)
    st.write(f"The predicted quality of the wine is: {pred}")