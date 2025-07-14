import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.title("🔥 Fire Weather Index (FWI) Prediction")

st.header("📥 Input Weather and Region Features")

## Load Models 
with open('scaler.pkl' , 'rb') as file1:
    scaler_model = pickle.load(file1)

with open('ridge.pkl' , 'rb') as file2:
    ridge_model = pickle.load(file2)

## Take Inputs
temp = st.number_input("temperature ", step = 1, min_value= 22 , max_value= 42)

RH =  st.number_input("Relative Humditiy ", step = 1 , min_value= 21 , max_value = 90)

ws = st.number_input("Wind speed in km/h ", step = 1 , min_value=6 , max_value=29)

rain = st.number_input('Rain: total day in mm' , step = 0.1 , min_value=0.0, max_value=16.8)
st.write(' Fire Weather (FWI) Components')
FFMC = st.number_input('Fine Fuel Moisture Code (FFMC) index from the FWI system', step = 0.1, min_value=28.6 , max_value=92.5)

DMC = st.number_input('Duff Moisture Code (DMC) index from the FWI system ', step = 0.1 , min_value=1.1 , max_value=65.9)

ISI = st.number_input('Initial Spread Index (ISI) index from the FWI system ' , step = 0.1 , min_value=0.0 , max_value=18.5)

classes = st.selectbox('Class , fire (1) or not fire (0)', [1 , 0] )
region = st.selectbox('Region' , ['Bejaia Region' , 'Sidi-Bel Abbes Region'])

region_encoded = 1 if region == 'Bejaia Region' else 0


## Convert features list to nd array
features = np.array([[temp , RH , ws , rain , FFMC , DMC , ISI , classes , region_encoded]])

## Prediction
if st.button("Predict Fire Weather Index(FWI)"):
    ## Step 1 : Standard Scaling 
    new_data_scaled_features = scaler_model.transform(features)

    # Step 2: Predict using ridge model
    prediction = ridge_model.predict(new_data_scaled_features)

    st.success(f"🌡️ Predicted Fire Weather Index (FWI): **{prediction[0]:.2f}**")


