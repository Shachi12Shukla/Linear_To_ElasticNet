import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.title("🔥 Fire Weather Index (FWI) Prediction")

st.header("📥 Input Weather and Region Features")

## Load Models 
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('ridge.pkl', 'rb') as f:
    ridge = pickle.load(f)

## Take Inputs
temp = st.number_input("Enter temperature [in range 22-42] (degree Celsius) ", step = 1, min_value= 22 , max_value= 42)

RH =  st.number_input("Enter Relative Humditiy [in range 21-90] (%)", step = 1 , min_value= 21 , max_value = 90)

ws = st.number_input("Enter Wind speed [in range 6-29] (km/h) ", step = 1 , min_value=6 , max_value=29)

rain = st.number_input('Rain: total day [in range 0.0-16.8] (mm)' , step = 0.1 , min_value=0.0, max_value=16.8)
st.write(' Fire Weather (FWI) Components')
FFMC = st.number_input('Fine Fuel Moisture Code (FFMC) index from the FWI system [in range 28.6-92.5] ', step = 0.1, min_value=28.6 , max_value=92.5)

DMC = st.number_input('Duff Moisture Code (DMC) index from the FWI system [in range 1.1-65.9]', step = 0.1 , min_value=1.1 , max_value=65.9)

ISI = st.number_input('Initial Spread Index (ISI) index from the FWI system [in range 0.0-18.5]' , step = 0.1 , min_value=0.0 , max_value=18.5)

classes = st.selectbox('Class , 1 for Fire or 0 for Not Fire', [1 , 0] )
region = st.selectbox('Region' , ['Bejaia Region' , 'Sidi-Bel Abbes Region'])

region_encoded = 1 if region == 'Bejaia Region' else 0


## Convert features list to nd array
features = np.array([[temp , RH , ws , rain , FFMC , DMC , ISI , classes , region_encoded]])

## Prediction

if st.button("Predict Fire Weather Index(FWI)"):
    ## Step 1 : Standard Scaling 
    new_features_transformed = scaler.transform(features)

    # Step 2: Predict using ridge model
    prediction = ridge.predict(new_features_transformed)

    st.success(f"🌡️ Predicted Fire Weather Index (FWI): **{prediction}**")


