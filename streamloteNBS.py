import streamlit as st
import joblib 
import pandas as pd
import numpy as np



st.title("Expresso Churn Prediction")
data={
    "Age":["K > 24 month","I 18-21 month","H 15-18 month","G 12-15 month", "J 21-24 month","F 9-12 month", "E 6-9 month", "D 3-6 month"],
    "label":[7,5,4,3,6,2,1,0]
}

df = pd.DataFrame(data)
df2 = pd.read_csv("VariableDefinitions.csv")
st.write("Don't forget : ")
st.table(df)
st.dataframe(df2)

with st.expander("Please fill in the following information", expanded=True):
    st.error("value required float or double")
    MONTANT = st.text_input("MONTANT")
    FREQUENCE_RECH = st.text_input("FREQUENCE_RECH")
    REVENUE = st.text_input("REVENUE")
    ARPU_SEGMENT = st.text_input("ARPU_SEGMENT")
    FREQUENCE = st.text_input("FREQUENCE")
    DATA_VOLUME = st.text_input("DATA_VOLUME")
    ON_NET = st.text_input("ON_NET")
    ORANGE = st.text_input("ORANGE")
    TIGO = st.text_input("TIGO")
    ZONE1 = st.text_input("ZONE1")
    ZONE2 = st.text_input("ZONE2")
    REGULARITY = st.text_input("REGULARITY")
    FREQ_TOP_PACK = st.text_input("FREQ_TOP_PACK")
    TENURE2 = st.text_input("TENURE2")


if st.button("Predict"):
    if MONTANT:
        value=[MONTANT,FREQUENCE_RECH,REVENUE,ARPU_SEGMENT,
            FREQUENCE,DATA_VOLUME,ON_NET,ORANGE,TIGO,
            ZONE1,ZONE2,REGULARITY,FREQ_TOP_PACK,TENURE2]
        array=[float(i) for i in value]
        X= np.array(array).reshape(1,-1)
        model = joblib.load("Expresso_churn.pkl")
        y=model.predict(X)
        st.write(f"Predict value = {y}")
        st.success("Form successfully submitted!")
    else:
        st.error("Please fill in all required fields.")