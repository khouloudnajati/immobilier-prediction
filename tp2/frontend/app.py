"""Interface : streamlit run frontend/app.py"""
import pandas as pd
import requests
import streamlit as st

API = "http://127.0.0.1:8000"
st.set_page_config(page_title="Estimation immobilière", page_icon="🏠")
st.title("🏠 Estimation du prix d'un bien (Californie)")

with st.form("form"):
    c1, c2 = st.columns(2)
    MedInc = c1.number_input("Revenu médian (x10 000 $)", 0.1, 20.0, 4.0, 0.1)
    HouseAge = c2.number_input("Âge médian des logements", 0.0, 100.0, 25.0)
    AveRooms = c1.number_input("Nombre moyen de pièces", 0.5, 50.0, 5.0, 0.1)
    AveBedrms = c2.number_input("Nombre moyen de chambres", 0.5, 20.0, 1.0, 0.1)
    Population = c1.number_input("Population du quartier", 1.0, 50000.0, 1200.0)
    AveOccup = c2.number_input("Occupation moyenne", 0.5, 50.0, 3.0, 0.1)
    Latitude = c1.number_input("Latitude", 32.0, 42.0, 34.05, 0.01)
    Longitude = c2.number_input("Longitude", -125.0, -114.0, -118.25, 0.01)
    submitted = st.form_submit_button("Estimer le prix")

if submitted:
    payload = dict(MedInc=MedInc, HouseAge=HouseAge, AveRooms=AveRooms, AveBedrms=AveBedrms,
                   Population=Population, AveOccup=AveOccup, Latitude=Latitude, Longitude=Longitude)
    try:
        r = requests.post(f"{API}/predict", json=payload, timeout=10)
        r.raise_for_status()
        st.success(f"Prix estimé : **{r.json()['estimated_price_usd']:,.0f} $**")
    except Exception as e:
        st.error(f"Erreur API : {e}")

st.subheader("Historique")
try:
    hist = requests.get(f"{API}/history", timeout=10).json()
    if hist:
        st.dataframe(pd.DataFrame([{"date": h["date"], **h["inputs"], "prix ($)": round(h["price"])} for h in hist]))
    else:
        st.info("Aucune estimation pour l'instant.")
except Exception:
    st.warning("API injoignable.")
