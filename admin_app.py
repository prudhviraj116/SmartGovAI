
import streamlit as st
import pandas as pd, os

st.set_page_config(page_title="SmartGovAI Admin", layout="wide")
st.title("SmartGovAI — Admin: Prioritization & Audit")

st.markdown("Tune weights for prioritization and upload sample audit logs to view.")

a = st.slider("alpha (risk weight)", 0.0, 1.0, 0.5)
b = st.slider("beta (predicted increase weight)", 0.0, 1.0, 0.3)
c = st.slider("gamma (resource avail weight)", 0.0, 1.0, 0.2)
st.write("Current weights:", (a,b,c))

uploaded = st.file_uploader("Upload audit CSV (optional)", type=['csv'])
if uploaded:
    df = pd.read_csv(uploaded)
    st.dataframe(df.head(200))
else:
    st.info("No audit log uploaded. You can create audit logs from the main app export.")
