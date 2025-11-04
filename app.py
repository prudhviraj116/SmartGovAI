import streamlit as st
import pandas as pd, numpy as np
import plotly.express as px
import os, json

from utils.data_cleaning import basic_clean, aggregate_counts
from models.predictor import SimpleTrendPredictor
from utils.prioritizer import compute_urgency

# --- OpenAI SDK ---
from openai import OpenAI

# --- Vertex AI Gemini Integration ---
from vertexai.preview.generative_models import GenerativeModel
import vertexai

# --- Initialize Streamlit page ---
st.set_page_config(page_title="SmartGovAI - Predictive Governance", layout="wide")
st.title("SmartGovAI — Predictive Citizen Service Dashboard (Full)")

st.markdown("""
Upload citizen complaint data (CSV).  
The app will visualize trends, train predictive models, and produce AI (Gemini/OpenAI) summaries.
""")

# --- Gemini summary function ---
def generate_ai_summary(df):
    try:
        vertexai.init(project="smartgovai-gcp", location="us-central1")

        model = GenerativeModel("gemini-1.5-flash")

        prompt = f"""
        You are analyzing a citizen service dataset for the Government of Maharashtra.
        Provide a concise AI-generated summary highlighting:
        1. Top complaint categories
        2. City with highest issues
        3. Percentage of high priority complaints
        4. Suggested improvements
        Dataset Preview:
        {df.head(10).to_string()}
        """

        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error: {str(e)}"

# --- Sidebar configuration ---
st.sidebar.header("Configuration")
freq = st.sidebar.selectbox("Aggregation frequency", options=['W','D','M'], index=0, 
                            format_func=lambda x: {'W':'Weekly','D':'Daily','M':'Monthly'}[x])
use_ai = st.sidebar.checkbox("Enable AI summaries (OpenAI/Gemini)", value=True)
use_vertex_gemini = st.sidebar.checkbox("Use Vertex Gemini (if available)", value=False)
show_admin = st.sidebar.checkbox("Show Admin (Prioritization weights) UI", value=False)

# --- File upload ---
uploaded_file = st.file_uploader("📁 Upload CSV file", type=['csv'])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df = basic_clean(df)
    st.success("✅ Data loaded and cleaned.")
    st.subheader("Dataset preview (anonymized)")
    st.dataframe(df.head(50))

    # --- Validation ---
    if 'date' not in df.columns or df['date'].isna().all():
        st.warning("No valid 'date' column found. Please include 'date' column parseable by pandas.")
    else:
        # --- Aggregation ---
        agg = aggregate_counts(df, freq=freq, date_col='date')
        st.subheader("Aggregated counts (sample)")
        st.dataframe(agg.sort_values('period_start').head(50))

        # --- Visualizations ---
        st.subheader("📊 Visualizations")

        # Time-series visualization for top categories
        top_cats = agg.groupby('category')['count'].sum().sort_values(ascending=False).head(5).index.tolist()
        if top_cats:
            ts = agg[agg['category'].isin(top_cats)].pivot_table(index='period_start', columns='category', values='count', aggfunc='sum').fillna(0)
            fig_ts = px.line(ts, x=ts.index, y=ts.columns, labels={'value':'count','period_start':'Period'}, title="Time Series of Top Categories")
            st.plotly_chart(fig_ts, use_container_width=True)

        # Bar chart by region
        region_totals = agg.groupby('region')['count'].sum().reset_index().sort_values('count', ascending=False)
        fig_reg = px.bar(region_totals, x='region', y='count', title="Total Complaints by Region")
        st.plotly_chart(fig_reg, use_container_width=True)

        # --- Predictive Model ---
        st.subheader("🔮 Predictive Model (Next Period Forecast)")
        predictor = SimpleTrendPredictor()
        predictor.fit(agg)
        preds = predictor.predict_next_period(agg)
        st.dataframe(preds.sort_values('predicted_count', ascending=False).head(20))

        # Compare recent data with predictions
        recent = agg.sort_values('period_start').groupby(['region','category']).tail(1).rename(columns={'count':'last_count'})
        combined = recent.merge(preds, on=['region','category'])
        combined['delta'] = combined['predicted_count'] - combined['last_count']
        st.markdown("**Top predicted increases (next period)**")
        st.dataframe(combined.sort_values('delta', ascending=False).head(15))

        fig_pred = px.bar(combined.sort_values('delta', ascending=False).head(10),
                          x='region', y='delta', color='category', title="Top Increases Predicted (Next Period)")
        st.plotly_chart(fig_pred, use_container_width=True)

        # --- Admin weights for prioritization ---
        if show_admin:
            st.sidebar.markdown("### Prioritization Weights (Admin)")
            a = st.sidebar.slider("alpha (risk weight)", 0.0, 1.0, 0.5)
            b = st.sidebar.slider("beta (predicted increase weight)", 0.0, 1.0, 0.3)
            c = st.sidebar.slider("gamma (resource availability weight)", 0.0, 1.0, 0.2)
            weights = (a,b,c)
            st.sidebar.write("Current weights:", weights)
        else:
            weights = (0.5,0.3,0.2)

        # --- Compute urgency ---
        combined = combined.reset_index(drop=True)
        combined['risk_score'] = np.clip(combined['delta'] / (combined['last_count']+1), 0, 1)
        combined['resource_availability'] = np.random.rand(len(combined))
        combined['urgency_score'] = combined.apply(
            lambda r: compute_urgency(r['risk_score'], r['delta']/ (r['last_count']+1), r['resource_availability'], weights),
            axis=1
        )

        st.subheader("🚨 Triage & Prioritization (Sample)")
        st.dataframe(
            combined[['region','category','last_count','predicted_count','delta',
                      'risk_score','resource_availability','urgency_score']]
            .sort_values('urgency_score', ascending=False)
            .head(20)
        )

        # --- AI Summaries ---
        if use_ai:
            st.subheader("🧠 AI-Generated Executive Summary")
            if use_vertex_gemini:
                st.info("Using Vertex Gemini model for summary (Gemini-1.5-flash)")
                summary = generate_ai_summary(df)
                st.write(summary)
            else:
                openai_key = st.secrets.get("OPENAI_API_KEY", None) if hasattr(st, "secrets") else os.getenv("OPENAI_API_KEY")
                if openai_key:
                    client = OpenAI(api_key=openai_key)
                    sample_rows = df[['date','region','category','description_anonymized']].head(10).to_string(index=False)
                    prompt = f"You are an assistant. Produce a concise executive summary (4-6 sentences) for municipal decision-makers based on these sample records:\n\n{sample_rows}"
                    try:
                        response = client.chat.completions.create(
                            model="gpt-4o-mini",  # Or gpt-3.5-turbo if preferred
                            messages=[{"role": "user", "content": prompt}],
                            max_tokens=200
                        )
                        st.info(response.choices[0].message.content)
                    except Exception as e:
                        st.error(f"OpenAI error: {e}")
                else:
                    st.warning("No OpenAI key found. Set OPENAI_API_KEY in Streamlit secrets or environment to enable AI summaries.")
        else:
            st.info("AI summaries disabled.")

        # --- Export predictions ---
        csv = preds.to_csv(index=False)
        st.download_button("⬇️ Download predictions CSV", data=csv, file_name="smartgovai_predictions.csv", mime="text/csv")
        # After computing `combined` or `preds`
        combined.to_csv("shared_data/latest_predictions.csv", index=False)

else:
    st.info("Upload a CSV to begin. Required columns: 'date' (YYYY-MM-DD), 'category', 'region', 'description'")
