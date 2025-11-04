from fastapi import FastAPI, UploadFile, File, Body, HTTPException
import pandas as pd
import numpy as np
from fastapi.middleware.cors import CORSMiddleware
from utils.data_cleaning import basic_clean, aggregate_counts
from models.predictor import SimpleTrendPredictor
from utils.prioritizer import compute_urgency
from vertexai.preview.generative_models import GenerativeModel
import vertexai, os
from openai import OpenAI
from pathlib import Path
import json

# ============================================================
# CONFIGURATION
# ============================================================

app = FastAPI(title="SmartGovAI Backend", version="1.0")

# File to store latest prediction results
PREDICTIONS_FILE = Path("latest_predictions.json")

# Allow all CORS (frontend React)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# ROOT HEALTH ENDPOINTS
# ============================================================

@app.get("/")
def root():
    return {"message": "✅ SmartGovAI API running successfully"}

@app.get("/health")
def health():
    return {"status": "ok"}

# ============================================================
# PREDICTIVE INSIGHTS
# ============================================================

@app.post("/predictive_insights")
async def predictive_insights(file: UploadFile = File(...)):
    """
    Upload CSV -> clean -> aggregate weekly counts -> predict trends.
    Results saved to latest_predictions.json for dashboard use.
    """
    try:
        df = pd.read_csv(file.file)
        df = basic_clean(df)
        agg = aggregate_counts(df, freq='W', date_col='date')

        predictor = SimpleTrendPredictor()
        predictor.fit(agg)
        preds = predictor.predict_next_period(agg)

        result = (
            preds.sort_values('predicted_count', ascending=False)
            .head(20)
            .to_dict(orient='records')
        )

        # Save results for dashboard
        with open(PREDICTIONS_FILE, "w") as f:
            json.dump(result, f)

        return {"status": "success", "predictions": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {e}")

@app.get("/api/predictions")
def get_predictions():
    """Return the latest saved prediction results (if any)."""
    if not PREDICTIONS_FILE.exists():
        return {"predictions": [], "message": "No stored predictions found."}
    with open(PREDICTIONS_FILE, "r") as f:
        data = json.load(f)
    return {"predictions": data}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Simplified prediction endpoint without saving results.
    """
    df = pd.read_csv(file.file)
    df = basic_clean(df)
    agg = aggregate_counts(df)
    predictor = SimpleTrendPredictor()
    predictor.fit(agg)
    preds = predictor.predict_next_period(agg)
    return preds.to_dict(orient="records")

# ============================================================
# AI SUMMARIZATION
# ============================================================

@app.post("/ai_summary")
async def ai_summary(file: UploadFile = File(...), model_type: str = "gemini"):
    """
    Generate AI summary of uploaded dataset using Gemini (default) or OpenAI fallback.
    """
    df = pd.read_csv(file.file)
    sample = df.head(10).to_string()

    if model_type == "gemini":
        try:
            vertexai.init(project="smartgovai-gcp", location="asia-south1")
            model = GenerativeModel("gemini-1.5-flash")

            prompt = f"""
            You are analyzing a citizen service dataset for the Government of Maharashtra.
            Provide a concise AI-generated summary highlighting:
            1. Top complaint categories
            2. City with highest issues
            3. Percentage of high priority complaints
            4. Suggested improvements
            Dataset Preview:
            {sample}
            """

            response = model.generate_content(prompt)
            return {"summary": response.text}

        except Exception as e:
            return {"error": str(e)}

    else:  # OpenAI fallback
        key = os.getenv("OPENAI_API_KEY")
        if not key:
            return {"error": "OpenAI key not found"}

        client = OpenAI(api_key=key)
        prompt = f"Summarize this citizen service dataset:\n{sample}"

        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=250,
            )
            return {"summary": response.choices[0].message.content}
        except Exception as e:
            return {"error": str(e)}

@app.post("/summary")
async def summary(file: UploadFile = File(...)):
    """
    Simpler Gemini-based summary endpoint for testing.
    """
    df = pd.read_csv(file.file)
    vertexai.init(project="smartgovai-gcp", location="asia-south1")
    model = GenerativeModel("gemini-1.5-flash")
    prompt = f"Summarize the key citizen service patterns:\n{df.head(10).to_string()}"
    response = model.generate_content(prompt)
    return {"summary": response.text}

# ============================================================
# URGENCY CALCULATION
# ============================================================

# Sample data for testing (replace with real predictions later)
sample_data = pd.DataFrame({
    "region": ["Pune", "Mumbai", "Nagpur"],
    "category": ["Water", "Roads", "Electricity"],
    "last_count": [23, 45, 31],
    "predicted_count": [30, 52, 40],
})

def compute_urgency_local(risk, delta_ratio, resource, weights):
    a, b, c = weights
    return round(a * risk + b * delta_ratio + c * (1 - resource), 3)

@app.post("/api/urgency")
def get_urgency(weights: dict = Body(...)):
    """
    Compute urgency scores using weighted risk, delta, and resource availability.
    Accepts JSON input: {"alpha": 0.5, "beta": 0.3, "gamma": 0.2}
    """
    try:
        alpha = float(weights.get("alpha", 0.5))
        beta = float(weights.get("beta", 0.3))
        gamma = float(weights.get("gamma", 0.2))
        weights_tuple = (alpha, beta, gamma)

        df = sample_data.copy()
        df["delta"] = df["predicted_count"] - df["last_count"]
        df["risk_score"] = np.clip(df["delta"] / (df["last_count"] + 1), 0, 1)
        df["resource_availability"] = np.random.rand(len(df))
        df["urgency_score"] = df.apply(
            lambda r: compute_urgency_local(
                r["risk_score"],
                r["delta"] / (r["last_count"] + 1),
                r["resource_availability"],
                weights_tuple
            ),
            axis=1,
        )

        return df.to_dict(orient="records")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error computing urgency: {e}")

# ============================================================
# PRIORITIZATION
# ============================================================

@app.post("/prioritize")
async def prioritize(file: UploadFile = File(...)):
    df = pd.read_csv(file.file)
    df = basic_clean(df)
    urgency_scores = compute_urgency(df)
    df["urgency_score"] = urgency_scores
    prioritized_df = df.sort_values(by="urgency_score", ascending=False)
    return prioritized_df.to_dict(orient="records")

# ============================================================
# FEEDBACK + INSIGHTS
# ============================================================

@app.post("/api/feedback_summary")
def feedback_summary(payload: dict = Body(...)):
    text = payload.get("text", "")
    if not text:
        return {"error": "No text provided"}
    summary = (
        "AI summary (mock): Based on feedback, key issues are roads, water, "
        "and electricity in urban areas."
    )
    return {"summary": summary}

@app.get("/api/insights")
def get_insights():
    trends = [
        {"date": "2025-10-01", "value": 120},
        {"date": "2025-10-02", "value": 160},
        {"date": "2025-10-03", "value": 200},
        {"date": "2025-10-04", "value": 180},
    ]
    summary = (
        "AI Insights: Citizen complaints increased 25% this week, mainly "
        "in water supply and waste management sectors."
    )
    return {"summary": summary, "trends": trends}

# ============================================================
# DEPARTMENTS & UPLOAD HANDLERS
# ============================================================

@app.get("/departments")
async def get_departments():
    data = [
        {"id": 1, "name": "Health", "urgency": 87, "status": "Active"},
        {"id": 2, "name": "Infrastructure", "urgency": 72, "status": "Active"},
        {"id": 3, "name": "Public Safety", "urgency": 54, "status": "Resolved"},
        {"id": 4, "name": "Education", "urgency": 68, "status": "Active"},
    ]
    return {"departments": data}

@app.post("/api/upload_csv")
async def upload_csv(file: UploadFile = File(...)):
    contents = await file.read()
    return {"message": f"Received CSV: {file.filename} ({len(contents)} bytes)"}

# ============================================================
# MAIN ENTRYPOINT
# ============================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
