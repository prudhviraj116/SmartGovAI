
# SmartGovAI — AI-Driven Governance & Predictive Analytics Platform

SmartGovAI is an AI/ML engineering platform designed to transform multi-sectoral public data into predictive, actionable intelligence for government decision-makers. Built with a lightweight Python backend architecture, it forecasts public service demand, prioritizes citizen issues using weighted risk scoring, and generates executive summaries using Google Gemini AI.

---

## 🌟 Key Features

* **Predictive AI Modeling:** Forecasts complaint volume and service demand for upcoming periods using time-series trend analysis.
* **Dynamic Prioritization Engine:** Automatically ranks municipal issues by computing an urgency score based on risk, predicted demand delta, and resource availability.
* **AI-Powered Executive Summaries:** Integrates Google Gemini (`google-generativeai`) to summarize dataset patterns and actionable insights for decision-makers.
* **Interactive Streamlit UI:** Live data visualization, time-series plotting, and threshold adjustments with immediate fallback to local summary generators.
* **FastAPI Backend Services:** Modular REST API endpoints for model inference, data ingestion, and batch data cleaning.

---

## ⚙️ Tech Stack

| Layer | Technology |
| :--- | :--- |
| **Frontend / Dashboard** | Streamlit, Plotly Express, Pandas |
| **Backend / API** | FastAPI, Uvicorn, Pydantic |
| **ML & AI Engine** | Python, Scikit-learn, NumPy, Google Gemini API (`google-generativeai`) |
| **Deployment** | Render (FastAPI Web Service), Streamlit Community Cloud (Dashboard UI) |

---

## 🧩 Architecture Flow

```text
[ Citizen CSV Upload / Data Source ]
                 │
                 ▼
     ┌───────────────────────┐
     │  Streamlit Dashboard  │ ◄── (Interactive UI / Visualizations)
     └───────────┬───────────┘
                 │ (REST API / HTTPS)
                 ▼
     ┌───────────────────────┐
     │    FastAPI Backend    │ ───► Data Cleaning & Anonymization
     └───────────┬───────────┘
                 ├───► ML Trend Predictor & Urgency Scoring
                 └───► Google Gemini API (Executive Summarization)
📁 Project Structure
Plaintext
SmartGovAI/
├── app.py                     # Streamlit frontend dashboard
├── main.py                    # FastAPI backend server & API endpoints
├── requirements.txt           # Python dependencies
├── .gitignore                 # Git ignore configuration
├── models/
│   └── predictor.py           # Trend prediction engine
├── utils/
│   ├── data_cleaning.py       # Data cleaning & column normalization
│   ├── prioritizer.py         # Urgency scoring algorithm
│   └── backup_ai.py           # Fallback local summary generator
└── shared_data/               # Storage for generated outputs
⚡ Quick Start (Local Setup)
1. Clone Repository
Bash
git clone [https://github.com/prudhviraj116/SmartGovAI.git](https://github.com/prudhviraj116/SmartGovAI.git)
cd SmartGovAI
2. Set Up Virtual Environment & Dependencies
PowerShell
# Create & activate environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install requirements
pip install -r requirements.txt
3. Environment Variables
Create a .env file in the project root:

Code snippet
GEMINI_API_KEY="YOUR_GEMINI_API_KEY"
4. Run Services
Start FastAPI Backend:

Bash
uvicorn main:app --reload --port 8000
API Docs will be live at http://127.0.0.1:8000/docs

Start Streamlit Dashboard:

Bash
streamlit run app.py
Dashboard will open at http://localhost:8501

☁️ Live Deployment
FastAPI Backend Service: Deployed on Render

https://smartgovai-backend.onrender.com

Interactive Frontend: Deployed on Streamlit Community Cloud

https://smartgovai.streamlit.app

👨‍💻 Author
S. Prudhviraj

AI Developer | Full-Stack & Systems Enthusiast

Email: prudhvirajsuthapalli@gmail.com

GitHub: @prudhviraj116

LinkedIn: linkedin.com/in/prudhvirajsuthapalli

📜 License
Distributed under the MIT License. See LICENSE for details.
