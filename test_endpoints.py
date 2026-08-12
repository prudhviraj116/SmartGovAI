import pathlib
import requests

base = "http://127.0.0.1:8001"

print("GET /", requests.get(base + "/", timeout=10).status_code)
print(requests.get(base + "/", timeout=10).json())
print("GET /health", requests.get(base + "/health", timeout=10).status_code)
print(requests.get(base + "/health", timeout=10).json())

files_to_test = ["data/citizen_complaints_10k.csv", "data/health_data.csv"]
for fpath in files_to_test:
    print(f"\nPOST /predict with {fpath}")
    with open(fpath, "rb") as fp:
        r = requests.post(base + "/predict", files={"file": ("data.csv", fp, "text/csv")}, timeout=60)
    print("status", r.status_code)
    if r.headers.get("content-type", "").startswith("application/json"):
        print("json keys", list(r.json().keys()) if isinstance(r.json(), dict) else "list", len(r.json()) if isinstance(r.json(), list) else "")
    else:
        print(r.text[:200])

print("\nPOST /predictive_insights with citizen_complaints_10k.csv")
with open("data/citizen_complaints_10k.csv", "rb") as fp:
    r = requests.post(base + "/predictive_insights", files={"file": ("data.csv", fp, "text/csv")}, timeout=120)
print("status", r.status_code)
print(r.json() if r.headers.get("content-type", "").startswith("application/json") else r.text[:200])

print("\nGET /api/predictions")
r = requests.get(base + "/api/predictions", timeout=10)
print(r.status_code, r.json())

print("\nPOST /summary with health_data.csv")
with open("data/health_data.csv", "rb") as fp:
    r = requests.post(base + "/summary", files={"file": ("data.csv", fp, "text/csv")}, timeout=120)
print(r.status_code)
print(r.text[:1000])

print("\nPOST /ai_summary missing OpenAI key")
with open("data/health_data.csv", "rb") as fp:
    r = requests.post(base + "/ai_summary", files={"file": ("data.csv", fp, "text/csv")}, data={"model_type": "openai"}, timeout=120)
print(r.status_code)
print(r.text[:1000])

print("\nPOST /prioritize with health_data.csv")
with open("data/health_data.csv", "rb") as fp:
    r = requests.post(base + "/prioritize", files={"file": ("data.csv", fp, "text/csv")}, timeout=120)
print(r.status_code)
print(r.text[:1000])

print("\nPOST /api/urgency")
r = requests.post(base + "/api/urgency", json={"alpha": 0.5, "beta": 0.3, "gamma": 0.2}, timeout=10)
print(r.status_code)
print(r.json())

print("\nPOST /api/feedback_summary")
r = requests.post(base + "/api/feedback_summary", json={"text": "Road issues and water shortages are rising"}, timeout=10)
print(r.status_code)
print(r.json())

print("\nGET /api/insights")
r = requests.get(base + "/api/insights", timeout=10)
print(r.status_code)
print(r.json())

print("\nGET /departments")
r = requests.get(base + "/departments", timeout=10)
print(r.status_code)
print(r.json())

print("\nPOST /predict invalid file")
with open("README.md", "rb") as fp:
    r = requests.post(base + "/predict", files={"file": ("readme.md", fp, "text/markdown")}, timeout=120)
print(r.status_code)
print(r.text[:1000])
