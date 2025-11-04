# vertex_model_setup.py

from google.cloud import aiplatform

# === CONFIGURATION ===
PROJECT_ID = "smartgovai-gcp"
LOCATION = "asia-south1"
BQ_SOURCE = "bq://smartgovai-gcp.gov_data.cleaned_health"
DISPLAY_NAME = "health_risk_predictor"
TARGET_COLUMN = "critical_cases"

def main():
    print("🚀 Initializing Vertex AI...")
    aiplatform.init(project=PROJECT_ID, location=LOCATION)

    # === Step 1: Create Dataset from BigQuery ===
    print("📦 Creating dataset from BigQuery source...")
    dataset = aiplatform.TabularDataset.create(
        display_name=f"{DISPLAY_NAME}_dataset",
        bq_source=BQ_SOURCE
    )
    print(f"✅ Dataset created: {dataset.resource_name}")

    # === Step 2: Train Model using AutoML Tables ===
    print("🧠 Training AutoML model...")
    model = aiplatform.AutoMLTabularTrainingJob(
        display_name=f"{DISPLAY_NAME}_training_job",
        optimization_prediction_type="regression",
        optimization_objective="minimize-rmse"
    ).run(
        dataset=dataset,
        target_column=TARGET_COLUMN,
        model_display_name=f"{DISPLAY_NAME}_model",
        budget_milli_node_hours=1000,  # ~1 hour budget
        disable_early_stopping=False,
    )
    print(f"✅ Model trained: {model.resource_name}")

    # === Step 3: Deploy Model to Endpoint ===
    print("🚀 Creating endpoint and deploying model...")
    endpoint = model.deploy(
        deployed_model_display_name=f"{DISPLAY_NAME}_endpoint",
        traffic_split={"0": 100}
    )
    print(f"✅ Model deployed at endpoint: {endpoint.resource_name}")

    # === Step 4: Test Prediction ===
    print("🧪 Testing model prediction...")
    test_instance = {
    "district": "Pune",
    "hospital_name": "Pune General Hospital",
    "patients_treated": "50",
    "critical_cases": "5",
    "reported_outbreaks": "2",
    "date": "2025-10-30"
    }


    prediction = endpoint.predict(instances=[test_instance])
    print("✅ Prediction result:", prediction.predictions)

if __name__ == "__main__":
    main()
