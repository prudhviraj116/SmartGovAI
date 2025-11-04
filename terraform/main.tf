
provider "google" {
  project = var.project_id
  region  = var.region
}

resource "google_bigquery_dataset" "smartgovai" {
  dataset_id = "smartgovai_dataset"
  location   = var.region
  deletion_protection = false
}
