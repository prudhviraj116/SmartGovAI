#!/bin/bash
PROJECT=$1
REGION=${2:-asia-south1}
IMAGE=gcr.io/${PROJECT}/smartgovai:latest
gcloud builds submit --tag ${IMAGE} --project ${PROJECT}
gcloud run deploy smartgovai --image ${IMAGE} --region ${REGION} --platform=managed --allow-unauthenticated --project ${PROJECT}
