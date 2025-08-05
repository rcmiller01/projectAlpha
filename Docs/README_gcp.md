# 🌤️ GCP Deployment Guide: Mia & Solene Emotional AI

## 1. Prerequisites
- ✅ Google Cloud Project with billing enabled
- ✅ Enable these APIs:
  - Cloud Build
  - Cloud Run
  - Artifact Registry
  - Firestore (Native mode)

## 2. Set Up Firestore (Symbolic Memory)
```bash
gcloud firestore databases create --region=us-central
```

## 3. Deploy Backend with Cloud Build
```bash
gcloud builds submit --config cloudbuild.yaml
```

## 4. Deploy Container to Cloud Run
```bash
gcloud run deploy mia-solene-backend   --image gcr.io/YOUR_PROJECT_ID/mia-solene-backend   --platform managed   --region us-central1   --allow-unauthenticated   --port 8000
```

## 5. (Optional) Upload Frontend to Cloud Storage
```bash
gsutil mb -l us-central1 gs://mia-companion-ui
gsutil -m cp -r web/* gs://mia-companion-ui
gsutil web set -m index.html -e 404.html gs://mia-companion-ui
```

Your app will be accessible at the Cloud Run URL and/or Cloud Storage frontend bucket.
