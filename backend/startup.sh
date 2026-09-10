#!/bin/bash
# Azure App Service startup script for EcoNavigators backend
# This file is set as the "Startup Command" in Azure App Service Configuration

gunicorn api.main:app \
  --workers 1 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000 \
  --timeout 120 \
  --keep-alive 5 \
  --log-level info
