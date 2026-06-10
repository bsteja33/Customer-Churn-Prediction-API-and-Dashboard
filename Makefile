# Makefile for Customer Churn Prediction project

PYTHON=python
PIP=pip
REQ=requirements.txt
PROJECT_ROOT=$(shell pwd)

.PHONY: train serve test docker-build

# Train the model using the default config
train:
	$(PYTHON) src/train.py

# Run the FastAPI service
serve:
	uvicorn api.app:app --host 0.0.0.0 --port 8000

# Run the test suite
test:
	$(PYTHON) -m pytest -vv

# Build the Docker image
docker-build:
	docker build -t churn-api:latest .
