# FastAPI Image Classification API

## Objective

Deploy a trained PyTorch image classification model using FastAPI and Docker.

## Features

- FastAPI REST API
- Image Upload Endpoint
- PyTorch Inference
- Docker Support
- Swagger Documentation

## Installation

```bash
pip install -r requirements.txt
```

## Run Locally

```bash
uvicorn app:app --reload
```

Open:

http://localhost:8000/docs

## Docker

Build:

```bash
docker build -t cifar-api .
```

Run:

```bash
docker run -p 8000:8000 cifar-api
```

## Endpoint

POST /predict

Example:

```bash
curl -X POST -F "file=@sample.jpg" http://localhost:8000/predict
```

Response:

```json
{
  "prediction": "cat"
}
```
