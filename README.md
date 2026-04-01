# 🔌 XOR FastAPI API

A simple REST API built with **FastAPI** to predict the result of the logical XOR operation using a neural network model trained with **PyTorch**.
This project demonstrates how to expose a machine learning model through a clean and modular backend API.

---

## 🚀 Features

* 🧠 PyTorch neural network trained on XOR logic
* 🔁 Prediction via a `/predict` POST endpoint
* 🐳 Fully dockerized for reproducible deployment
* ⚙️ Clean architecture (API, services, schemas, dependencies)
* 📊 JSON response: input, raw_output (probability), prediction

---

## 📁 Project Structure

```text
app/
├── api/
│   ├── endpoints/        # API routes
│   │   └── inference_route.py
│   └── router.py         # Central router
│
├── dependencies/         # Dependency injection
│   └── inference.py
│
├── schemas/              # Pydantic models (DTO)
│   ├── input_data.py
│   └── prediction_response.py
│
├── services/             # Business logic
│   └── inference_service.py

config/                   # Configuration (env variables)
└── config.py

datasets/                 # Training & test data
├── test_data.py
└── train_data.py

ml/                       # Model structure and weights 
├── model/
│   └── xor_net.py
└── model_weights/
    └── xor_model.pth

scripts/                  # Model Training and testing scripts 
├── test.py
└── train.py

.dockerignore
.gitignore
Dockerfile
main.py
README.md
requirements.txt
```

---

## ⚙️ Environment Variables

Create a `.env` file:

```env
MODEL_PATH=ml/model_weights/xor_model.pth
```

---

## 🧠 Train the model (optional)

```bash
python -m scripts.train
```
The trained model is automatically saved to: ml/model_weights/xor_model.pth

---

## 🧪 Test the model

```bash
python -m scripts.test
```

---

## 🚀 Run the API locally (without Docker)

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Start the server

```bash
uvicorn main:app --reload
```

👉 API docs:
http://localhost:8000/docs

---

## 🐳 Run with Docker

### 1. Build the image

```bash
docker build -t xor-api .
```

### 2. Run the container

```bash
docker run --env-file .env -p 8000:8000 xor-api
```

👉 API docs:
http://localhost:8000/docs

---

## 📬 Test the API

```bash
curl -X POST http://localhost:8000/predict \
-H "Content-Type: application/json" \
-d '{"x1": 1, "x2": 0}'
```

### Example response

```json
{
  "user_input": [1, 0],
  "raw_output": 0.9998396635055542,
  "prediction": 1
}
```

---

## 📌 Technologies Used

* Python 3.11
* FastAPI
* PyTorch
* Docker
* Matplotlib 

---

## 🧪 Notes

* 🔄 Model loaded once using FastAPI dependency injection (efficient inference)
* The trained model weights are included for reproducibility
* Training, testing scripts and datasets are excluded from the Docker image
* Environment variables are injected at runtime (`--env-file`)

---

## 🚀 Future Improvements

* Logging & error handling
* Database for predictions
* Authentication (JWT)

---