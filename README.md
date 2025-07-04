# 📰 Abstractive News Summarization with T5 + QLoRA

Fine-tuned a lightweight T5 model using **QLoRA** on AWS SageMaker for high-quality, abstractive news summarization. The model is deployed using a **SageMaker Endpoint** and accessible via an **AWS Lambda-triggered API Gateway**. A lightweight client is also integrated and hosted via **Heroku**.

---

## 🌟 Features

- 🔄 **Abstractive Summarization**: Generates human-like summaries for long-form news content.
- 🚀 **QLoRA Fine-Tuning**: Parameter-efficient fine-tuning for resource-constrained environments.
- 🔧 **Cloud-Native Deployment**: Deployed using SageMaker + API Gateway + Lambda for high availability.
- ⚡ **CI/CD Integrated**: Automated GitHub Actions workflows for model deployment and app updates.
- 💡 **Client Integration**: Minimal frontend hosted on Heroku with secure API integration.

---

## 🖼️ Screenshot

> 📌 Replace this with your actual image once added:

![App Screenshot] <img width="804" alt="Screenshot 2025-07-04 at 4 18 44 AM" src="https://github.com/user-attachments/assets/a5b9988a-586e-4aad-b3dd-e5901404356f" />


---

## 🔗 Hosted Demo

> 📌 Replace this with your deployed client URL:

**🔗 Try it live:** [https://news-summarizer-app-07a98f43784c.herokuapp.com/](https://news-summarizer-app-07a98f43784c.herokuapp.com/)

---

## 🛠️ Tech Stack

| Component       | Tech Used                      |
|----------------|---------------------------------|
| Model           | T5-small / T5-base              |
| Fine-Tuning     | QLoRA (Parameter-efficient)     |
| Cloud Training  | AWS SageMaker                   |
| Deployment      | SageMaker Endpoint + Lambda + API Gateway |
| CI/CD           | GitHub Actions                  |
| Client          | Python + Streamlit (or Flask) on Heroku |
| Containerization| Docker                          |

---

## 🧪 Example Usage

**Input:**
```
The new climate bill passed in the U.S. Senate today is expected to cut carbon emissions by 40% by 2030...
```

**Generated Summary:**
```
The U.S. Senate passed a climate bill aimed at reducing carbon emissions by 40% by 2030.
```

---

## 🚀 How It Works

1. **Training**:
    - Trained a T5 model using QLoRA on AWS SageMaker with CNN/DailyMail dataset.
2. **Model Hosting**:
    - Model saved to S3 and deployed as a SageMaker endpoint.
3. **Inference Pipeline**:
    - A Lambda function handles input and sends it to the endpoint.
    - API Gateway exposes the Lambda as a public HTTP API.
4. **Frontend**:
    - Client application interacts with the API to display summaries.

---

## 📁 Project Structure

```
├── lambda_function.py         # Lambda handler for API calls
├── inference_client.py        # Streamlit or Flask-based UI
├── Dockerfile                 # For client app container
├── training_job/              # SageMaker training scripts
├── screenshots/               # Place your UI screenshots here
├── .github/workflows/         # CI/CD pipelines (GitHub Actions)
└── README.md
```

---

## 📦 Setup Instructions

### 🐍 Local Setup (Client Only)

```bash
pip install -r requirements.txt
python inference_client.py
```

### 🐳 Dockerized Run

```bash
docker build -t summarizer-client .
docker run -p 8501:8501 summarizer-client
```

---
