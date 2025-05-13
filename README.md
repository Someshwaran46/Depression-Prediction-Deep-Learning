# 🧠 Predicting Depression from Mental Health Survey Data using Deep Learning

---
## 📄 Project Overview
This project aims to predict whether an individual may experience depression based on demographic information, lifestyle choices, and medical history. Using a custom deep learning model built with PyTorch and deployed via Streamlit, the solution provides real-time mental health assessments.
---

## 💡 Skills You Will Learn
- Building deep learning models using PyTorch.
- Data preprocessing: missing value handling, categorical encoding, and normalization.
- Designing a classification neural network architecture.
- Creating an interactive Streamlit app for real-time predictions.
- Deploying applications on AWS (EC2) or Streamlit Cloud.

---

## Folder Structure
```
Depression-Prediction-Deep-learning/
|
├── Data Cleaning.ipynb
├── Data Preprocessing.ipynb
├── Data modelling.ipynb
├── EDA.ipynb
├── README.md
├── requirements.txt
├── depression_model.pt
├── scaler.pkl
├── page.py
└── app.py
```
---

## 🌍 Domain
**Mental Health** and **Healthcare AI**

---

## ❓ Problem Statement
Develop a model that can predict the likelihood of depression using survey responses. The system must handle healthcare data challenges such as bias, noise, and demographic diversity, while offering fair and accurate predictions.

---

## 💼 Business Use Cases

### 🏥 Healthcare Providers
Early detection of mental health risks for improved intervention and patient outcomes.

### 🧠 Mental Health Clinics
Support for therapists and counselors in designing personalized care plans.

### 🏢 Corporate Wellness Programs
Track employee mental well-being and offer proactive support.

### 🏛️ Government and NGOs
Identify at-risk populations and allocate resources effectively for mental health outreach.

---

## ⚙️ Approach

### 🔹 Data Preprocessing
- Load and inspect the dataset.
- Handle missing data.
- Encode categorical variables.
- Normalize numeric values.

### 🔹 Model Development
- Build a custom PyTorch deep learning model (MLP).
- Use binary classification with a sigmoid output.
- Train, evaluate, and fine-tune the model.

### 🔹 Pipelines
- End-to-end pipeline for preprocessing, training, and evaluation.
- Use scikit-learn metrics for model evaluation (accuracy, precision, recall, F1-score).

### 🔹 Model Deployment
- Build a Streamlit app to collect user input and predict outcomes.
- Deploy on AWS EC2 (or Streamlit Cloud).

---

## 📈 Evaluation Metrics
- **Accuracy** – Correct predictions across all samples.
- **Precision** – True positive rate over predicted positives.
- **Recall** – True positive rate over actual positives.
- **F1-Score** – Harmonic mean of precision and recall.

---

## 🖥️ Streamlit Application
A web interface allows users to enter lifestyle and health data to receive a prediction for depression risk in real-time.

---

## 🚀 AWS Deployment Guide

### ✅ Setup Instructions
#### Update and install dependencies
```bash
sudo apt update
```
```bash
sudo apt-get update
```
```bash
sudo apt upgrade -y
```
```bash
sudo apt install git curl unzip tar make sudo vim wget -y
```
#### Clone the repository
```bash
git clone https://github.com/Someshwaran46/depression-prediction-deep-learning
```
```bash
cd depression-prediction-deep-learning
```
#### Install Python and pip
```bash
sudo apt install python3-pip -y
```
#### (Optional) Remove existing venv
```bash
rm -rf .venv
```
#### Install venv module
```bash
sudo apt install python3-venv python3-full -y
```
#### Create and activate virtual environment
```bash
python3 -m venv .venv
```
```bash
source .venv/bin/activate
```
#### Install Python dependencies
```bash
pip install -r requirements.txt
````

### ▶️ Run the Application

#### 🔹 Temporary (testing)

```bash
python3 -m streamlit run app.py
```

#### 🔹 Permanent (background mode)

```bash
nohup python3 -m streamlit run app.py &
```

### 🌐 Accessing the App

* Open your browser and go to: `http://<your-ec2-public-ip>:8501`
* Make sure port **8501** is open in your EC2 Security Group (inbound rules).

