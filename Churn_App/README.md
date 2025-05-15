# 🔍 Customer Churn Prediction & Monitoring App

**Why do customers leave?**  
This Streamlit app explores that question, not just by predicting churn, but by explaining **why** it happens and **when** the model itself starts to drift.

I’m an AI enthusiast with a deep interest in **trustworthy AI** and **AI governance**. I built this project using a bank churn dataset to demonstrate that models require more than just high accuracy — they also need **oversight**, **interpretability**, and **adaptability**.

---

## 🚀 What This App Does

✅ **Predicts customer churn** using a trained Random Forest model  
🔍 **Explains predictions** with feature importance  
📉 **Monitors model drift** using Kolmogorov–Smirnov (KS) statistical tests  
📊 **Visualises drift** so you know when to retrain

---

## 🎯 Why It Matters

A model that was 98% accurate today might be misleading tomorrow.

Real-world data evolves, and behaviours shift. If your model isn't monitored, it quietly starts making bad decisions. This app shows how to **build machine learning systems that can be trusted over time**.

---

## 🛠️ How to Run It Locally

```bash
# Step 1: Clone this repository
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>

# Step 2: Install dependencies
pip install -r requirements.txt

# Step 3: Launch the app
streamlit run app.py
