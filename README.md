# Student Dropout/Performance Prediction

### Deep Learning-Based Student Dropout and Academic Success Prediction

An AI-based early-warning system that uses Machine Learning and Deep Learning to predict student academic outcomes: Dropout, Enrolled, or Graduate.

## 📌 Project Overview

Student dropout is a major challenge for educational institutions. This project aims to identify students who may be at risk of dropping out by analyzing demographic, admission, financial, and academic information.

The system uses a Deep Neural Network (DNN) to predict student outcomes and estimate the probability of each class. A Streamlit web application provides an interactive interface for generating predictions.

## 🎯 Objectives

- Predict student outcomes using Machine Learning and Deep Learning.
- Classify students into Dropout, Enrolled, and Graduate categories.
- Estimate the probability of each predicted outcome.
- Compare Logistic Regression and Deep Neural Network models.
- Develop an interactive early-warning application using Streamlit.

## 📊 Dataset

**Source:** UCI Machine Learning Repository

**Dataset:** Predict Students' Dropout and Academic Success

- Total student records: 4,424
- Input features: 36
- Target classes: 3
- Classes: Dropout, Enrolled, Graduate
- Original dataset institution: Portugal

The dataset contains demographic, academic, admission, and financial information.

Note: The dataset is not included in this repository. Please obtain it from the UCI Machine Learning Repository.

## ⚙️ Methodology

1. Dataset Loading
2. Data Preprocessing
3. Categorical Feature Encoding using One-Hot Encoding
4. Numerical Feature Scaling using StandardScaler
5. Train-Test Split
6. Logistic Regression Baseline
7. Deep Neural Network Development
8. Model Evaluation
9. Streamlit Application Integration

## 🤖 Models Implemented

- Logistic Regression – Baseline Machine Learning model
- Initial Deep Neural Network
- Tuned Deep Neural Network
- Full Academic Deep Neural Network

## 🛠️ Technologies Used

- Python
- TensorFlow / Keras
- Scikit-learn
- Pandas
- NumPy
- Streamlit
- Matplotlib

## ✨ Application Features

- Interactive student information form
- Student outcome prediction
- Dropout probability estimation
- Risk-level indication
- Class probability distribution
- Recommended actions based on prediction

## 📁 Project Structure

```text
Student-Dropout-Performance-Prediction/
│
├── app.py
├── early_warning_model.keras
├── early_warning_preprocessor.pkl
├── class_names.npy
├── requirements.txt
├── README.md
└── .gitignore
```

## 🚀 Installation and Execution

### 1. Clone the repository

```bash
git clone https://github.com/gauridyappa/Student-Dropout-Performance-Prediction.git
```

### 2. Navigate to the project folder

```bash
cd Student-Dropout-Performance-Prediction
```

### 3. Create a virtual environment

```bash
python3 -m venv .venv
```

### 4. Activate the environment

```bash
source .venv/bin/activate
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

### 6. Run the Streamlit application

```bash
python -m streamlit run app.py
```

The application will open in your browser.

## 📈 Evaluation Metrics

The models are evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

## 🔮 Future Scope

- Integration with real-time university data
- Attendance and Learning Management System (LMS) integration
- Continuous academic performance monitoring
- Cloud deployment
- Personalized student support recommendations

## ⚠️ Disclaimer

This system provides predictive insights based on historical data. Predictions are intended only as early-warning indicators and should not be treated as definitive decisions about students.

## 👩‍💻 Author

**Gauri Dyappa**

MSc Artificial Intelligence and Machine Learning

---

⭐ If you find this project interesting, feel free to explore the repository!
