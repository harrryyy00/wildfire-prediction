# 🔥 Wildfire Risk Prediction & Hotspot Analysis

A machine learning project for analyzing wildfire patterns, predicting wildfire risk levels, and identifying potential hotspot regions using multi-year MODIS wildfire data.

## 📌 Project Overview

Wildfires can cause significant environmental and economic damage. This project analyzes historical wildfire data and uses machine learning to classify wildfire risk levels into:

- 🟢 Low
- 🟡 Moderate
- 🟠 High
- 🔴 Extreme

The project uses **Random Forest** as the best-performing model after comparing multiple classification algorithms.

## 🎯 Objectives

- Analyze historical wildfire patterns.
- Process multi-year MODIS wildfire datasets.
- Predict wildfire risk levels using machine learning.
- Compare different classification models.
- Identify high-risk geographic regions.
- Generate wildfire risk summaries and hotspot visualizations.
- Provide an interactive map for wildfire hotspot analysis.

## 🤖 Machine Learning Models

Three classification models were evaluated:

| Model | Accuracy | Macro Precision | Macro Recall | Macro F1 |
|---|---:|---:|---:|---:|
| 🏆 Random Forest | **88.74%** | **87.22%** | **80.60%** | **83.39%** |
| Logistic Regression | 87.89% | 84.86% | 79.95% | 82.10% |
| Decision Tree | 84.17% | 79.14% | 77.41% | 78.19% |

### 🏆 Best Model

**Random Forest**

- Accuracy: **88.74%**
- Macro Precision: **87.22%**
- Macro Recall: **80.60%**
- Macro F1 Score: **83.39%**

Random Forest achieved the highest overall accuracy and macro F1 score among the evaluated models.

## 🛰️ Dataset

The project uses MODIS wildfire datasets covering multiple years from **2012 to 2024**.

The data contains wildfire-related information used for:

- Fire event analysis
- Geographic analysis
- Fire radiative power analysis
- Brightness analysis
- Confidence analysis
- Wildfire risk classification

## ⚙️ Project Workflow

```text
MODIS Wildfire Data
        ↓
Data Collection & Preprocessing
        ↓
Exploratory Data Analysis
        ↓
Feature Engineering
        ↓
Time-Series Analysis
        ↓
Model Training
        ↓
Random Forest / Logistic Regression / Decision Tree
        ↓
Model Evaluation
        ↓
Wildfire Risk Classification
        ↓
Risk-Zone Analysis
        ↓
Interactive Hotspot Visualization
```

## 🗺️ Key Features

### 🔥 Wildfire Risk Prediction
Classifies wildfire conditions into different risk levels using machine learning.

### 🤖 Random Forest Classification
Uses Random Forest to predict wildfire risk based on wildfire-related features.

### 📊 Model Comparison
Compares Random Forest, Logistic Regression, and Decision Tree using accuracy, precision, recall, and F1 score.

### 🗺️ Hotspot Mapping
Generates an interactive map to visualize wildfire hotspot locations.

### ⚠️ Risk-Zone Analysis
Analyzes geographic regions and generates wildfire risk-zone summaries.

### 📈 Data Analysis
Analyzes historical wildfire trends using multi-year datasets.

## 🛠️ Technologies Used

- **Python**
- **Pandas**
- **NumPy**
- **Scikit-learn**
- **Matplotlib**
- **Jupyter Notebook**
- **Random Forest**
- **Machine Learning**
- **Time-Series Analysis**
- **MODIS Satellite Data**
- **Data Visualization**

## 📁 Project Structure

```text
wildfire-prediction/
│
├── wildfire_prediction.ipynb
├── wildfire_prediction_executed.ipynb
├── wildfire_risk_pipeline.py
├── run_wildfire_pipeline.py
│
├── DataCollection/
│
├── outputs/
│   ├── model_comparison.csv
│   ├── best_model_predictions.csv
│   ├── best_model_metrics.json
│   ├── risk_zone_summary.csv
│   └── wildfire_hotspot_map.html
│
├── *.csv
│
└── README.md
```

## 🚀 How to Run

### 1. Clone the repository

```bash
git clone https://github.com/harrryyy00/wildfire-prediction.git
cd wildfire-prediction
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

### 3. Activate the environment

**macOS/Linux:**

```bash
source .venv/bin/activate
```

**Windows:**

```bash
.venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install pandas numpy scikit-learn matplotlib jupyter
```

### 5. Run the notebook

```bash
jupyter notebook
```

Open:

```text
wildfire_prediction.ipynb
```

## 📊 Results

The Random Forest model achieved the best performance:

> **88.74% Accuracy**

with:

> **83.39% Macro F1 Score**

The results demonstrate that Random Forest performed better than Logistic Regression and Decision Tree for the wildfire risk classification task.

## 🗺️ Visualization

The project also generates an interactive wildfire hotspot map and risk-zone analysis outputs that can be used to explore geographic wildfire patterns.

## 🔮 Future Improvements

- Integrate real-time satellite/fire data.
- Add weather information such as temperature, humidity, wind speed, and rainfall.
- Experiment with advanced ensemble and deep-learning models.
- Develop a web dashboard for real-time wildfire monitoring.
- Deploy the prediction system as an online application.
- Add automated alerts for high-risk regions.

## 👨‍💻 Author

**Hargovind Singh**

B.Tech Computer Science & Engineering (Artificial Intelligence)

Interested in **AI/ML, Software Development, Data Science, and AI-powered applications**.

### 🔗 Connect

- GitHub: https://github.com/harrryyy00
- LinkedIn: https://www.linkedin.com/in/hargovind-singh-356b20279

---

⭐ If you find this project useful, consider giving the repository a star!
