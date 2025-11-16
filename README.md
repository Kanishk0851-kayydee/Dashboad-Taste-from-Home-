# Taste From Home - Marketing Analytics Dashboard

## 🍽️ Project Overview
This is a comprehensive Streamlit dashboard for analyzing market survey data for Taste From Home, a home-cooked meal delivery service in Dubai.

## 📊 Features
1. **Marketing Insights Dashboard** - 5 interactive charts with filters by Status, Location, and Nationality
2. **ML Algorithms & Performance** - Classification, Clustering, and Regression models with metrics
3. **Customer Prediction** - Predict individual customer interest
4. **Upload & Predict** - Upload new datasets and download predictions

## 🚀 How to Run Locally

### Step 1: Install Requirements
```bash
pip install -r requirements.txt
```

### Step 2: Run the Dashboard
```bash
streamlit run app.py
```

The dashboard will open in your browser at `http://localhost:8501`

## 📦 Deploy to Streamlit Cloud

### Step 1: Push to GitHub
1. Create a new repository on GitHub
2. Clone the repo locally:
   ```bash
   git clone https://github.com/your-username/your-repo-name.git
   cd your-repo-name
   ```
3. Copy all files to the repo folder
4. Add, commit, and push:
   ```bash
   git add .
   git commit -m "Initial dashboard commit"
   git push origin main
   ```

### Step 2: Deploy on Streamlit Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with GitHub
3. Click "New app"
4. Select your repository, branch (main), and main file (app.py)
5. Click "Deploy"

## 📁 Files Included
- `app.py` - Main Streamlit dashboard application
- `taste_from_home_survey_data.csv` - Survey dataset (600 responses)
- `requirements.txt` - Python dependencies
- `README.md` - This file

## 🎯 Dashboard Tabs

### Tab 1: Marketing Insights
- 5 complex visualization charts
- Interactive filters (Status, Location, Nationality)
- Key metrics and actionable insights

### Tab 2: ML Algorithms
- Classification models (Logistic Regression, Decision Tree, Random Forest, Gradient Boosting)
- Performance metrics (Accuracy, Precision, Recall, F1-Score)
- K-Means Clustering for customer segmentation
- Linear Regression for WTP prediction

### Tab 3: Customer Prediction
- Input customer details
- Get interest prediction and recommendations

### Tab 4: Upload & Predict
- Upload new CSV dataset
- Download predictions with labels

## 📧 Support
For questions or issues, please contact your team members.
