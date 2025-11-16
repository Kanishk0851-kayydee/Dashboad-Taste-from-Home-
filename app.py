import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
    classification_report, mean_squared_error, r2_score, mean_absolute_error
)
import statsmodels.api as sm
import os
import warnings

warnings.filterwarnings('ignore')
st.set_page_config(page_title="Taste From Home Dashboard", page_icon="🍽️", layout="wide")

# --- Custom CSS for Branding ---
st.markdown("""
<style>
.main-header {font-size:42px;font-weight:bold;color:#FF6B6B;text-align:center;padding:20px;background:linear-gradient(90deg,#FF6B6B 0%,#4ECDC4 100%);-webkit-background-clip:text;-webkit-text-fill-color:transparent;}
.stTabs [data-baseweb="tab-list"] {gap:24px;}
.stTabs [data-baseweb="tab"] {height:50px;background-color:#f0f2f6;border-radius:10px 10px 0 0;padding:10px 20px;}
</style>
""", unsafe_allow_html=True)

# --- Data Loader ---
@st.cache_data
def load_data():
    possible_paths = [
        'taste_from_home_survey_data.csv',
        './taste_from_home_survey_data.csv',
        'data/taste_from_home_survey_data.csv',
    ]
    for path in possible_paths:
        if os.path.exists(path):
            return pd.read_csv(path)
    st.warning("⚠️ CSV file not found. Generating sample data...")
    return generate_sample_data()

def generate_sample_data():
    np.random.seed(42)
    n = 600
    data = {
        'Age_Group': np.random.choice(['Under 18','18-24','25-34','35-44','45-54','55+'], n, p=[0.02,0.35,0.38,0.15,0.07,0.03]),
        'Gender': np.random.choice(['Male','Female','Prefer not to say'], n, p=[0.548,0.427,0.025]),
        'Nationality': np.random.choice(['Indian Subcontinent','Middle East/North Africa','Southeast Asia','East Asia','Africa','Europe','Americas','Other'], n, p=[0.35,0.25,0.17,0.08,0.07,0.04,0.03,0.01]),
        'Status': np.random.choice(['International University/College Student','Local University/College Student','Working Professional (Bachelor/Single)','Working Professional (Married)','Freelancer/Entrepreneur','Other'], n, p=[0.3,0.15,0.25,0.18,0.1,0.02]),
        'Location': np.random.choice(['International City','Dubai Academic City','JLT','Dubai Marina','Bur Dubai/Deira','Sharjah','Ajman','Abu Dhabi','Other'], n, p=[0.15,0.18,0.16,0.1,0.14,0.12,0.06,0.07,0.02]),
        'Living_Situation': np.random.choice(['University dormitory/hostel','Shared apartment with roommates','Rented studio/apartment (alone)','Living with family','Company-provided accommodation','Other'], n, p=[0.22,0.3,0.2,0.15,0.1,0.03]),
        'Monthly_Food_Budget_AED': np.random.choice(['Less than 500','500-1000','1000-1500','1500-2000','2000-3000','More than 3000'], n, p=[0.1,0.25,0.3,0.2,0.1,0.05]),
        'Cooking_Frequency': np.random.choice(['Daily','4-6 times a week','2-3 times a week','Once a week','Rarely/Never'], n, p=[0.15,0.18,0.25,0.22,0.2]),
        'Current_Spending_Per_Meal_AED': np.random.choice(['Less than 10','10-15','15-20','20-30','30-50','More than 50'], n, p=[0.08,0.2,0.28,0.25,0.15,0.04]),
        'Delivery_Frequency': np.random.choice(['Multiple times a day','Once a day','4-6 times a week','2-3 times a week','Once a week','Rarely','Never'], n, p=[0.03,0.1,0.18,0.3,0.22,0.15,0.02]),
        'Interest_Level': np.random.choice([1,2,3,4,5], n, p=[0.08,0.12,0.18,0.3,0.32]),
        'Subscription_Preference': np.random.choice([1,2,3,4,5], n, p=[0.1,0.15,0.2,0.35,0.2]),
        'WTP_Per_Meal_AED': np.round(np.random.normal(27,8,n),2),
        'Meals_Per_Week': np.random.choice(['1-2','3-4','5-7','8-10','More than 10','Would not order'], n, p=[0.2,0.3,0.28,0.12,0.05,0.05]),
        'Taste_Satisfaction': np.random.choice([1,2,3,4,5], n, p=[0.25,0.3,0.25,0.15,0.05]),
        'Healthiness_Satisfaction': np.random.choice([1,2,3,4,5], n, p=[0.2,0.28,0.3,0.18,0.04]),
        'Affordability_Satisfaction': np.random.choice([1,2,3,4,5], n, p=[0.15,0.25,0.35,0.2,0.05]),
        'Convenience_Satisfaction': np.random.choice([1,2,3,4,5], n, p=[0.1,0.18,0.3,0.3,0.12]),
        'Variety_Satisfaction': np.random.choice([1,2,3,4,5], n, p=[0.18,0.25,0.32,0.2,0.05])
    }
    df = pd.DataFrame(data)
    df['WTP_Per_Meal_AED'] = df['WTP_Per_Meal_AED'].clip(lower=10, upper=60)
    df['Interested'] = (df['Interest_Level'] >= 4).astype(int)
    return df

df = load_data()

st.markdown('<h1 class="main-header">🍽️ Taste From Home: Marketing Analytics Dashboard</h1>', unsafe_allow_html=True)
st.markdown("### Data-Driven Insights for Home-Cooked Meal Delivery Service in Dubai")
st.markdown("---")

# --- Main Dashboard Tabs ---
main_tab, ml_tab, pred_tab, upload_tab = st.tabs([
    "📊 Marketing Insights & Charts",
    "🤖 ML Algorithms & Performance",
    "🎯 Customer Prediction",
    "📤 Upload & Predict"
])

# --- Tab 1: Marketing Insights ---
with main_tab:
    st.header("Marketing Insights Dashboard")
    # ... Charts and filters can go here (not repeated for brevity; see earlier instructions) ...

# --- Tab 2: ML Algorithms & Performance ---
with ml_tab:
    st.header("Machine Learning Algorithms & Performance")
    ml_subtab1, ml_subtab2, ml_subtab3 = st.tabs([
        "🎯 Classification",
        "🔍 Clustering",
        "💰 Regression"
    ])

    # --- Classification ---
    with ml_subtab1:
        st.header("Classification Models")
        if st.button("🚀 Run Classification Algorithms", key="run_classify"):
            df_ml = df.copy()
            le = LabelEncoder()
            categorical_cols = [
                'Age_Group','Gender','Nationality','Status','Location','Living_Situation',
                'Monthly_Food_Budget_AED','Cooking_Frequency','Current_Spending_Per_Meal_AED',
                'Delivery_Frequency','Meals_Per_Week'
            ]
            for col in categorical_cols:
                df_ml[col + '_Encoded'] = le.fit_transform(df_ml[col])

            feature_cols = [col for col in df_ml.columns if col.endswith('_Encoded')] + [
                'Interest_Level', 'Subscription_Preference', 'WTP_Per_Meal_AED',
                'Taste_Satisfaction', 'Healthiness_Satisfaction', 'Affordability_Satisfaction',
                'Convenience_Satisfaction', 'Variety_Satisfaction']

            X = df_ml[feature_cols]
            y = df_ml['Interested']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

            models = {
                'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
                'Decision Tree': DecisionTreeClassifier(random_state=42),
                'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
                'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
            }

            results = []
            for model_name, model in models.items():
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred)
                recall = recall_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred)
                cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
                results.append({
                    'Model': model_name,
                    'Accuracy': accuracy,
                    'Precision': precision,
                    'Recall': recall,
                    'F1-Score': f1,
                    'CV Score (mean)': cv_scores.mean(),
                    'CV Score (std)': cv_scores.std()
                })

            results_df = pd.DataFrame(results)
            st.dataframe(results_df.style.highlight_max(axis=0, subset=['Accuracy','Precision','Recall','F1-Score']), use_container_width=True)

            # Feature importance for Random Forest (as example)
            best_model = models['Random Forest']
            best_model.fit(X_train, y_train)
            feature_importance = pd.DataFrame({
                'Feature': feature_cols,
                'Importance': best_model.feature_importances_
            }).sort_values('Importance', ascending=False).head(10)
            fig_imp = px.bar(feature_importance, x='Importance', y='Feature', orientation='h',
                            title='Top 10 Most Important Features',
                            color='Importance', color_continuous_scale='Viridis')
            st.plotly_chart(fig_imp, use_container_width=True)
            st.success("Classification models executed.")

    # --- Clustering ---
    with ml_subtab2:
        st.header("K-Means Clustering")
        if st.button("🔍 Run Clustering Algorithm", key="run_cluster"):
            cluster_features = ['WTP_Per_Meal_AED','Interest_Level','Subscription_Preference',
                               'Taste_Satisfaction','Affordability_Satisfaction']
            scaler = StandardScaler()
            X_cluster = df[cluster_features]
            X_scaled = scaler.fit_transform(X_cluster)
            kmeans = KMeans(n_clusters=4, random_state=42)
            clusters = kmeans.fit_predict(X_scaled)
            df['Cluster'] = clusters
            st.dataframe(df.groupby('Cluster')[cluster_features].mean().round(1))
            fig_clus = px.scatter(
                df, x='WTP_Per_Meal_AED', y='Interest_Level', color='Cluster',
                title='Customer Segments by WTP & Interest',
                labels={'Cluster':'Segment'}
            )
            st.plotly_chart(fig_clus, use_container_width=True)
            st.success("Clustering executed.")

    # --- Regression ---
    with ml_subtab3:
        st.header("Regression Analysis")
        if st.button("💰 Run Regression Algorithm", key="run_regress"):
            df_ml = df.copy()
            le = LabelEncoder()
            categorical_cols = [
                'Age_Group','Gender','Nationality','Status','Location','Living_Situation',
                'Monthly_Food_Budget_AED','Cooking_Frequency','Current_Spending_Per_Meal_AED',
                'Delivery_Frequency','Meals_Per_Week'
            ]
            for col in categorical_cols:
                df_ml[col + '_Encoded'] = le.fit_transform(df_ml[col])
            feature_cols = [col for col in df_ml.columns if col.endswith('_Encoded')] + [
                'Interest_Level', 'Subscription_Preference', 'Taste_Satisfaction',
                'Healthiness_Satisfaction', 'Affordability_Satisfaction',
                'Convenience_Satisfaction', 'Variety_Satisfaction'
            ]
            y_reg = df_ml['WTP_Per_Meal_AED']
            X_reg = df_ml[feature_cols]
            X_train, X_test, y_train, y_test = train_test_split(X_reg, y_reg, test_size=0.3, random_state=42)
            lr = LinearRegression()
            lr.fit(X_train, y_train)
            y_pred = lr.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            col1, col2, col3 = st.columns(3)
            col1.metric("RMSE", f"{rmse:.2f} AED")
            col2.metric("MAE", f"{mae:.2f} AED")
            col3.metric("R² Score", f"{r2:.3f}")
            reg_comparison = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
            fig_reg = px.scatter(reg_comparison, x='Actual', y='Predicted',
                                title='Actual vs Predicted WTP',
                                labels={'Actual':'Actual WTP (AED)','Predicted':'Predicted WTP (AED)'},
                                trendline='ols')
            st.plotly_chart(fig_reg, use_container_width=True)
            st.success("Regression analysis executed.")

# --- Tab 3, 4: Customer Prediction & Upload ---
# (Retain your earlier code for prediction and uploading datasets. Not repeated for brevity.)

st.markdown("---")
st.markdown("**Taste From Home Dashboard** | Built with Streamlit | © 2025")
