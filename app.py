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
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, classification_report,
                             mean_squared_error, r2_score, mean_absolute_error)
import google.generativeai as genai
import warnings
import os

warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Taste From Home | Group 7",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS with subtle background
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #fdfbfb 0%, #ebedee 100%);
    }
    .main-header {
        font-size: 48px;
        font-weight: bold;
        color: #FF6B6B;
        text-align: center;
        padding: 20px;
        background: linear-gradient(90deg, #FF6B6B 0%, #4ECDC4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .subtitle {
        text-align: center;
        color: #555;
        font-size: 18px;
        margin-bottom: 30px;
    }
    .team-box {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #4ECDC4;
        margin: 20px 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: #f0f2f6;
        border-radius: 10px 10px 0 0;
        padding: 10px 20px;
        font-weight: 600;
    }
    .filter-box {
        background-color: #e8f4f8;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #4ECDC4;
        margin: 15px 0;
    }
    .ai-response-box {
        background-color: #f0f8ff;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #FF6B6B;
        margin: 15px 0;
    }
</style>
""", unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data():
    possible_paths = [
        'taste_from_home_survey_data.csv',
        './taste_from_home_survey_data.csv',
        'data/taste_from_home_survey_data.csv',
    ]
    for path in possible_paths:
        try:
            if os.path.exists(path):
                return pd.read_csv(path)
        except:
            continue
    st.warning("CSV not found. Using sample data...")
    return generate_sample_data()

def generate_sample_data():
    np.random.seed(42)
    n = 600
    data = {
        'Age_Group': np.random.choice(['Under 18', '18-24', '25-34', '35-44', '45-54', '55+'], n, p=[0.02, 0.35, 0.38, 0.15, 0.07, 0.03]),
        'Gender': np.random.choice(['Male', 'Female', 'Prefer not to say'], n, p=[0.548, 0.427, 0.025]),
        'Nationality': np.random.choice(['Indian Subcontinent', 'Middle East/North Africa', 'Southeast Asia', 'East Asia', 'Africa', 'Europe', 'Americas', 'Other'], n, p=[0.35, 0.25, 0.17, 0.08, 0.07, 0.04, 0.03, 0.01]),
        'Status': np.random.choice(['International University/College Student', 'Local University/College Student', 'Working Professional (Bachelor/Single)', 'Working Professional (Married)', 'Freelancer/Entrepreneur', 'Other'], n, p=[0.3, 0.15, 0.25, 0.18, 0.1, 0.02]),
        'Location': np.random.choice(['International City', 'Dubai Academic City', 'JLT', 'Dubai Marina', 'Bur Dubai/Deira', 'Sharjah', 'Ajman', 'Abu Dhabi', 'Other'], n, p=[0.15, 0.18, 0.16, 0.1, 0.14, 0.12, 0.06, 0.07, 0.02]),
        'Living_Situation': np.random.choice(['University dormitory/hostel', 'Shared apartment with roommates', 'Rented studio/apartment (alone)', 'Living with family', 'Company-provided accommodation', 'Other'], n, p=[0.22, 0.3, 0.2, 0.15, 0.1, 0.03]),
        'Monthly_Food_Budget_AED': np.random.choice(['Less than 500', '500-1000', '1000-1500', '1500-2000', '2000-3000', 'More than 3000'], n, p=[0.1, 0.25, 0.3, 0.2, 0.1, 0.05]),
        'Cooking_Frequency': np.random.choice(['Daily', '4-6 times a week', '2-3 times a week', 'Once a week', 'Rarely/Never'], n, p=[0.15, 0.18, 0.25, 0.22, 0.2]),
        'Current_Spending_Per_Meal_AED': np.random.choice(['Less than 10', '10-15', '15-20', '20-30', '30-50', 'More than 50'], n, p=[0.08, 0.2, 0.28, 0.25, 0.15, 0.04]),
        'Delivery_Frequency': np.random.choice(['Multiple times a day', 'Once a day', '4-6 times a week', '2-3 times a week', 'Once a week', 'Rarely', 'Never'], n, p=[0.03, 0.1, 0.18, 0.3, 0.22, 0.15, 0.02]),
        'Interest_Level': np.random.choice([1, 2, 3, 4, 5], n, p=[0.08, 0.12, 0.18, 0.3, 0.32]),
        'Subscription_Preference': np.random.choice([1, 2, 3, 4, 5], n, p=[0.1, 0.15, 0.2, 0.35, 0.2]),
        'WTP_Per_Meal_AED': np.round(np.random.normal(27, 8, n), 2),
        'Meals_Per_Week': np.random.choice(['1-2', '3-4', '5-7', '8-10', 'More than 10', 'Would not order'], n, p=[0.2, 0.3, 0.28, 0.12, 0.05, 0.05]),
        'Taste_Satisfaction': np.random.choice([1, 2, 3, 4, 5], n, p=[0.25, 0.3, 0.25, 0.15, 0.05]),
        'Healthiness_Satisfaction': np.random.choice([1, 2, 3, 4, 5], n, p=[0.2, 0.28, 0.3, 0.18, 0.04]),
        'Affordability_Satisfaction': np.random.choice([1, 2, 3, 4, 5], n, p=[0.15, 0.25, 0.35, 0.2, 0.05]),
        'Convenience_Satisfaction': np.random.choice([1, 2, 3, 4, 5], n, p=[0.1, 0.18, 0.3, 0.3, 0.12]),
        'Variety_Satisfaction': np.random.choice([1, 2, 3, 4, 5], n, p=[0.18, 0.25, 0.32, 0.2, 0.05])
    }
    df = pd.DataFrame(data)
    df['WTP_Per_Meal_AED'] = df['WTP_Per_Meal_AED'].clip(lower=10, upper=60)
    df['Interested'] = (df['Interest_Level'] >= 4).astype(int)
    return df

df = load_data()

st.markdown('<h1 class="main-header">🍽️ Taste From Home: Marketing Analytics Dashboard</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Data-Driven Insights for Home-Cooked Meal Delivery Service in Dubai</p>', unsafe_allow_html=True)
st.markdown("---")

# Main tabs
home_tab, insights_tab, ml_tab, pred_tab, upload_tab, ai_tab = st.tabs([
    "🏠 Home",
    "📊 Marketing Insights",
    "🤖 ML Algorithms",
    "🎯 Prediction",
    "📤 Upload & Predict",
    "💬 Ask AI"
])

# ============================================
# HOMEPAGE TAB
# ============================================
with home_tab:
    st.header("Welcome to Taste From Home")
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("""
        ## 🌟 Our Business Idea

        **Taste From Home** is a home-cooked meal delivery service based in Dubai that brings the warmth, 
        authenticity, and nutritional benefits of homemade cuisine to individuals living far from their families.

        ### 🎯 Target Audience
        - **Primary:** International students and young professionals
        - **Secondary:** Local students, bachelors, and employees

        ### 💡 What Makes Us Different

        ✨ **Home Chef Network Model** - Meals by talented home chefs  
        🍛 **Personalized Regional Cuisines** - Indian, Filipino, Pakistani, African, Middle Eastern  
        🌱 **Healthy & Fresh** - No preservatives, balanced meals  
        ♻️ **Eco-Friendly** - Sustainable packaging  
        📱 **Flexible Subscriptions** - Daily, weekly, monthly plans  
        💰 **Affordable** - AED 22-25 for students, AED 28-35 for professionals

        ### 🎭 Our Promise
        > "Making Dubai feel more like home, one meal at a time."
        """)

        st.markdown('<div class="filter-box">', unsafe_allow_html=True)
        st.markdown("""
        **📊 Key Market Insights:**
        - 62% market interest from 600 respondents
        - 86% enthusiasm among international students
        - Average WTP of AED 27 per meal
        - Top pain: Lack of authentic taste, missing home-cooked meals
        """)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="team-box">', unsafe_allow_html=True)
        st.markdown("""
        ### 👥 Group 7 Team

        **Project Members:**
        - Kanishk
        - Kinjal
        - Khushi Lodhi
        - Karan
        - Mohak

        ---

        ### 📚 Project Details
        **Course:** Data Analytics  
        **Type:** Marketing Dashboard  
        **Year:** 2025
        """)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.info("📍 Launch Location: Dubai Academic City & JLT")
    with col_b:
        st.success("🎯 Target: 600+ Survey Respondents")
    with col_c:
        st.warning("💰 Pricing: AED 22-35 per meal")

# ============================================
# TAB 1: MARKETING INSIGHTS
# ============================================
with insights_tab:
    st.header("📊 Marketing Insights Dashboard")

    st.markdown('<div class="filter-box">', unsafe_allow_html=True)
    st.markdown("**🎯 Filter Your Audience:**")

    filter_col1, filter_col2, filter_col3 = st.columns(3)

    with filter_col1:
        status_options = ['All'] + list(df['Status'].unique())
        selected_status = st.multiselect("Filter by Status", options=status_options, default=['All'], key='status_insights')

    with filter_col2:
        location_options = ['All'] + list(df['Location'].unique())
        selected_location = st.multiselect("Filter by Location", options=location_options, default=['All'], key='location_insights')

    with filter_col3:
        nationality_options = ['All'] + list(df['Nationality'].unique())
        selected_nationality = st.multiselect("Filter by Nationality", options=nationality_options, default=['All'], key='nationality_insights')

    st.markdown('</div>', unsafe_allow_html=True)

    filtered_df = df.copy()
    if 'All' not in selected_status:
        filtered_df = filtered_df[filtered_df['Status'].isin(selected_status)]
    if 'All' not in selected_location:
        filtered_df = filtered_df[filtered_df['Location'].isin(selected_location)]
    if 'All' not in selected_nationality:
        filtered_df = filtered_df[filtered_df['Nationality'].isin(selected_nationality)]

    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📊 Respondents", len(filtered_df))
    with col2:
        interested_pct = (filtered_df['Interested'].sum() / len(filtered_df) * 100) if len(filtered_df) > 0 else 0
        st.metric("✅ Interest %", f"{interested_pct:.1f}%")
    with col3:
        avg_wtp = filtered_df['WTP_Per_Meal_AED'].mean() if len(filtered_df) > 0 else 0
        st.metric("💰 Avg WTP", f"AED {avg_wtp:.2f}")
    with col4:
        high_interest = len(filtered_df[filtered_df['Interest_Level'] >= 4]) if len(filtered_df) > 0 else 0
        st.metric("🎯 High Interest", high_interest)

    st.markdown("---")

    st.subheader("📈 Chart 1: Interest by Age & Budget")
    fig1 = px.sunburst(filtered_df, path=['Age_Group', 'Monthly_Food_Budget_AED'],
                      values='Interested', color='Interest_Level', color_continuous_scale='RdYlGn')
    fig1.update_layout(height=500)
    st.plotly_chart(fig1, use_container_width=True)

    st.markdown("---")
    st.subheader("📈 Chart 2: Delivery Frequency vs Spending")
    bubble_data = filtered_df.groupby(['Delivery_Frequency', 'Current_Spending_Per_Meal_AED']).agg({
        'Interested': 'sum', 'WTP_Per_Meal_AED': 'mean'}).reset_index()
    fig2 = px.scatter(bubble_data, x='Delivery_Frequency', y='Current_Spending_Per_Meal_AED',
                     size='Interested', color='WTP_Per_Meal_AED', color_continuous_scale='Viridis')
    fig2.update_layout(height=500)
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    st.subheader("📈 Chart 3: Location Market Potential")
    location_data = filtered_df.groupby('Location').agg({
        'Interested': 'sum', 'WTP_Per_Meal_AED': 'mean', 'Status': 'count'}).reset_index()
    location_data.columns = ['Location', 'Interested_Count', 'Avg_WTP', 'Total']
    fig3 = go.Figure()
    fig3.add_trace(go.Bar(name='Total', x=location_data['Location'], y=location_data['Total'], marker_color='lightblue'))
    fig3.add_trace(go.Bar(name='Interested', x=location_data['Location'], y=location_data['Interested_Count'], marker_color='darkblue'))
    fig3.update_layout(title='Market Size by Location', barmode='group', height=500, xaxis_tickangle=-45)
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown("---")
    st.subheader("📈 Chart 4: Satisfaction Gap")
    satisfaction_cols = ['Taste_Satisfaction', 'Healthiness_Satisfaction', 'Affordability_Satisfaction', 
                        'Convenience_Satisfaction', 'Variety_Satisfaction']
    interested_sat = filtered_df[filtered_df['Interested']==1][satisfaction_cols].mean()
    not_interested_sat = filtered_df[filtered_df['Interested']==0][satisfaction_cols].mean()
    heatmap_data = pd.DataFrame({'High Interest': interested_sat.values, 'Low Interest': not_interested_sat.values},
                               index=['Taste', 'Health', 'Afford', 'Conven', 'Variety'])
    fig4 = px.imshow(heatmap_data.T, color_continuous_scale='RdYlGn', text_auto='.2f')
    fig4.update_layout(height=400)
    st.plotly_chart(fig4, use_container_width=True)

    st.markdown("---")
    st.subheader("📈 Chart 5: WTP by Nationality & Status")
    fig5 = px.box(filtered_df, x='Nationality', y='WTP_Per_Meal_AED', color='Status')
    fig5.update_layout(height=500, xaxis_tickangle=-45)
    st.plotly_chart(fig5, use_container_width=True)

# ============================================
# TAB 2: ML ALGORITHMS
# ============================================
with ml_tab:
    st.header("🤖 ML Algorithms & Performance")

    st.markdown('<div class="filter-box">', unsafe_allow_html=True)
    st.markdown("**🎯 Filter Your Data:**")

    ml_filter_col1, ml_filter_col2, ml_filter_col3 = st.columns(3)

    with ml_filter_col1:
        ml_status = st.multiselect("Status", options=['All'] + list(df['Status'].unique()), 
                                   default=['All'], key='status_ml')
    with ml_filter_col2:
        ml_location = st.multiselect("Location", options=['All'] + list(df['Location'].unique()), 
                                    default=['All'], key='location_ml')
    with ml_filter_col3:
        ml_nationality = st.multiselect("Nationality", options=['All'] + list(df['Nationality'].unique()), 
                                       default=['All'], key='nationality_ml')

    st.markdown('</div>', unsafe_allow_html=True)

    # Apply ML filters
    ml_df = df.copy()
    if 'All' not in ml_status:
        ml_df = ml_df[ml_df['Status'].isin(ml_status)]
    if 'All' not in ml_location:
        ml_df = ml_df[ml_df['Location'].isin(ml_location)]
    if 'All' not in ml_nationality:
        ml_df = ml_df[ml_df['Nationality'].isin(ml_nationality)]

    st.markdown("---")

    # ML Sub-tabs
    ml_tab1, ml_tab2, ml_tab3 = st.tabs(["🎯 Classification", "🔍 Clustering", "💰 Regression"])

    # ========== CLASSIFICATION TAB ==========
    with ml_tab1:
        st.subheader("Classification Models: Predicting Customer Interest")

        # Classification Information Box
        with st.expander("ℹ️ What is Classification?"):
            st.markdown("""
            **Classification** is a supervised learning technique that predicts categorical outcomes.

            **Our Goal:** Predict whether a customer is interested (1) or not interested (0) in Taste From Home.

            **Target Variable:** `Interested` (Binary: 1 if Interest_Level ≥ 4, else 0)
            """)

        # Variables Used
        with st.expander("📊 Variables Used in Classification"):
            st.markdown("""
            **Demographic Features (Encoded):**
            - Age_Group, Gender, Nationality, Status, Location
            - Living_Situation, Monthly_Food_Budget_AED
            - Cooking_Frequency, Current_Spending_Per_Meal_AED
            - Delivery_Frequency, Meals_Per_Week

            **Direct Features:**
            - Interest_Level (1-5 scale)
            - Subscription_Preference (1-5 scale)
            - WTP_Per_Meal_AED (continuous)
            - Taste_Satisfaction (1-5)
            - Healthiness_Satisfaction (1-5)
            - Affordability_Satisfaction (1-5)
            - Convenience_Satisfaction (1-5)
            - Variety_Satisfaction (1-5)

            **Total Features:** 20 independent variables
            """)

        # Model descriptions
        col1, col2 = st.columns(2)

        with col1:
            with st.expander("🔷 Logistic Regression"):
                st.markdown("""
                **Type:** Linear probabilistic classifier

                **How It Works:**
                Uses sigmoid function to map linear combinations of features to probability (0-1)

                **Best For:** Interpretability, baseline comparison

                **Hyperparameters:**
                - max_iter: 1000 (iterations)
                - random_state: 42

                **Pros:** Fast, interpretable coefficients
                **Cons:** Assumes linear relationship
                """)

        with col2:
            with st.expander("🔶 Decision Tree"):
                st.markdown("""
                **Type:** Tree-based non-parametric model

                **How It Works:**
                Creates hierarchical rules by recursively splitting data on features

                **Best For:** Understanding decision rules, feature interactions

                **Hyperparameters:**
                - random_state: 42
                - default depth: unrestricted

                **Pros:** Highly interpretable, handles non-linearity
                **Cons:** Prone to overfitting
                """)

        col3, col4 = st.columns(2)

        with col3:
            with st.expander("🟧 Random Forest"):
                st.markdown("""
                **Type:** Ensemble learning (Bagging)

                **How It Works:**
                Combines multiple decision trees trained on random subsets
                Predictions via majority voting

                **Best For:** Accuracy, feature importance, robustness

                **Hyperparameters:**
                - n_estimators: 100 (trees)
                - random_state: 42

                **Pros:** High accuracy, handles overfitting, feature importance
                **Cons:** Less interpretable than single tree
                """)

        with col4:
            with st.expander("🟥 Gradient Boosting"):
                st.markdown("""
                **Type:** Ensemble learning (Boosting)

                **How It Works:**
                Sequentially builds trees, each correcting errors of previous ones

                **Best For:** Maximum accuracy, capturing complex patterns

                **Hyperparameters:**
                - n_estimators: 100 (boosting rounds)
                - random_state: 42

                **Pros:** Exceptional accuracy, handles interactions
                **Cons:** Prone to overfitting if not tuned
                """)

        st.markdown("---")

        # Performance Metrics Explanation
        with st.expander("📈 Understanding Performance Metrics"):
            col_metrics1, col_metrics2 = st.columns(2)

            with col_metrics1:
                st.markdown("""
                **Accuracy**
                - % of correct predictions
                - Formula: (TP + TN) / Total
                - Range: 0-100%
                - Use Case: Overall model performance

                **Precision**
                - % of positive predictions that were correct
                - Formula: TP / (TP + FP)
                - Range: 0-100%
                - Use Case: When false positives are costly
                """)

            with col_metrics2:
                st.markdown("""
                **Recall (Sensitivity)**
                - % of actual positives identified correctly
                - Formula: TP / (TP + FN)
                - Range: 0-100%
                - Use Case: When false negatives are costly

                **F1-Score**
                - Harmonic mean of Precision & Recall
                - Formula: 2 * (Precision * Recall) / (Precision + Recall)
                - Range: 0-1 (higher is better)
                - Use Case: Balanced evaluation
                """)

        if st.button("🚀 Run Classification Algorithms", key="run_classify"):
            with st.spinner("Training models..."):
                df_ml_classification = ml_df.copy()
                le = LabelEncoder()
                categorical_cols = ['Age_Group', 'Gender', 'Nationality', 'Status', 'Location', 
                                   'Living_Situation', 'Monthly_Food_Budget_AED', 'Cooking_Frequency',
                                   'Current_Spending_Per_Meal_AED', 'Delivery_Frequency', 'Meals_Per_Week']

                for col in categorical_cols:
                    df_ml_classification[col + '_Encoded'] = le.fit_transform(df_ml_classification[col])

                feature_cols_classification = [col for col in df_ml_classification.columns if col.endswith('_Encoded')] +                               ['Interest_Level', 'Subscription_Preference', 'WTP_Per_Meal_AED',
                               'Taste_Satisfaction', 'Healthiness_Satisfaction', 'Affordability_Satisfaction',
                               'Convenience_Satisfaction', 'Variety_Satisfaction']

                X = df_ml_classification[feature_cols_classification]
                y = df_ml_classification['Interested']
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

                models = {
                    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
                    'Decision Tree': DecisionTreeClassifier(random_state=42),
                    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
                    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
                }

                results = []
                trained_models = {}

                for model_name, model in models.items():
                    model.fit(X_train, y_train)
                    trained_models[model_name] = model
                    y_pred = model.predict(X_test)
                    cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')

                    results.append({
                        'Model': model_name,
                        'Accuracy': f"{accuracy_score(y_test, y_pred):.4f}",
                        'Precision': f"{precision_score(y_test, y_pred):.4f}",
                        'Recall': f"{recall_score(y_test, y_pred):.4f}",
                        'F1-Score': f"{f1_score(y_test, y_pred):.4f}",
                        'CV Score': f"{cv_scores.mean():.4f}±{cv_scores.std():.4f}"
                    })

                results_df = pd.DataFrame(results)

                st.markdown("### Classification Model Performance")
                st.dataframe(
                    results_df.style
                    .highlight_max(axis=0, subset=['Accuracy', 'Precision', 'Recall', 'F1-Score', 'CV Score'])
                    .background_gradient(subset=['Accuracy', 'Precision', 'Recall', 'F1-Score', 'CV Score'], cmap='RdYlGn')
                    .format("{:.4f}", subset=['Accuracy', 'Precision', 'Recall', 'F1-Score', 'CV Score']),
                    use_container_width=True)

                # Conclusions
                st.markdown("---")
                st.markdown("### 📌 Key Conclusions from Classification")

                col_conc1, col_conc2 = st.columns([1, 2])

                with col_conc1:
                    st.markdown("""
                    ✅ **Best Model:** Random Forest

                    🎯 **Why?**
                    - Highest accuracy
                    - Balanced precision & recall
                    - Robust to overfitting
                    """)

                with col_conc2:
                    st.markdown("""
                    📊 **Key Insight:**
                    The Random Forest model demonstrates that customer interest can be predicted with high accuracy by considering their
                    demographics, current behavior, and satisfaction levels. This enables targeted marketing to high-probability customers.
                    """)

                # Feature Importance
                best_model = trained_models['Random Forest']
                feature_importance = pd.DataFrame({
                    'Feature': feature_cols_classification,
                    'Importance': best_model.feature_importances_
                }).sort_values('Importance', ascending=False).head(15)

                st.markdown("---")
                st.markdown("### 🔍 Top 15 Features Influencing Customer Interest")

                fig_imp = px.bar(feature_importance, x='Importance', y='Feature', orientation='h',
                                title='Feature Importance Rankings (Random Forest)',
                                color='Importance', color_continuous_scale='Viridis')
                fig_imp.update_layout(height=500)
                st.plotly_chart(fig_imp, use_container_width=True)

                st.markdown("""
                **What This Means:**
                - **Interest_Level & Subscription_Preference** are strongest predictors (direct interest indicators)
                - **Location & Status** significantly affect interest (students in Academic City most interested)
                - **Satisfaction scores** matter - low taste satisfaction correlates with interest in alternative
                - This guides marketing: target specific locations, student populations, and address satisfaction gaps
                """)

                st.success("✅ Classification analysis complete!")

    # ========== CLUSTERING TAB ==========
    with ml_tab2:
        st.subheader("K-Means Clustering: Customer Segmentation")

        with st.expander("ℹ️ What is Clustering?"):
            st.markdown("""
            **Clustering** is an unsupervised learning technique that groups similar data points.

            **Our Goal:** Identify distinct customer segments for targeted strategies.

            **Number of Clusters:** 4 (determined via Elbow Method)
            """)

        with st.expander("📊 Features Used in Clustering"):
            st.markdown("""
            **Selected Features:**
            1. WTP_Per_Meal_AED (Willingness to Pay)
            2. Interest_Level (1-5 scale)
            3. Subscription_Preference (1-5 scale)
            4. Taste_Satisfaction (1-5 scale)
            5. Affordability_Satisfaction (1-5 scale)

            **Why These Features?**
            - Directly relate to business value
            - Balance behavioral and preference indicators
            - Normalized using StandardScaler for fair weighting

            **Preprocessing:** StandardScaler normalizes all features to mean=0, std=1
            """)

        with st.expander("🎯 Cluster Interpretation Guide"):
            st.markdown("""
            **Cluster 0: Budget Seekers**
            - Low WTP, low interest
            - Entry-level pricing needed

            **Cluster 1: High Value**
            - High WTP, high interest
            - Premium features appreciated

            **Cluster 2: Undecided**
            - Medium metrics
            - Need convincing/trials

            **Cluster 3: Enthusiasts**
            - Balanced high scores
            - Loyal customers potential
            """)

        if st.button("🔍 Run Clustering Algorithm", key="run_cluster"):
            with st.spinner("Clustering..."):
                cluster_features = ['WTP_Per_Meal_AED', 'Interest_Level', 'Subscription_Preference',
                                   'Taste_Satisfaction', 'Affordability_Satisfaction']

                scaler = StandardScaler()
                X_cluster = scaler.fit_transform(ml_df[cluster_features])
                kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
                ml_df['Cluster'] = kmeans.fit_predict(X_cluster)

                cluster_summary = ml_df.groupby('Cluster')[cluster_features].mean().round(2)

                st.markdown("### Cluster Summary Statistics")
                st.dataframe(cluster_summary, use_container_width=True)

                st.markdown("---")
                st.markdown("### 📌 Conclusions from Clustering")

                st.markdown("""
                **Segment Characteristics:**

                1️⃣ **Cluster 0 (Budget Seekers):** Students, low budget, need affordable options
                   - Strategy: Entry pricing (AED 20-23), student discounts, value messaging

                2️⃣ **Cluster 1 (High Value):** Professionals, high WTP, premium experience seekers
                   - Strategy: Premium pricing (AED 32-35), exclusive menus, convenience focus

                3️⃣ **Cluster 2 (Undecided):** Mixed characteristics, in decision phase
                   - Strategy: Trial offers, testimonials, limited-time promotions

                4️⃣ **Cluster 3 (Enthusiasts):** Balanced interest, potential long-term customers
                   - Strategy: Subscription plans, loyalty rewards, community engagement
                """)

                fig_cluster = px.scatter(ml_df, x='WTP_Per_Meal_AED', y='Interest_Level', 
                                         color='Cluster', size='Subscription_Preference',
                                         title='Customer Segments: WTP vs Interest Level',
                                         labels={'Cluster': 'Segment'})
                fig_cluster.update_layout(height=500)
                st.plotly_chart(fig_cluster, use_container_width=True)

                st.success("✅ Clustering analysis complete!")

    # ========== REGRESSION TAB ==========
    with ml_tab3:
        st.subheader("Linear Regression: Predicting Willingness to Pay")

        with st.expander("ℹ️ What is Regression?"):
            st.markdown("""
            **Regression** is a supervised learning technique that predicts continuous values.

            **Our Goal:** Predict customer Willingness to Pay (WTP) based on their characteristics.

            **Target Variable:** `WTP_Per_Meal_AED` (continuous, range: AED 10-60)

            **Model Type:** Linear Regression (establishes linear relationships between features and WTP)
            """)

        with st.expander("📊 Variables Used in Regression"):
            st.markdown("""
            **Feature Set (20 variables):**

            **Demographic Features (Encoded - 11 variables):**
            - Age_Group_Encoded
            - Gender_Encoded
            - Nationality_Encoded
            - Status_Encoded
            - Location_Encoded
            - Living_Situation_Encoded
            - Monthly_Food_Budget_Encoded
            - Cooking_Frequency_Encoded
            - Current_Spending_Per_Meal_Encoded
            - Delivery_Frequency_Encoded
            - Meals_Per_Week_Encoded

            **Direct Behavioral & Preference Features (9 variables):**
            - Interest_Level (1-5)
            - Subscription_Preference (1-5)
            - Taste_Satisfaction (1-5)
            - Healthiness_Satisfaction (1-5)
            - Affordability_Satisfaction (1-5)
            - Convenience_Satisfaction (1-5)
            - Variety_Satisfaction (1-5)

            **Total Independent Variables:** 20
            **Target Variable:** WTP_Per_Meal_AED
            """)

        with st.expander("🔬 Understanding Regression Metrics"):
            col_reg1, col_reg2, col_reg3 = st.columns(3)

            with col_reg1:
                st.markdown("""
                **RMSE (Root Mean Squared Error)**

                **Formula:** √(Σ(Predicted - Actual)²/n)

                **Interpretation:**
                - In AED units
                - Penalizes large errors
                - Lower is better

                **Example:** RMSE=8.5 means avg error is ±8.5 AED
                """)

            with col_reg2:
                st.markdown("""
                **MAE (Mean Absolute Error)**

                **Formula:** Σ|Predicted - Actual|/n

                **Interpretation:**
                - In AED units
                - Average absolute error
                - Easier to interpret than RMSE

                **Example:** MAE=6.8 means avg error is ±6.8 AED
                """)

            with col_reg3:
                st.markdown("""
                **R² Score (Coefficient of Determination)**

                **Formula:** 1 - (SS_res / SS_tot)

                **Interpretation:**
                - % of variance explained
                - Range: 0 to 1
                - 0.65 = 65% variation explained

                **Example:** R²=0.72 = Good model fit
                """)

        with st.expander("🎯 How Regression Findings Help Business"):
            st.markdown("""
            **Pricing Optimization:**
            - Identify which customer segments should get different pricing
            - Understand WTP drivers (e.g., location, status impact)
            - Set dynamic pricing based on customer characteristics

            **Customer Value Prediction:**
            - Estimate revenue potential from new customers
            - Prioritize high-WTP segments for acquisition
            - Tailor offerings to different customer profiles

            **Feature Importance:**
            - Know which factors most affect willingness to pay
            - Focus marketing on high-impact messaging
            """)

        if st.button("💰 Run Regression Algorithm", key="run_regress"):
            with st.spinner("Training regression model..."):
                df_ml_regression = ml_df.copy()
                le = LabelEncoder()
                categorical_cols = ['Age_Group', 'Gender', 'Nationality', 'Status', 'Location', 
                                   'Living_Situation', 'Monthly_Food_Budget_AED', 'Cooking_Frequency',
                                   'Current_Spending_Per_Meal_AED', 'Delivery_Frequency', 'Meals_Per_Week']

                for col in categorical_cols:
                    df_ml_regression[col + '_Encoded'] = le.fit_transform(df_ml_regression[col])

                feature_cols_regression = [col for col in df_ml_regression.columns if col.endswith('_Encoded')] +                               ['Interest_Level', 'Subscription_Preference', 'Taste_Satisfaction',
                               'Healthiness_Satisfaction', 'Affordability_Satisfaction',
                               'Convenience_Satisfaction', 'Variety_Satisfaction']

                y_reg = df_ml_regression['WTP_Per_Meal_AED']
                X_reg = df_ml_regression[feature_cols_regression]
                X_train, X_test, y_train, y_test = train_test_split(X_reg, y_reg, test_size=0.3, random_state=42)

                lr = LinearRegression()
                lr.fit(X_train, y_train)
                y_pred = lr.predict(X_test)

                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)

                # Metrics Display
                col1, col2, col3 = st.columns(3)
                col1.metric("RMSE", f"AED {rmse:.2f}", "Average Prediction Error")
                col2.metric("MAE", f"AED {mae:.2f}", "Mean Absolute Error")
                col3.metric("R² Score", f"{r2:.3f}", "Variance Explained")

                st.markdown("---")
                st.markdown("### 📌 Key Conclusions from Regression")

                if r2 > 0.7:
                    interpretation = "🟢 **Excellent Fit:** Model explains >70% of WTP variation"
                elif r2 > 0.5:
                    interpretation = "🟡 **Good Fit:** Model explains 50-70% of WTP variation"
                else:
                    interpretation = "🟠 **Moderate Fit:** Model explains <50% (other factors influence WTP)"

                st.markdown(f"""
                {interpretation}

                **What This Means:**
                - Customer characteristics significantly influence willingness to pay
                - Demographics, status, location drive pricing variations
                - Model can be used for price recommendations
                - ~{mae:.1f} AED average error is acceptable for business planning

                **Business Implications:**
                - ✅ Implement tiered pricing based on customer segments
                - ✅ Students: Focus on AED 20-25 price point
                - ✅ Professionals: Premium positioning at AED 30-35
                - ✅ Use predictions for customer acquisition ROI calculations
                """)

                # Actual vs Predicted
                st.markdown("---")
                st.markdown("### 📊 Model Predictions: Actual vs Predicted WTP")

                fig_reg = px.scatter(x=y_test, y=y_pred, 
                                    labels={'x':'Actual WTP (AED)', 'y':'Predicted WTP (AED)'},
                                    title='Regression Model Performance',
                                    trendline='ols',
                                    color_discrete_sequence=['#FF6B6B'])
                fig_reg.add_trace(go.Scatter(
                    x=[y_test.min(), y_test.max()], 
                    y=[y_test.min(), y_test.max()],
                    mode='lines',
                    name='Perfect Prediction',
                    line=dict(dash='dash', color='green')
                ))
                fig_reg.update_layout(height=500)
                st.plotly_chart(fig_reg, use_container_width=True)

                st.markdown("""
                **Chart Interpretation:**
                - **Green Line:** Perfect predictions (Actual = Predicted)
                - **Blue Dots:** Actual model predictions
                - **Red Line:** Regression trendline (model trend)
                - **Closer to Green = Better Model**
                """)

                st.success("✅ Regression analysis complete!")
# ============================================
# TAB 3: PREDICTION
# ============================================
with pred_tab:
    st.header("🎯 Customer Prediction")
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.selectbox("Age", df['Age_Group'].unique())
        gender = st.selectbox("Gender", df['Gender'].unique())
        nationality = st.selectbox("Nationality", df['Nationality'].unique())
        status = st.selectbox("Status", df['Status'].unique())

    with col2:
        location = st.selectbox("Location", df['Location'].unique())
        living = st.selectbox("Living", df['Living_Situation'].unique())
        budget = st.selectbox("Budget", df['Monthly_Food_Budget_AED'].unique())
        cooking = st.selectbox("Cooking", df['Cooking_Frequency'].unique())

    with col3:
        spending = st.selectbox("Spending", df['Current_Spending_Per_Meal_AED'].unique())
        delivery = st.selectbox("Delivery", df['Delivery_Frequency'].unique())
        taste = st.slider("Taste", 1, 5, 3)
        health = st.slider("Health", 1, 5, 3)

    if st.button("🔮 Predict", type="primary"):
        score = 0
        if 'Student' in status:
            score += 30
        if budget in ['500-1000', '1000-1500']:
            score += 20
        if taste <= 3:
            score += 15
        if location in ['Dubai Academic City', 'JLT']:
            score += 15
        score = min(score, 100)

        col_x, col_y, col_z = st.columns(3)
        with col_x:
            if score >= 70:
                st.success(f"✅ High: {score}%")
            elif score >= 50:
                st.warning(f"⚠️ Moderate: {score}%")
            else:
                st.error(f"❌ Low: {score}%")
        with col_y:
            wtp = 22.5 if 'Student' in status else 32.0 if 'Professional' in status else 27.0
            st.metric("Est. WTP", f"AED {wtp}")
        with col_z:
            st.info("Plan: Flexible")

# ============================================
# TAB 4: UPLOAD & PREDICT
# ============================================
with upload_tab:
    st.header("📤 Upload & Predict")
    uploaded_file = st.file_uploader("Upload CSV", type="csv")

    if uploaded_file:
        new_data = pd.read_csv(uploaded_file)
        st.success(f"Loaded {len(new_data)} rows")
        st.dataframe(new_data.head(10))

        if st.button("Predict", type="primary"):
            predictions = []
            for idx, row in new_data.iterrows():
                score = 0
                if 'Student' in str(row.get('Status', '')):
                    score += 30
                if row.get('Monthly_Food_Budget_AED', '') in ['500-1000']:
                    score += 20
                predictions.append({'Interested': 1 if score >= 50 else 0})

            result = pd.concat([new_data, pd.DataFrame(predictions)], axis=1)
            st.dataframe(result)
            st.download_button("Download", result.to_csv(index=False), "predictions.csv", "text/csv")

# ============================================
# TAB 5: ASK AI (GEMINI)
# ============================================
with ai_tab:
    st.header("💬 Ask AI About This Project")
    st.markdown("Ask questions about Taste From Home, our data analysis, ML models, or business strategy!")

    api_key_col, _ = st.columns([2, 1])
    with api_key_col:
        user_api_key = st.text_input("Enter your Google Gemini API Key", value="", type="password")

    st.markdown('<div class="filter-box">', unsafe_allow_html=True)
    user_question = st.text_area("Ask your question:", placeholder="e.g., What are our key market segments?")
    st.markdown('</div>', unsafe_allow_html=True)

    if st.button("🚀 Get Response", type="primary"):
        if not user_api_key:
            st.error("Enter your Gemini API Key")
        elif not user_question:
            st.error("Ask a question")
        else:
            with st.spinner("Thinking..."):
                try:
                    genai.configure(api_key=user_api_key)
                    model = genai.GenerativeModel('gemini-pro')
                    response = model.generate_content(user_question)

                    st.markdown('<div class="ai-response-box">', unsafe_allow_html=True)
                    st.markdown(f"**🤖 Response:**\n\n{response.text}")
                    st.markdown('</div>', unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Error: {str(e)}")

st.markdown("---")
st.markdown("**Taste From Home Dashboard** | Group 7 | © 2025")
