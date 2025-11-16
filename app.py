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
home_tab, insights_tab, ml_tab, pred_tab, upload_tab = st.tabs([
    "🏠 Home",
    "📊 Marketing Insights",
    "🤖 ML Algorithms",
    "🎯 Prediction",
    "📤 Upload & Predict"
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
        - Khushi
        - Karan
        - Mohak

        ---

        ### 📚 Project Details
        **Course:** Data Analytics  
        **Type:** Marketing Dashboard  
        **Year:** 2025
        **Institution:** SP Jain School of Global Management, Dubai
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
# TAB 1: MARKETING INSIGHTS (WITH FILTERS INSIDE)
# ============================================
with insights_tab:
    st.header("📊 Marketing Insights Dashboard")

    # FILTERS DIRECTLY IN THIS TAB (NOT SIDEBAR)
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

    # Apply filters
    filtered_df = df.copy()
    if 'All' not in selected_status:
        filtered_df = filtered_df[filtered_df['Status'].isin(selected_status)]
    if 'All' not in selected_location:
        filtered_df = filtered_df[filtered_df['Location'].isin(selected_location)]
    if 'All' not in selected_nationality:
        filtered_df = filtered_df[filtered_df['Nationality'].isin(selected_nationality)]

    st.markdown("---")

    # Metrics
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

    # Chart 1
    st.subheader("📈 Chart 1: Interest by Age & Budget")
    fig1 = px.sunburst(filtered_df, path=['Age_Group', 'Monthly_Food_Budget_AED'],
                      values='Interested', color='Interest_Level', color_continuous_scale='RdYlGn')
    fig1.update_layout(height=500)
    st.plotly_chart(fig1, use_container_width=True)

    # Chart 2
    st.markdown("---")
    st.subheader("📈 Chart 2: Delivery Frequency vs Spending")
    bubble_data = filtered_df.groupby(['Delivery_Frequency', 'Current_Spending_Per_Meal_AED']).agg({
        'Interested': 'sum', 'WTP_Per_Meal_AED': 'mean'}).reset_index()
    fig2 = px.scatter(bubble_data, x='Delivery_Frequency', y='Current_Spending_Per_Meal_AED',
                     size='Interested', color='WTP_Per_Meal_AED', color_continuous_scale='Viridis')
    fig2.update_layout(height=500)
    st.plotly_chart(fig2, use_container_width=True)

    # Chart 3
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

    # Chart 4
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

    # Chart 5
    st.markdown("---")
    st.subheader("📈 Chart 5: WTP by Nationality & Status")
    fig5 = px.box(filtered_df, x='Nationality', y='WTP_Per_Meal_AED', color='Status')
    fig5.update_layout(height=500, xaxis_tickangle=-45)
    st.plotly_chart(fig5, use_container_width=True)

# ============================================
# TAB 2: ML ALGORITHMS (WITH FILTERS INSIDE)
# ============================================
with ml_tab:
    st.header("🤖 ML Algorithms & Performance")

    # FILTERS DIRECTLY IN THIS TAB
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

    # Classification
    with ml_tab1:
        st.subheader("Classification Models")
        if st.button("🚀 Run Classification", key="run_classify"):
            with st.spinner("Training..."):
                df_ml = ml_df.copy()
                le = LabelEncoder()
                categorical_cols = ['Age_Group', 'Gender', 'Nationality', 'Status', 'Location', 
                                   'Living_Situation', 'Monthly_Food_Budget_AED', 'Cooking_Frequency',
                                   'Current_Spending_Per_Meal_AED', 'Delivery_Frequency', 'Meals_Per_Week']
                for col in categorical_cols:
                    df_ml[col + '_Encoded'] = le.fit_transform(df_ml[col])
                feature_cols = [col for col in df_ml.columns if col.endswith('_Encoded')] +                               ['Interest_Level', 'Subscription_Preference', 'WTP_Per_Meal_AED',
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
                    results.append({
                        'Model': model_name,
                        'Accuracy': accuracy_score(y_test, y_pred),
                        'Precision': precision_score(y_test, y_pred),
                        'Recall': recall_score(y_test, y_pred),
                        'F1-Score': f1_score(y_test, y_pred),
                        'CV Score': cross_val_score(model, X, y, cv=5, scoring='accuracy').mean()
                    })

                results_df = pd.DataFrame(results)
                st.dataframe(results_df.style.highlight_max(axis=0), use_container_width=True)

                best_model = models['Random Forest']
                best_model.fit(X_train, y_train)
                feature_importance = pd.DataFrame({
                    'Feature': feature_cols,
                    'Importance': best_model.feature_importances_
                }).sort_values('Importance', ascending=False).head(10)

                fig_imp = px.bar(feature_importance, x='Importance', y='Feature', orientation='h',
                                title='Top 10 Features', color='Importance', color_continuous_scale='Viridis')
                st.plotly_chart(fig_imp, use_container_width=True)
                st.success("✅ Done!")

    # Clustering
    with ml_tab2:
        st.subheader("K-Means Clustering")
        if st.button("🔍 Run Clustering", key="run_cluster"):
            with st.spinner("Clustering..."):
                cluster_features = ['WTP_Per_Meal_AED', 'Interest_Level', 'Subscription_Preference',
                                   'Taste_Satisfaction', 'Affordability_Satisfaction']
                scaler = StandardScaler()
                X_cluster = scaler.fit_transform(ml_df[cluster_features])
                kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
                ml_df['Cluster'] = kmeans.fit_predict(X_cluster)

                cluster_summary = ml_df.groupby('Cluster')[cluster_features].mean().round(1)
                st.dataframe(cluster_summary)

                fig_clus = px.scatter(ml_df, x='WTP_Per_Meal_AED', y='Interest_Level', 
                                     color='Cluster', title='Customer Segments')
                st.plotly_chart(fig_clus, use_container_width=True)
                st.success("✅ Done!")

    # Regression
    with ml_tab3:
        st.subheader("Regression Analysis")
        if st.button("💰 Run Regression", key="run_regress"):
            with st.spinner("Training..."):
                df_ml = ml_df.copy()
                le = LabelEncoder()
                categorical_cols = ['Age_Group', 'Gender', 'Nationality', 'Status', 'Location', 
                                   'Living_Situation', 'Monthly_Food_Budget_AED', 'Cooking_Frequency',
                                   'Current_Spending_Per_Meal_AED', 'Delivery_Frequency', 'Meals_Per_Week']
                for col in categorical_cols:
                    df_ml[col + '_Encoded'] = le.fit_transform(df_ml[col])
                feature_cols = [col for col in df_ml.columns if col.endswith('_Encoded')] +                               ['Interest_Level', 'Subscription_Preference', 'Taste_Satisfaction',
                               'Healthiness_Satisfaction', 'Affordability_Satisfaction',
                               'Convenience_Satisfaction', 'Variety_Satisfaction']
                y_reg = df_ml['WTP_Per_Meal_AED']
                X_reg = df_ml[feature_cols]
                X_train, X_test, y_train, y_test = train_test_split(X_reg, y_reg, test_size=0.3, random_state=42)

                lr = LinearRegression()
                lr.fit(X_train, y_train)
                y_pred = lr.predict(X_test)

                col1, col2, col3 = st.columns(3)
                col1.metric("RMSE", f"{np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
                col2.metric("MAE", f"{mean_absolute_error(y_test, y_pred):.2f}")
                col3.metric("R²", f"{r2_score(y_test, y_pred):.3f}")

                fig_reg = px.scatter(x=y_test, y=y_pred, labels={'x':'Actual', 'y':'Predicted'},
                                    title='Actual vs Predicted', trendline='ols')
                st.plotly_chart(fig_reg, use_container_width=True)
                st.success("✅ Done!")

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

st.markdown("---")
st.markdown("**Taste From Home Dashboard** | Group 7 | © 2025")
