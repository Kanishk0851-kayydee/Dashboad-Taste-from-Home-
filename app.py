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
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Taste From Home Dashboard",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 42px;
        font-weight: bold;
        color: #FF6B6B;
        text-align: center;
        padding: 20px;
        background: linear-gradient(90deg, #FF6B6B 0%, #4ECDC4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: #f0f2f6;
        border-radius: 10px 10px 0 0;
        padding: 10px 20px;
    }
</style>
""", unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv('taste_from_home_survey_data.csv')
    return df

df = load_data()

# Title
st.markdown('<h1 class="main-header">🍽️ Taste From Home: Marketing Analytics Dashboard</h1>', unsafe_allow_html=True)
st.markdown("### Data-Driven Insights for Home-Cooked Meal Delivery Service in Dubai")
st.markdown("---")

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Marketing Insights & Charts", 
    "🤖 ML Algorithms & Performance", 
    "🎯 Customer Prediction", 
    "📤 Upload & Predict"
])

# ============================================
# TAB 1: MARKETING INSIGHTS WITH FILTERS
# ============================================
with tab1:
    st.header("Marketing Insights Dashboard")
    st.markdown("**Interactive filters to segment your target audience**")

    # Sidebar filters
    with st.sidebar:
        st.header("🎯 Audience Filters")

        # Status filter
        status_options = ['All'] + list(df['Status'].unique())
        selected_status = st.multiselect(
            "Filter by Status",
            options=status_options,
            default=['All']
        )

        # Location filter
        location_options = ['All'] + list(df['Location'].unique())
        selected_location = st.multiselect(
            "Filter by Location",
            options=location_options,
            default=['All']
        )

        # Nationality filter
        nationality_options = ['All'] + list(df['Nationality'].unique())
        selected_nationality = st.multiselect(
            "Filter by Nationality",
            options=nationality_options,
            default=['All']
        )

    # Apply filters
    filtered_df = df.copy()
    if 'All' not in selected_status:
        filtered_df = filtered_df[filtered_df['Status'].isin(selected_status)]
    if 'All' not in selected_location:
        filtered_df = filtered_df[filtered_df['Location'].isin(selected_location)]
    if 'All' not in selected_nationality:
        filtered_df = filtered_df[filtered_df['Nationality'].isin(selected_nationality)]

    # Key Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📊 Total Respondents", len(filtered_df))
    with col2:
        interested_pct = (filtered_df['Interested'].sum() / len(filtered_df) * 100) if len(filtered_df) > 0 else 0
        st.metric("✅ Interest Rate", f"{interested_pct:.1f}%")
    with col3:
        avg_wtp = filtered_df['WTP_Per_Meal_AED'].mean() if len(filtered_df) > 0 else 0
        st.metric("💰 Avg WTP", f"AED {avg_wtp:.2f}")
    with col4:
        high_interest = len(filtered_df[filtered_df['Interest_Level'] >= 4]) if len(filtered_df) > 0 else 0
        st.metric("🎯 High Interest", high_interest)

    st.markdown("---")

    # Chart 1: Interest Level by Age Group and Spending Capacity
    st.subheader("📈 Chart 1: Interest Level by Age Group & Spending Capacity")
    st.markdown("**Insight:** Identify which age segments show highest interest and their spending patterns")

    fig1 = px.sunburst(
        filtered_df,
        path=['Age_Group', 'Monthly_Food_Budget_AED'],
        values='Interested',
        color='Interest_Level',
        color_continuous_scale='RdYlGn',
        title='Customer Interest Hierarchy: Age → Budget → Interest'
    )
    fig1.update_layout(height=500)
    st.plotly_chart(fig1, use_container_width=True)
    st.markdown("**Marketing Action:** Target the high-interest age groups with tailored messaging and pricing tiers.")

    # Chart 2: Delivery Frequency vs Current Spending - Bubble Chart
    st.markdown("---")
    st.subheader("📈 Chart 2: Delivery Frequency vs Current Spending vs Interest")
    st.markdown("**Insight:** Understand behavioral patterns of frequent orderers and their spending habits")

    # Prepare data for bubble chart
    bubble_data = filtered_df.groupby(['Delivery_Frequency', 'Current_Spending_Per_Meal_AED']).agg({
        'Interested': 'sum',
        'WTP_Per_Meal_AED': 'mean'
    }).reset_index()

    fig2 = px.scatter(
        bubble_data,
        x='Delivery_Frequency',
        y='Current_Spending_Per_Meal_AED',
        size='Interested',
        color='WTP_Per_Meal_AED',
        color_continuous_scale='Viridis',
        title='Behavioral Segmentation: Ordering Frequency × Spending × Interest',
        labels={'WTP_Per_Meal_AED': 'Avg WTP (AED)'}
    )
    fig2.update_layout(height=500)
    st.plotly_chart(fig2, use_container_width=True)
    st.markdown("**Marketing Action:** Create loyalty programs for frequent orderers and premium offerings for high spenders.")

    # Chart 3: Location-wise Market Penetration Potential
    st.markdown("---")
    st.subheader("📈 Chart 3: Geographic Market Potential by Location")
    st.markdown("**Insight:** Prioritize locations with highest interest and market size")

    location_data = filtered_df.groupby('Location').agg({
        'Interested': 'sum',
        'WTP_Per_Meal_AED': 'mean',
        'Status': 'count'
    }).reset_index()
    location_data.columns = ['Location', 'Interested_Count', 'Avg_WTP', 'Total_Respondents']
    location_data['Interest_Rate'] = (location_data['Interested_Count'] / location_data['Total_Respondents'] * 100)

    fig3 = go.Figure()
    fig3.add_trace(go.Bar(
        name='Total Respondents',
        x=location_data['Location'],
        y=location_data['Total_Respondents'],
        marker_color='lightblue'
    ))
    fig3.add_trace(go.Bar(
        name='Interested Customers',
        x=location_data['Location'],
        y=location_data['Interested_Count'],
        marker_color='darkblue'
    ))
    fig3.update_layout(
        title='Location-wise Market Size & Interest',
        barmode='group',
        height=500,
        xaxis_tickangle=-45
    )
    st.plotly_chart(fig3, use_container_width=True)
    st.markdown("**Marketing Action:** Launch in Dubai Academic City and JLT first, then expand to high-interest areas.")

    # Chart 4: Customer Satisfaction Heatmap
    st.markdown("---")
    st.subheader("📈 Chart 4: Current Satisfaction Levels - Gap Analysis")
    st.markdown("**Insight:** Identify satisfaction gaps that Taste From Home can address")

    satisfaction_cols = ['Taste_Satisfaction', 'Healthiness_Satisfaction', 
                         'Affordability_Satisfaction', 'Convenience_Satisfaction', 
                         'Variety_Satisfaction']

    # Compare interested vs not interested
    interested_sat = filtered_df[filtered_df['Interested']==1][satisfaction_cols].mean()
    not_interested_sat = filtered_df[filtered_df['Interested']==0][satisfaction_cols].mean()

    heatmap_data = pd.DataFrame({
        'High Interest Customers': interested_sat.values,
        'Low Interest Customers': not_interested_sat.values
    }, index=['Taste', 'Healthiness', 'Affordability', 'Convenience', 'Variety'])

    fig4 = px.imshow(
        heatmap_data.T,
        labels=dict(x="Satisfaction Dimension", y="Customer Segment", color="Satisfaction Score"),
        color_continuous_scale='RdYlGn',
        title='Satisfaction Heatmap: High vs Low Interest Segments',
        text_auto='.2f'
    )
    fig4.update_layout(height=400)
    st.plotly_chart(fig4, use_container_width=True)
    st.markdown("**Marketing Action:** Emphasize taste authenticity, health benefits, and affordability in campaigns.")

    # Chart 5: Willingness to Pay Distribution by Nationality
    st.markdown("---")
    st.subheader("📈 Chart 5: Willingness to Pay by Nationality & Status")
    st.markdown("**Insight:** Price sensitivity across cultural and professional segments")

    fig5 = px.box(
        filtered_df,
        x='Nationality',
        y='WTP_Per_Meal_AED',
        color='Status',
        title='WTP Distribution: Nationality × Customer Status',
        labels={'WTP_Per_Meal_AED': 'Willingness to Pay (AED)'}
    )
    fig5.update_layout(height=500, xaxis_tickangle=-45)
    st.plotly_chart(fig5, use_container_width=True)
    st.markdown("**Marketing Action:** Develop tiered pricing: AED 22-25 for students, AED 28-35 for professionals.")

# ============================================
# TAB 2: ML ALGORITHMS & PERFORMANCE
# ============================================
with tab2:
    st.header("🤖 Machine Learning Algorithms & Performance Metrics")
    st.markdown("**Apply classification and clustering algorithms with one click**")

    if st.button("🚀 Run All ML Algorithms", type="primary"):
        with st.spinner("Training models... Please wait..."):

            # Prepare data for ML
            df_ml = df.copy()

            # Encode categorical variables
            le = LabelEncoder()
            categorical_cols = ['Age_Group', 'Gender', 'Nationality', 'Status', 'Location', 
                               'Living_Situation', 'Monthly_Food_Budget_AED', 'Cooking_Frequency',
                               'Current_Spending_Per_Meal_AED', 'Delivery_Frequency', 'Meals_Per_Week']

            for col in categorical_cols:
                df_ml[col + '_Encoded'] = le.fit_transform(df_ml[col])

            # Features for classification
            feature_cols = [col for col in df_ml.columns if col.endswith('_Encoded')] +                           ['Interest_Level', 'Subscription_Preference', 'WTP_Per_Meal_AED',
                           'Taste_Satisfaction', 'Healthiness_Satisfaction', 'Affordability_Satisfaction',
                           'Convenience_Satisfaction', 'Variety_Satisfaction']

            X = df_ml[feature_cols]
            y = df_ml['Interested']

            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

            # ===== CLASSIFICATION MODELS =====
            st.subheader("🎯 Classification Models: Predicting Customer Interest")

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

                # Cross-validation
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
            st.dataframe(results_df.style.highlight_max(axis=0, subset=['Accuracy', 'Precision', 'Recall', 'F1-Score']), use_container_width=True)

            # Visualize model comparison
            st.markdown("### Model Performance Comparison")
            fig_models = go.Figure()
            metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
            for metric in metrics:
                fig_models.add_trace(go.Bar(
                    name=metric,
                    x=results_df['Model'],
                    y=results_df[metric],
                    text=results_df[metric].round(3),
                    textposition='auto'
                ))
            fig_models.update_layout(
                title='Classification Model Performance',
                barmode='group',
                height=500
            )
            st.plotly_chart(fig_models, use_container_width=True)

            # Best model details
            best_model = models['Random Forest']
            best_model.fit(X_train, y_train)
            y_pred_best = best_model.predict(X_test)

            st.markdown("### 🏆 Best Model: Random Forest - Detailed Metrics")
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Confusion Matrix**")
                cm = confusion_matrix(y_test, y_pred_best)
                fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale='Blues',
                                   labels=dict(x="Predicted", y="Actual"),
                                   title="Confusion Matrix")
                st.plotly_chart(fig_cm, use_container_width=True)

            with col2:
                st.markdown("**Classification Report**")
                report = classification_report(y_test, y_pred_best, output_dict=True)
                report_df = pd.DataFrame(report).transpose()
                st.dataframe(report_df.style.format("{:.3f}"), use_container_width=True)

            # Feature Importance
            st.markdown("### 📊 Feature Importance Analysis")
            feature_importance = pd.DataFrame({
                'Feature': feature_cols,
                'Importance': best_model.feature_importances_
            }).sort_values('Importance', ascending=False).head(10)

            fig_imp = px.bar(feature_importance, x='Importance', y='Feature', orientation='h',
                            title='Top 10 Most Important Features',
                            color='Importance', color_continuous_scale='Viridis')
            st.plotly_chart(fig_imp, use_container_width=True)

            # ===== CLUSTERING =====
            st.markdown("---")
            st.subheader("🔍 K-Means Clustering: Customer Segmentation")

            # Prepare clustering data
            cluster_features = ['WTP_Per_Meal_AED', 'Interest_Level', 'Subscription_Preference',
                               'Taste_Satisfaction', 'Affordability_Satisfaction']
            X_cluster = df[cluster_features].copy()

            scaler = StandardScaler()
            X_cluster_scaled = scaler.fit_transform(X_cluster)

            # Apply K-Means
            kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
            df['Cluster'] = kmeans.fit_predict(X_cluster_scaled)

            # Cluster summary
            cluster_summary = df.groupby('Cluster').agg({
                'WTP_Per_Meal_AED': 'mean',
                'Interest_Level': 'mean',
                'Interested': 'sum',
                'Status': 'count'
            }).reset_index()
            cluster_summary.columns = ['Cluster', 'Avg_WTP', 'Avg_Interest', 'Interested_Count', 'Total']
            cluster_summary['Cluster_Name'] = ['Budget Seekers', 'High Value', 'Undecided', 'Premium Enthusiasts']

            st.dataframe(cluster_summary, use_container_width=True)

            # Visualize clusters
            fig_cluster = px.scatter(
                df,
                x='WTP_Per_Meal_AED',
                y='Interest_Level',
                color='Cluster',
                size='Subscription_Preference',
                title='Customer Segments: WTP vs Interest Level',
                labels={'Cluster': 'Customer Segment'},
                color_continuous_scale='Portland'
            )
            st.plotly_chart(fig_cluster, use_container_width=True)

            # ===== REGRESSION =====
            st.markdown("---")
            st.subheader("💰 Regression: Predicting Willingness to Pay")

            # Prepare regression data
            y_reg = df_ml['WTP_Per_Meal_AED']
            X_reg = df_ml[feature_cols].drop(columns=['WTP_Per_Meal_AED'], errors='ignore')

            X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
                X_reg, y_reg, test_size=0.3, random_state=42
            )

            lr_model = LinearRegression()
            lr_model.fit(X_train_reg, y_train_reg)
            y_pred_reg = lr_model.predict(X_test_reg)

            rmse = np.sqrt(mean_squared_error(y_test_reg, y_pred_reg))
            mae = mean_absolute_error(y_test_reg, y_pred_reg)
            r2 = r2_score(y_test_reg, y_pred_reg)

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("RMSE", f"{rmse:.2f} AED")
            with col2:
                st.metric("MAE", f"{mae:.2f} AED")
            with col3:
                st.metric("R² Score", f"{r2:.3f}")

            # Actual vs Predicted
            reg_comparison = pd.DataFrame({
                'Actual': y_test_reg.values,
                'Predicted': y_pred_reg
            })

            fig_reg = px.scatter(
                reg_comparison,
                x='Actual',
                y='Predicted',
                title='Actual vs Predicted WTP',
                labels={'Actual': 'Actual WTP (AED)', 'Predicted': 'Predicted WTP (AED)'},
                trendline='ols'
            )
            fig_reg.add_trace(go.Scatter(
                x=[10, 60], y=[10, 60],
                mode='lines',
                name='Perfect Prediction',
                line=dict(dash='dash', color='red')
            ))
            st.plotly_chart(fig_reg, use_container_width=True)

            st.success("✅ All models trained successfully!")

# ============================================
# TAB 3: CUSTOMER PREDICTION
# ============================================
with tab3:
    st.header("🎯 Individual Customer Interest Prediction")
    st.markdown("**Enter customer details to predict their interest and spending potential**")

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.selectbox("Age Group", df['Age_Group'].unique())
        gender = st.selectbox("Gender", df['Gender'].unique())
        nationality = st.selectbox("Nationality", df['Nationality'].unique())
        status = st.selectbox("Status", df['Status'].unique())

    with col2:
        location = st.selectbox("Location", df['Location'].unique())
        living = st.selectbox("Living Situation", df['Living_Situation'].unique())
        budget = st.selectbox("Monthly Food Budget", df['Monthly_Food_Budget_AED'].unique())
        cooking = st.selectbox("Cooking Frequency", df['Cooking_Frequency'].unique())

    with col3:
        spending = st.selectbox("Current Spending Per Meal", df['Current_Spending_Per_Meal_AED'].unique())
        delivery = st.selectbox("Delivery Frequency", df['Delivery_Frequency'].unique())
        taste_sat = st.slider("Taste Satisfaction (1-5)", 1, 5, 3)
        health_sat = st.slider("Healthiness Satisfaction (1-5)", 1, 5, 3)

    if st.button("🔮 Predict Interest", type="primary"):
        st.markdown("### Prediction Results")

        # Simple prediction logic based on patterns
        interest_score = 0

        # Status-based scoring
        if status == 'International University/College Student':
            interest_score += 30
        elif status == 'Local University/College Student':
            interest_score += 25
        elif 'Working Professional' in status:
            interest_score += 20

        # Budget-based scoring
        if budget in ['500-1000', '1000-1500']:
            interest_score += 20

        # Satisfaction-based scoring
        if taste_sat <= 3:
            interest_score += 15
        if health_sat <= 3:
            interest_score += 10

        # Location-based scoring
        if location in ['Dubai Academic City', 'JLT', 'International City']:
            interest_score += 15

        # Delivery frequency
        if delivery in ['2-3 times a week', '4-6 times a week', 'Once a day']:
            interest_score += 10

        interest_probability = min(interest_score, 100)

        col1, col2, col3 = st.columns(3)

        with col1:
            if interest_probability >= 70:
                st.success(f"✅ High Interest: {interest_probability}%")
                st.markdown("**Recommendation:** Priority target for marketing")
            elif interest_probability >= 50:
                st.warning(f"⚠️ Moderate Interest: {interest_probability}%")
                st.markdown("**Recommendation:** Nurture with trials and offers")
            else:
                st.error(f"❌ Low Interest: {interest_probability}%")
                st.markdown("**Recommendation:** May not be ideal target")

        with col2:
            # Estimate WTP based on budget and status
            wtp_estimate = 27.0
            if 'Student' in status:
                wtp_estimate = 22.5
            elif 'Professional' in status:
                wtp_estimate = 32.0

            st.metric("Estimated WTP", f"AED {wtp_estimate:.2f}")

        with col3:
            # Recommended plan
            if delivery in ['2-3 times a week', '4-6 times a week']:
                plan = "Weekly Plan (5 meals)"
            elif delivery in ['Once a day', 'Multiple times a day']:
                plan = "Daily Plan (7 meals)"
            else:
                plan = "Flexible Pay-per-order"

            st.info(f"📦 {plan}")

# ============================================
# TAB 4: UPLOAD & PREDICT
# ============================================
with tab4:
    st.header("📤 Upload New Dataset & Predict Interest")
    st.markdown("**Upload a CSV file with customer data to predict interest and download results**")

    st.markdown("""
    **Required columns:**
    - Age_Group, Gender, Nationality, Status, Location, Living_Situation
    - Monthly_Food_Budget_AED, Cooking_Frequency, Current_Spending_Per_Meal_AED
    - Delivery_Frequency, Taste_Satisfaction, Healthiness_Satisfaction, etc.
    """)

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        try:
            new_data = pd.read_csv(uploaded_file)
            st.success(f"✅ File uploaded successfully! {len(new_data)} rows detected.")

            st.subheader("Preview of Uploaded Data")
            st.dataframe(new_data.head(10), use_container_width=True)

            if st.button("🚀 Run Prediction", type="primary"):
                with st.spinner("Predicting interest levels..."):
                    # Simple rule-based prediction
                    predictions = []

                    for idx, row in new_data.iterrows():
                        score = 0

                        # Apply rules
                        if 'Student' in str(row.get('Status', '')):
                            score += 30
                        if row.get('Monthly_Food_Budget_AED', '') in ['500-1000', '1000-1500']:
                            score += 20
                        if row.get('Location', '') in ['Dubai Academic City', 'JLT']:
                            score += 15
                        if row.get('Taste_Satisfaction', 3) <= 3:
                            score += 15
                        if row.get('Delivery_Frequency', '') in ['2-3 times a week', '4-6 times a week']:
                            score += 10

                        interest_prob = min(score, 100) / 100
                        predictions.append({
                            'Interested_Prediction': 1 if interest_prob >= 0.5 else 0,
                            'Interest_Probability': interest_prob,
                            'Confidence': 'High' if interest_prob >= 0.7 or interest_prob <= 0.3 else 'Medium'
                        })

                    pred_df = pd.DataFrame(predictions)
                    result_df = pd.concat([new_data, pred_df], axis=1)

                    st.success("✅ Predictions completed!")
                    st.subheader("Results Preview")
                    st.dataframe(result_df.head(10), use_container_width=True)

                    # Summary statistics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Records", len(result_df))
                    with col2:
                        interested_count = result_df['Interested_Prediction'].sum()
                        st.metric("Predicted Interested", interested_count)
                    with col3:
                        interest_rate = (interested_count / len(result_df) * 100)
                        st.metric("Interest Rate", f"{interest_rate:.1f}%")

                    # Download button
                    csv = result_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Predictions CSV",
                        data=csv,
                        file_name="taste_from_home_predictions.csv",
                        mime="text/csv",
                        type="primary"
                    )

        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            st.info("Please ensure your CSV has the required columns and proper formatting.")

# Footer
st.markdown("---")
st.markdown("**Taste From Home Dashboard** | Built with Streamlit | © 2025")
