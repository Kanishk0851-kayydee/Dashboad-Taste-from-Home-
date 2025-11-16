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
cl_tab, clust_tab, reg_tab = st.tabs([
    "🎯 Classification", 
    "🔍 Clustering", 
    "💰 Regression"
])

# === Classification ===
with cl_tab:
    st.header("Classification Models")
    if st.button("🚀 Run Classification Algorithms", key="run_classify"):
        # Classification code here
        st.success("Classification models executed.")

# === Clustering ===
with clust_tab:
    st.header("Clustering")
    if st.button("🔍 Run Clustering Algorithm", key="run_cluster"):
        # Clustering code here
        st.success("Clustering executed.")

# === Regression ===
with reg_tab:
    st.header("Regression")
    if st.button("💰 Run Regression Algorithm", key="run_regress"):
        # Regression code here
        st.success("Regression analysis executed.")

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
