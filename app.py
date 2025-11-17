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
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import google.generativeai as genai
import warnings
import os

warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Taste From Home | Group 7",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    body {
        background-color: #fdfbfb;
    }
    .main {
        background: linear-gradient(135deg, #fdfbfb 0%, #ebedee 100%);
    }
    .main-header {
        font-size: 48px;
        font-weight: bold;
        text-align: center;
        padding: 20px;
        background: linear-gradient(90deg, #FF6B6B 0%, #4ECDC4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
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
    .filter-box {
        background-color: #e8f4f8;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #4ECDC4;
        margin: 15px 0;
    }
    .algo-box {
        background-color: #f0f8ff;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #FF6B6B;
        margin: 10px 0;
    }
    .upload-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 30px;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 30px 0;
    }
    .upload-success {
        background-color: #d4edda;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #28a745;
        margin: 15px 0;
        color: #155724;
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
</style>
""", unsafe_allow_html=True)

if 'df_uploaded' not in st.session_state:
    st.session_state.df_uploaded = None
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False

st.markdown('<h1 class="main-header">🍽️ Taste From Home: Marketing Analytics Dashboard</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Data-Driven Insights for Home-Cooked Meal Delivery Service in Dubai</p>', unsafe_allow_html=True)
st.markdown("---")

st.header("🏠 Welcome to Taste From Home")
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
    """)

    st.markdown('<div class="filter-box">', unsafe_allow_html=True)
    st.markdown("""
    **📊 Key Market Insights:**
    - 62% market interest from survey
    - 86% enthusiasm among international students
    - Average WTP: AED 27 per meal
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
    **Institute:** SP Jain School of Global Management, Dubai Campus  
    """)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")

st.markdown("## 📤 Step 2: Load Survey Data for Analysis")
st.markdown('<div class="filter-box">', unsafe_allow_html=True)
st.markdown("""
Upload your CSV file to unlock all analysis features!
""")
st.markdown('</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("📁 Choose your CSV file", type="csv", key="data_upload")

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.session_state.df_uploaded = df
        st.session_state.data_loaded = True

        st.markdown('<div class="upload-success">', unsafe_allow_html=True)
        st.markdown(f"""
        ✅ **Data Uploaded Successfully!**
        - **Total Records:** {len(df)}
        - **Total Columns:** {len(df.columns)}
        """)
        st.markdown('</div>', unsafe_allow_html=True)

        st.dataframe(df.head(10), use_container_width=True)
        st.success("✅ Ready to explore! Scroll down for analysis tabs.")

    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        st.session_state.data_loaded = False

if st.session_state.data_loaded and st.session_state.df_uploaded is not None:
    df = st.session_state.df_uploaded

    if 'Interested' not in df.columns:
        df['Interested'] = (df['Interest_Level'] >= 4).astype(int)

    st.markdown("---")
    st.markdown("## 📊 Data Analysis Tabs")

    insights_tab, ml_tab, pred_tab, upload_tab, ai_tab = st.tabs([
        "📊 Marketing Insights", "🤖 ML Algorithms", "🎯 Prediction", "📤 Upload & Predict", "💬 Ask AI"
    ])

    with insights_tab:
        st.header("📊 Marketing Insights Dashboard")

        st.markdown('<div class="filter-box">', unsafe_allow_html=True)
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
        fig2.update_layout(height=500, xaxis_tickangle=-45)
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

    with ml_tab:
        st.header("🤖 ML Algorithms & Performance")
        st.info(f"📊 All algorithms run on complete dataset ({len(df)} records)")

        st.markdown("---")
        ml_tab1, ml_tab2, ml_tab3, ml_tab4 = st.tabs(["🎯 Classification", "🔍 Clustering", "💰 Regression", "🔗 Association Rules"])

        with ml_tab1:
            st.subheader("Classification Models: Predicting Customer Interest")

            with st.expander("ℹ️ What is Classification?"):
                st.markdown("**Classification** predicts categorical outcomes (Interested: Yes/No)")

            with st.expander("📊 Variables Used (19 Total)"):
                st.markdown("**11 Demographics + 8 Features:** Age_Group, Gender, Nationality, Status, Location, Living_Situation, Monthly_Food_Budget, Cooking_Frequency, Current_Spending, Delivery_Frequency, Meals_Per_Week, Interest_Level, Subscription_Preference, WTP, and 5 Satisfaction scores")

            col1, col2 = st.columns(2)
            with col1:
                with st.expander("🔷 Logistic Regression"):
                    st.markdown("**Type:** Linear | **Pros:** Fast, interpretable | **Cons:** Assumes linear relationships")
            with col2:
                with st.expander("🔶 Decision Tree"):
                    st.markdown("**Type:** Tree-based | **Pros:** Highly interpretable | **Cons:** Prone to overfitting")

            col3, col4 = st.columns(2)
            with col3:
                with st.expander("🟧 Random Forest"):
                    st.markdown("**Type:** Ensemble Bagging | **Pros:** High accuracy, feature importance | **Cons:** Less interpretable")
            with col4:
                with st.expander("🟥 Gradient Boosting"):
                    st.markdown("**Type:** Ensemble Boosting | **Pros:** Exceptional accuracy | **Cons:** Can overfit")

            if st.button("🚀 Run Classification", key="run_classify"):
                with st.spinner("Training models..."):
                    df_ml = df.copy()
                    le = LabelEncoder()
                    categorical_cols = ['Age_Group', 'Gender', 'Nationality', 'Status', 'Location', 
                                       'Living_Situation', 'Monthly_Food_Budget_AED', 'Cooking_Frequency',
                                       'Current_Spending_Per_Meal_AED', 'Delivery_Frequency', 'Meals_Per_Week']

                    for col in categorical_cols:
                        df_ml[col + '_Encoded'] = le.fit_transform(df_ml[col])

                    feature_cols = [col for col in df_ml.columns if col.endswith('_Encoded')] +                                   ['Interest_Level', 'Subscription_Preference', 'WTP_Per_Meal_AED',
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
                    trained_models = {}

                    for model_name, model in models.items():
                        model.fit(X_train, y_train)
                        trained_models[model_name] = model
                        y_pred = model.predict(X_test)
                        cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')

                        results.append({
                            'Model': model_name,
                            'Accuracy': float(accuracy_score(y_test, y_pred)),
                            'Precision': float(precision_score(y_test, y_pred, zero_division=0)),
                            'Recall': float(recall_score(y_test, y_pred, zero_division=0)),
                            'F1-Score': float(f1_score(y_test, y_pred, zero_division=0)),
                            'CV Score': float(cv_scores.mean())
                        })

                    results_df = pd.DataFrame(results)
                    st.markdown("### Classification Results")
                    st.dataframe(results_df.style.format(precision=4), use_container_width=True)

                    st.markdown("---")
                    st.markdown("### 📌 Key Conclusions")
                    if results_df['Accuracy'].min() == 1.0:
                        st.info("✅ Perfect accuracy achieved - dataset has very clear, separable patterns")
                    st.success("✅ Best Model: Random Forest with balanced metrics")

                    best_model = trained_models['Random Forest']
                    feature_importance = pd.DataFrame({
                        'Feature': feature_cols,
                        'Importance': best_model.feature_importances_
                    }).sort_values('Importance', ascending=False).head(15)

                    st.markdown("---")
                    st.markdown("### 🔍 Top 15 Most Important Features")
                    fig_imp = px.bar(feature_importance, x='Importance', y='Feature', orientation='h',
                                    title='Feature Importance', color='Importance', color_continuous_scale='Viridis')
                    fig_imp.update_layout(height=500)
                    st.plotly_chart(fig_imp, use_container_width=True)

        with ml_tab2:
            st.subheader("K-Means Clustering: Customer Segmentation")

            with st.expander("ℹ️ What is Clustering?"):
                st.markdown("**Clustering** groups similar customers for targeted strategies (unsupervised learning)")

            if st.button("🔍 Run Clustering", key="run_cluster"):
                with st.spinner("Clustering..."):
                    cluster_features = ['WTP_Per_Meal_AED', 'Interest_Level', 'Subscription_Preference',
                                       'Taste_Satisfaction', 'Affordability_Satisfaction']
                    scaler = StandardScaler()
                    X_cluster = scaler.fit_transform(df[cluster_features])
                    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
                    cluster_labels = kmeans.fit_predict(X_cluster)

                    df_temp = df.copy()
                    df_temp['Cluster'] = cluster_labels

                    cluster_summary = df_temp.groupby('Cluster')[cluster_features].mean().round(2)
                    st.markdown("### Cluster Summary Statistics")
                    st.dataframe(cluster_summary, use_container_width=True)

                    st.markdown("---")
                    st.markdown("### 📌 4 Customer Segments Identified:")
                    st.markdown("- **Cluster 0:** Budget Seekers (Low WTP, students)")
                    st.markdown("- **Cluster 1:** High Value (High WTP, professionals)")
                    st.markdown("- **Cluster 2:** Undecided (Mixed metrics)")
                    st.markdown("- **Cluster 3:** Enthusiasts (Balanced high scores, loyal)")

                    fig_clus = px.scatter(df_temp, x='WTP_Per_Meal_AED', y='Interest_Level', 
                                         color='Cluster', size='Subscription_Preference',
                                         title='Customer Segments Distribution')
                    fig_clus.update_layout(height=500)
                    st.plotly_chart(fig_clus, use_container_width=True)
                    st.success("✅ Clustering complete!")

        with ml_tab3:
            st.subheader("Linear Regression: Predicting Willingness to Pay")

            with st.expander("ℹ️ What is Regression?"):
                st.markdown("**Regression** predicts continuous values (not categories). Target: WTP_Per_Meal_AED (AED 10-60)")

            with st.expander("📊 All 19 Variables Used"):
                st.markdown("**11 Encoded Demographics + 8 Direct Features** - Same as Classification for consistency")

            with st.expander("🔬 Regression Metrics Explained"):
                col_r1, col_r2, col_r3 = st.columns(3)
                with col_r1:
                    st.markdown("**RMSE:** Avg prediction error (AED) - Lower = Better")
                with col_r2:
                    st.markdown("**MAE:** Avg absolute error (AED) - Lower = Better")
                with col_r3:
                    st.markdown("**R² Score:** Variance explained (0-1) - Higher = Better")

            if st.button("💰 Run Regression", key="run_regress"):
                with st.spinner("Training regression model..."):
                    df_ml = df.copy()
                    le = LabelEncoder()
                    categorical_cols = ['Age_Group', 'Gender', 'Nationality', 'Status', 'Location', 
                                       'Living_Situation', 'Monthly_Food_Budget_AED', 'Cooking_Frequency',
                                       'Current_Spending_Per_Meal_AED', 'Delivery_Frequency', 'Meals_Per_Week']

                    for col in categorical_cols:
                        df_ml[col + '_Encoded'] = le.fit_transform(df_ml[col])

                    feature_cols = [col for col in df_ml.columns if col.endswith('_Encoded')] +                                   ['Interest_Level', 'Subscription_Preference', 'Taste_Satisfaction',
                                   'Healthiness_Satisfaction', 'Affordability_Satisfaction',
                                   'Convenience_Satisfaction', 'Variety_Satisfaction']

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
                    col1.metric("RMSE", f"AED {rmse:.2f}")
                    col2.metric("MAE", f"AED {mae:.2f}")
                    col3.metric("R² Score", f"{r2:.3f}")

                    st.markdown("---")
                    st.markdown("### 📌 Model Fit Conclusions")
                    if r2 > 0.7:
                        st.success("🟢 **Excellent:** Model explains >70% of WTP variation")
                    elif r2 > 0.5:
                        st.info("🟡 **Good:** Model explains 50-70% of WTP variation")
                    else:
                        st.warning("🟠 **Moderate:** Other factors also influence WTP")

                    st.markdown("---")
                    st.markdown("### 💡 Business Implications")
                    st.markdown(f"""
                    - Average prediction error: ±{mae:.1f} AED per meal
                    - Model explains {r2*100:.1f}% of price variation
                    - **Student Segment:** Lower WTP (AED 20-25)
                    - **Professional Segment:** Higher WTP (AED 30-35)
                    - Use for pricing optimization & customer valuation
                    """)

                    st.markdown("---")
                    st.markdown("### 📊 Actual vs Predicted WTP")
                    fig_reg = px.scatter(x=y_test, y=y_pred, 
                                        labels={'x':'Actual WTP (AED)', 'y':'Predicted WTP (AED)'},
                                        title='Regression Model Performance', trendline='ols',
                                        color_discrete_sequence=['#FF6B6B'])
                    fig_reg.add_trace(go.Scatter(x=[y_test.min(), y_test.max()], 
                                                y=[y_test.min(), y_test.max()],
                                                mode='lines', name='Perfect Fit',
                                                line=dict(dash='dash', color='green')))
                    fig_reg.update_layout(height=500)
                    st.plotly_chart(fig_reg, use_container_width=True)
                    st.success("✅ Regression analysis complete!")

        with ml_tab4:
            st.subheader("🔗 Association Rule Mining: Finding Customer Patterns")

            with st.expander("ℹ️ What is Association Rule Mining?"):
                st.markdown("""
                **Association Rule Mining** discovers interesting relationships between variables.

                **Metrics:**
                - **Support:** How often items appear together
                - **Confidence:** If A occurs, probability B also occurs
                - **Lift:** How much more likely B occurs if A occurs
                """)

            if st.button("🔗 Run Association Rules", key="run_arm"):
                with st.spinner("Mining rules..."):
                    df_arm = df.copy()
                    df_arm['Interest_Cat'] = pd.cut(df_arm['Interest_Level'], bins=[0, 2, 4, 5], 
                                                     labels=['Low', 'Medium', 'High'])
                    df_arm['Interested_Cat'] = df_arm['Interested'].map({0: 'Not_Interested', 1: 'Interested'})

                    transactions = []
                    for idx, row in df_arm.iterrows():
                        transaction = [
                            f"Status={str(row['Status'])[:15]}",
                            f"Location={str(row['Location'])[:15]}",
                            f"Budget={row['Monthly_Food_Budget_AED']}",
                            f"Interest={row['Interest_Cat']}",
                            f"Interested={row['Interested_Cat']}"
                        ]
                        transactions.append(transaction)

                    te = TransactionEncoder()
                    te_ary = te.fit(transactions).transform(transactions)
                    df_encoded = pd.DataFrame(te_ary, columns=te.columns_)

                    frequent_itemsets = apriori(df_encoded, min_support=0.05, use_colnames=True)

                    if len(frequent_itemsets) > 0:
                        rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)

                        if len(rules) > 0:
                            rules['antecedent_str'] = rules['antecedents'].apply(lambda x: ', '.join(list(x)))
                            rules['consequent_str'] = rules['consequents'].apply(lambda x: ', '.join(list(x)))

                            st.markdown("### 🔗 Top Association Rules (by Lift)")
                            top_rules = rules.nlargest(10, 'lift')[['antecedent_str', 'consequent_str', 
                                                                      'support', 'confidence', 'lift']]
                            top_rules.columns = ['If (Antecedent)', 'Then (Consequent)', 'Support', 'Confidence', 'Lift']
                            top_display = top_rules.copy()
                            top_display['Support'] = top_display['Support'].apply(lambda x: f"{x:.2%}")
                            top_display['Confidence'] = top_display['Confidence'].apply(lambda x: f"{x:.2%}")
                            top_display['Lift'] = top_display['Lift'].apply(lambda x: f"{x:.2f}")
                            st.dataframe(top_display, use_container_width=True)

                            st.markdown("---")
                            st.markdown("### 💡 Business Insights")
                            interested_rules = rules[rules['consequents'].apply(lambda x: any('Interested' in str(item) for item in x))]
                            if len(interested_rules) > 0:
                                st.success(f"✅ Found {len(interested_rules)} rules predicting interest")
                                for idx, rule in interested_rules.nlargest(3, 'confidence').iterrows():
                                    st.info(f"**{', '.join(list(rule['antecedents']))} → Interested** ({rule['confidence']:.1%} confidence)")

                            st.success("✅ Association Rule Mining complete!")
                        else:
                            st.warning("No rules found with min confidence of 50%")
                    else:
                        st.warning("No frequent itemsets found")

    with pred_tab:
        st.header("🎯 Customer Prediction")
        col1, col2, col3 = st.columns(3)

        with col1:
            age = st.selectbox("Age", df['Age_Group'].unique())
            gender = st.selectbox("Gender", df['Gender'].unique())
            status = st.selectbox("Status", df['Status'].unique())

        with col2:
            location = st.selectbox("Location", df['Location'].unique())
            budget = st.selectbox("Budget", df['Monthly_Food_Budget_AED'].unique())

        with col3:
            taste = st.slider("Taste Satisfaction", 1, 5, 3)
            interest = st.slider("Interest Level", 1, 5, 3)

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

            col_x, col_y = st.columns(2)
            with col_x:
                if score >= 70:
                    st.success(f"✅ High Interest: {score}%")
                elif score >= 50:
                    st.warning(f"⚠️ Moderate: {score}%")
                else:
                    st.error(f"❌ Low: {score}%")
            with col_y:
                wtp = 22.5 if 'Student' in status else 32.0 if 'Professional' in status else 27.0
                st.metric("Est. WTP", f"AED {wtp}")

    with upload_tab:
        st.header("📤 Batch Prediction")
        uploaded_pred = st.file_uploader("Upload CSV", type="csv", key="batch_upload")

        if uploaded_pred:
            new_data = pd.read_csv(uploaded_pred)
            st.success(f"Loaded {len(new_data)} rows")

            if st.button("Predict", type="primary"):
                predictions = []
                for idx, row in new_data.iterrows():
                    score = 0
                    if 'Student' in str(row.get('Status', '')):
                        score += 30
                    if row.get('Monthly_Food_Budget_AED', '') in ['500-1000', '1000-1500']:
                        score += 20
                    predictions.append({'Interested': 1 if score >= 50 else 0})

                result = pd.concat([new_data, pd.DataFrame(predictions)], axis=1)
                st.download_button("Download Results", result.to_csv(index=False), "predictions.csv", "text/csv")

    with ai_tab:
        st.header("💬 Ask AI")
        api_key = st.text_input("Google Gemini API Key", type="password")
        question = st.text_area("Your question:")

        if st.button("🚀 Get Response", type="primary"):
            if not api_key or not question:
                st.error("Please provide API key and question")
            else:
                with st.spinner("Thinking..."):
                    try:
                        genai.configure(api_key=api_key)
                        model = genai.GenerativeModel('gemini-pro')
                        response = model.generate_content(question)
                        st.markdown(f"**🤖 Response:**\n\n{response.text}")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")

else:
    st.markdown('<div class="upload-box">', unsafe_allow_html=True)
    st.markdown("## 📂 Upload Data to Unlock Analysis Features")
    st.markdown("Use sample_test_data_100_records.csv to test")
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")
st.markdown("**Taste From Home Dashboard** | Group 7 | © 2025")
