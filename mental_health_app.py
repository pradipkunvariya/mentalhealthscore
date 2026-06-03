import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage, dendrogram

# ─── APP CONFIGURATION ────────────────────────────────────────────
st.set_page_config(
    page_title="Mental Health Score Predictor Dashboard",
    page_icon="🧠",
    layout="wide"
)

# ─── HEADER ──────────────────────────────────────────────────────
st.title("🧠 Student Mental Health Analysis & Predictive Analytics Dashboard")
st.markdown("""
This web application is a direct interactive deployment of the **Data Science & Machine Learning Research Framework**.
It analyses how academic factors, lifestyle variables, and demographics affect overall mental welfare scores,
enabling structural predictive inference using optimised Regression trees.
""")

# ─── DATA LOADING & PREPROCESSING ───────────────────────────────
@st.cache_data
def load_and_preprocess_data():
    # FIX 1: Load the real CSV instead of mock data
    raw_df = pd.read_csv("student_mental_health_v3.csv")

    processed_df = raw_df.copy()

    # FIX 2: Fill nulls using actual dataset mean/median (12 missing in 3 columns)
    processed_df['sleep_hours'] = processed_df['sleep_hours'].fillna(processed_df['sleep_hours'].mean())
    processed_df['screen_time'] = processed_df['screen_time'].fillna(processed_df['screen_time'].mean())
    processed_df['study_hours'] = processed_df['study_hours'].fillna(processed_df['study_hours'].median())

    # FIX 3: Use LabelEncoder exactly as in your notebook (mirrors In[14]-[15])
    le = LabelEncoder()
    for col in ['gender', 'education', 'exercise', 'diet', 'stress_frequency', 'anxiety_frequency']:
        processed_df[col] = le.fit_transform(processed_df[col])

    return raw_df, processed_df

raw_df, df = load_and_preprocess_data()

# ─── FEATURE / TARGET SPLIT (used across all tabs) ───────────────
X = df.drop(columns=['mental_health_score'])
y = df['mental_health_score']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ─── TABS ─────────────────────────────────────────────────────────
# FIX 4: Added 2 extra tabs for Clustering and Linear Regression
# to match all the notebook sections you wrote
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Exploratory Data Analysis",
    "📉 Linear Regression",
    "🤖 Tree Models",
    "🌿 Clustering & Dendrogram",
    "🔮 Custom Score Predictor"
])

# ═══════════════════════════════════════════════════════════════════
# TAB 1 — EXPLORATORY DATA ANALYSIS
# ═══════════════════════════════════════════════════════════════════
with tab1:
    st.header("📊 Dataset Exploratory Analysis & Metrics Summary")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Observed Profiles",    f"{len(df)} Records")
    col2.metric("Mean Daily Sleep Hours",      f"{raw_df['sleep_hours'].mean():.2f} Hrs")
    col3.metric("Mean Active Screen Time",     f"{raw_df['screen_time'].mean():.2f} Hrs")
    col4.metric("Mean Target Wellness Score",  f"{df['mental_health_score'].mean():.2f} / 10")

    st.subheader("📋 Sample Snapshot Output Table (Raw Data Preview)")
    st.dataframe(raw_df.head(10), use_container_width=True)

    st.subheader("📈 Statistical Feature Overview")
    st.dataframe(raw_df.describe().round(2), use_container_width=True)

    st.subheader("🔍 Null Values Check")
    null_counts = raw_df.isnull().sum().reset_index()
    null_counts.columns = ["Feature", "Missing Count"]
    st.dataframe(null_counts, use_container_width=True, hide_index=True)

    st.subheader("📈 Statistical Feature Distributions")
    feature_to_plot = st.selectbox(
        "Choose Target Behavioral Attribute column for Distribution Visual Analysis:",
        ['sleep_hours', 'screen_time', 'study_hours', 'mental_health_score']
    )
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.histplot(raw_df[feature_to_plot], kde=True, color="#4A90E2", bins=20, ax=ax)
    ax.set_title(
        f"Empirical Density Representation for: {feature_to_plot.replace('_', ' ').title()}",
        fontsize=14
    )
    ax.set_xlabel(feature_to_plot.replace('_', ' ').title())
    ax.set_ylabel("Frequency Count")
    st.pyplot(fig)

    st.subheader("🔥 Correlation Heatmap")
    fig_corr, ax_corr = plt.subplots(figsize=(10, 6))
    sns.heatmap(
        df.corr().round(2),
        annot=True, fmt=".2f",
        cmap="coolwarm", linewidths=0.5,
        ax=ax_corr
    )
    ax_corr.set_title("Feature Correlation Matrix", fontsize=14)
    st.pyplot(fig_corr)

# ═══════════════════════════════════════════════════════════════════
# TAB 2 — LINEAR REGRESSION  (mirrors In[25]-[32] of your notebook)
# ═══════════════════════════════════════════════════════════════════
with tab2:
    st.header("📉 Linear Regression Model")
    st.markdown("Mirrors **In[25]–[32]** of your Jupyter Notebook.")

    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    lr_pred  = lr_model.predict(X_test)
    lr_mae   = mean_absolute_error(y_test, lr_pred)

    c1, c2 = st.columns(2)
    c1.success("**Model:** Linear Regression")
    c2.metric("Mean Absolute Error (MAE)", f"{lr_mae:.4f}")

    st.subheader("📊 Actual vs Predicted Values")
    compare_df = pd.DataFrame({
        "Actual Score":    y_test.values,
        "Predicted Score": lr_pred.round(2)
    }).reset_index(drop=True)
    st.dataframe(compare_df.head(20), use_container_width=True, hide_index=True)

    fig_lr, ax_lr = plt.subplots(figsize=(8, 4))
    ax_lr.scatter(y_test, lr_pred, color="#4A90E2", alpha=0.6, edgecolors="white")
    ax_lr.plot(
        [y_test.min(), y_test.max()],
        [y_test.min(), y_test.max()],
        color="red", linestyle="--", linewidth=1.5, label="Perfect Fit"
    )
    ax_lr.set_xlabel("Actual Mental Health Score")
    ax_lr.set_ylabel("Predicted Score")
    ax_lr.set_title("Linear Regression — Actual vs Predicted", fontsize=13)
    ax_lr.legend()
    st.pyplot(fig_lr)

    st.subheader("📐 Model Coefficients")
    coef_df = pd.DataFrame({
        "Feature":     X.columns,
        "Coefficient": lr_model.coef_.round(4)
    }).sort_values("Coefficient", ascending=False)
    st.dataframe(coef_df, use_container_width=True, hide_index=True)

# ═══════════════════════════════════════════════════════════════════
# TAB 3 — DECISION TREE & RANDOM FOREST (mirrors In[33]-[47])
# ═══════════════════════════════════════════════════════════════════
with tab3:
    st.header("🤖 Model Performance Matrices & Architecture Insights")
    st.markdown("Mirrors **In[33]–[47]** of your Jupyter Notebook.")

    model_choice = st.radio(
        "Select Machine Learning Regressor Framework Strategy:",
        ("Decision Tree Regressor", "Random Forest Regressor")
    )

    if model_choice == "Decision Tree Regressor":
        # FIX 5: Train once here and store in session_state so Tab 5 can reuse it
        dt_model = DecisionTreeRegressor(max_depth=3)
        dt_model.fit(X_train, y_train)
        dt_pred = dt_model.predict(X_test)
        dt_mae  = mean_absolute_error(y_test, dt_pred)
        st.session_state["trained_model"]      = dt_model
        st.session_state["trained_model_name"] = "Decision Tree Regressor"

        c1, c2 = st.columns(2)
        c1.success(f"**Selected Model:** {model_choice}")
        c2.metric("Mean Absolute Error (MAE)", f"{dt_mae:.4f}")

        st.subheader("🌳 Decision Tree Structure Graph Output Visual")
        fig_tree, ax_tree = plt.subplots(figsize=(16, 8))
        plot_tree(
            dt_model,
            feature_names=X.columns,
            filled=True, rounded=True,
            fontsize=10, ax=ax_tree
        )
        ax_tree.set_title(
            "Decision Tree Regressor (Depth Constrained to Level 3)",
            fontsize=14
        )
        st.pyplot(fig_tree)

    else:
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)
        rf_pred = rf_model.predict(X_test)
        rf_mae  = mean_absolute_error(y_test, rf_pred)
        st.session_state["trained_model"]      = rf_model
        st.session_state["trained_model_name"] = "Random Forest Regressor"

        c1, c2 = st.columns(2)
        c1.success(f"**Selected Model:** {model_choice}")
        c2.metric("Mean Absolute Error (MAE)", f"{rf_mae:.4f}")

        st.subheader("📊 Ensemble Feature Importance Score Weights")
        importances = rf_model.feature_importances_
        indices     = np.argsort(importances)

        fig_imp, ax_imp = plt.subplots(figsize=(10, 6))
        ax_imp.barh(
            X.columns[indices], importances[indices],
            color="#2ECC71", edgecolor="black"
        )
        ax_imp.set_title(
            "Calculated Feature Importance Profile Scores Matrix",
            fontsize=14, fontweight="bold"
        )
        ax_imp.set_xlabel("Relative Gini/Variance Reduction Scale Importance Weight")
        ax_imp.set_ylabel("Features Evaluated")
        st.pyplot(fig_imp)

# ═══════════════════════════════════════════════════════════════════
# TAB 4 — CLUSTERING & DENDROGRAM (mirrors In[48]-[55])
# ═══════════════════════════════════════════════════════════════════
with tab4:
    st.header("🌿 Agglomerative Clustering & Dendrogram")
    st.markdown("Mirrors **In[48]–[55]** of your Jupyter Notebook.")

    st.subheader("🌲 Hierarchical Dendrogram (Ward Linkage)")
    st.caption("Using the same linkage(X, method='ward') call from your notebook.")

    # FIX 6: Use a sample for dendrogram so it renders fast on Streamlit Cloud
    # Full dataset dendrograms can be very slow; sample 80 rows for display
    X_sample = X.sample(n=80, random_state=42)

    Z = linkage(X_sample, method='ward')

    fig_dend, ax_dend = plt.subplots(figsize=(14, 7))
    dendrogram(Z, ax=ax_dend, leaf_rotation=90, leaf_font_size=8, color_threshold=0)
    ax_dend.set_title("Hierarchical Dendrogram (Ward Linkage — 80 sample rows)", fontsize=14)
    ax_dend.set_xlabel("Sample Index")
    ax_dend.set_ylabel("Distance")
    st.pyplot(fig_dend)

    st.subheader("🔵 Agglomerative Clustering (n_clusters = 2)")
    st.caption("Mirrors: model = AgglomerativeClustering(n_clusters=2) → fit_predict(X)")

    # FIX 7: Use same X (full) for clustering, match your notebook exactly
    agg_model  = AgglomerativeClustering(n_clusters=2)
    cluster_labels = agg_model.fit_predict(X)

    sil_score = silhouette_score(X, cluster_labels)

    c1, c2 = st.columns(2)
    c1.metric("Silhouette Score", f"{sil_score:.4f}")
    c2.metric("Number of Clusters", "2")

    st.subheader("📋 Cluster Label Assignment (First 20 Rows)")
    cluster_df = raw_df.copy().head(20)
    cluster_df["Cluster"] = cluster_labels[:20]
    st.dataframe(cluster_df, use_container_width=True, hide_index=True)

    st.subheader("📊 Cluster Distribution Plot")
    fig_clust, ax_clust = plt.subplots(figsize=(6, 4))
    ax_clust.bar(
        ["Cluster 0", "Cluster 1"],
        [(cluster_labels == 0).sum(), (cluster_labels == 1).sum()],
        color=["#4A90E2", "#E24A4A"], edgecolor="white"
    )
    ax_clust.set_title("Agglomerative Clustering — Cluster Sizes", fontsize=13)
    ax_clust.set_ylabel("Number of Records")
    st.pyplot(fig_clust)

# ═══════════════════════════════════════════════════════════════════
# TAB 5 — CUSTOM SCORE PREDICTOR (mirrors Tab 3 original)
# ═══════════════════════════════════════════════════════════════════
with tab5:
    st.header("🔮 Interactive Personal Profile Input Variable Predictor Panel")
    st.markdown(
        "Modify individual features on the parameters grid to process instantaneous "
        "predictive calculations against the backend regression framework:"
    )

    p_col1, p_col2 = st.columns(2)

    with p_col1:
        st.markdown("### 📋 Demographics & Academic Track Status")
        input_age      = st.slider("Select Student Age Profile:", 15, 30, 21)
        input_gender   = st.selectbox(
            "Identified Gender:",
            # FIX 8: Added 'Prefer not to say' to match actual dataset
            ["Male", "Female", "Prefer not to say"]
        )
        input_edu      = st.selectbox(
            "Current Formal Education Level Status:",
            ["School", "Undergraduate", "Postgraduate"]
        )
        input_study    = st.slider(
            "Average Self Study Breakdown Hours (Daily Target):",
            1.0, 12.0, 3.5, step=0.5
        )
        input_exercise = st.radio(
            "Performs Routine Physical Training Fitness/Exercise?",
            ["Yes", "No"]
        )

    with p_col2:
        st.markdown("### 📱 Lifestyle Metrics & Psychological Frequency Logs")
        input_sleep   = st.slider(
            "Average Quality Sleep Scale (Hours Rest Metrics Logged):",
            3.0, 12.0, 7.0, step=0.5
        )
        input_screen  = st.slider(
            "Observed Display/Screen Time Exposure (Daily Tracking Metrics):",
            0.5, 16.0, 4.5, step=0.5
        )
        input_diet    = st.selectbox(
            "Typical Nutritional Profile Assessment Quality Intake:",
            ["Healthy", "Average", "Unhealthy"]
        )
        input_stress  = st.select_slider(
            "Felt Stress Episodes Logged Frequency:",
            # FIX 9: Added 'Often' to match actual dataset values
            options=["Never", "Rarely", "Sometimes", "Often", "Always"],
            value="Sometimes"
        )
        input_anxiety = st.select_slider(
            "Anxiety Attacks/Felt Symptoms Log Matrix Range:",
            options=["Never", "Rarely", "Sometimes", "Often", "Always"],
            value="Never"
        )

    # FIX 10: Use same LabelEncoder order as the dataset
    # LabelEncoder sorts alphabetically, so we replicate that order here
    gender_order   = sorted(raw_df['gender'].unique())        # Female, Male, Prefer not to say
    edu_order      = sorted(raw_df['education'].unique())     # Postgraduate, School, Undergraduate
    exercise_order = sorted(raw_df['exercise'].unique())      # No, Yes
    diet_order     = sorted(raw_df['diet'].unique())          # Average, Healthy, Unhealthy
    stress_order   = sorted(raw_df['stress_frequency'].unique())   # Always, Never, Often, Rarely, Sometimes
    anxiety_order  = sorted(raw_df['anxiety_frequency'].unique())  # Always, Never, Often, Rarely, Sometimes

    gender_enc   = gender_order.index(input_gender)
    edu_enc      = edu_order.index(input_edu)
    exercise_enc = exercise_order.index(input_exercise)
    diet_enc     = diet_order.index(input_diet)
    stress_enc   = stress_order.index(input_stress)
    anxiety_enc  = anxiety_order.index(input_anxiety)

    custom_vector_features = pd.DataFrame([{
        'age':               float(input_age),
        'gender':            int(gender_enc),
        'education':         int(edu_enc),
        'sleep_hours':       float(input_sleep),
        'screen_time':       float(input_screen),
        'study_hours':       float(input_study),
        'exercise':          int(exercise_enc),
        'diet':              int(diet_enc),
        'stress_frequency':  int(stress_enc),
        'anxiety_frequency': int(anxiety_enc)
    }])

    if st.button("🚀 Calculate Estimated Profile Mental Health Score Output", use_container_width=True):
        # FIX 11: Reuse the model trained in Tab 3 if available,
        # otherwise fall back to Decision Tree trained on full data
        if "trained_model" in st.session_state:
            inference_model      = st.session_state["trained_model"]
            inference_model_name = st.session_state["trained_model_name"]
        else:
            inference_model = DecisionTreeRegressor(max_depth=3)
            inference_model.fit(X, y)
            inference_model_name = "Decision Tree Regressor (default)"

        predicted_result_score = inference_model.predict(custom_vector_features)[0]
        predicted_result_score = float(np.clip(predicted_result_score, 1.0, 10.0))

        st.markdown("---")
        st.subheader("🎯 Result Evaluation Prediction:")

        score_col, desc_col = st.columns([1, 2])
        score_col.metric(
            label=f"Calculated Welfare Index Score ({inference_model_name})",
            value=f"{predicted_result_score:.2f} / 10.0"
        )

        if predicted_result_score >= 7.5:
            desc_col.success(
                "🌟 **Strong Positive Indicators:** This mental profile matches lifestyle "
                "parameters indicative of low stress levels, structured balance, and healthy "
                "coping mechanisms."
            )
        elif predicted_result_score >= 5.0:
            desc_col.warning(
                "⚠️ **Moderate Warning Range Indicators:** Balanced baseline markers; "
                "consider scheduling lower overall screen time constraints or shifting daily "
                "physical active metrics upwards to boost individual score levels."
            )
        else:
            desc_col.error(
                "🚨 **High Clinical Risk Priority Warning:** Data array conditions flag profile "
                "markers indexing high anxiety exposure frequency paired with inadequate sleep "
                "patterns or high screen strain averages."
            )

        st.markdown("#### 📋 Your Input Summary")
        st.dataframe(
            pd.DataFrame({
                "Feature": [
                    "Age", "Gender", "Education", "Sleep Hours",
                    "Screen Time", "Study Hours", "Exercise", "Diet",
                    "Stress Frequency", "Anxiety Frequency"
                ],
                "Your Input": [
                    input_age, input_gender, input_edu, input_sleep,
                    input_screen, input_study, input_exercise, input_diet,
                    input_stress, input_anxiety
                ]
            }),
            use_container_width=True, hide_index=True
        )
