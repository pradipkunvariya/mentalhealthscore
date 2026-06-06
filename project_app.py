import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage, dendrogram

# ── Page config ───────────────────────────────────────────────────
st.set_page_config(
    page_title="Student Mental Health Project",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Student Mental Health Score — ML Project")
st.markdown("**By Pradip Kunvariya** · B.Tech Computer Engineering · Gandhinagar University")
st.divider()

# ── Load & preprocess data (Cells 2–14) ──────────────────────────
@st.cache_data
def load_data():
    # Cell 2
    df = pd.read_csv("student_mental_health_v3.csv")
    raw = df.copy()

    # Cell 8, 10
    df['sleep_hours'] = df['sleep_hours'].fillna(df['sleep_hours'].mean())
    df['screen_time'] = df['screen_time'].fillna(df['screen_time'].mean())
    df['study_hours'] = df['study_hours'].fillna(df['study_hours'].median())

    # Cell 12–14
    le = LabelEncoder()
    for col in ['gender', 'education', 'exercise', 'diet',
                'stress_frequency', 'anxiety_frequency']:
        df[col] = le.fit_transform(df[col])

    return raw, df

raw_df, df = load_data()

# Cell 17
X = df.drop(columns=['mental_health_score'])
y = df['mental_health_score']

# Cell 19
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ── Tabs ──────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📋 Dataset",
    "📉 Linear Regression",
    "🌳 Decision Tree",
    "🌲 Random Forest",
    "🔵 Clustering",
])

# ════════════════════════════════════════════════════════════════
# TAB 1 — DATASET (Cells 3–16)
# ════════════════════════════════════════════════════════════════
with tab1:
    st.header("📋 Dataset Overview")

    # Cell 3 — df.head()
    st.subheader("df.head()")
    st.dataframe(raw_df.head(), use_container_width=True)

    # Cell 4 — df.tail()
    st.subheader("df.tail()")
    st.dataframe(raw_df.tail(), use_container_width=True)

    # Cell 5 — df.describe()
    st.subheader("df.describe()")
    st.dataframe(raw_df.describe().round(2), use_container_width=True)

    # Cell 6 — df.isnull().sum()
    st.subheader("df.isnull().sum() — Before filling")
    null_before = pd.DataFrame({
        "Feature": raw_df.columns,
        "Missing": raw_df.isnull().sum().values
    })
    st.dataframe(null_before, use_container_width=True, hide_index=True)

    # Cell 11 — after filling
    st.subheader("df.isnull().sum() — After filling")
    null_after = pd.DataFrame({
        "Feature": df.columns,
        "Missing": df.isnull().sum().values
    })
    st.dataframe(null_after, use_container_width=True, hide_index=True)

    # Cell 15 — encoded df.head()
    st.subheader("df.head() — After Label Encoding")
    st.dataframe(df.head(), use_container_width=True)

    # Cell 16 — df.info()
    st.subheader("df.info()")
    info_df = pd.DataFrame({
        "Column":   df.columns,
        "Dtype":    [str(df[c].dtype) for c in df.columns],
        "Non-Null": [df[c].notna().sum() for c in df.columns],
    })
    st.dataframe(info_df, use_container_width=True, hide_index=True)

# ════════════════════════════════════════════════════════════════
# TAB 2 — LINEAR REGRESSION (Cells 24–31)
# ════════════════════════════════════════════════════════════════
with tab2:
    st.header("📉 Linear Regression")

    # Cell 25–28
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)

    # Cell 30–31
    mae_lr = mean_absolute_error(y_test, y_pred_lr)

    c1, c2 = st.columns(2)
    c1.success("**Model:** LinearRegression()")
    c2.metric("Mean Absolute Error (MAE)", f"{mae_lr:.4f}")
    st.caption("Notebook output — Cell 31: `1.6939997231244106`")

    # Predicted values table (Cell 28 — y_pred)
    st.subheader("y_pred (first 10 values)")
    pred_df = pd.DataFrame({
        "Actual":    y_test.values[:10].round(2),
        "Predicted": y_pred_lr[:10].round(4)
    })
    st.dataframe(pred_df, use_container_width=True, hide_index=True)

    # Coefficients
    st.subheader("Model Coefficients")
    coef_df = pd.DataFrame({
        "Feature":     X.columns,
        "Coefficient": lr.coef_.round(4)
    }).sort_values("Coefficient", ascending=False)
    st.dataframe(coef_df, use_container_width=True, hide_index=True)

# ════════════════════════════════════════════════════════════════
# TAB 3 — DECISION TREE (Cells 32–38)
# ════════════════════════════════════════════════════════════════
with tab3:
    st.header("🌳 Decision Tree Regressor")

    # Cell 33
    dt = DecisionTreeRegressor(max_depth=3)
    dt.fit(X_train, y_train)

    # Cell 34–36
    y_pred_dt = dt.predict(X_test)
    mae_dt = mean_absolute_error(y_test, y_pred_dt)

    c1, c2 = st.columns(2)
    c1.success("**Model:** DecisionTreeRegressor(max_depth=3)")
    c2.metric("Mean Absolute Error (MAE)", f"{mae_dt:.4f}")
    st.caption("Notebook output — Cell 36: `1.6773517740429502`")

    # Cell 38 — plot_tree
    st.subheader("Decision Tree Plot — plt.figure(figsize=(12,6))")
    fig_tree, ax_tree = plt.subplots(figsize=(12, 6))
    plot_tree(dt, feature_names=X.columns, filled=True, rounded=True, ax=ax_tree)
    ax_tree.set_title("Decision Tree Regressor")
    st.pyplot(fig_tree)

# ════════════════════════════════════════════════════════════════
# TAB 4 — RANDOM FOREST (Cells 39–47)
# ════════════════════════════════════════════════════════════════
with tab4:
    st.header("🌲 Random Forest Regressor")

    # Cell 40
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    # Cell 41–43
    y_pred_rf = rf.predict(X_test)
    mae_rf = mean_absolute_error(y_test, y_pred_rf)

    c1, c2 = st.columns(2)
    c1.success("**Model:** RandomForestRegressor(n_estimators=100)")
    c2.metric("Mean Absolute Error (MAE)", f"{mae_rf:.4f}")
    st.caption("Notebook output — Cell 43: `1.757166666666667`")

    # Cell 44–46 — feature importance
    imp = rf.feature_importances_
    st.subheader("Feature Importances — plt.barh(X.columns, imp)")
    st.caption("Notebook output — Cell 45: array([0.0717, 0.0504, 0.0802, ...])")

    fig_imp, ax_imp = plt.subplots(figsize=(10, 5))
    ax_imp.barh(X.columns, imp, color="#2ECC71", edgecolor="black")
    ax_imp.set_title("Feature Importance")
    ax_imp.set_xlabel("Importance Score")
    st.pyplot(fig_imp)

    # Cell 47 — one tree from the forest
    st.subheader("One Tree from the Forest — model.estimators_[0]")
    fig_one, ax_one = plt.subplots(figsize=(20, 8))
    plot_tree(
        rf.estimators_[0],
        feature_names=X_train.columns,
        filled=True, max_depth=3,
        ax=ax_one
    )
    ax_one.set_title("Random Forest — estimators_[0] (depth limited to 3 for display)")
    st.pyplot(fig_one)

# ════════════════════════════════════════════════════════════════
# TAB 5 — CLUSTERING (Cells 48–55)
# ════════════════════════════════════════════════════════════════
with tab5:
    st.header("🔵 Agglomerative Clustering")

    # Cell 49 — Z = linkage(X, method='ward')
    st.subheader("Dendrogram — linkage(X, method='ward')")
    st.caption("Cell 49–51: Z = linkage(X, method='ward') → dendrogram(Z)")

    # Sample 80 rows so it renders fast (full 300 rows is very slow)
    X_sample = X.sample(n=80, random_state=42)
    Z = linkage(X_sample, method='ward')

    fig_dend, ax_dend = plt.subplots(figsize=(12, 6))
    dendrogram(Z, ax=ax_dend, leaf_rotation=90, leaf_font_size=7)
    ax_dend.set_title("Dendrogram (80 sample rows — Ward Linkage)")
    ax_dend.set_xlabel("Sample Index")
    ax_dend.set_ylabel("Distance")
    st.pyplot(fig_dend)

    # Cell 52–55
    st.subheader("AgglomerativeClustering(n_clusters=2)")
    st.caption("Cell 52: model = AgglomerativeClustering(n_clusters=2) → fit_predict(X)")

    agg = AgglomerativeClustering(n_clusters=2)
    cluster_y = agg.fit_predict(X)

    # Cell 54–55 — silhouette score
    sil = silhouette_score(X, cluster_y)

    c1, c2, c3 = st.columns(3)
    c1.metric("Silhouette Score", f"{sil:.4f}")
    c2.metric("Cluster 0 Size",   f"{(cluster_y == 0).sum()}")
    c3.metric("Cluster 1 Size",   f"{(cluster_y == 1).sum()}")
    st.caption(f"Notebook output — Cell 55: `{sil:.17f}`")

    # Cluster distribution bar chart
    fig_cl, ax_cl = plt.subplots(figsize=(5, 3))
    ax_cl.bar(["Cluster 0", "Cluster 1"],
              [(cluster_y == 0).sum(), (cluster_y == 1).sum()],
              color=["#3498DB", "#E74C3C"], edgecolor="white")
    ax_cl.set_title("Cluster Distribution")
    ax_cl.set_ylabel("Count")
    st.pyplot(fig_cl)

    # Cell 53 — y values preview
    st.subheader("Cluster Labels — y (first 20)")
    label_df = raw_df.head(20).copy()
    label_df["Cluster"] = cluster_y[:20]
    st.dataframe(label_df, use_container_width=True, hide_index=True)

# ── Footer ────────────────────────────────────────────────────────
st.divider()
st.caption(
    "🧠 Student Mental Health Score · "
    "Linear Regression MAE: 1.6940 · "
    "Decision Tree MAE: 1.6774 · "
    "Random Forest MAE: 1.7572 · "
    "Silhouette Score: 0.1644"
)
