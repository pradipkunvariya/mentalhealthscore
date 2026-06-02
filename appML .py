import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error
)

import matplotlib.pyplot as plt

# ----------------------------------
# PAGE SETTINGS
# ----------------------------------

st.set_page_config(
    page_title="AI Multi Prediction Dashboard",
    layout="wide"
)

st.title("🚀 AI Multi Prediction Dashboard")

st.markdown("---")

# ----------------------------------
# PREDICTION TYPE
# ----------------------------------

prediction_type = st.selectbox(
    "Select Prediction Type",
    [
        "House Price Prediction",
        "Gold Price Prediction",
        "Stock Price Prediction"
    ]
)

st.info(
    f"Upload a CSV dataset for {prediction_type}"
)

# ----------------------------------
# FILE UPLOAD
# ----------------------------------

uploaded_file = st.file_uploader(
    "Upload CSV File",
    type=["csv"]
)

# ----------------------------------
# MAIN PROCESS
# ----------------------------------

if uploaded_file:

    try:

        # Load dataset
        df = pd.read_csv(uploaded_file)

        # Remove missing values
        df = df.dropna()

        st.subheader("📊 Dataset Preview")
        st.dataframe(df.head())

        # Dataset size check
        if len(df) < 20:
            st.error(
                "Dataset should contain at least 20 rows."
            )
            st.stop()

        st.markdown("---")

        # Target selection
        target_column = st.selectbox(
            "Select Target Column",
            df.columns
        )

        # Features and target
        X = df.drop(columns=[target_column])
        y = df[target_column]

        # Convert categorical columns
        X = pd.get_dummies(X)

        feature_columns = X.columns

        # Train Test Split
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42
        )

        # Model
        model = RandomForestRegressor(
            n_estimators=300,
            random_state=42
        )

        model.fit(X_train, y_train)

        # Prediction
        y_pred = model.predict(X_test)

        # Metrics
        r2 = r2_score(y_test, y_pred)

        rmse = np.sqrt(
            mean_squared_error(y_test, y_pred)
        )

        mae = mean_absolute_error(
            y_test,
            y_pred
        )

        st.markdown("---")

        st.subheader("📈 Model Performance")

        col1, col2, col3 = st.columns(3)

        col1.metric("R² Score", f"{r2:.4f}")
        col2.metric("RMSE", f"{rmse:.4f}")
        col3.metric("MAE", f"{mae:.4f}")

        st.markdown("---")

        # ----------------------------------
        # USER INPUTS
        # ----------------------------------

        st.subheader("🔮 Enter Values For Prediction")

        user_values = {}

        original_features = df.drop(
            columns=[target_column]
        )

        for col in original_features.columns:

            if pd.api.types.is_numeric_dtype(
                original_features[col]
            ):

                user_values[col] = st.number_input(
                    col,
                    value=float(
                        original_features[col].mean()
                    )
                )

            else:

                options = list(
                    original_features[col].unique()
                )

                user_values[col] = st.selectbox(
                    col,
                    options
                )

        # ----------------------------------
        # PREDICT BUTTON
        # ----------------------------------

        if st.button("Predict"):

            input_df = pd.DataFrame(
                [user_values]
            )

            input_df = pd.get_dummies(
                input_df
            )

            input_df = input_df.reindex(
                columns=feature_columns,
                fill_value=0
            )

            prediction = model.predict(
                input_df
            )

            st.success(
                f"Predicted Value: {prediction[0]:,.2f}"
            )

            result_df = pd.DataFrame(
                {
                    "Prediction":
                    [prediction[0]]
                }
            )

            csv = result_df.to_csv(
                index=False
            )

            st.download_button(
                label="📥 Download Prediction",
                data=csv,
                file_name="prediction.csv",
                mime="text/csv"
            )

        st.markdown("---")

        # ----------------------------------
        # ACTUAL VS PREDICTED
        # ----------------------------------

        st.subheader("📉 Actual vs Predicted")

        fig1, ax1 = plt.subplots(
            figsize=(8, 6)
        )

        ax1.scatter(
            y_test,
            y_pred,
            alpha=0.6
        )

        ax1.plot(
            [y_test.min(), y_test.max()],
            [y_test.min(), y_test.max()],
            "r--"
        )

        ax1.set_xlabel("Actual Values")
        ax1.set_ylabel("Predicted Values")
        ax1.set_title(
            "Actual vs Predicted"
        )

        st.pyplot(fig1)

        st.markdown("---")

        # ----------------------------------
        # FEATURE IMPORTANCE
        # ----------------------------------

        st.subheader("⭐ Feature Importance")

        importance_df = pd.DataFrame(
            {
                "Feature":
                feature_columns,
                "Importance":
                model.feature_importances_
            }
        )

        importance_df = importance_df.sort_values(
            by="Importance",
            ascending=False
        )

        fig2, ax2 = plt.subplots(
            figsize=(10, 6)
        )

        ax2.barh(
            importance_df["Feature"],
            importance_df["Importance"]
        )

        ax2.set_title(
            "Feature Importance"
        )

        ax2.invert_yaxis()

        st.pyplot(fig2)

        st.dataframe(
            importance_df,
            use_container_width=True
        )

        st.markdown("---")

        st.subheader("📋 Dataset Statistics")

        st.dataframe(
            df.describe(),
            use_container_width=True
        )

    except Exception as e:

        st.error(
            f"Error: {str(e)}"
        )

else:

    st.warning(
        "Please upload a CSV dataset to continue."
    )
