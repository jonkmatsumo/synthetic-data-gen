"""ACH Risk Inspector Dashboard.

A Streamlit dashboard for fraud risk analysis with three modes:
- Live Scoring: Real-time transaction evaluation via API
- Historical Analytics: Analysis of historical data from database
- Model Lab: Train models and manage the model registry

NOTE: This service is isolated and does NOT import from src.model or src.generator.
"""

import json
import os
import time

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from data_service import (
    fetch_daily_stats,
    fetch_fraud_summary,
    fetch_recent_alerts,
    fetch_transaction_details,
    predict_risk,
)
from mlflow_utils import (
    check_mlflow_connection,
    get_experiment_runs,
    get_production_model_version,
    promote_to_production,
)
from plotly.subplots import make_subplots

# Configuration from environment
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://synthetic:synthetic_dev_password@localhost:5432/synthetic_data",
)

# Page configuration
st.set_page_config(
    page_title="ACH Risk Inspector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)


def render_live_scoring() -> None:
    """Render the Live Scoring (API) page.

    This page allows users to submit transactions for real-time
    fraud risk evaluation via the signal API.
    """
    st.header("Live Scoring")
    st.markdown("Submit transactions for real-time fraud risk evaluation.")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Transaction Input")

        user_id = st.text_input("User ID", value="user_001")
        amount = st.number_input(
            "Amount", min_value=0.01, value=100.00, step=0.01, format="%.2f"
        )
        currency = st.text_input("Currency", value="USD", disabled=True)

        analyze_clicked = st.button("Analyze Risk", type="primary")

    with col2:
        st.subheader("Risk Assessment")

        if analyze_clicked:
            # Measure API latency
            start_time = time.time()
            result = predict_risk(user_id, amount, currency)
            elapsed_ms = (time.time() - start_time) * 1000

            if result is None:
                st.error("API request failed. Is the API server running?")
                st.caption(f"Latency: {elapsed_ms:.0f}ms")
            else:
                score = result.get("score", 0)

                # Score gauge with color-coded risk level
                if score < 10:
                    st.markdown(
                        "<h1 style='color: #2ecc71; text-align: center;'>LOW RISK</h1>",
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        f"<h2 style='color: #2ecc71; text-align: center;'>"
                        f"Score: {score}</h2>",
                        unsafe_allow_html=True,
                    )
                elif score < 80:
                    st.markdown(
                        "<h1 style='color: #f39c12; text-align: center;'>"
                        "MEDIUM RISK</h1>",
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        f"<h2 style='color: #f39c12; text-align: center;'>"
                        f"Score: {score}</h2>",
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        "<h1 style='color: #e74c3c; text-align: center;'>"
                        "HIGH RISK</h1>",
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        f"<h2 style='color: #e74c3c; text-align: center;'>"
                        f"Score: {score}</h2>",
                        unsafe_allow_html=True,
                    )

                st.markdown("---")

                # Risk components
                risk_components = result.get("risk_components", [])
                if risk_components:
                    st.markdown("**Risk Factors:**")
                    for component in risk_components:
                        label = component.get("label", "unknown")
                        st.caption(f"- {label}")
                else:
                    st.caption("No specific risk factors identified.")

                # Latency display
                st.markdown("---")
                st.caption(f"Latency: {elapsed_ms:.0f}ms")

                # Raw JSON expander
                with st.expander("View Raw API Response"):
                    st.json(json.dumps(result, indent=2, default=str))
        else:
            st.markdown("*Submit a transaction to see results*")


def render_analytics() -> None:
    """Render the Historical Analytics (DB) page.

    This page provides analytics and visualizations based on
    historical transaction data stored in the database.
    """
    st.header("Historical Analytics")
    st.markdown("Analyze historical transaction patterns and fraud metrics.")

    # Fetch data
    summary = fetch_fraud_summary()
    daily_stats = fetch_daily_stats(days=30)
    transactions = fetch_transaction_details(days=7)
    alerts = fetch_recent_alerts(limit=50)

    # --- Global Metrics ---
    st.subheader("Global Metrics")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            label="Total Transactions Analyzed",
            value=f"{summary['total_transactions']:,}",
        )

    with col2:
        fraud_delta = (
            f"{summary['fraud_rate']:.2f}%" if summary["fraud_rate"] > 0 else None
        )
        st.metric(
            label="Detected Fraud (High Risk)",
            value=f"{summary['total_fraud']:,}",
            delta=fraud_delta,
            delta_color="inverse",
        )

    with col3:
        # Estimate false positive rate based on alerts vs actual fraud
        # This is a rough estimate: alerts that aren't fraud / total alerts
        if len(alerts) > 0 and "is_fraudulent" in alerts.columns:
            true_positives = alerts["is_fraudulent"].sum()
            false_positives = len(alerts) - true_positives
            fpr = (false_positives / len(alerts) * 100) if len(alerts) > 0 else 0
            st.metric(
                label="False Positive Rate (Est)",
                value=f"{fpr:.1f}%",
                help="Percentage of high-risk alerts that are not actual fraud",
            )
        else:
            st.metric(
                label="False Positive Rate (Est)",
                value="--",
                help="No alert data available",
            )

    st.markdown("---")

    # --- Time Series Visualization ---
    st.subheader("Transaction Volume & Fraud Trends")

    if len(daily_stats) > 0:
        # Create dual-axis chart: bars for volume, line for fraud
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        # Bar chart for transaction volume
        fig.add_trace(
            go.Bar(
                x=daily_stats["date"],
                y=daily_stats["total_transactions"],
                name="Transaction Volume",
                marker_color="#3498db",
                opacity=0.7,
            ),
            secondary_y=False,
        )

        # Line chart for fraud count
        fig.add_trace(
            go.Scatter(
                x=daily_stats["date"],
                y=daily_stats["fraud_count"],
                name="Fraud Count",
                mode="lines+markers",
                line={"color": "#e74c3c", "width": 3},
                marker={"size": 8},
            ),
            secondary_y=True,
        )

        fig.update_layout(
            title="Daily Transaction Volume with Fraud Overlay",
            xaxis_title="Date",
            legend={"orientation": "h", "yanchor": "bottom", "y": 1.02},
            height=400,
            hovermode="x unified",
        )
        fig.update_yaxes(title_text="Transaction Count", secondary_y=False)
        fig.update_yaxes(title_text="Fraud Count", secondary_y=True)

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No daily statistics available. Generate data to see trends.")

    st.markdown("---")

    # --- Amount Distribution ---
    st.subheader("Transaction Amount Distribution")

    if len(transactions) > 0 and "amount" in transactions.columns:
        # Create a copy and add fraud label for visualization
        viz_data = transactions.copy()
        if "is_fraudulent" in viz_data.columns:
            viz_data["Fraud Status"] = viz_data["is_fraudulent"].map(
                {True: "Fraudulent", False: "Legitimate"}
            )
        else:
            viz_data["Fraud Status"] = "Unknown"

        fig = px.histogram(
            viz_data,
            x="amount",
            color="Fraud Status",
            nbins=50,
            title="Transaction Amount Distribution by Fraud Status",
            labels={"amount": "Transaction Amount ($)", "count": "Frequency"},
            color_discrete_map={
                "Fraudulent": "#e74c3c",
                "Legitimate": "#2ecc71",
                "Unknown": "#95a5a6",
            },
            opacity=0.7,
            barmode="overlay",
        )
        fig.update_layout(height=400)

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No transaction data available. Generate data to see distribution.")

    st.markdown("---")

    # --- Recent Alerts Table ---
    st.subheader("Recent High-Risk Alerts")

    if len(alerts) > 0:
        # Format the display columns
        display_cols = [
            "record_id",
            "user_id",
            "created_at",
            "amount",
            "computed_risk_score",
            "is_fraudulent",
            "fraud_type",
        ]
        available_cols = [c for c in display_cols if c in alerts.columns]

        if available_cols:
            display_df = alerts[available_cols].copy()

            # Rename columns for display
            column_names = {
                "record_id": "Record ID",
                "user_id": "User ID",
                "created_at": "Timestamp",
                "amount": "Amount ($)",
                "computed_risk_score": "Risk Score",
                "is_fraudulent": "Confirmed Fraud",
                "fraud_type": "Fraud Type",
            }
            display_df = display_df.rename(
                columns={k: v for k, v in column_names.items() if k in display_df}
            )

            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True,
            )

            st.caption(f"Showing {len(alerts)} high-risk transactions (score >= 80)")
        else:
            st.warning("Alert data structure unexpected.")
    else:
        st.info(
            "No high-risk alerts found. This could mean no risky transactions "
            "or no data has been generated yet."
        )


def render_model_lab() -> None:
    """Render the Model Lab page.

    This page provides model training and registry management:
    - Train new models with configurable hyperparameters
    - View experiment runs and metrics
    - Promote models to production
    """
    st.header("Model Lab")
    st.markdown("Train models and manage the model registry.")

    # Check MLflow connection
    mlflow_connected = check_mlflow_connection()
    if not mlflow_connected:
        st.error(
            "Cannot connect to MLflow tracking server. "
            "Make sure the MLflow service is running."
        )
        return

    st.success("Connected to MLflow tracking server")

    # --- Section A: Train New Model ---
    st.subheader("Train New Model")

    col1, col2 = st.columns(2)

    with col1:
        max_depth = st.slider(
            "Max Depth",
            min_value=2,
            max_value=12,
            value=6,
            step=1,
            help="Maximum depth of XGBoost trees",
        )

    with col2:
        training_window = st.slider(
            "Training Window (days)",
            min_value=7,
            max_value=90,
            value=30,
            step=7,
            help="Number of days before today for training cutoff",
        )

    train_clicked = st.button("Start Training", type="primary")

    if train_clicked:
        with st.spinner("Training model... This may take a moment."):
            try:
                import requests

                response = requests.post(
                    f"{API_BASE_URL}/train",
                    json={
                        "max_depth": max_depth,
                        "training_window_days": training_window,
                    },
                    timeout=300,  # Training can take a while
                )
                result = response.json()

                if result.get("success"):
                    st.success(f"Training complete! Run ID: `{result.get('run_id')}`")
                    st.balloons()
                else:
                    st.error(f"Training failed: {result.get('error')}")
            except requests.exceptions.RequestException as e:
                st.error(f"API request failed: {e}")

    st.markdown("---")

    # --- Section B: Model Registry ---
    st.subheader("Model Registry")

    # Show current production model
    prod_version = get_production_model_version()
    if prod_version:
        st.info(f"Current Production Model: Version {prod_version}")
    else:
        st.warning("No production model deployed yet.")

    # Fetch and display experiment runs
    runs_df = get_experiment_runs()

    if len(runs_df) > 0:
        st.markdown("**Experiment Runs** (sorted by PR-AUC)")

        st.dataframe(
            runs_df,
            use_container_width=True,
            hide_index=True,
        )

        st.markdown("---")

        # Promote to production
        st.markdown("**Promote to Production**")

        run_ids = runs_df["Run ID"].tolist()
        selected_run = st.selectbox(
            "Select Run ID to promote",
            options=run_ids,
            index=0,
            help="Choose a model run to promote to production",
        )

        promote_clicked = st.button("Promote to Production", type="secondary")

        if promote_clicked and selected_run:
            with st.spinner("Promoting model..."):
                result = promote_to_production(selected_run)

            if result["success"]:
                st.success(result["message"])
            else:
                st.error(result["message"])
    else:
        st.info("No experiment runs found. Train a model to see results here.")


def main() -> None:
    """Main application entry point."""
    # Sidebar navigation
    st.sidebar.title("🔍 ACH Risk Inspector")
    st.sidebar.markdown("---")

    page = st.sidebar.radio(
        "Navigation",
        options=["Live Scoring (API)", "Historical Analytics (DB)", "Model Lab"],
        index=0,
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Configuration")
    st.sidebar.text(f"API: {API_BASE_URL}")
    db_display = DATABASE_URL.split("@")[-1] if "@" in DATABASE_URL else "configured"
    st.sidebar.text(f"DB: {db_display}")

    # Render selected page
    if page == "Live Scoring (API)":
        render_live_scoring()
    elif page == "Historical Analytics (DB)":
        render_analytics()
    else:
        render_model_lab()


if __name__ == "__main__":
    main()
