import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# -----------------------------
# App Title & Description
# -----------------------------
st.set_page_config(page_title="Customer Segmentation Dashboard", layout="wide")

st.title("🟢 Customer Segmentation Dashboard")
st.write(
    "This system uses **K-Means Clustering** to group customers based on their "
    "purchasing behavior and similarities."
)

# -----------------------------
# Load Dataset
# -----------------------------
st.sidebar.header("📂 Data Upload")
uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head())

    # Select only numerical columns
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

    # -----------------------------
    # Sidebar Controls
    # -----------------------------
    st.sidebar.header("⚙️ Clustering Controls")

    feature_1 = st.sidebar.selectbox("Select Feature 1", numeric_cols)
    feature_2 = st.sidebar.selectbox("Select Feature 2", numeric_cols)

    k = st.sidebar.slider("Number of Clusters (K)", 2, 10, 3)
    random_state = st.sidebar.number_input(
        "Random State (Optional)", value=42, step=1
    )

    run_btn = st.sidebar.button("🟦 Run Clustering")

    # -----------------------------
    # Run K-Means
    # -----------------------------
    if run_btn:
        X = df[[feature_1, feature_2]]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        kmeans = KMeans(n_clusters=k, random_state=random_state)
        clusters = kmeans.fit_predict(X_scaled)

        df["Cluster"] = clusters

        # -----------------------------
        # Visualization Section
        # -----------------------------
        st.subheader("📈 Cluster Visualization")

        fig, ax = plt.subplots()
        scatter = ax.scatter(
            X[feature_1],
            X[feature_2],
            c=clusters,
            cmap="viridis"
        )

        centers = scaler.inverse_transform(kmeans.cluster_centers_)
        ax.scatter(
            centers[:, 0],
            centers[:, 1],
            marker="X",
            s=200
        )

        ax.set_xlabel(feature_1)
        ax.set_ylabel(feature_2)
        ax.set_title("K-Means Clustering Result")

        st.pyplot(fig)

        # -----------------------------
        # Cluster Summary Section
        # -----------------------------
        st.subheader("📋 Cluster Summary")

        summary = (
            df.groupby("Cluster")[[feature_1, feature_2]]
            .agg(["mean", "count"])
            .reset_index()
        )

        summary.columns = ["Cluster", f"{feature_1} Avg", f"{feature_1} Count",
                           f"{feature_2} Avg", f"{feature_2} Count"]

        st.dataframe(summary)

        # -----------------------------
        # Business Interpretation
        # -----------------------------
        st.subheader("💡 Business Interpretation")

        for i in range(k):
            avg_f1 = df[df["Cluster"] == i][feature_1].mean()
            avg_f2 = df[df["Cluster"] == i][feature_2].mean()

            st.write(
                f"🟢 **Cluster {i}**: Customers with average "
                f"{feature_1} ≈ {avg_f1:.2f} and "
                f"{feature_2} ≈ {avg_f2:.2f}. "
                "These customers show similar purchasing behavior."
            )

        # -----------------------------
        # User Guidance Box
        # -----------------------------
        st.info(
            "Customers in the same cluster exhibit similar purchasing behaviour "
            "and can be targeted with similar business strategies."
        )

else:
    st.warning("⬅️ Please upload a CSV file to begin clustering.")
