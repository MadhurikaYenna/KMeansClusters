import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# -------------------------------------------------
# Page Config & Theme
# -------------------------------------------------
st.set_page_config(
    page_title="Customer Segmentation Dashboard",
    layout="wide"
)

st.markdown(
    """
    <style>
    .main { background-color: #f7f9fc; }
    h1 { color: #2c7be5; }
    h2 { color: #1f2937; }
    .block-container { padding-top: 2rem; }
    </style>
    """,
    unsafe_allow_html=True
)

# -------------------------------------------------
# App Title & Description
# -------------------------------------------------
st.title("🟢 Customer Segmentation Dashboard (K-Means)")

st.markdown(
    """
    **This dashboard uses K-Means Clustering to group customers based on their
    purchasing behavior and similarities.**

    👉 *Discover hidden customer segments and support business decision-making.*
    """
)

# -------------------------------------------------
# Load Dataset
# -------------------------------------------------
df1 = pd.read_csv("Wholesale customers data.csv")

st.success("✅ Dataset loaded successfully")

with st.expander("🔍 Dataset Preview"):
    st.dataframe(df1.head())

# -------------------------------------------------
# Feature Selection
# -------------------------------------------------
st.header("📌 Feature Selection")

features = [
    "Fresh",
    "Milk",
    "Grocery",
    "Frozen",
    "Detergents_Paper",
    "Delicassen"
]

X = df1[features]

# -------------------------------------------------
# Data Scaling
# -------------------------------------------------
st.header("⚙️ Data Scaling & Verification")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_scaled_df = pd.DataFrame(X_scaled, columns=features)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Mean (≈ 0)")
    st.dataframe(X_scaled_df.mean())

with col2:
    st.subheader("Standard Deviation (≈ 1)")
    st.dataframe(X_scaled_df.std())

# -------------------------------------------------
# Elbow Method
# -------------------------------------------------
st.header("📉 Elbow Method for Optimal K")

wcss = []
for i in range(1, 11):
    kmeans = KMeans(
        n_clusters=i,
        init="k-means++",
        random_state=0,
        n_init=10
    )
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

fig1, ax1 = plt.subplots()
ax1.plot(range(1, 11), wcss, marker="o", color="#2c7be5")
ax1.set_title("Elbow Method", fontsize=14)
ax1.set_xlabel("Number of Clusters (K)")
ax1.set_ylabel("WCSS")
ax1.grid(True, linestyle="--", alpha=0.5)
st.pyplot(fig1)

st.info("📌 The optimal K is chosen where the WCSS curve starts flattening.")

# -------------------------------------------------
# K-Means Clustering
# -------------------------------------------------
st.header("🧠 K-Means Clustering (K = 5)")

kmeans = KMeans(
    n_clusters=5,
    init="k-means++",
    random_state=0,
    n_init=10
)
y_kmeans = kmeans.fit_predict(X_scaled)

# -------------------------------------------------
# Cluster Visualization
# -------------------------------------------------
st.subheader("🎨 Customer Cluster Visualization")

fig2, ax2 = plt.subplots(figsize=(7, 5))

colors = ["#ef4444", "#3b82f6", "#22c55e", "#06b6d4", "#a855f7"]

for i in range(5):
    ax2.scatter(
        X_scaled[y_kmeans == i, 0],
        X_scaled[y_kmeans == i, 1],
        s=60,
        c=colors[i],
        label=f"Cluster {i}",
        alpha=0.7
    )

ax2.scatter(
    kmeans.cluster_centers_[:, 0],
    kmeans.cluster_centers_[:, 1],
    s=300,
    c="gold",
    marker="X",
    edgecolor="black",
    label="Centroids"
)

ax2.set_title("Customer Segments", fontsize=14)
ax2.set_xlabel("Feature 1 (Scaled)")
ax2.set_ylabel("Feature 2 (Scaled)")
ax2.legend()
ax2.grid(True, linestyle="--", alpha=0.4)
st.pyplot(fig2)

# -------------------------------------------------
# Cluster Profiling
# -------------------------------------------------
st.header("📊 Cluster Profiling")

cluster_profile = df1.groupby(y_kmeans)[features].mean()

st.subheader("Average Spending per Category")
st.dataframe(cluster_profile.style.background_gradient(cmap="Blues"))

dominant_category = cluster_profile.idxmax(axis=1)

st.subheader("Dominant Purchase Category")
st.dataframe(dominant_category.rename("Dominant Category"))

# -------------------------------------------------
# Business Insights (Fixed)
# -------------------------------------------------
st.header("💼 Business Insights & Strategies")

st.markdown("""
### 🟢 Cluster 0 – Balanced Buyers
- Bundle promotions to increase basket value

### 🔵 Cluster 1 – Household Essentials Buyers
- Inventory prioritization & bulk discounts

### 🟠 Cluster 2 – Premium & Fresh Buyers
- Personalized pricing & premium offers

### 🟣 Cluster 3 – Frozen / Bulk Buyers
- Optimize cold storage & promote bulk buying

### 🔴 Cluster 4 – Low-Value Buyers
- Targeted promotions & re-engagement campaigns
""")

st.info(
    "Customers within the same cluster exhibit similar purchasing behaviour "
    "and can be targeted with similar business strategies."
)

# -------------------------------------------------
# Stability Check
# -------------------------------------------------
st.header("🔁 Clustering Stability Check")

kmeans_alt = KMeans(n_clusters=5, random_state=99, n_init=10)
y_kmeans_alt = kmeans_alt.fit_predict(X_scaled)

changed = np.sum(y_kmeans != y_kmeans_alt)

st.warning(
    f"🔄 **{changed} customers** changed their cluster assignment when the random state was modified."
)
