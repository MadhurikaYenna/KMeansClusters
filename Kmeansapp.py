import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# -------------------------------------------------
# App Title
# -------------------------------------------------
st.set_page_config(layout="wide")
st.title("🟢 Customer Segmentation Dashboard (K-Means)")

st.write(
    "This system uses K-Means Clustering to group customers based on "
    "their purchasing behavior and similarities."
)

# -------------------------------------------------
# Load Dataset (Directly from path)
# -------------------------------------------------
df1 = pd.read_csv("Wholesale customers data.csv")


st.success("Dataset loaded successfully")

st.subheader("Dataset Preview")
st.dataframe(df1.head())

# -------------------------------------------------
# Feature Selection
# -------------------------------------------------
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
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_scaled_df = pd.DataFrame(X_scaled, columns=features)

st.subheader("Feature Scaling Verification")
st.write("Mean of scaled features (≈ 0):")
st.write(X_scaled_df.mean())

st.write("Standard deviation of scaled features (≈ 1):")
st.write(X_scaled_df.std())

# -------------------------------------------------
# Elbow Method
# -------------------------------------------------
st.subheader("Elbow Method")

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
ax1.plot(range(1, 11), wcss, marker="o")
ax1.set_title("The Elbow Method")
ax1.set_xlabel("Number of Clusters")
ax1.set_ylabel("WCSS")
st.pyplot(fig1)

# -------------------------------------------------
# K-Means Clustering (K = 5)
# -------------------------------------------------
st.subheader("K-Means Clustering (K = 5)")

kmeans = KMeans(
    n_clusters=5,
    init="k-means++",
    random_state=0,
    n_init=10
)
y_kmeans = kmeans.fit_predict(X_scaled)

# -------------------------------------------------
# Cluster Visualization (2D)
# -------------------------------------------------
fig2, ax2 = plt.subplots(figsize=(7, 5))

colors = ["red", "blue", "green", "cyan", "magenta"]

for i in range(5):
    ax2.scatter(
        X_scaled[y_kmeans == i, 0],
        X_scaled[y_kmeans == i, 1],
        s=80,
        c=colors[i],
        label=f"Cluster {i}"
    )

ax2.scatter(
    kmeans.cluster_centers_[:, 0],
    kmeans.cluster_centers_[:, 1],
    s=250,
    c="yellow",
    marker="X",
    label="Centroids"
)

ax2.set_title("Customer Clusters")
ax2.set_xlabel("Feature 1 (Scaled)")
ax2.set_ylabel("Feature 2 (Scaled)")
ax2.legend()
st.pyplot(fig2)

# -------------------------------------------------
# Cluster Profiling
# -------------------------------------------------
st.subheader("Cluster Profiling")

cluster_profile = df1.groupby(y_kmeans)[features].mean()
st.write("Average spending per category (per cluster):")
st.dataframe(cluster_profile)

dominant_category = cluster_profile.idxmax(axis=1)

st.write("Dominant purchase category per cluster:")
st.dataframe(dominant_category.rename("Dominant Category"))

# -------------------------------------------------
# Business Insights
# -------------------------------------------------
st.subheader("Business-Friendly Insights")

for cluster in cluster_profile.index:
    st.markdown(
        f"🟢 **Cluster {cluster}:** Customers in this cluster show strong preference "
        f"for **{dominant_category[cluster]}** products and can be targeted with "
        "focused marketing strategies."
    )

st.info(
    "Customers in the same cluster exhibit similar purchasing behaviour "
    "and can be targeted with similar business strategies."
)

# -------------------------------------------------
# Stability Check
# -------------------------------------------------
st.subheader("Clustering Stability Check")

kmeans_alt = KMeans(n_clusters=5, random_state=99, n_init=10)
y_kmeans_alt = kmeans_alt.fit_predict(X_scaled)

changed = np.sum(y_kmeans != y_kmeans_alt)

st.write(
    f"Number of customers with changed cluster assignment "
    f"after changing random state: **{changed}**"
)

