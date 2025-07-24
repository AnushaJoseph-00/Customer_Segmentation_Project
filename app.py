import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import numpy as np

st.set_page_config(page_title="Customer Segmentation", layout="wide")

st.title("📊 Customer Segmentation Dashboard")

# Load data
@st.cache_data
def load_data():
    return pd.read_csv("customers_segmented.csv")  # Update path if needed

df = load_data()

# Features used
features = ['age', 'income', 'spending_score', 'membership_years', 'purchase_frequency', 'last_purchase_amount']

# Perform PCA if not already done
if 'PCA1' not in df.columns or 'PCA2' not in df.columns:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[features])
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    df['PCA1'] = X_pca[:, 0]
    df['PCA2'] = X_pca[:, 1]

# Segment filter
segments = df['Segment'].unique()
selected_segments = st.multiselect("🧠 Select Customer Segments to View", options=segments, default=segments)

filtered_df = df[df['Segment'].isin(selected_segments)]

# PCA Scatter Plot
st.subheader("📍 PCA Projection of Segments")
fig = px.scatter(
    filtered_df,
    x='PCA1',
    y='PCA2',
    color='Segment',
    hover_data=features,
    title='PCA Scatter Plot by Segment',
    labels={'PCA1': 'PCA Component 1', 'PCA2': 'PCA Component 2'}
)
st.plotly_chart(fig, use_container_width=True)

# Segment Profiles Radar Chart
st.subheader("📈 Segment Profile Comparison")

profile_data = df.groupby("Segment")[features].mean().reset_index()
radar_fig = go.Figure()

for i, row in profile_data.iterrows():
    radar_fig.add_trace(go.Scatterpolar(
        r=row[features].values,
        theta=features,
        fill='toself',
        name=f'Segment {row["Segment"]}'
    ))

radar_fig.update_layout(
    polar=dict(radialaxis=dict(visible=True)),
    showlegend=True
)
st.plotly_chart(radar_fig, use_container_width=True)

# Segment Descriptions
st.subheader("📌 Segment Descriptions")

with st.expander("📋 View Customer Segment Profiles"):
    st.markdown("""
    ### Segment 0 - **Mid Spenders**
    - Moderate age (~46)
    - Medium income (~$66K)
    - Lower spending score (~31)
    - Average loyalty (~4.6 years)
    - Spend more per purchase (~$746)

    ### Segment 1 - **Wealthy Minimalists**
    - High income (~$110K)
    - Low spending score (~35)
    - Low purchase amounts (~$244)
    - Likely occasional big-ticket buyers

    ### Segment 2 - **Top Customers**
    - Highest income (~$114K)
    - High spending score (~66)
    - Loyal (6.5+ years)
    - Highest purchase frequency & amount (~$725)
    - *Target for premium offerings*

    ### Segment 3 - **Frequent Moderate Spenders**
    - Moderate income (~$61K)
    - Highest spending score (~73)
    - Shop often (~25 purchases)
    - Medium purchase amount (~$326)
    - *Likely promo-sensitive or value-seekers*
    """)

# Segment Table
st.subheader("🧾 Segment Data Table")
st.dataframe(filtered_df.head(20), use_container_width=True)

# Download data
csv = filtered_df.to_csv(index=False).encode()
st.download_button("📥 Download Selected Segment Data", csv, "selected_segments.csv", "text/csv")

# Segment-wise prediction (new input)
st.subheader("🔮 Predict Segment for New Customer")
with st.form("predict_form"):
    age = st.slider("Age", 18, 70, 35)
    income = st.number_input("Income", min_value=10000, max_value=200000, value=50000)
    spending_score = st.slider("Spending Score", 0, 100, 50)
    membership_years = st.slider("Membership Years", 0, 10, 5)
    purchase_frequency = st.slider("Purchase Frequency", 1, 50, 10)
    last_purchase_amount = st.number_input("Last Purchase Amount", 0.0, 10000.0, 500.0)

    submitted = st.form_submit_button("Predict Segment")

    if submitted:
        # Fit KMeans on full data to make predictions
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df[features])
        kmeans = KMeans(n_clusters=len(segments), random_state=42, n_init='auto')
        kmeans.fit(X_scaled)

        new_customer = np.array([[age, income, spending_score, membership_years, purchase_frequency, last_purchase_amount]])
        new_scaled = scaler.transform(new_customer)
        predicted_segment = kmeans.predict(new_scaled)[0]
        st.success(f"🎯 Predicted Segment: {predicted_segment}")

        # Segment interpretation
        segment_descriptions = {
            0: "Mid Spenders — average loyalty, medium income, lower spending score.",
            1: "Wealthy Minimalists — high income, spend infrequently.",
            2: "Top Customers — loyal, high income, high frequency and spending.",
            3: "Frequent Moderate Spenders — spend often, likely value-seeking."
        }

        if predicted_segment in segment_descriptions:
            st.info(f"📝 Description: {segment_descriptions[predicted_segment]}")
