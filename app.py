"""
app.py
Streamlit app to explore CORD-19 dataset.
"""

import streamlit as st
import pandas as pd
import analysis  # our helper script
import matplotlib.pyplot as plt

# =========================
# Load Data
# =========================
@st.cache_data
def load_and_clean(filepath: str):
    df = analysis.load_data(filepath)
    df = analysis.clean_data(df)
    return df


st.title(" CORD-19 Data Explorer")
st.write("Interactive exploration of COVID-19 research papers metadata.")

# Filepath to your dataset
DATA_PATH = "data/metadata.csv"

# Load data
df = load_and_clean(DATA_PATH)

# =========================
# Sidebar Filters
# =========================
st.sidebar.header("Filters")
years = st.sidebar.slider("Select Year Range", int(df['year'].min()), int(df['year'].max()), 
                          (2020, 2021))
filtered_df = df[(df['year'] >= years[0]) & (df['year'] <= years[1])]

# =========================
# Data Overview
# =========================
st.subheader("Dataset Overview")
st.write(filtered_df.head())

st.markdown(f"**Total papers in selected range:** {len(filtered_df)}")

# =========================
# Visualizations
# =========================
st.subheader("Publications by Year")
fig1 = analysis.plot_publications_per_year(filtered_df)
st.pyplot(fig1)

st.subheader("Top Journals")
top_n = st.sidebar.number_input("Number of Top Journals", 5, 20, 10)
fig2 = analysis.plot_top_journals(filtered_df, n=top_n)
st.pyplot(fig2)

st.subheader("Word Cloud of Titles")
fig3 = analysis.generate_wordcloud(filtered_df)
st.pyplot(fig3)
