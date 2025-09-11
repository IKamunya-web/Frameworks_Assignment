"""
analysis.py
Data loading, cleaning, and analysis functions for the CORD-19 dataset.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# =========================
# 1. Loading & Exploring Dataset
# =========================
def load_data(filepath: str) -> pd.DataFrame:
    """Load metadata.csv into a Pandas DataFrame."""
    df = pd.read_csv(filepath, low_memory=False)
    return df


def basic_exploration(df: pd.DataFrame):
    """Print basic dataset info and missing values summary."""
    print("Shape:", df.shape)
    print("\nInfo:")
    print(df.info())
    print("\nMissing Values:")
    print(df.isnull().sum())


# =========================
# 2. Data Cleaning
# =========================
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean dataset: handle dates, missing values, and add helper columns."""
    # Dropping rows without publication date
    df = df.dropna(subset=['publish_time'])

    # Converting date column
    df['publish_time'] = pd.to_datetime(df['publish_time'], errors='coerce')

    # Extracting year
    df['year'] = df['publish_time'].dt.year

    # Dropping rows where year is missing
    df = df.dropna(subset=['year'])

    # Adding abstract word count (optional)
    df['abstract_word_count'] = df['abstract'].fillna("").apply(lambda x: len(x.split()))

    return df


# =========================
# 3. Analysis Functions
# =========================
def publications_per_year(df: pd.DataFrame):
    """Return number of publications per year."""
    return df['year'].value_counts().sort_index()


def top_journals(df: pd.DataFrame, n=10):
    """Return top N journals by publication count."""
    return df['journal'].value_counts().head(n)


def plot_publications_per_year(df: pd.DataFrame):
    """Bar plot of publications per year."""
    year_counts = publications_per_year(df)
    plt.figure(figsize=(8, 5))
    sns.barplot(x=year_counts.index, y=year_counts.values, color="skyblue")
    plt.title("Publications by Year")
    plt.xlabel("Year")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.tight_layout()
    return plt


def plot_top_journals(df: pd.DataFrame, n=10):
    """Bar plot of top N journals."""
    top = top_journals(df, n)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=top.values, y=top.index, palette="viridis")
    plt.title(f"Top {n} Journals Publishing COVID-19 Research")
    plt.xlabel("Count")
    plt.ylabel("Journal")
    plt.tight_layout()
    return plt


def generate_wordcloud(df: pd.DataFrame):
    """Generate a word cloud from paper titles."""
    text = " ".join(df['title'].dropna().astype(str).tolist())
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title("Word Cloud of Paper Titles")
    return plt
