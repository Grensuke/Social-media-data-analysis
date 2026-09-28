# %% [markdown]
# # Sentiment Dataset Preprocessing and EDA
# This notebook covers standard preprocessing steps and exploratory data analysis (EDA) for the sentiment dataset.

# %%
import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns

# Configure plotting
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

# %% [markdown]
# ## 1. Data Preprocessing

# %%
def clean_text(text):
    if not isinstance(text, str):
        return text
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE) # Remove URLs
    text = re.sub(r'[^\w\s]', '', text) # Remove punctuation
    text = re.sub(r'\s+', ' ', text).strip() # Remove extra spaces
    return text

# Load dataset
input_file = "sentimentdataset.csv"
df = pd.read_csv(input_file)
print(f"Original shape: {df.shape}")

# Drop unnecessary columns
columns_to_drop = ['Unnamed: 0', 'Unnamed: 0.1', 'Timestamp', 'User', 'Year', 'Month', 'Day', 'Hour']
existing_cols = [col for col in columns_to_drop if col in df.columns]
df = df.drop(columns=existing_cols)
print(f"Dropped columns: {existing_cols}")

# Handle missing values
df = df.dropna(subset=['Text', 'Sentiment'])

# Clean text data
df['Text'] = df['Text'].apply(clean_text)

# Strip whitespace from categorical columns
for col in ['Sentiment', 'Platform', 'Country', 'Hashtags']:
    if col in df.columns:
        df[col] = df[col].astype(str).str.strip()

print(f"Preprocessed shape: {df.shape}")
display(df.head())

# %% [markdown]
# ## 2. Exploratory Data Analysis (EDA)

# %% [markdown]
# ### 2.1 Sentiment Distribution

# %%
plt.figure(figsize=(10, 8))
order = df['Sentiment'].value_counts().index
top_n = min(20, len(order))
sns.countplot(y='Sentiment', data=df, order=order[:top_n], hue='Sentiment', legend=False, palette='viridis')
plt.title('Top 20 Sentiments Distribution')
plt.xlabel('Count')
plt.ylabel('Sentiment')
plt.show()

# %% [markdown]
# ### 2.2 Platform Distribution

# %%
plt.figure(figsize=(8, 5))
sns.countplot(x='Platform', data=df, order=df['Platform'].value_counts().index, hue='Platform', legend=False, palette='Set2')
plt.title('Distribution of Platforms')
plt.xlabel('Platform')
plt.ylabel('Count')
plt.show()

# %% [markdown]
# ### 2.3 Top 10 Countries by Tweet Volume

# %%
plt.figure(figsize=(10, 6))
country_order = df['Country'].value_counts().index[:10]
sns.countplot(y='Country', data=df, order=country_order, hue='Country', legend=False, palette='magma')
plt.title('Top 10 Countries by Tweet Volume')
plt.xlabel('Count')
plt.ylabel('Country')
plt.show()

# %% [markdown]
# ### 2.4 Engagement Metrics by Platform (Likes and Retweets)

# %%
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

sns.boxplot(ax=axes[0], x='Platform', y='Likes', data=df, hue='Platform', legend=False, palette='pastel')
axes[0].set_title('Likes Distribution by Platform')

sns.boxplot(ax=axes[1], x='Platform', y='Retweets', data=df, hue='Platform', legend=False, palette='pastel')
axes[1].set_title('Retweets Distribution by Platform')

plt.tight_layout()
plt.show()

# %% [markdown]
# ### 2.5 Summary Statistics

# %%
print("Summary Statistics for Likes and Retweets:")
display(df[['Likes', 'Retweets']].describe())
