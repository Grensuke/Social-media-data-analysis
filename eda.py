import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def perform_eda(input_file):
    print(f"Loading data from {input_file} for EDA...")
    df = pd.read_csv(input_file)
    
    # Create an output directory for the plots
    output_dir = "eda_visualizations"
    os.makedirs(output_dir, exist_ok=True)
    
    sns.set_theme(style="whitegrid")
    
    # 1. Distribution of Sentiments
    plt.figure(figsize=(10, 8))
    order = df['Sentiment'].value_counts().index
    top_n = min(20, len(order)) # Plot top 20 sentiments
    sns.countplot(y='Sentiment', data=df, order=order[:top_n], hue='Sentiment', legend=False, palette='viridis')
    plt.title('Top 20 Sentiments Distribution')
    plt.xlabel('Count')
    plt.ylabel('Sentiment')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sentiment_distribution.png'))
    plt.close()
    print("Saved sentiment_distribution.png")
    
    # 2. Distribution of Platforms
    plt.figure(figsize=(8, 5))
    sns.countplot(x='Platform', data=df, order=df['Platform'].value_counts().index, hue='Platform', legend=False, palette='Set2')
    plt.title('Distribution of Platforms')
    plt.xlabel('Platform')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'platform_distribution.png'))
    plt.close()
    print("Saved platform_distribution.png")
    
    # 3. Top 10 Countries by Tweet Volume
    plt.figure(figsize=(10, 6))
    country_order = df['Country'].value_counts().index[:10]
    sns.countplot(y='Country', data=df, order=country_order, hue='Country', legend=False, palette='magma')
    plt.title('Top 10 Countries by Tweet Volume')
    plt.xlabel('Count')
    plt.ylabel('Country')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'country_distribution.png'))
    plt.close()
    print("Saved country_distribution.png")

    # 4. Likes and Retweets by Platform
    plt.figure(figsize=(10, 5))
    sns.boxplot(x='Platform', y='Likes', data=df, hue='Platform', legend=False, palette='pastel')
    plt.title('Likes Distribution by Platform')
    # Use log scale if there's high variance, but let's try linear first unless we hit errors
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'likes_by_platform.png'))
    plt.close()
    print("Saved likes_by_platform.png")
    
    plt.figure(figsize=(10, 5))
    sns.boxplot(x='Platform', y='Retweets', data=df, hue='Platform', legend=False, palette='pastel')
    plt.title('Retweets Distribution by Platform')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'retweets_by_platform.png'))
    plt.close()
    print("Saved retweets_by_platform.png")
    
    # Generate a textual summary
    summary_path = os.path.join(output_dir, "eda_summary.txt")
    with open(summary_path, 'w') as f:
        f.write("EDA Summary for Sentiment Dataset\n")
        f.write("="*35 + "\n")
        f.write(f"Total Records: {len(df)}\n\n")
        f.write("Sentiment Value Counts (Top 15):\n")
        f.write(f"{df['Sentiment'].value_counts().head(15)}\n\n")
        f.write("Platform Value Counts:\n")
        f.write(f"{df['Platform'].value_counts()}\n\n")
        f.write("Country Value Counts (Top 10):\n")
        f.write(f"{df['Country'].value_counts().head(10)}\n\n")
        f.write("Basic Statistics for Likes and Retweets:\n")
        f.write(f"{df[['Likes', 'Retweets']].describe()}\n")
    print(f"Saved textual summary to {summary_path}")
    
    print("EDA Visualizations generation complete!")

if __name__ == "__main__":
    perform_eda("preprocessed_sentimentdataset.csv")
