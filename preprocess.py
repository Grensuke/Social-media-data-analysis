import pandas as pd
import re

def clean_text(text):
    if not isinstance(text, str):
        return text
    # Convert to lowercase
    text = text.lower()
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def preprocess_data(input_file, output_file):
    print(f"Loading data from {input_file}...")
    df = pd.read_csv(input_file)
    
    # 1. Drop unnecessary columns
    columns_to_drop = ['Unnamed: 0', 'Unnamed: 0.1', 'Timestamp', 'User', 'Year', 'Month', 'Day', 'Hour']
    existing_cols_to_drop = [col for col in columns_to_drop if col in df.columns]
    if existing_cols_to_drop:
        df = df.drop(columns=existing_cols_to_drop)
        print(f"Dropped columns: {existing_cols_to_drop}")

    # 2. Handle missing values
    initial_shape = df.shape
    # Drop rows where 'Text' or 'Sentiment' is missing
    df = df.dropna(subset=['Text', 'Sentiment'])
    print(f"Dropped {initial_shape[0] - df.shape[0]} rows due to missing values in Text or Sentiment.")

    # 3. Clean 'Text' column
    print("Cleaning text data...")
    df['Text'] = df['Text'].apply(clean_text)
    
    # Clean string columns (strip extra spaces from Sentiment, Platform, Country)
    for col in ['Sentiment', 'Platform', 'Country', 'Hashtags']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
    
    # 4. Save to new CSV
    df.to_csv(output_file, index=False)
    print(f"Preprocessing complete. Saved to {output_file}")
    print(f"Final shape: {df.shape}")

if __name__ == "__main__":
    input_filename = "sentimentdataset.csv"
    output_filename = "preprocessed_sentimentdataset.csv"
    preprocess_data(input_filename, output_filename)
