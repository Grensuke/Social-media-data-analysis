# Social Media Data Analysis

This repository contains Python scripts and a Jupyter Notebook for preprocessing and performing Exploratory Data Analysis (EDA) on a social media sentiment dataset.

## Project Structure

- `sentimentdataset.csv`: The raw dataset containing social media posts, sentiments, and engagement metrics.
- `preprocess.py`: Script to clean the text data, handle missing values, and drop unnecessary columns. Outputs `preprocessed_sentimentdataset.csv`.
- `eda.py`: Script to generate visualizations (bar charts, box plots) based on the preprocessed data. Saves outputs to the `eda_visualizations/` folder.
- `sentiment_analysis_eda.ipynb`: A Jupyter Notebook containing both preprocessing and EDA steps for interactive exploration.
- `sentiment_analysis_eda.py`: The Python script version of the Jupyter Notebook.

## Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Sahith0077/Social-media-data-analysis.git
   cd Social-media-data-analysis
   ```

2. **Install dependencies:**
   It is recommended to use a virtual environment.
   ```bash
   pip install -r requirements.txt
   ```

## Usage

**Step 1: Preprocess the data**
```bash
python preprocess.py
```
This will read `sentimentdataset.csv` and generate `preprocessed_sentimentdataset.csv`.

**Step 2: Run Exploratory Data Analysis (EDA)**
```bash
python eda.py
```
This will read the preprocessed data and generate visualizations in the `eda_visualizations/` directory.

Alternatively, you can open `sentiment_analysis_eda.ipynb` in Jupyter Notebook or JupyterLab to run the analysis interactively.
