import pandas as pd

# Load dataset
df = pd.read_csv("fake_or_real_news.csv")

# Check column names
print("Columns:", df.columns)

# Check if 'label' exists
if 'label' not in df.columns:
    print("Error: 'label' column not found!")
else:
    print("Unique values in 'label':", df['label'].unique())

# Convert 'label' to uppercase only if it exists
if 'label' in df.columns:
    df['label'] = df['label'].astype(str).str.upper()
    print("Conversion successful!")
