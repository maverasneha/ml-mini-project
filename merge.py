import pandas as pd

# 1. Load both CSV files
true_df = pd.read_csv('data/True.csv')
fake_df = pd.read_csv('data/Fake.csv')

# 2. Add a label column to each dataset
true_df['label'] = 'real'
fake_df['label'] = 'fake'

# 3. Combine them into one dataset and shuffle the rows randomly
combined_df = pd.concat([true_df, fake_df], ignore_index=True)
combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)

# 4. Save to the final file expected by your training pipeline
combined_df.to_csv('data/fake_or_real_news.csv', index=False)

print("✅ Successfully merged True.csv and Fake.csv into data/fake_or_real_news.csv!")
