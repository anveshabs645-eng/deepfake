import pandas as pd

df = pd.read_csv('scores.csv')
print(f"Before: {len(df)} rows")

df = df[~df['filename'].isin(['us5.mp4', 'us18.mp4'])]
print(f"After: {len(df)} rows")

df.to_csv('scores.csv', index=False)
print("Done — scores.csv updated")