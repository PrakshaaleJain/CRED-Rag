import pandas as pd
df = pd.read_csv('OmegaSquared/data/CRED/filteredpromax_kaggle_with_year_month_CRED.csv')
y = df['label'].values
print("y dtype:", y.dtype)
print("y[0] type:", type(y[0]))
