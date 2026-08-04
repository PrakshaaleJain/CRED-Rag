import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, accuracy_score

RATING_MAP = {
    'AAA': 20, 'AA+': 19, 'AA': 18, 'AA-': 17, 'A+': 16, 'A': 15, 'A-': 14,
    'BBB+': 13, 'BBB': 12, 'BBB-': 11, 'BB+': 10, 'BB': 9, 'BB-': 8,
    'B+': 7, 'B': 6, 'B-': 5, 'CCC+': 4, 'CCC': 3, 'CCC-': 2, 'CC': 1, 'C': 0, 'D': 0
}

df = pd.read_csv('OmegaSquared/data/CRED/filteredpromax_kaggle_with_year_month_CRED.csv')
y_val = df['label'].values[:10]
# simulate predictions
preds = df['label'].values[1:11] 

y_val_num = np.array([RATING_MAP.get(v, 0) for v in y_val])
preds_num = np.array([RATING_MAP.get(p, 0) for p in preds])

mae = mean_absolute_error(y_val_num, preds_num)
abs_diff = np.abs(y_val_num - preds_num)
within_1_bucket = np.mean(abs_diff <= 1)

print(y_val[:5])
print(y_val_num[:5])
print(preds[:5])
print(preds_num[:5])
print(mae, within_1_bucket)
