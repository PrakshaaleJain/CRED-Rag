import numpy as np
from sklearn.metrics import mean_absolute_error
y_val = ['BBB-', 'A-', 'BB+']
preds = ['BBB-', 'A-', 'BB+']
RATING_MAP = {
    'AAA': 20, 'AA+': 19, 'AA': 18, 'AA-': 17, 'A+': 16, 'A': 15, 'A-': 14,
    'BBB+': 13, 'BBB': 12, 'BBB-': 11, 'BB+': 10, 'BB': 9, 'BB-': 8,
    'B+': 7, 'B': 6, 'B-': 5, 'CCC+': 4, 'CCC': 3, 'CCC-': 2, 'CC': 1, 'C': 0, 'D': 0
}
y_val_num = np.array([RATING_MAP.get(v, 0) for v in y_val])
preds_num = np.array([RATING_MAP.get(p, 0) for p in preds])
mae = mean_absolute_error(y_val_num, preds_num)
abs_diff = np.abs(y_val_num - preds_num)
within_1_bucket = np.mean(abs_diff <= 1)
print(y_val_num, preds_num)
print("MAE", mae, "Within 1", within_1_bucket)
