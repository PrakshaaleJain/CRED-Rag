import pandas as pd
import numpy as np
classes = np.array(['BBB-', 'A-', 'BB+'])
mapping = {c: i for i, c in enumerate(classes)}
inv_mapping = {i: c for i, c in enumerate(classes)}

preds_mapped = np.array([0, 1, 2])
preds = np.array([inv_mapping.get(p, -1) for p in preds_mapped])
print(preds)

preds_mapped2 = np.array([0.0, 1.0, 2.0])
preds2 = np.array([inv_mapping.get(p, -1) for p in preds_mapped2])
print(preds2)
