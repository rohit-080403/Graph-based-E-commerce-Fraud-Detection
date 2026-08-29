# This script is used to check the feature importance of the trained XGBoost model.
# Which feature model learned on the most.

import joblib
import pandas as pd

model = joblib.load("src/baseline/xgb_baseline.model")
importance = model.feature_importances_
features = model.get_booster().feature_names

df = pd.DataFrame({"feature": features, "importance": importance})
df = df.sort_values("importance", ascending=False)
print(df.head(15))