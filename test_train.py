# test_train.py

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import joblib
import warnings
# Import the custom class from the new utility file
from utils import TargetEncoder 

warnings.filterwarnings('ignore')

# -------------------------- CONFIG --------------------------
RANDOM_STATE = 42
DATA_PATH = "beer-servings.csv" 
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)
# ------------------------------------------------------------

print("Starting model training and artifact generation...")

def load_and_preprocess_data():
    """Loads and cleans the dataset."""
    try:
        df = pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        print(f"FATAL: Data file not found at: {DATA_PATH}.")
        return pd.DataFrame()
    
    df['continent'] = df['continent'].fillna('Unknown')
    df = df.dropna()
    return df

df = load_and_preprocess_data()
if df.empty:
    exit()

X = df[['country', 'beer_servings', 'spirit_servings', 'wine_servings', 'continent']]
y = df['total_litres_of_pure_alcohol']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE
)

# 1. Target Encode 'country'
country_encoder = TargetEncoder()
country_encoder.fit(X_train['country'], y_train) 
X_train['country_encoded'] = country_encoder.transform(X_train['country']) 
X_test['country_encoded'] = country_encoder.transform(X_test['country'])

# 2. One-Hot Encode 'continent'
ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
continent_ohe_train = ohe.fit_transform(X_train[['continent']])
continent_cols = [f'cont_{c}' for c in ohe.categories_[0]]

X_train_cont = pd.DataFrame(continent_ohe_train, columns=continent_cols, index=X_train.index)
X_test_cont = pd.DataFrame(ohe.transform(X_test[['continent']]), columns=continent_cols, index=X_test.index)

X_train_final = pd.concat([X_train.drop(columns=['continent', 'country']), X_train_cont], axis=1)
X_test_final = pd.concat([X_test.drop(columns=['continent', 'country']), X_test_cont], axis=1)

# 3. Scale continuous features
scaler = StandardScaler()
cont_features = ['beer_servings', 'spirit_servings', 'wine_servings', 'country_encoded']
X_train_final[cont_features] = scaler.fit_transform(X_train_final[cont_features])
X_test_final[cont_features] = scaler.transform(X_test_final[cont_features])

# --- Training and Evaluation ---
ridge = Ridge(random_state=RANDOM_STATE)
ridge_cv = GridSearchCV(ridge, {'alpha': [0.1, 1.0, 10.0]}, cv=5, scoring='r2', n_jobs=-1)
ridge_cv.fit(X_train_final, y_train)

best_name = 'Ridge Regression'
best_model = ridge_cv.best_estimator_
preds = best_model.predict(X_test_final)
r2 = r2_score(y_test, preds)

print(f"\nTraining Complete. Best model selected: {best_name} (R2 on test set: {r2:.4f})")

# --- Save Artifacts (CRITICAL STEP) ---
try:
    print(f"Saving artifacts to {MODELS_DIR}...")
    
    # 1. SAVE THE MODEL AS A DICTIONARY (This definitively fixes the 'name' error)
    joblib.dump(
        {'model': best_model, 'name': best_name, 'r2_test': r2}, 
        MODELS_DIR / 'best_model.joblib'
    )
    
    # 2. Save the preprocessors
    joblib.dump(country_encoder, MODELS_DIR / 'country_encoder.joblib')
    joblib.dump(ohe, MODELS_DIR / 'cont_ohe.joblib')
    joblib.dump(scaler, MODELS_DIR / 'scaler.joblib')

    print("SUCCESS: All model artifacts saved!")

except Exception as e:
    print(f"ERROR: Could not save model artifacts! {e}")