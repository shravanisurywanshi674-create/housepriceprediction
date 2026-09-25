from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'dataset' / 'house_price_dataset.csv'
MODEL_DIR = ROOT / 'models'
MODEL_DIR.mkdir(exist_ok=True)

df = pd.read_csv(DATA)
X = df.drop(columns=['price_inr'])
y = df['price_inr']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

models = {
    'Linear Regression': Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('model', LinearRegression()),
    ]),
    'Random Forest': Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('model', RandomForestRegressor(n_estimators=250, max_depth=16, min_samples_leaf=2, random_state=42, n_jobs=-1)),
    ]),
    'Gradient Boosting': Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('model', GradientBoostingRegressor(n_estimators=250, learning_rate=0.05, max_depth=3, random_state=42, loss='huber')),
    ]),
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    results[name] = {
        'MAE_INR': float(mean_absolute_error(y_test, pred)),
        'RMSE_INR': float(mean_squared_error(y_test, pred) ** 0.5),
        'R2': float(r2_score(y_test, pred)),
    }

# Select by lowest RMSE, then highest R2.
best_name = min(results, key=lambda n: (results[n]['RMSE_INR'], -results[n]['R2']))
best_model = models[best_name]
joblib.dump(best_model, MODEL_DIR / 'house_price_model.joblib')

metadata = {
    'best_model': best_name,
    'features': list(X.columns),
    'target': 'price_inr',
    'test_size': 0.20,
    'random_state': 42,
    'metrics': results,
}
(MODEL_DIR / 'metrics.json').write_text(json.dumps(metadata, indent=2))

print('\nModel comparison:')
for name, m in results.items():
    print(f"{name}: MAE=₹{m['MAE_INR']:,.0f}, RMSE=₹{m['RMSE_INR']:,.0f}, R²={m['R2']:.4f}")
print(f'\nSaved best model: {best_name}')
