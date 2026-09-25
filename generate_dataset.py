import numpy as np
import pandas as pd
from pathlib import Path

rng = np.random.default_rng(42)
n = 3000

area = rng.integers(500, 4501, n)
# Correlated but realistic house characteristics
bedrooms = np.clip(np.round(area / 750 + rng.normal(0, 0.7, n)), 1, 6).astype(int)
bathrooms = np.clip(np.round(bedrooms * 0.65 + rng.normal(0, 0.45, n)), 1, 5).astype(int)
stories = rng.integers(1, 4, n)
parking = rng.integers(0, 4, n)
age = rng.integers(0, 41, n)
distance = np.round(rng.uniform(1, 30, n), 2)
location_score = np.round(rng.uniform(1, 10, n), 1)
condition_score = np.round(rng.uniform(1, 10, n), 1)
has_garden = rng.integers(0, 2, n)
has_basement = rng.integers(0, 2, n)

# Price in Indian Rupees; deterministic relationship + noise.
price = (
    1800000
    + area * 8500
    + bedrooms * 850000
    + bathrooms * 650000
    + stories * 350000
    + parking * 300000
    - age * 55000
    - distance * 85000
    + location_score * 850000
    + condition_score * 500000
    + has_garden * 450000
    + has_basement * 600000
    + rng.normal(0, 1400000, n)
)
price = np.maximum(price, 1200000).round(-3).astype(int)

df = pd.DataFrame({
    'area_sqft': area,
    'bedrooms': bedrooms,
    'bathrooms': bathrooms,
    'stories': stories,
    'parking': parking,
    'age_years': age,
    'distance_to_city_km': distance,
    'location_score': location_score,
    'condition_score': condition_score,
    'has_garden': has_garden,
    'has_basement': has_basement,
    'price_inr': price,
})

out = Path(__file__).resolve().parent / 'dataset' / 'house_price_dataset.csv'
out.parent.mkdir(exist_ok=True)
df.to_csv(out, index=False)
print(f'Saved {len(df)} rows to {out}')
