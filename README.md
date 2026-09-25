# 🏠 House Price Prediction — End-to-End Machine Learning Project

A complete regression machine-learning project that predicts the estimated price of a house from its characteristics and exposes the trained model through a Streamlit web application.

## Project Requirements Covered

1. Data collection / dataset creation and preprocessing
2. Model building
3. Model evaluation
4. Model saving
5. Streamlit UI
6. Prediction and interpretation
7. GitHub-ready README and project structure

## Dataset

The included dataset is `dataset/house_price_dataset.csv` with **3,000 records**. It was generated specifically for this educational project using realistic ranges for common house-price variables. The target is `price_inr`.

Features:

- `area_sqft` — house area in square feet
- `bedrooms` — number of bedrooms
- `bathrooms` — number of bathrooms
- `stories` — number of floors/stories
- `parking` — parking spaces
- `age_years` — age of the house
- `distance_to_city_km` — distance from city center
- `location_score` — location quality score from 1 to 10
- `condition_score` — property condition score from 1 to 10
- `has_garden` — 0/1 indicator
- `has_basement` — 0/1 indicator
- `price_inr` — target house price in Indian Rupees

> Note: This project uses a generated educational dataset rather than a live real-estate market dataset. Therefore, the predictions demonstrate the ML workflow and should not be treated as real market valuations.

## Data Preprocessing

The training script:

- separates features and target
- performs an 80/20 train-test split
- uses median imputation through a scikit-learn pipeline
- standardizes features for Linear Regression
- trains tree-based models without unnecessary scaling

## Models Compared

- Linear Regression
- Random Forest Regressor
- Gradient Boosting Regressor

The final model is selected using the lowest test RMSE, with R² as a tie-breaker. The selected pipeline is saved as `models/house_price_model.joblib`.

## Evaluation Metrics

The project reports:

- **MAE (Mean Absolute Error):** average absolute prediction error in rupees
- **RMSE (Root Mean Squared Error):** penalizes larger errors more strongly
- **R² Score:** proportion of target variance explained by the model

The exact values generated on your machine are stored in `models/metrics.json` and shown in the Streamlit application.

## Project Structure

```text
House-Price-Prediction/
├── dataset/
│   └── house_price_dataset.csv
├── models/
│   ├── house_price_model.joblib
│   └── metrics.json
├── screenshots/
│   └── README.md
├── notebooks/
│   └── README.md
├── app.py
├── generate_dataset.py
├── train_model.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Installation

Use Python 3.10+ in a virtual environment.

```bash
python -m venv venv
```

Windows:

```bash
venv\Scripts\activate
```

macOS/Linux:

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the Project

If you want to regenerate the dataset:

```bash
python generate_dataset.py
```

Train and save the models:

```bash
python train_model.py
```

Start the Streamlit application:

```bash
streamlit run app.py
```

The application will open in your browser.

## Streamlit Application

The UI accepts the house characteristics, sends them to the saved model, and displays an estimated price. It also displays the model-comparison evaluation metrics.

For screenshots required by an assignment, run the application locally and capture:

1. Home/input screen
2. Prediction result screen
3. Evaluation metrics screen

Save them inside the `screenshots/` directory before pushing to GitHub.

## Model Saving

The trained scikit-learn pipeline is saved with Joblib. Only load model files that you trust, because pickle-based formats such as Joblib can execute code during loading.

## GitHub Commands

Create an empty GitHub repository named `House-Price-Prediction`, then run from this project folder:

```bash
git init
git add .
git commit -m "Build end-to-end house price prediction ML project"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/House-Price-Prediction.git
git push -u origin main
```

Replace `YOUR_USERNAME` with your GitHub username.

## Future Improvements

- Replace the generated dataset with a verified real-world real-estate dataset.
- Add location/category features.
- Add cross-validation and hyperparameter tuning.
- Add prediction explanation using feature importance or SHAP.
- Deploy the Streamlit app online.

## References

The project structure and Streamlit implementation use standard scikit-learn and Streamlit workflows. Streamlit provides numeric and selection input widgets for interactive applications, while scikit-learn documents model persistence approaches such as Joblib. See the official documentation linked in the project notes.
