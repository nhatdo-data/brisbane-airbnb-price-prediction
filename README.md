# Brisbane Airbnb Price Prediction

Predicting nightly Airbnb listing prices in Brisbane, Australia, using gradient boosting and stacking models with engineered sentiment, geospatial and amenity features.

The approach is adapted from Milunovich & Nasrabadi (2024), *Airbnb Pricing in Sydney: Predictive Modelling and Explainable AI*, which found that ensemble methods and engineered location, amenity and sentiment features improve price forecasts.

> Group project for a Master of Business Analytics predictive analytics unit at Macquarie University.

---

## Project overview

| | |
|---|---|
| **Goal** | Predict the nightly price of a Brisbane Airbnb listing |
| **Data** | 3,735 training listings and 1,601 test listings |
| **Target** | `log(price)`, with predictions converted back using `exp()` |
| **Final feature set** | 69 columns (21 original numerical, 33 engineered, 15 dummy columns) |
| **Evaluation** | 5-fold cross-validation, MAE on log-price |

---

## Workflow

1. **Data cleaning and exploration**
   - Converted price to numeric and plotted its distribution (strongly right-skewed, so log-transformed).
   - Summarised missing values in the training and test sets, and imputed them.
2. **Feature engineering**
   - **Sentiment:** VADER compound scores for the listing description, neighbourhood overview and host bio.
   - **Geospatial:** haversine distance to the nearest tourist attraction, major highway and train station, plus distance to the CBD.
   - **Amenities:** amenities grouped into low, medium and high price tiers using the average listing price for each amenity. Each listing gets a count per tier.
   - **Property and size features:** bedroom-to-bathroom ratio, guests per bedroom, luxury density, and neighbourhood median price.
   - **Interactions:** distance × size, and tourist area × accommodates.
   - **Binary flags:** pool, fireplace, BBQ, beachfront.
3. **Preprocessing**
   - Capped skewed numerical features at the 99th percentile, with the cap calculated from the training set only.
   - One-hot encoded the top categories for neighbourhood, property type and room type.
   - Ordinal-encoded host response time.
   - Standardised numerical features.
4. **Modelling**
   - Tuned Gradient Boosting, XGBoost and CatBoost with grid search and 5-fold cross-validation.
   - Tested stacking ensembles with a Ridge meta-model.
5. **Prediction**
   - Retrained the best model on the full training set and predicted test prices.

---

## Results

Tuned models, 5-fold CV MAE on log-price (lower is better):

| Model | CV MAE |
|---|---|
| XGBoost | 0.0641 |
| Gradient Boosting | 0.0656 |
| CatBoost | 0.0764 |
| Stacking (GB + XGB + CatBoost) | _fill in_ |
| Stacking (XGBoost + Random Forest) | _fill in_ |

**Final model:** _fill in_
**Leaderboard score:** _fill in_

---

## Repository structure

```
brisbane-airbnb-price-prediction/
├── notebook.ipynb      # full analysis, feature engineering and modelling
├── README.md
├── requirements.txt
├── .gitignore
└── data/               # not included, see "Data" below
```

---

## Data

The data was provided for a university predictive analytics competition and is **not included in this repository**. To run the notebook, place `train.csv` and `test.csv` in a `data/` folder and update the file paths at the top of the notebook.

---

## How to run

```bash
git clone https://github.com/<your-username>/brisbane-airbnb-price-prediction.git
cd brisbane-airbnb-price-prediction
pip install -r requirements.txt
jupyter notebook notebook.ipynb
```

---

## Tech stack

Python, pandas, NumPy, scikit-learn, XGBoost, CatBoost, vaderSentiment, matplotlib, seaborn

---

## Reference

Milunovich, G., & Nasrabadi, D. (2024). *Airbnb Pricing in Sydney: Predictive Modelling and Explainable AI.* SSRN. https://doi.org/10.2139/ssrn.4805859
