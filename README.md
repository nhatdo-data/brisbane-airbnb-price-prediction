# Predicting Airbnb Listing Prices in Brisbane

**BUSA8001 Group Project · Semester 1, 2026 · Kaggle in-class competition**

Team **BUSA8001_HTQ**

| Member | Responsibility |
|---|---|
| Member 1 | Task 1: problem framing and EDA |
| Nhat Thanh Thuy Do | Task 2: data cleaning and feature engineering |
| Member 2 (team leader) | Task 3: modelling, tuning and submission |

We predict the nightly price (AUD) of Brisbane Airbnb listings from property, host, location, availability and review data. Submissions are scored by **Mean Absolute Percentage Error (MAPE)**.

**Final result:** our best model stacks **XGBoost and a Random Forest** under a Ridge meta-learner. It has a cross-validated MAPE of **6.24%** and scored **0.062 (6.2%)** on Kaggle's private leaderboard. Tuned XGBoost on its own scored 6.36% in cross-validation and 0.065 on the private leaderboard. Both models scored 0.077 on the public leaderboard.

---

## Contents

1. [Problem](#1-problem)
2. [Data](#2-data)
3. [Exploratory analysis](#3-exploratory-analysis)
4. [Data preparation and feature engineering](#4-data-preparation-and-feature-engineering)
5. [Modelling](#5-modelling)
6. [Results](#6-results)
7. [Limitations and lessons learned](#7-limitations-and-lessons-learned)
8. [Reproducing the work](#8-reproducing-the-work)

---

## 1. Problem

This is a supervised regression task: predict `price` for 1,601 unseen listings. Accurate price estimates are useful to several groups:

- **Hosts and investors** can benchmark their prices and avoid under- or over-pricing.
- **Property managers** can price a whole portfolio consistently.
- **Guests** benefit from more transparent pricing.
- **Platforms** can power smart-pricing and recommendation tools.
- **Policymakers** can track the short-term rental market and its effect on housing.

The framing follows *hedonic pricing*: a listing's price is treated as the sum of the values of its attributes (size, location, amenities, host reputation).

### Evaluation metric

$$
\text{MAPE} = \frac{1}{n}\sum_{i=1}^{n}\left|\frac{A_i - F_i}{A_i}\right| \times 100
$$

MAPE does not depend on scale, so a $50 error on a $100 room counts more than the same error on a $500 house. It is also easy to interpret. Its weaknesses are an asymmetric penalty (it punishes over-prediction more than under-prediction) and instability on very cheap listings.

## 2. Data

| File | Rows | Columns | Notes |
|---|---|---|---|
| `train.csv` | 3,735 | 65 | Includes `price` (stored as text, e.g. `"$205.00"`) |
| `test.csv` | 1,601 | 64 | Price withheld |
| `metaData.csv` | – | – | Column dictionary (Inside Airbnb schema) |
| `sample_solution.csv` | 1,601 | 2 | Submission format: `ID, price` |

**Missing values.** The training set has 13,302 missing cells and the test set has 5,582. The pattern is almost identical in both sets:

- **Optional host text is mostly missing:** `host_neighbourhood` (96%), `neighborhood_overview` (53%), `host_about` (40%), `host_location` (22%).
- **Review fields are missing together for about 11.7% of listings.** These are listings with no reviews yet, so the gap itself carries information.
- **Core property fields** (`bedrooms`, `beds`, `bathrooms`) are less than 0.2% missing.

## 3. Exploratory analysis

**Price** is heavily right-skewed (skewness 7.48). The mean is $230, the median is $193, and the maximum is $5,000. A log transform brings skewness down to about 0.24, so all models are trained on `log(price)`.

**Market structure**

- Entire homes make up 76.4% of listings and private rooms 23.4%. Shared rooms (4) and hotel rooms (2) are negligible.
- 77.1% of hosts respond within an hour. Only 1.8% take a few days or more.
- `availability_30` has a spike at 0 days (about 480 listings that are fully booked or blocked), a median of 11 days, and a second spike at 30 days.
- 1-bedroom listings make up 50.8%. Most listings require a minimum stay of 1–2 nights.
- Review scores have a ceiling effect: the median is 4.83 out of 5.
- Listings cluster in the inner city. The top five suburbs (Brisbane City, South Brisbane, Fortitude Valley, Newstead–Bowen Hills, Spring Hill) hold about 38% of listings.

**Relationships with price**

| Driver | Evidence |
|---|---|
| Size | `accommodates` r = 0.52, `bedrooms` 0.50, `beds` 0.49, `bathrooms` 0.42 |
| Room type | Median $217 for an entire home vs $91 for a private room (a 138% premium) |
| Neighbourhood | Median $238 in Brisbane City vs $90 in Sunnybank |
| Host response time | Non-monotonic: hosts who reply "within a day" have the highest median ($203) and "a few days or more" the lowest ($140.50) |
| Weak linear signal | Review scores, `minimum_nights`, response and acceptance rates (\|r\| < 0.1) |

We kept the weak features anyway, because tree models can pick up non-linear effects that a correlation coefficient misses.

**The 20 selected base features:** `accommodates`, `bedrooms`, `beds`, `bathrooms`, `room_type`, `property_type`, `neighbourhood_cleansed`, `latitude`, `longitude`, `review_scores_rating`, `review_scores_location`, `review_scores_cleanliness`, `reviews_per_month`, `host_response_rate`, `host_acceptance_rate`, `host_response_time`, `minimum_nights`, `availability_30`, `number_of_reviews` and `amenities`. The model also uses `estimated_revenue_l365d`, `estimated_occupancy_l365d`, `maximum_nights`, `availability_60` and `number_of_reviews_ltm`.

## 4. Data preparation and feature engineering

All statistics used to transform the data (medians, top-5 categories, percentile caps, scaler parameters, neighbourhood medians) are learned **from the training set only** and then applied to both train and test.

### Cleaning

- Stripped `$` and `,` from `price` and converted it to a number.
- Converted `host_response_rate` and `host_acceptance_rate` from `"93%"` strings to proportions (0.93).

### Engineered features (33 in total)

| Group | Features | Idea |
|---|---|---|
| Amenities | `amenity_count`, `low_amenities`, `med_amenities`, `high_amenities` | Amenities are normalised with regex (e.g. all TV variants become `tv`) and ranked by the average price of listings that have them. Each listing then gets a count of low-, mid- and high-tier amenities. `high_amenities` has r = 0.24. |
| Luxury flags | `has_pool`, `has_fireplace`, `has_bbq`, `has_beachfront` | Direct signals of premium listings |
| Text sentiment | `sentiment_description`, `sentiment_neighbourhood`, `sentiment_host` | VADER compound score (−1 to +1); missing text is set to 0 |
| Location | `distance_to_cbd`, `dist_to_attraction`, `dist_to_highway`, `dist_to_station`, `neighbourhood_price_median`, `is_tourist_area` | Haversine distances to the CBD and the nearest landmark, motorway and train station, plus target-based neighbourhood medians |
| Room configuration | `bedroom_bathroom_ratio`, `accommodates_per_bedroom`, `beds_per_bedroom`, `excess_beds`, `luxury_density` | How spacious or crowded the listing is |
| Interactions | `location_size_score`, `entire_home_x_bedrooms`, `entire_home_x_bathrooms`, `size_x_bathrooms`, `bedroom_quality_score`, `distance_x_*`, `tourist_x_accommodates` | Location × size and size × quality effects |
| Revenue | `avg_nightly_rate` = `estimated_revenue_l365d / (estimated_occupancy_l365d + 1)` | Realised nightly rate (see [Limitations](#7-limitations-and-lessons-learned)) |

The strongest engineered features are `location_size_score` (r = 0.56), `entire_home_x_bedrooms` (0.53), `size_x_bathrooms` (0.53) and `avg_nightly_rate` (0.53).

### Missing values

A correlation heatmap of the features with gaps guided the strategy.

1. **Infer from related fields.** Review scores fill each other's gaps. `bedrooms` is set to `accommodates / 2`, `bathrooms` to `bedrooms × 0.5`, and `beds` to `accommodates`.
2. **Mode** for `host_response_time` (`"within an hour"`).
3. **Training-set median** for any remaining numeric gaps, including engineered features that inherited gaps from their inputs.

### Encoding

- `neighbourhood_cleansed` and `property_type` keep their 5 most frequent categories; everything else becomes `other`. Both are then one-hot encoded.
- `room_type` is one-hot encoded.
- `host_response_time` is ordinal-encoded: within an hour = 4, within a few hours = 3, within a day = 2, a few days or more = 1.
- Train and test columns are aligned with `DataFrame.align`.

### Final preparation

- Capped 27 skewed features at the training set's 99th percentile, including all distance features, `minimum_nights` and `maximum_nights`.
- Target: `log_price = log(price)`.
- Applied `StandardScaler` to 48 continuous columns. Binary flags and dummy columns are left unscaled.
- **Final design matrix:** `X_train` is 3,735 × 69 and `X_test` is 1,601 × 69, with no missing values.

## 5. Modelling

### Candidate models

| Model | How it works | Why it suits this problem | Drawbacks |
|---|---|---|---|
| Gradient Boosting (sklearn) | Builds trees one after another, each fitting the previous trees' residuals | Captures non-linear effects and interactions | Slow; no built-in regularisation |
| XGBoost | Gradient boosting with L1/L2 regularisation, row/column subsampling and a fast histogram method | Strong on tabular data; less prone to overfitting | More hyperparameters to tune |
| CatBoost | Gradient boosting with symmetric trees and ordered boosting | Robust with little tuning | Its native categorical handling goes unused because the data is already one-hot encoded |

### Evaluation strategy

- `KFold(n_splits=5, shuffle=True, random_state=42)`.
- Models are fitted on log price. In each validation fold, predictions are converted back to dollars and scored with `mean_absolute_percentage_error`, so model selection uses the **same metric as Kaggle**.
- Overfitting is checked by comparing training MAPE with cross-validated MAPE.

### Hyperparameters

The values below came from an earlier 5-fold `GridSearchCV` (over number of trees, learning rate, depth, minimum leaf size and subsampling) and are hard-coded in the final notebook.

| Model | Parameters |
|---|---|
| Gradient Boosting | `n_estimators=300, learning_rate=0.05, max_depth=7, min_samples_leaf=1` |
| XGBoost | `n_estimators=300, learning_rate=0.05, max_depth=7, subsample=0.8, colsample_bytree=1.0` |
| CatBoost | `iterations=300, learning_rate=0.1, depth=8` |

### Improvement: stacking

We stacked the tuned XGBoost model with a Random Forest (`n_estimators=200, max_depth=15, min_samples_leaf=2`). A `Ridge(alpha=0.01)` meta-learner is trained on 5-fold out-of-fold predictions. The idea was that bagging (Random Forest) and boosting (XGBoost) make partly different errors, so blending them would reduce variance.

## 6. Results

| Model | CV MAPE | Train MAPE | Kaggle public | Kaggle private |
|---|---|---|---|---|
| Gradient Boosting | 6.75% | 0.86% | – | – |
| XGBoost | 6.36% | 0.78% | 0.077 | 0.065 |
| CatBoost | 7.84% | 2.61% | – | – |
| **XGBoost + RF stack (best)** | **6.24%** | – | 0.077 | **0.062** |

- **XGBoost was the best single model.** Gradient Boosting was close behind and CatBoost clearly worse.
- **All three models overfit to some degree.** Training error is about 0.8% against roughly 6.4% in cross-validation, because trees of depth 7 memorise the training set. Validation error is still low, and there is no sign of underfitting.
- **Stacking was the most effective improvement.** It lowered cross-validated MAPE from 6.36% to 6.24% and private-leaderboard MAPE from 0.065 to 0.062. The public leaderboard, which is calculated on only part of the test set, showed no difference (0.077 for both). That is why stacking looked ineffective while the competition was running.
- **Cross-validation predicted the private result better than the public leaderboard did.** It ranked the two models in the same order as the private leaderboard. The lesson: trust a robust cross-validation setup over a small public split.
- **Only the two selected submissions count toward the final standing.** These are the stack and XGBoost files uploaded by Nhat Thanh Thuy Do, and they are the scores in the table above.
  - The same notebook was also run and uploaded from a second team member's machine. Those copies scored slightly differently: stack 0.063 private / 0.076 public, XGBoost 0.067 / 0.075. This is the library-version drift described under [Reproducing the work](#8-reproducing-the-work).
  - Earlier experimental versions reached 0.059 private, but those versions are not the final notebook and were not selected.
- **Test predictions look realistic.** XGBoost predicts $41–$1,066 and the stack $41–$1,041, both with a median of about $191. This matches the training distribution without its extreme tail.

## 7. Limitations and lessons learned

- **Target leakage through `avg_nightly_rate`.** In this dataset, `estimated_revenue_l365d / estimated_occupancy_l365d` equals `price` exactly for every listing with non-zero occupancy (3,124 of 3,735 training rows). The feature is present in the test file, so using it was valid within the competition, and it explains much of the low error. A model meant for pricing a *new* listing, which would have no revenue history, could not use it. Without it, expect noticeably higher error.
- **Inverse transform mismatch.** The target is built with `np.log`, but predictions are converted back with `np.expm1` (which is `exp − 1`). Every prediction is therefore exactly $1 too low. The effect is small (about 0.5% on a $200 listing), and because MAPE penalises over-prediction more, it may even have helped slightly. The correct inverse is `np.exp`, or the target should be built with `np.log1p`.
- **Overfitting.** Shallower trees, `min_child_weight`, stronger `reg_lambda` or early stopping might narrow the gap between training and validation error.
- **Missing signals.** The data has no photos, interior quality, views or seasonal calendar pricing. Luxury outliers up to $5,000 are rare, and there are only 3,735 training rows.
- **Coarse categories.** Grouping neighbourhood and property type into top 5 + "other" discards detail. The `neighbourhood_price_median` feature partly makes up for this.
- **Public vs private leaderboard.** The public score (0.077) was noticeably worse than the private score (0.062–0.065). This suggests the public subset contained harder listings, such as more cheap ones where percentage errors grow large.

## 8. Reproducing the work

```bash
pip install pandas numpy scikit-learn xgboost catboost vaderSentiment matplotlib seaborn scipy
```

Put `train.csv` and `test.csv` in `asba-predictive-analytics-competition-s-1-2026/`, then run the notebook from top to bottom. It writes two files:

- `submission_mape_optimised.csv`: tuned XGBoost (private MAPE 0.065)
- `submission_xgb_rf_stack_finalised.csv`: the XGBoost + RF stack (**best**, private MAPE 0.062)

> **Verification.** The notebook's saved outputs match both submitted CSV files exactly. A fresh run with newer libraries reproduces Gradient Boosting, CatBoost and the stack's cross-validation scores. XGBoost output moves by roughly 1% between library versions, which was enough to change the Kaggle score by 0.001–0.002 when the same notebook was run on two different machines. Pin the original package versions (`pip freeze > requirements.txt`) for bit-for-bit reproduction.

```
├── Group-Project-S1-2026.ipynb        # final notebook (Copy2_2)
├── README.md
├── kaggle_score.png / kaggle_score2.png
└── asba-predictive-analytics-competition-s-1-2026/
    ├── train.csv
    ├── test.csv
    ├── metaData.csv
    └── sample_solution.csv
```
