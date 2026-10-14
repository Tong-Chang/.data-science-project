# House Sales in King County, USA

Predicting residential sale prices in King County (Seattle area) from structural and
location attributes.

Author: Tong Chang (mc63803@um.edu.mo) · Course: CISC7204 · Status: Completed

## Problem / research question

A real-estate investment trust wants to value residential properties before making an
offer. How accurately can a house's sale price in King County be predicted from its
structural attributes (living area, grade, bedrooms, bathrooms, floors, basement, view,
waterfront) and its location (latitude, longitude, zipcode), and which attributes carry
the most signal?

## Dataset source and usage permissions

`kc_house_data_NaN.csv` — 21,613 residential sales in King County, Washington, May 2014
to May 2015. Originally King County Assessor records, redistributed via the
[Kaggle "House Sales in King County" dataset](https://www.kaggle.com/datasets/harlfoxem/housesalesprediction);
this analysis uses the course-modified copy hosted by IBM Skills Network at
`https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-SkillsNetwork/labs/FinalModule_Coursera/data/kc_house_data_NaN.csv`.

The dataset is published for public, non-commercial educational use. The notebook reads it
directly from that URL; no data file is redistributed here. It contains property records
only — no personal identifiers, credentials, or API keys are stored in this repository.

## Methods and tools

Python, pandas, NumPy, matplotlib, seaborn, scikit-learn, Jupyter Notebook.

1. Load the CSV from the public URL.
2. Drop `id` and `Unnamed: 0`; impute the missing `bedrooms` (13) and `bathrooms` (10)
   values with the column mean.
3. EDA: floor counts, price boxplot by waterfront, price-vs-`sqft_above` regression plot,
   correlation of every numeric feature with price.
4. Fit: linear regression on `sqft_living`; linear regression on 11 features; a
   `StandardScaler → PolynomialFeatures → LinearRegression` pipeline.
5. Split 18,371 train / 3,242 test (`test_size=0.15`, `random_state=1`) and fit
   `Ridge(alpha=0.1)` with and without a 2nd-degree polynomial expansion.

## Key findings

Correlation with price: `sqft_living` 0.702, `grade` 0.667, `sqft_above` 0.606,
`sqft_living15` 0.585, `bathrooms` 0.526, `bedrooms` 0.309, `lat` 0.307, `long` 0.022,
`zipcode` −0.053.

| Model | R² | Evaluated on |
| --- | --- | --- |
| Linear regression, `sqft_living` only | 0.4929 | in-sample |
| Linear regression, 11 features | 0.6577 | in-sample |
| Scaler + polynomial + linear pipeline | 0.7513 | in-sample |
| Ridge (α=0.1) | 0.6479 | held-out test |
| Ridge + 2nd-degree polynomial | 0.7003 | held-out test |

- Living area is the strongest single predictor; size, grade and above-ground area dominate.
- Neighbourhood quality beats room count: `sqft_living15` (0.585) predicts price far better
  than `bedrooms` (0.309).
- Raw geography is misleading in a linear model: `long` 0.022 and `zipcode` −0.053.
- Non-linear terms give the largest gain, lifting test R² from 0.648 to 0.700.
- Waterfront homes show more extreme high-price outliers than non-waterfront homes.

### Limitation

The first three R² values are in-sample, so 0.7513 is optimistic and cannot be compared
directly with the 0.7003 test-set result. Only the two Ridge models were evaluated on
held-out data, using a single fixed split with no cross-validation.

## How to run

```bash
git clone https://github.com/Tong-Chang/.data-science-project.git
cd .data-science-project
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook     # open house_price_prediction_king_county.ipynb → Run All
```

An internet connection is needed — the notebook downloads the dataset itself.
