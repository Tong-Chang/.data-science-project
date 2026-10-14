# House Sales in King County, USA — Price Analysis & Prediction

Predicting residential sale prices in King County (Seattle area) from structural and
location attributes, using exploratory data analysis and regularised linear models.

| | |
| --- | --- |
| **Author** | Tong Chang (mc63803@um.edu.mo) |
| **Course** | CISC7204 — Data Visualisation / Data Analysis |
| **Status** | Completed |
| **Notebook** | `house_price_prediction_king_county.ipynb` |

---

## 1. Problem and research question

A real-estate investment trust wants to enter the residential housing market and needs a
defensible, data-driven way to value a property before making an offer.

> **Research question.** How accurately can the sale price of a house in King County be
> predicted from its structural attributes (living area, construction grade, number of
> bedrooms/bathrooms, floors, basement, view, waterfront) and its location (latitude,
> longitude, zipcode) — and which of these attributes carry the most predictive signal?

This is framed as a supervised regression problem. `price` is the target; the remaining
columns are candidate predictors.

---

## 2. Dataset source and usage permissions

| Item | Detail |
| --- | --- |
| **File** | `kc_house_data_NaN.csv` |
| **Content** | 21,613 residential property sales in King County, Washington (includes Seattle) |
| **Period** | May 2014 – May 2015 |
| **Origin** | King County Assessor records, redistributed through the [Kaggle "House Sales in King County" dataset](https://www.kaggle.com/datasets/harlfoxem/housesalesprediction) |
| **Version used** | Course-modified copy (`_NaN` variant) hosted by IBM Skills Network for *Data Analysis with Python* |
| **Download URL** | `https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-SkillsNetwork/labs/FinalModule_Coursera/data/kc_house_data_NaN.csv` |
| **Licence / permission** | The underlying Kaggle dataset is published for public, non-commercial educational and analysis use. This repository redistributes **no raw data** — the notebook reads the CSV straight from the public URL at runtime. |
| **Privacy** | Property transaction records only. No personal identifiers, no credentials, no API keys are stored in this repository. |

Columns used: `price`, `bedrooms`, `bathrooms`, `sqft_living`, `sqft_lot`, `floors`,
`waterfront`, `view`, `condition`, `grade`, `sqft_above`, `sqft_basement`, `yr_built`,
`yr_renovated`, `zipcode`, `lat`, `long`, `sqft_living15`, `sqft_lot15`.
Identifier columns `id` and `Unnamed: 0` are dropped during wrangling.

---

## 3. Methods and tools

**Tools:** Python 3, pandas, NumPy, matplotlib, seaborn, scikit-learn, Jupyter Notebook.

**Pipeline**

1. **Import** — load the CSV directly from the public URL via `pandas.read_csv`.
2. **Data wrangling** —
   - drop `id` and `Unnamed: 0`;
   - missing values found in `bedrooms` (13) and `bathrooms` (10);
   - both imputed with the column mean, then re-checked (0 remaining NaN).
3. **Exploratory data analysis** —
   - frequency table of `floors` with `value_counts()`;
   - seaborn boxplot of `price` grouped by `waterfront` to compare outlier spread;
   - seaborn `regplot` of `price` vs `sqft_above` with fitted regression line;
   - full numeric correlation matrix, ranking all features against `price`.
4. **Model development** — four progressively richer specifications:
   - `LinearRegression` on `sqft_living` alone;
   - `LinearRegression` on 11 features (`floors`, `waterfront`, `lat`, `bedrooms`,
     `sqft_basement`, `view`, `bathrooms`, `sqft_living15`, `sqft_above`, `grade`,
     `sqft_living`);
   - `Pipeline([StandardScaler → PolynomialFeatures → LinearRegression])` on the same
     11 features.
5. **Evaluation and refinement** —
   - `train_test_split(test_size=0.15, random_state=1)` → 18,371 train / 3,242 test rows;
   - `Ridge(alpha=0.1)` on the 11 features, scored on the held-out test set;
   - `Ridge(alpha=0.1)` on 2nd-degree `PolynomialFeatures` expansion, scored on the same
     test set.

---

## 4. Key findings

**Feature correlation with `price`** (Pearson *r*):

| Feature | r | Feature | r |
| --- | --- | --- | --- |
| `sqft_living` | 0.702 | `bedrooms` | 0.309 |
| `grade` | 0.667 | `lat` | 0.307 |
| `sqft_above` | 0.606 | `waterfront` | 0.266 |
| `sqft_living15` | 0.585 | `floors` | 0.257 |
| `bathrooms` | 0.526 | `view` | 0.397 |
| `sqft_basement` | 0.324 | `long` | 0.022 |
| `zipcode` | −0.053 | `sqft_lot` | 0.090 |

**Model performance (R²):**

| # | Model | R² | Evaluated on |
| --- | --- | --- | --- |
| 1 | Linear regression, `sqft_living` only | 0.4929 | full data (in-sample) |
| 2 | Linear regression, 11 features | 0.6577 | full data (in-sample) |
| 3 | Scaler + polynomial + linear pipeline | 0.7513 | full data (in-sample) |
| 4 | Ridge (α=0.1), 11 features | 0.6479 | held-out test set |
| 5 | Ridge (α=0.1) + 2nd-degree polynomial | 0.7003 | held-out test set |

**What the analysis shows**

1. **Size is the dominant driver.** Living area alone accounts for roughly half the
   variance in price (r = 0.702; R² = 0.493). Construction grade (0.667) and above-ground
   area (0.606) follow.
2. **Neighbourhood quality beats raw room counts.** `sqft_living15` — the living area of
   the 15 nearest neighbours — correlates 0.585 with price, far above `bedrooms` (0.309).
   Where a house sits matters more than how many rooms it has.
3. **Location is real but non-linear.** Latitude helps (r = 0.307) while longitude is
   almost useless in a linear model (r = 0.022), and `zipcode` treated as a number is
   actively misleading (r = −0.053). Geographic signal needs encoding, not raw numbers.
4. **Waterfront properties have more extreme high-price outliers** than non-waterfront
   ones (see Figure 1), confirming the segment behaves differently rather than simply
   costing more.
5. **Non-linear terms are where the gains are.** Adding a 2nd-degree polynomial expansion
   on the held-out test set lifts R² from 0.648 to 0.700 — about a 5-point improvement —
   the single largest gain from any modelling change.
6. **The market is low-rise.** 87.5% of sales are 1- or 2-storey homes
   (10,680 + 8,241 of 21,613).

### Limitation

**The headline figures are not measured on equal footing.** Models 1–3 report *in-sample*
R² computed on the entire dataset, so 0.7513 is optimistic and cannot be compared
directly with the 0.7003 test-set result of Model 5 — the pipeline's advantage is partly
an artefact of evaluating on data it was fitted on. Only Models 4 and 5 were evaluated on
a held-out split, and even those use a single fixed `random_state=1` split with no
cross-validation, so the reported values carry no confidence interval.

Three further constraints: the data covers a single 12-month window in one county, so
nothing here generalises to other markets or to post-2015 prices; bedrooms and bathrooms
were mean-imputed, which shrinks variance and ignores the fact that missingness may not be
random; and the entire model family is linear in its parameters, so it cannot capture
localised price dynamics without explicit feature engineering.

---

## 5. Instructions for running the analysis

The notebook downloads the dataset itself, so no data file needs to be prepared. An
internet connection is required on first run.

```bash
# 1. Clone
git clone https://github.com/Tong-Chang/.data-science-project.git
cd .data-science-project

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch Jupyter and run all cells
jupyter notebook
# open house_price_prediction_king_county.ipynb → Kernel → Restart & Run All
```

Expected runtime: under one minute on a laptop. All four figures are committed under
`images/`, so the results can be inspected without executing anything.

### Repository structure

```
.data-science-project/
├── README.md
├── requirements.txt
├── house_price_prediction_king_county.ipynb   # full analysis (figures 1-2 are produced here)
├── make_figures.py                            # regenerates the two summary figures
└── images/
    ├── fig1_price_by_waterfront.png           # price spread by waterfront (EDA)
    ├── fig2_price_vs_sqft_above.png           # price vs above-ground area (EDA)
    ├── fig3_model_r2_comparison.png           # R² across the five models
    └── fig4_feature_correlation.png           # correlation of each feature with price
```

To redraw the two summary figures without running the notebook:

```bash
pip install -r requirements.txt
python make_figures.py
```

---

## 6. Compliance note

No passwords, API keys, tokens, personal data, or restricted datasets are included in this
repository. All analysis inputs are publicly available and cited above.
