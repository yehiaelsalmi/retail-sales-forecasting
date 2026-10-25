# Retail Sales Forecasting

Forecasting daily unit sales for a retail chain so that stock can be planned ahead of demand. The model predicts sales for every store and product across a full year it has never seen, and is compared against a simple "same day last year" baseline.

**Result:** XGBoost reduces forecast error (MAE) from **8.55 to 6.93 units per store, item and day**, about **19% lower than the baseline**, with no systematic over- or under-forecasting.

## Business problem

A retailer with several stores needs to decide how much of each product to order.

- Order too little: shelves are empty and sales are lost.
- Order too much: money is tied up in unsold stock.

**Question:** how many units of each product will each store sell on each day of the coming year?

## Data

[Store Item Demand Forecasting Challenge](https://www.kaggle.com/competitions/demand-forecasting-kernels-only) (Kaggle).

| | |
|---|---|
| Records | about 913,000 daily rows |
| Stores | 10 |
| Products | 50 |
| Period | 1 Jan 2013 to 31 Dec 2017 (5 years) |
| Columns | `date`, `store`, `item`, `sales` |

The raw data is not included in this repository. Download it from Kaggle and place it in the notebook's input folder.

## Approach

1. **Exploratory analysis.** Looked for trend, yearly seasonality and weekly patterns.
2. **Time-based split.** Trained on 2013 to 2016 and tested on 2017. A random split was avoided on purpose, because it would let the model see the future (data leakage).
3. **Baseline.** A seasonal-naive forecast: predict each day's sales as the sales from 364 days earlier, which lands on the same weekday.
4. **Feature engineering.** Calendar features and one lag feature (see below).
5. **Model.** XGBoost regressor (300 trees, learning rate 0.1, max depth 6).
6. **Evaluation.** MAE on the full 2017 test year, plus a check for bias and an error breakdown by month, weekday and store.

### Features

| Feature | Purpose |
|---|---|
| `store`, `item` | Each store and product has its own sales level |
| `year` | Long-term trend |
| `month`, `dayofyear` | Yearly seasonality |
| `dayofweek` | Weekly pattern |
| `lag_364` | Sales on the same day one year earlier |

**Why only a 364-day lag:** the goal is to forecast the whole of 2017 using data up to the end of 2016. Shorter lags (such as 7 or 28 days) would not be known at prediction time for most of the year, so using them would be leakage. A lag of 364 days is always available, and it keeps the weekday aligned.

## Findings from the data

- **Trend:** total sales grew every year, from about 8 million units in 2013 to about 10.7 million in 2017.
- **Yearly seasonality:** sales peak in the summer months (May to August).
- **Weekly pattern:** weekends sell more than weekdays.

## Results

Test set: all of 2017 (about 182,500 store-item-day rows). Average daily sales per store and item: 58.8 units.

| Model | MAE | Error as % of average sales |
|---|---|---|
| Seasonal naive baseline (same day last year) | 8.55 | 14.5% |
| XGBoost | **6.93** | **11.8%** |

- **No bias:** the average prediction (58.7) is almost identical to the average actual sales (58.8), so the model does not systematically under- or over-forecast.
- **What drives the forecast:** feature importance shows `lag_364` carries most of the weight (about 0.9). The calendar features and store/item add smaller refinements.
- **Where the error is highest:** July, Sundays and store 2 have the largest absolute errors, while January, Mondays and store 7 have the smallest. This follows sales volume, since busier periods have larger numbers.

## Limitations and next steps

- Most of the accuracy comes from last year's sales, so the model depends on having at least one year of history.
- Forecasting the whole year at once means short-term information (recent weeks) is not used. A shorter forecast horizon would allow shorter lags and probably lower error.
- No holidays, promotions or prices are available in this dataset. Adding them would likely help.
- Next ideas: smoothed last-year features (rolling averages around the same date), hyperparameter tuning, and comparing against LightGBM or Prophet.

## How to run

1. Download the dataset from Kaggle and open the notebook in Kaggle or locally.
2. Install the requirements:

```bash
pip install pandas numpy matplotlib scikit-learn xgboost
```

3. Run the notebook from top to bottom.

## Tools

Python, Pandas, NumPy, Matplotlib, Scikit-learn, XGBoost

## Author

Yehia El Salmy
[LinkedIn](https://linkedin.com/in/yehia-el-salmy-59b5ba307) | [GitHub](https://github.com/yehiaelsalmi)
