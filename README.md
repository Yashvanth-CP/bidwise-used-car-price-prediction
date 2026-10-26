# BidWise: Used-Car Price Prediction with Prediction Intervals

Predicts Indian used-car prices and returns a 90% price range, not just one number.

## Results (held-out test set, 11,475 cars)
- Typical error: about 17% (test MAE 0.1572 on the log scale)
- Median error: ₹52,206
- 90% prediction intervals covered 89.8% of unseen cars
- Coverage stayed between 88.7% and 90.5% in every price group

![Predicted vs actual](reports/figures/pred_vs_actual.png)
![Interval examples](reports/figures/interval_examples.png)
![Error by group](reports/figures/error_by_group.png)
![Model comparison](reports/figures/model_comparison.png)

## Model comparison (5-fold CV on the training set, MAE on log price)
| Model | MAE |
|---|---|
| Median baseline | about 0.70 (single split) |
| Ridge | 0.1621 ± 0.0016 |
| HistGB (tuned) | 0.1504 ± 0.0015 |
| Random Forest | 0.1493 ± 0.0016 |

Random Forest and tuned HistGB are statistically tied. I chose Random Forest because it is simpler and needed no tuning (a tuning grid showed a plateau).

## Method
1. Cleaning with fixed rules only (safe before the split)
2. Train/test split first; all learned steps (imputing, scaling, encoding) are inside a scikit-learn Pipeline
3. Target is log1p(price), so errors are percentages
4. Split conformal prediction, calibrated separately for 5 predicted-price groups

## Data
Dataset: Shashi (2025), *Indian Used Car Listings Dataset (57,480 Records)*, Kaggle: <paste link>.
Place `indian_used_cars_raw_57480.csv` in `data/`. I used the raw file, because the numeric file encodes brand using average price, which leaks the target.

Issues found and fixed: 28 missing fuel types, 77 impossible mileage values (over 100,000 km/year), duplicate brand spellings, a fake body type "15" (relabelled "Unknown"), and repeated model name formats.

## Limitations
- Rare luxury brands (fewer than 10 listings, like Rolls-Royce) are predicted poorly
- Some listing prices look like data entry errors, which limits accuracy for cheap cars
- Intervals are widest for the cheapest cars (about ±68%)
- `city` added no accuracy and was dropped

## How to run
pip install -r requirements.txt, then run 02_clean.py, 03_baseline.py, 04_compare.py, 05_intervals.py, 06_final_test.py in that order.