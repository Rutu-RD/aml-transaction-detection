# AML Transaction Detection
 
I built this to see whether a model could beat rules-based screening on money laundering detection, and by how much.
 
Short answer: at a fixed budget of 500 alerts, my model finds 227 laundering transactions where the rules baseline finds 18. That's a 12.6x improvement in investigator efficiency, measured on data the model never saw during development.
 
The longer answer, including the parts that didn't work, is below.
 
---
 
## Contents
 
1. [Why I built it this way](#1-why-i-built-it-this-way)
2. [The data](#2-the-data)
3. [Pipeline](#3-pipeline)
4. [Cleaning, and two things I nearly missed](#4-cleaning-and-two-things-i-nearly-missed)
5. [What the data actually showed](#5-what-the-data-actually-showed)
6. [Features](#6-features)
7. [How I kept leakage out](#7-how-i-kept-leakage-out)
8. [The rules baseline](#8-the-rules-baseline)
9. [Training](#9-training)
10. [Tuning](#10-tuning)
11. [Final evaluation](#11-final-evaluation)
12. [Serving](#12-serving)
13. [Where this falls short](#13-where-this-falls-short)
14. [What production would need](#14-what-production-would-need)
15. [Running it](#15-running-it)
---
 
## 1. Why I built it this way
 
I work in AML reporting, so I've seen what detection actually looks like inside a bank: rules. An investigator can read a rule, verify it by hand, and defend it to a regulator. That auditability is why rules survive despite mediocre performance.
 
The weakness is that each rule fires on its own. A rules engine can't say "high fan-in matters only when the amount is also in the structuring band." That interaction is exactly what a tree model picks up, and it's the gap I wanted to measure.
 
So I built the rules baseline first. Every model number in this README is reported as a multiple of it, because "PR-AUC 0.21" on its own tells you nothing about whether the thing is worth deploying.
 
The other constraint shaping everything: 0.11% of transactions are laundering. Roughly 1 in 900. That drives the metric choice, the threshold, the sampling, and what counts as a decent result.
 
---
 
## 2. The data
 
[IBM Transactions for Anti Money Laundering (HI-Small)](https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml). IBM simulated a world of people, companies and banks, then ran illicit funds through placement, layering and integration and labelled the results.
 
| | |
|---|---|
| Raw transactions | 5,078,345 |
| After cleaning | 5,077,237 |
| Unique accounts | 518,581 |
| Time span | 1–10 September 2022 |
| Laundering rate | 0.0891% (4,522 positives) |
| Payment formats | ACH, Cheque, Credit Card, Cash, Bitcoin, Wire, Reinvestment |
| Currencies | 15 |
 
I use three files: transactions, an accounts reference table, and a typology labels file. The typology labels I use for evaluation only, never for training. More on why in section 7.
 
---
 
## 3. Pipeline
 
```
data/raw/HI-Small_Trans.csv          5.08M rows
        │
        ▼  dataset_loader.py         clean, drop simulation tail, flag burn-in
data/interim/transaction/
        │
        ▼  dataset_loader_accounts.py  merge account attributes
data/interim/merged/
        │
        ▼  features.py               16 features, leakage-safe
data/processed/transactions_features.parquet
        │
        ▼  splitting.py              temporal 3-way split + row exclusions
data/processed/splits/{train,val,test}.parquet
        │
        ├──▶ baseline.py             7-rule incumbent, PR-AUC 0.0126
        │
        ▼  modeling/train.py         LogReg · RandomForest · XGBoost
        ▼  modeling/tune.py          GridSearchCV with TimeSeriesSplit
        ▼  modeling/evaluate.py      held-out test, model registration
        │
        ▼  api/app.py                FastAPI + Pydantic
```
 
Each stage is a DVC stage, so `dvc repro` only rebuilds what changed.
 
Stack: Python, pandas, scikit-learn, XGBoost, MLflow, DVC, FastAPI, Pydantic, pytest.
 
---
 
## 4. Cleaning, and two things I nearly missed
 
Both of these came from looking at daily counts before touching a model. Neither was visible from the schema.
 
### 4.1 The last eight days aren't real days
 
I plotted transactions per day with the laundering rate and the tail looked wrong:
 
| Date | Rows | Positives | Rate |
|---|---|---|---|
| 01 Sep | 1,114,921 | 322 | 0.03% |
| 02 Sep | 754,449 | 408 | 0.05% |
| … | … | … | … |
| 10 Sep | 208,325 | 442 | 0.21% |
| 11 Sep | 396 | 232 | 58.6% |
| 12 Sep | 281 | 170 | 60.5% |
| … | … | … | … |
| 18 Sep | 11 | 8 | 72.7% |
 
Days 11 to 18 hold 1,108 rows at 58–73% laundering, against roughly 0.1% everywhere else.
 
What's happening is that the simulator stops generating background traffic after day 10, but lets already-started laundering chains finish. So the ordinary transactions vanish and only the tail ends of chains remain. The tell isn't the high rate, it's that the volume collapses from 500,000 a day to a few hundred.
 
I dropped those rows. If I'd kept them, a temporal split would put a 58%-positive region into my test set and every metric would have looked wonderful and meant nothing. The features would also have learned "active after 11 September equals laundering", which is a calendar artifact. The chains aren't lost, since their earlier legs sit in days 1–10.
 
### 4.2 The first two days are quiet for the wrong reason
 
| Period | Rows | Positives | Rate |
|---|---|---|---|
| Days 1–2 | 1,869,370 | 730 | 0.039% |
| Days 3–10 | 3,207,867 | 3,792 | 0.118% |
 
Three times less laundering at the start. Not because those days are cleaner, but because chains take days to unfold. On day 1 only the first steps exist. By day 5, chains started on days 1 through 4 are all producing steps at once.
 
I flag these rows rather than deleting them. They're excluded as training examples, because training across two different base rates miscalibrates the model. But they stay in the frame, because computing "transactions in the prior 7 days" for a day-3 row needs days 1 and 2 to look back at.
 
This costs 730 of 2,297 training positives, which is 32% and not nothing. I implemented it as a flag so I could test the decision rather than just assert it.
 
### 4.3 Account attributes
 
I join the accounts file onto both sides of every transaction. Two attributes survive cleaning:
 
- `entity_type`, parsed from free text: `"Corporation #33520"` becomes `Corporation`
- `bank_country`, parsed from bank names: `"Portugal Bank #4507"` becomes `Portugal`
My first attempt at the country parse split on the word "Bank" and produced categories like `National` and `Savings`, because some banks are called "National Bank of Harrisburg" with no country in the name at all. I switched to an anchored regex and bucketed the unmatched ones as `Named`.
 
I drop the identifiers at source: `Entity ID`, `Account Number`, and the numeric suffix in `Entity Name`. If the simulator created laundering entities in a block, those ids encode the answer. I checked, and the suffix correlates with the label at 0.0012, so there's no actual leakage. I still drop them, because a customer id carries no signal that transfers to real data.
 
The merge has three guards on it, because a fan-out merge doesn't raise an error. It just quietly multiplies your rows:
 
1. Account keys asserted unique before merging
2. `validate="many_to_one"` on the merge
3. Row count checked afterwards
---
 
## 5. What the data actually showed
 
Everything in this section is computed on the training split only. Using the full frame would let test-set information shape my feature choices, and I'd never see the inflation it caused.
 
### 5.1 Payment format
 
| Format | Rows | Positives | Rate | Lift |
|---|---|---|---|---|
| ACH | 399,953 | 2,381 | 0.595% | 7.3x |
| Cheque | 1,243,542 | 226 | 0.018% | 0.2x |
| Credit Card | 883,291 | 139 | 0.016% | 0.2x |
| Cash | 327,864 | 74 | 0.023% | 0.3x |
| Bitcoin | 102,085 | 36 | 0.035% | 0.4x |
| Reinvestment | 481,056 | 0 | 0% | 0 |
| Wire | 116,275 | 0 | 0% | 0 |
 
ACH holds 83% of all laundering from 11% of the rows. Two formats have zero positives across 597,000 rows between them.
 
I exclude Reinvestment and Wire from training rows, since a format with no positive examples can only teach the model that the format is safe. They stay in the scoring path, because a production system has to score everything that arrives.
 
### 5.2 Amount is a band, not a size
 
Raw amount is useless across rows. The currency medians span six orders of magnitude:
 
| Currency | Rows | Median amount |
|---|---|---|
| Bitcoin | 102,067 | 0.073 |
| US Dollar | 1,325,768 | 962 |
| Yuan | 149,563 | 6,360 |
| Rupee | 133,220 | 69,166 |
| Yen | 108,672 | 101,537 |
 
To see whether there's a real amount signal underneath that, I looked at US Dollar transactions alone:
 
| Percentile | Normal | Laundering | Ratio |
|---|---|---|---|
| 25% | 153 | 1,898 | 12x |
| 50% | 960 | 4,995 | 5x |
| 75% | 6,024 | 12,817 | 2x |
| 95% | 224,277 | 27,523 | 0.12x |
 
Laundering is rarely small and rarely very large. Normal traffic has a huge mass of tiny amounts and a long tail into the millions. Laundering has neither.
 
That's structuring: big enough to be worth moving, small enough to stay under the scrutiny the largest transactions attract. It also means the relationship isn't monotonic, which is a concrete reason to prefer trees over a linear model here.
 
### 5.3 A hypothesis I tested and threw away
 
Currency conversion is a recognised layering technique, so I expected cross-currency transactions to show an elevated rate. What I found was zero laundering across 47,865 cross-currency transactions, where independence would predict about 38.
 
| Test | Statistic | p |
|---|---|---|
| Chi-square | 38.01 | 7.03e-10 |
| Fisher's exact | OR = 0.0000 | 3.47e-17 |
 
Both reject the null comfortably. I used Fisher's exact because one expected cell count is small and the chi-square approximation strains there. You can see it straining, in fact, since the two p-values differ by seven orders of magnitude.
 
I dropped the feature anyway.
 
Significance tells you the pattern isn't chance. It doesn't tell you the feature is useful. Two things bothered me. The direction is backwards from everything I know about how layering works, and a perfectly empty cell across 47,865 rows looks like a generator rule rather than a behavioural tendency. Real data would show at least some overlap.
 
A model that keeps this learns "cross-currency means clean" and would ignore exactly the transactions an investigator should be looking at. The statistics measured the effect. Domain knowledge decided what to do about it.
 
### 5.4 Time
 
The rate peaks between 11:00 and 16:00 at about 0.0013, against 0.0002 at hour 0. A six-fold swing, and a bank knows the hour at scoring time, so I use it. I encode it cyclically so 23:00 and 00:00 sit next to each other instead of 23 units apart.
 
I rejected day of week. The file covers 10 days, so each weekday appears once or twice, and the highest-rate days were also the lowest-volume days. That's a fact about which calendar dates the simulation happened to cover, not a weekly banking rhythm.
 
Hour 0 has roughly four times the volume of any other hour, which reads to me like a default timestamp for records with no precise time rather than a genuine midnight surge. I flag it separately.
 
---
 
## 6. Features
 
Sixteen features across four families. Spread is the ratio between the highest and lowest decile laundering rate.
 
| Feature | Spread | Shape | What it captures |
|---|---|---|---|
| `src_secs_since_last_log` | 60x | U-shaped | Seconds since the sender's last transaction |
| `amt_pct_ccy` | 40x | Band | Amount percentile within its currency |
| `src_dormant_wake` | 10.3x | Boolean | Silent 2.3+ days, then active |
| `dst_fanin_ratio` | 8.7x | Threshold | Distinct senders / payments received |
| `payment_format` | 7.3x | Categorical | Channel |
| `hour_sin`, `hour_cos` | 6x | Cyclical | Time of day |
| `src_fanout_ratio` | 4.2x | Threshold | Distinct receivers / payments sent |
| `src_log_dev` | 3.5x | Monotonic | Amount vs the account's own baseline |
| `src_rapid_txn` | 3.3x | Boolean | Previous transaction within 60s |
| `dst_high_fanin` | — | Boolean | Fan-in above 0.667 |
| `is_hour_zero` | — | Boolean | Timestamp likely unknown |
| `src_entity_type`, `dst_entity_type` | ~1.3x | Categorical | Corporation / Partnership / etc |
| `src_bank_country`, `dst_bank_country` | ~1.0x | Categorical | Jurisdiction |
 
### 6.1 Timing turned out to be the strongest signal
 
| Gap since last transaction | Laundering rate |
|---|---|
| Under 2 minutes | 0.0015 |
| 5–15 minutes | 0.00002 |
| Over 22 hours | 0.0029 |
 
A 60x spread, wider than anything else I found, and U-shaped. There are two different behaviours sitting at the two ends.
 
Money arriving and leaving within minutes means the account is a conduit rather than a destination. That's the layering step. A dormant account suddenly moving money is classic mule behaviour. The middle is just someone paying a few bills over an afternoon.
 
I swept the thresholds rather than picking round numbers:
 
| Rapid | Lift | | Dormant | Positives | Lift |
|---|---|---|---|---|---|
| ≤ 60s | 3.26 | | > 200,000s | 570 | 10.33 |
| < 90s | 2.24 | | > 300,000s | 378 | 20.34 |
| < 300s | 1.40 | | > 400,000s | 208 | 24.60 |
| < 600s | 0.97 | | > 500,000s | 41 | 17.52 |
 
Lift decays smoothly on both sides instead of jumping around, which told me there was real structure there rather than noise I'd fitted to. It turns over at 500,000s where only 41 positives are left.
 
I picked 200,000s over the peak at 400,000s. The peak has better lift but only holds 7% of positives, while 200,000s holds 20%. Coverage matters more to me than a headline number.
 
One thing I found while sweeping: thresholds of 10s, 30s and 60s return byte-identical results. The timestamps are minute-granular, so "within 60 seconds" really means "same or adjacent minute". Sub-minute distinctions aren't observable in this data.
 
### 6.2 Network shape
 
Laundering has a shape. A normal customer pays the same landlord and the same utilities over and over. A mule account pays fifty accounts it has never dealt with before.
 
Fan-out is one account splitting money across many, breaking a sum into pieces small enough not to attract attention. Fan-in is many accounts feeding one, putting it back together.
 
```
dst_fanin_ratio  = distinct senders so far   / payments received so far
src_fanout_ratio = distinct receivers so far / payments sent so far
```
 
Both are NaN below three prior transactions. Before I added that guard, laundering spiked at exactly 1.0, 0.5 and 0.667, and it took me a moment to realise those were 1/1, 1/2 and 2/3 from accounts with almost no history. That's newness, not fan-out behaviour. About 26% of rows are NaN as a result, which XGBoost handles natively.
 
The implementation is worth mentioning. pandas has no expanding `nunique`, and the obvious `groupby.apply` version takes minutes across 518,000 accounts. What works instead is marking the first appearance of each (sender, receiver) pair and cumsumming those markers per sender. A repeat pair contributes zero, so the running count only rises on genuinely new counterparties, and subtracting the current marker excludes the current row. That runs in about five seconds on 3.5M rows.
 
### 6.3 The feature that took three attempts
 
I wanted to know whether a transaction was unusually large for that particular account. A 50,000 transfer is routine for a business and alarming for an account that has only ever sent 500.
 
My first version divided the amount by the account's prior mean. The values ran up to 3.5e7 and the plot was unreadable.
 
Second version took the log of that ratio, which fixed the scale. But the whole distribution centred on -3 instead of 0, meaning a typical transaction looked like about 5% of the account's average. That can't be right, and the -3 was the clue.
 
The problem was the mean. One huge transaction drags an arithmetic mean upward and makes every normal transaction afterwards look tiny by comparison. Averaging the logs instead gives a geometric mean, which outliers can't dominate. For an account sending 100, 100, 100 and once 1,000,000, the arithmetic mean is 250,000 and the geometric mean is about 560.
 
Third version centres at 0 and lifted separation from 2.5x to 3.5x.
 
The general lesson I took from it: for anything multiplicative and heavy-tailed like money, work in log space for both the value and the average.
 
I also tried an absolute-value version, treating unusually small the same as unusually large. It scored 2.2x against 3.5x for the signed one. Direction carries most of the signal.
 
### 6.4 Redundancy
 
Spearman rather than Pearson, since these are skewed and several relationships are monotonic without being linear.
 
| Pair | ρ | Why |
|---|---|---|
| `src_unique_dst_prior` / `src_txn_count_prior` | 0.92 | Numerator and denominator of the fan-out ratio |
| `dst_txn_count_prior` / `dst_fanin_ratio` | -0.81 | Bigger denominator pushes the ratio down |
| `dst_txn_count_prior` / `dst_unique_src_prior` | 0.78 | Same thing, incoming side |
| `amt_pct_ccy` / `src_log_dev` | 0.76 | Both amount-based |
| `dst_fanin_ratio` / `src_fanout_ratio` | 0.72 | Highly connected accounts score high on both |
 
I dropped the four raw counts, since they're the ratio components. I kept both fan ratios despite the 0.72, because they describe opposite ends of a chain and the correlation is a shared population of busy accounts rather than the same measurement twice.
 
`src_secs_since_last_log` doesn't correlate with anything above 0.40, which fits with it being the strongest feature.
 
I didn't run VIF. It detects multicollinearity that destabilises linear model coefficients, and XGBoost has no coefficients. It handles correlated features by picking one at each split and the predictions don't suffer. Pairwise correlation was enough for what I needed.
 
### 6.5 What I threw out
 
| Feature | Why |
|---|---|
| `is_cross_currency`, `amount_delta` | 0 positives in 47,865 rows, real association but backwards direction |
| `is_self_loop` | 0 positives, excluded the rows instead |
| `amt_logz_ccy`, `log_amount` | ρ = 0.955 with `amt_pct_ccy` |
| Four raw prior-counts | ρ up to 0.92 with the ratios |
| `src_log_dev_abs` | 2.2x against 3.5x signed |
| `src_amount_vs_own_mean` | Arithmetic mean inflated by outliers |
| `src_is_new`, `dst_is_new` | Lower rate than base, which is mechanical: a first transaction has no history so it can't be part of a chain yet |
| `day_of_week` | 10-day window, 1–2 observations per weekday |
 
Everything I cut falls into one of four buckets: artifact, redundant, superseded by a better version, or mechanical. Nothing got cut just for being weak.
 
---
 
## 7. How I kept leakage out
 
Rare-event detection is easy to get wrong in ways that look like success, so this got more attention than any other part of the project.
 
**Temporal splits, never random.** Laundering is a chain across days. `train_test_split` puts earlier legs in train and later legs in test, so the model sees part of the pattern it's meant to detect. I cut on timestamp quantiles and `validate_splits()` asserts there's no time overlap between partitions.
 
**Features built on the full frame, before splitting.** History features look backwards, so a day-8 transaction needs to see that account's day-1 activity. Building them per split would leave every validation and test row with no history at all.
 
**Every history feature is prior-only.** I use two vectorised identities, `cumsum − current` for sums and first-occurrence cumsum for distinct counts. Both are fast and neither is obviously correct by reading it, so `tests/test_features.py` recomputes 30 to 40 random rows by brute force using only rows strictly before each one, and asserts they match.
 
That test paid for itself. It caught a bug where timestamps stored as `datetime64[us]` were converted with the usual `astype("int64") // 10**9`, which assumes nanoseconds and quietly produced kiloseconds. Every rolling window ended up wider than the entire dataset. The feature looked completely plausible and was completely wrong.
 
**Typology labels never touch training.** The dataset ships ground-truth pattern labels (FAN-IN, CYCLE, SCATTER-GATHER). No bank has those. Using them for sampling or features would build a model that can't exist in production, so I use them only to report per-typology recall at evaluation.
 
**The test split gets read once.** Model selection, tuning and threshold selection all run against validation. `load_split("test")` logs a warning, and I deliberately left `test.parquet` out of the training stage's DVC dependencies so it can't creep in.
 
### The splits
 
| Split | Rows | Positives | Rate | Period |
|---|---|---|---|---|
| Train | 1,109,155 | 1,563 | 0.141% | 03 Sep – 06 Sep |
| Validation | 954,954 | 1,081 | 0.113% | 06 Sep – 08 Sep |
| Test | 1,015,418 | 1,143 | 0.113% | 08 Sep – 10 Sep |
 
Exclusions (Reinvestment, Wire, self-loops, burn-in) apply to train and validation only. Test keeps everything, because production scores everything that arrives. Filtering the test set would mean measuring an easier problem than the real one.
 
---
 
## 8. The rules baseline
 
Seven rules, each one derived from something I found in the EDA. This is the incumbent the model has to beat.
 
### Each rule on its own, measured on validation
 
| Rule | Fires | % of rows | Caught | Precision | Recall | Lift |
|---|---|---|---|---|---|---|
| `dormant_reactivation` | 16,046 | 1.7% | 259 | 1.61% | 24.0% | 14.26x |
| `high_fanin` | 16,061 | 1.7% | 173 | 1.08% | 16.0% | 9.52x |
| `high_risk_format` | 106,760 | 11.2% | 923 | 0.86% | 85.4% | 7.64x |
| `structuring_band` | 340,381 | 35.6% | 853 | 0.25% | 78.9% | 2.21x |
| `rapid_passthrough` | 160,537 | 16.8% | 378 | 0.24% | 35.0% | 2.08x |
| `amount_spike` | 106,113 | 11.1% | 217 | 0.20% | 20.1% | 1.81x |
| `business_hours` | 268,180 | 28.1% | 411 | 0.15% | 38.0% | 1.35x |
 
### Combined, by how many rules match
 
| Threshold | Alerts | Alert rate | Caught | Precision | Recall | Lift |
|---|---|---|---|---|---|---|
| ≥1 rule | 653,643 | 68.5% | 1,076 | 0.16% | 99.5% | 1.45x |
| ≥2 rules | 270,337 | 28.3% | 984 | 0.36% | 91.0% | 3.22x |
| ≥3 rules | 71,574 | 7.5% | 710 | 0.99% | 65.7% | 8.76x |
| ≥4 rules | 15,849 | 1.7% | 364 | 2.30% | 33.7% | 20.29x |
| ≥5 rules | 2,598 | 0.3% | 75 | 2.89% | 6.9% | 25.50x |
 
Baseline PR-AUC comes out at 0.0126, which is 11.1x random given the 0.0011 base rate.
 
Worth noting that rules produce only 8 distinct scores, so the precision-recall curve has 8 points and everything within a score is tied. That coarseness is a real limitation of rules, and it's part of where the model's advantage comes from. Some of the improvement is continuous ranking inside groups the rules can't separate, and some is genuine interaction learning. I'd rather say that than pretend it's all the latter.
 
I also tested dropping `business_hours`, since 1.35x lift while firing on 28% of transactions looked like poor value. At comparable alert volumes the 7-rule set did equal or better, and removing it collapsed the score range from 0–7 to 0–6 and cost me operating points. So it stayed. Weak on its own, useful as a tiebreaker.
 
---
 
## 9. Training
 
Four configurations, all measured against the baseline. These are validation numbers.
 
| Model | PR-AUC | vs baseline | ROC-AUC | Precision@500 | Recall@1000 |
|---|---|---|---|---|---|
| XGBoost + class weight | 0.3251 | 25.8x | 0.9715 | 57.6% | 34.7% |
| XGBoost + SMOTE | 0.2439 | 19.4x | 0.9729 | 43.4% | 27.8% |
| Logistic Regression | 0.1510 | 12.0x | 0.9577 | 32.6% | 19.2% |
| Random Forest | 0.1491 | 11.8x | 0.9726 | 30.6% | 18.4% |
 
Three things worth pulling out of that table.
 
**SMOTE lost**, 0.2439 against 0.3251 for plain class weighting. I expected this but reported it either way. SMOTE interpolates between neighbours in feature space, and a lot of my features are ratios and boolean flags where the midpoint between two real transactions isn't a plausible transaction. I run it inside an `imblearn.Pipeline` so resampling happens during fit and never during predict, since generating synthetic positives from validation neighbours would be a silent leak.
 
**ROC-AUC shows why it's the wrong metric here.** Every model lands between 0.95 and 0.97 while PR-AUC ranges from 0.15 to 0.33. At a 0.11% positive rate, ROC-AUC is dominated by the enormous pile of easy negatives, so it can look excellent while the model is useless at any threshold you'd actually operate at.
 
**Logistic regression did better than I expected.** Every strong feature here is non-monotonic, and a linear model structurally cannot represent "both extremes are risky". Reaching 12x baseline anyway suggests the features carry enough independent signal that even a linear combination separates the classes reasonably.
 
### Checking for leakage
 
I check feature-importance concentration on every run, since one dominant feature is usually how leakage announces itself.
 
| Model | Top feature | Share | Top 3 |
|---|---|---|---|
| XGBoost + class weight | `payment_format_ACH` | 38.0% | 44.9% |
| Random Forest | `payment_format_ACH` | 27.8% | 46.2% |
| XGBoost + SMOTE | `payment_format_ACH` | 24.7% | 41.6% |
 
Top features for the winning model:
 
| Feature | Importance |
|---|---|
| `payment_format_ACH` | 0.3796 |
| `payment_format_Credit Card` | 0.0365 |
| `src_bank_country_Named` | 0.0331 |
| `payment_format_Cheque` | 0.0289 |
| `src_secs_since_last_log` | 0.0267 |
| `src_fanout_ratio` | 0.0194 |
| `dst_fanin_ratio` | 0.0182 |
| `dst_bank_country_Saudi Arabia` | 0.0177 |
| `payment_format_Bitcoin` | 0.0175 |
| `amt_pct_ccy` | 0.0168 |
| `src_rapid_txn` | 0.0159 |
| `src_entity_type_Corporation` | 0.0137 |
 
ACH dominating made me look twice, but it has an explanation I can defend: it carries 87% of positives across all three splits. The same feature tops all three tree models, so it's stable rather than an artifact of one fit, and the rest of the importance spreads out widely. I concluded it's real signal.
 
---
 
## 10. Tuning
 
`GridSearchCV` with `TimeSeriesSplit` rather than `KFold`. This is the detail that would have quietly wrecked everything: KFold shuffles rows into folds, which splits a laundering chain across train and validation folds and lets backward-looking features see rows from later folds.
 
```
fold 1:  train [-------]  val [--]
fold 2:  train [---------]  val [--]
fold 3:  train [-----------]  val [--]
```
 
I score on `average_precision`, which is PR-AUC and the metric I'm reporting. Grids live in `params.yaml` so DVC tracks them and changing one re-runs the stage.
 
| Model | CV PR-AUC | Val PR-AUC | CV–val gap | Best params |
|---|---|---|---|---|
| cv_xgboost | 0.3599 ± 0.0010 | 0.3646 | -0.0047 | `lr=0.1, max_depth=8, min_child_weight=10` |
| cv_logreg | 0.1610 ± 0.0187 | 0.1511 | +0.0099 | `C=0.01` |
 
I report two numbers because they answer different questions. The CV score picks the hyperparameters. The validation score, which the search never saw, is what I report. If those two diverged much I'd know the grid had been fitted to the fold structure rather than to anything generalisable. A gap of -0.005 says the folds are representative.
 
Tuning took XGBoost from 0.3251 to 0.3646, about a 12% relative gain.
 
---
 
## 11. Final evaluation
 
Run once, on the held-out test split.
 
### The headline
 
| Metric | Value |
|---|---|
| PR-AUC | 0.2129 |
| vs rules baseline | 16.9x |
| vs random | 189.2x |
| ROC-AUC | 0.9574 |
| Brier score | 0.00267 |
| Positives | 1,143 of 1,015,418 |
 
### Against the baseline at the same alert volume
 
This is the comparison that would matter to an AML team. Investigator capacity is fixed, so the question isn't what the AUC is, it's how much more laundering you surface for the same amount of review work.
 
| Alerts | Model caught | Rules caught | Model precision | Rules precision | Improvement |
|---|---|---|---|---|---|
| 500 | 227 | 18 | 45.4% | 3.6% | 12.6x |
| 1,000 | 313 | 35 | 31.3% | 3.5% | 8.9x |
| 5,000 | 510 | 113 | 10.2% | 2.3% | 4.5x |
| 10,000 | 623 | 191 | 6.2% | 1.9% | 3.3x |
 
### At the operating threshold
 
I picked the threshold (0.0877) on validation to hit 60% recall, then applied it unchanged to test. Choosing it on test would have inflated precision by fitting the evaluation set, which is the same mistake as tuning on test, just smaller and easier to miss.
 
| | |
|---|---|
| Alerts | 9,877 (0.97% of transactions) |
| True positives | 622 |
| False positives | 9,255 |
| False negatives | 521 |
| Precision | 6.30% |
| Recall | 54.4% |
| Alerts per genuine case | 15.9 |
 
It's deliberately not 0.5. A missed report is a regulatory failure and a false positive costs review time, so the threshold leans toward recall.
 
### Recall by payment format, which is the table I'd want someone to read
 
| Format | Rows | Positives | Caught | Alerts | Recall | Precision |
|---|---|---|---|---|---|---|
| ACH | 138,205 | 999 | 621 | 9,238 | 62.2% | 6.7% |
| Cheque | 411,720 | 71 | 1 | 432 | 1.4% | 0.2% |
| Credit Card | 291,707 | 38 | 0 | 25 | 0% | 0% |
| Cash | 108,697 | 22 | 0 | 53 | 0% | 0% |
| Bitcoin | 28,804 | 13 | 0 | 124 | 0% | 0% |
| Wire | 36,285 | 0 | 0 | 5 | — | 0% |
 
What I've built is an ACH detector. The aggregate 54% recall hides the fact that it catches essentially nothing on four of six channels, and anyone deploying this would need to know which typologies it's blind to.
 
The cause is data quantity rather than modelling. Cheque has 71 positives across 412,000 rows. Credit Card has 38. Cash has 22. There's very little there to learn from, and no amount of feature work fixes that.
 
### Recall by entity type
 
| Entity type | Positives | Caught | Recall | Precision |
|---|---|---|---|---|
| Partnership | 394 | 241 | 61.2% | 6.9% |
| Corporation | 426 | 222 | 52.1% | 5.1% |
| Sole Proprietorship | 316 | 159 | 50.3% | 8.1% |
 
Reasonably even, so no single entity type is disproportionately unmonitored.
 
### The drop from validation to test
 
Validation PR-AUC 0.3646, test 0.2129. A drop of 0.152, which is large enough that I went looking for the cause.
 
It isn't distribution drift. ACH share of positives is stable across the three splits at 88.3%, 85.4% and 87.4%, and base rates are 0.141%, 0.113% and 0.113%. What it looks like is the tuned hyperparameters, `max_depth=8` in particular, fitting the validation period specifically. The tuned figure was optimistic.
 
I'm reporting both numbers because that's the whole point of holding a test set back. A project that only reports 0.3646 hasn't measured anything.
 
### Calibration
 
| Score decile | Mean predicted | Observed rate | Gap |
|---|---|---|---|
| 0–8 | ~0.0000 | ~0.0000 | ~0 |
| 9 (highest) | 0.0447 | 0.0099 | +0.0349 |
 
The scores rank correctly but they aren't calibrated. Class weighting with `scale_pos_weight` around 709 inflates predicted probabilities by design. That's fine for ordering an alert queue, which is all I'm doing with them, but it would be wrong if the scores fed into a downstream risk calculation. Platt scaling or isotonic regression on a held-out slice would fix it.
 
---
 
## 12. Serving
 
FastAPI with Pydantic validation.
 
| Method | Path | Purpose |
|---|---|---|
| GET | `/` | Landing page with metrics, threshold rationale, limitations |
| GET | `/health` | Liveness and model state |
| GET | `/model/info` | Feature list, threshold, test metrics, limitations |
| POST | `/predict` | Score one transaction |
| POST | `/predict/batch` | Score up to 10,000 |
| GET | `/docs` | Interactive OpenAPI schema |
 
The request body is the engineered feature vector, not a raw transaction, and that's the decision I'd most expect to be asked about.
 
Several of my features are account-level aggregates built from history: seconds since the account's last transaction, distinct counterparties so far, deviation from that account's own typical amount. A single incoming transaction can't reconstruct any of that from its own payload, because it depends on everything the account did beforehand. In production those values would come from a feature store maintained incrementally as transactions land. This API sits downstream of that store rather than pretending to replace it.
 
Three things guard against train-serve skew:
 
Preprocessing lives inside the sklearn Pipeline, so the fitted transformations get pickled with the estimator and reapplied at inference instead of recomputed on whatever arrives.
 
The model is logged with an MLflow signature inferred from training data, 16 inputs with dtypes, so a malformed request gets rejected rather than silently mis-scored.
 
`TransactionFeatures.model_fields` is checked against `FEATURE_COLUMNS` at startup, and logs an error if the schema and the training features have drifted apart.
 
I set `extra="forbid"` on the Pydantic model, so an unexpected field returns a 422 rather than being quietly ignored. If the caller and the feature store disagree about the schema, I want that to surface.
 
Nulls are allowed only on `dst_fanin_ratio`, `src_fanout_ratio` and `src_log_dev`, where they genuinely mean "this account has fewer than three prior transactions". Filling those with a number would assert something false about a new account.
 
---
 
## 13. Where this falls short
 
**Simulator artifacts.** Zero laundering on Wire, Reinvestment and cross-currency transactions are properties of the generator, not of banking. All three are real laundering channels. Any model trained here inherits those blind spots.
 
**It only really covers one channel.** 62% recall on ACH and close to nothing elsewhere.
 
**Geographic features carry signal I wouldn't trust.** `dst_bank_country_Saudi Arabia` sits in the top ten by importance, and Saudi Arabia shows 4.5x the base rate on 17,000 rows. That's the generator's design showing through, not observed banking behaviour. Scoring transactions higher because of counterparty jurisdiction raises fair-treatment questions and would need explicit review against real typology evidence before going anywhere near production.
 
**Window dependence.** Ten days of data. My dormancy threshold of 200,000 seconds is about 23% of the whole observation window, so it would need refitting on anything longer, even if the underlying signal holds.
 
**Timestamp resolution.** Minute-granular, so sub-minute distinctions don't exist.
 
**Chain truncation.** Chains starting near day 10 get cut off by the file boundary, so some of my positives are partial patterns. I'm training mostly on completed chains, while production would see chains still in progress, and recall on in-flight laundering would likely be worse.
 
**Calibration.** Scores rank but overstate probability.
 
**Transaction level only.** Each transaction is scored on its own. Nothing aggregates risk per customer and nothing links connected accounts into a single case.
 
---
 
## 14. What production would need
 
**A feature store.** Account aggregates maintained incrementally and read at inference, since recomputing over five million rows per request isn't feasible. The offline definitions in `features.py` are the spec that the online store has to reproduce, and keeping the two consistent is the main skew risk in this design.
 
**Entity-level aggregation.** Rolling scores up per customer, because an account with forty medium-risk transactions may well be more suspicious than one with a single high-risk transaction. Sum of excess over a rolling window, normalised by expected volume for the customer segment, is a reasonable starting point. A plain mean is not, since structuring is specifically designed to hide in volume.
 
**A network layer.** Linking accounts that transact with each other so an investigator opens one case covering a cluster instead of forty separate alerts. That's where the remaining signal in this problem lives, and it's one of the few places graph methods genuinely beat gradient boosting.
 
**Drift monitoring.** Typologies change. A 0.15 gap between validation and test on ten days of data suggests this would need retraining fairly regularly.
 
**Model risk documentation.** Feature justifications, rejected alternatives, fair-lending analysis, challenger comparison, monitoring plan. Most of that is what this README already is.
 
---
 
## 15. Running it
 
```bash
git clone https://github.com/Rutu-RD/aml-transaction-detection
cd aml-transaction-detection
 
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e . && pip install -r requirements.txt
```
 
Download `HI-Small_Trans.csv` and `HI-Small_accounts.csv` from [Kaggle](https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml) into `data/raw/`, then:
 
```bash
dvc repro                 # clean -> merge -> features -> split -> train -> tune
pytest tests/ -v          # leakage regression tests
mlflow ui                 # experiment tracking
uvicorn aml_detection.api.app:app --reload
```
 
Individual stages:
 
```bash
python -m aml_detection.dataset_loader
python -m aml_detection.dataset_loader_accounts
python -m aml_detection.features
python -m aml_detection.splitting
python -m aml_detection.modeling.train
python -m aml_detection.modeling.tune
python -m aml_detection.modeling.evaluate     # reads the test split, run once
```
 
### Layout
 
```
aml_detection/
├── config.py                   paths
├── dataset_loader.py           load, clean, tail removal, burn-in flag
├── dataset_loader_accounts.py  account attribute merge
├── eda.py                      reusable analysis helpers
├── features.py                 16 features plus why the others were cut
├── splitting.py                temporal 3-way split
├── baseline.py                 7-rule incumbent
├── modeling/
│   ├── train.py                model comparison
│   ├── tune.py                 TimeSeriesSplit grid search
│   └── evaluate.py             held-out test, registration
└── api/app.py                  FastAPI service
 
notebooks/
├── 01_eda.ipynb                exploratory analysis
├── 02_baseline.ipynb           rules evaluation
└── 03_model_review.ipynb       importance and diagnostics
 
tests/test_features.py          leakage regression tests
params.yaml                     tracked hyperparameters
dvc.yaml                        pipeline definition
```