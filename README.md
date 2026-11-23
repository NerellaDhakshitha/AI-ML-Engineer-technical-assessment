
# Claim Denial Risk Model – Code Review

## About This Assessment

I reviewed the given Python model that predicts whether a medical insurance claim may be denied. The idea is to catch problems before submitting claims, while keeping the billing team's workload within 150 reviews per day.

I ran the original script, checked how it prepares data and evaluates predictions, and made a corrected example to address the biggest issues.

---

## 1. Original Code and Results

The original `claim_denial_model.py` generated 10,000 closed claims and 500 pending claims.

Here is what it printed:

| Measure | Original Result |
|---|---|
| Closed claims | 10,000 |
| Pending claims | 500 |
| Historical denial rate | 11.5% |
| Test ROC AUC | 0.917 |
| Selected threshold | 0.90 |
| Reported precision | 0.928 |
| Reported recall | 0.928 |
| Pending claims held | 41 of 500 |

These numbers look promising, but I would not rely on them for deployment because of the problems described below.

### How the Script Works

1. **`load_claims()`** creates sample historical and pending claims. Missing authorization, late filing, and patient risk influence the generated denial outcomes.

2. **`add_features()`** creates a late-filing flag, scales the billed amount, and calculates each patient's denial rate. It fills missing features with zero.

3. **`balance()`** duplicates denied claims to match the number of paid claims.

4. The script splits the data 80/20 and trains a Logistic Regression model.

5. It calculates test metrics, searches for a threshold, and predicts which pending claims to hold.

---

## 2. Main Problems I Found

### Critical Issues

#### 1. Payment Information Leaks the Outcome

The model uses `days_to_payment` as an input feature.

This information is only available after the insurance company processes the claim. In the sample data, it is also generated based on whether the claim was denied.

Since the model needs to predict denials before submission, this feature should not be used.

**Fix:** Remove `days_to_payment` from the model features.

#### 2. Patient Denial History Uses Future Information

The `patient_denial_rate` feature is calculated using all claims before splitting the data.

This means it can include the current claim's result and future claim outcomes.

As a result, the model gets information that would not be available when making a prediction.

**Fix:** Calculate patient denial history using only claims already resolved at prediction time, or remove the feature.

#### 3. Balancing Happens Before Splitting

The code duplicates denied claims before dividing the dataset into training and testing sets.

Because of this, the same claim can appear in both sets.

The test set also no longer reflects the actual denial rate of around 11.5%.

This makes the evaluation results unreliable.

**Fix:** Split the original claims first. If balancing is needed, apply it only to the training data.

#### 4. Daily Review Limit Is Not Enforced

The billing team can review a maximum of 150 claims per day.

However, the original code does not enforce this limit.

If the model selects more than 150 claims on a particular day, the billing team may not be able to review all of them.

**Fix:** Rank claims by predicted risk for each processing day and hold no more than 150.

### Other Important Issues

#### 5. The Date Sorting Does Not Work

The original code uses:

```python
data.sort_values("service_date")
```

However, the sorted result is not assigned back to `data`.

So the test set is not actually the newest 20% of claims, even though the output says so.

**Fix:** Save the sorted result and use chronological training, validation, and testing sets.

```python
data = data.sort_values("service_date")
```

#### 6. Recall Is Calculated Incorrectly

The original code calculates recall using:

```python
recall = tp / max(tp + fp, 1)
```

This is actually the formula for precision.

The correct recall formula is:

```python
recall = tp / max(tp + fn, 1)
```

Because of this mistake, the original code reports the same value for precision and recall.

**Fix:** Use Scikit-learn's built-in precision, recall, and F1 score functions.

#### 7. The Selected Threshold Is Not Used

The model selects a threshold of 0.90 during evaluation.

However, when predicting pending claims, it uses `model.predict()`, which applies the default threshold of 0.5.

The threshold is also selected using the test data, making the reported evaluation less reliable.

**Fix:** Choose the decision rule using validation data and apply it to predicted probabilities.

#### 8. Some Inputs May Not Be Available Before Submission

The model uses `days_to_submit` to identify late filing.

However, pending claims have not been submitted yet, so their actual submission delay may not be known.

**Fix:** Calculate filing age using the date the claim is being reviewed, or use a confirmed planned submission date.

#### 9. Missing Values Need Better Handling

The original code replaces all missing feature values with zero.

But missing information does not always mean zero.

This can affect the model's predictions.

**Fix:** Handle missing values based on what each feature represents.

#### 10. Model Evaluation Needs Improvement

The original model mainly reports ROC AUC, precision, and recall.

For this business problem, I would also check:

- How many actual denials are identified within the 150-claim review limit.
- How many held claims are actually denied.
- Whether the predicted probabilities are reliable.
- Whether reviewing claims saves more money than it costs.

The dataset is also synthetic, so real historical claims would be needed before deployment.

---

## 3. Corrected Code and Results

I created `claim_denial_model_fixed.py` as a demonstration to address the main issues.

### Changes Made

1. Removed `days_to_payment` and `patient_denial_rate` to prevent the main data leakage issues.
2. Used chronological training, validation, and testing splits.
3. Divided the historical data into 60% training, 20% validation, and 20% testing.
4. Kept the original denial distribution instead of balancing the entire dataset.
5. Corrected the precision and recall calculations.
6. Used predicted probabilities to identify high-risk claims.
7. Added a daily review limit of 150 claims.

### Corrected Model Results

I ran the corrected script using the original synthetic data generator.

The model selected a threshold of **0.23** using the validation set.

| Measure | Validation | Untouched Test |
|---|---|---|
| Claims | 2,000 | 2,000 |
| Denial rate | 11.4% | 12.0% |
| ROC AUC | 0.570 | 0.602 |
| Average Precision (AP) | 0.197 | 0.255 |
| Precision | 0.325 | 0.450 |
| Recall | 0.220 | 0.224 |
| Brier score | 0.098 | 0.100 |

### Original vs. Corrected Model

| Measure | Original Model | Corrected Model |
|---|---|---|
| ROC AUC | 0.917 | 0.602 |
| Precision | 0.928 | 0.450 |
| Recall | 0.928 (incorrect formula) | 0.224 |
| Selected threshold | 0.90 | 0.23 |
| Pending claims held | 41 | 27 |

The corrected scores are lower because the model no longer uses the main leaked information, and the test data is no longer mixed with duplicated training examples.

This gives a more useful evaluation of the model.

However, the corrected model still needs further testing before it can be used in production.

### Pending Claims Held by Day

The corrected script ranks pending claims by predicted risk and applies the daily review limit.

| Date Used for Demonstration | Pending Claims | Held for Review |
|---|---|---|
| 2025-07-01 | 111 | 5 |
| 2025-07-02 | 84 | 3 |
| 2025-07-03 | 98 | 6 |
| 2025-07-04 | 101 | 8 |
| 2025-07-05 | 106 | 5 |
| **Total** | **500** | **27** |

No group exceeds the daily limit of 150 claims.

The script also saves `pending_claim_decisions.csv`, which contains claim IDs, predicted probabilities, and hold decisions.

### Limitations of the Corrected Version

The corrected code is a demonstration using synthetic data, not a production-ready model.

It still uses `days_to_submit` as if it were known when making predictions.

In a real system, filing age should be calculated using the actual date when the claim is being reviewed.

Also, the sample dataset does not contain a separate processing date for pending claims.

Because of this, the corrected code uses `service_date` to demonstrate daily grouping.

These assumptions would need to be changed before real deployment.

---

## 4. How to Run the Project

### Requirements

- Python 3.10 or newer
- NumPy
- Pandas
- Scikit-learn

### Install Dependencies

```bash
pip install numpy pandas scikit-learn
```

### Run the Original Model

```bash
python claim_denial_model.py
```

### Run the Corrected Model

```bash
python claim_denial_model_fixed.py
```

Both Python files should be in the same folder because the corrected script imports `load_claims()` from the original script.

### Project Files

```text
claim-denial-risk-model-review/
│
├── README.md
├── claim_denial_model.py
├── claim_denial_model_fixed.py
└── pending_claim_decisions.csv
```

---

## 5. My Recommendation

**I would not recommend deploying the original model in the next sprint.**

Even though it reports a high ROC AUC of 0.917, the results are affected by data leakage and testing mistakes.

The original model also does not properly enforce the billing team's daily review limit.

Before deployment, I would recommend:

1. Testing the model using real historical claims.
2. Making sure every feature is available at the time of prediction.
3. Using properly separated training, validation, and testing data.
4. Comparing the model against simple authorization and late-filing rules.
5. Measuring how many denials can be identified within 150 reviews per day.
6. Checking whether the model actually helps reduce billing delays and manual work.

I would also suggest running the model alongside the existing billing process before allowing it to hold claims automatically.

This would help us understand how it performs on real pending claims without affecting the normal workflow.

---

## 6. Conclusion

After reviewing the code, I understood that a high ROC AUC score does not always mean a machine learning model is reliable.

The way data is prepared, the features used for training, and the evaluation process are equally important.

For this problem, the model should not only predict claim denials but also help the billing team identify the most important claims within their daily review capacity.

The original model is a good starting point, but the identified issues need to be fixed before it can be used in production.

The corrected version addresses the main problems and provides a better starting point for further testing.

---

