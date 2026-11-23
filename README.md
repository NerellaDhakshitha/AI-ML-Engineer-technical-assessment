## About the Project

This project is part of an AI/ML Engineer technical assessment. The task was to review an existing machine learning model that predicts whether medical insurance claims might get denied.

The main goal is to identify high-risk claims before they are submitted so the billing team can check and correct possible issues.

The billing team can review a maximum of 150 claims per day.

For this assessment, I ran the given code, understood how the model works, identified the problems, and created a corrected version.

## 1. Original Model Results

After running `claim_denial_model.py`, I got the following results:

| Metric | Result |
|---|---|
| Closed Claims | 10,000 |
| Pending Claims | 500 |
| Denial Rate | 11.5% |
| ROC AUC | 0.917 |
| Threshold | 0.90 |
| Precision | 0.928 |
| Recall | 0.928 |
| Claims Held | 41 out of 500 |

The results looked good initially, especially the ROC AUC of 0.917. However, after reviewing the code, I found some issues that make these results unreliable.

## 2. Understanding the Original Code

### Data Generation

The `load_claims()` function creates 10,000 historical claims and 500 pending claims.

The dataset includes patient details, service types, authorization status, billed amounts, submission delays, and claim outcomes.

The denial probability mainly depends on missing authorization, late filing, and patient risk.

### Feature Engineering

The `add_features()` function creates three features:

- `late_filing` – Checks whether a claim was submitted after 90 days.
- `billed_amount_k` – Converts the billed amount into thousands.
- `patient_denial_rate` – Calculates the average denial rate for each patient.

Missing values are replaced with zero.

### Data Balancing

The `balance()` function duplicates denied claims to make the number of denied and paid claims equal.

### Model Training

The code divides the data into 80% training and 20% testing.

It uses Logistic Regression to predict claim denials based on six features.

### Model Evaluation

The model calculates ROC AUC, precision, and recall. It also tries different thresholds to select the one with the best F1 score.

### Pending Claim Predictions

Finally, the model predicts which pending claims should be held for review.

The original model holds 41 out of 500 claims.

## 3. Issues Found in the Code

I grouped the issues based on their impact on the model.

### Critical Issues

#### 1. Data Leakage from `days_to_payment`

The model uses `days_to_payment` as a feature.

But this information is only available after the insurance company processes the claim.

Since the model is supposed to predict denials before submission, this feature should not be used.

It also contains information related to the actual denial outcome, which can make the model perform better during testing than it would in real life.

**Fix:** Remove `days_to_payment` from the input features.

#### 2. Data Leakage from `patient_denial_rate`

The code calculates each patient's denial rate using the entire dataset before splitting it into training and testing data.

This means the calculation can include the current claim's result and future claim outcomes.

The model is indirectly getting information it should not know.

**Fix:** Calculate denial history using only previously resolved claims, or remove the feature until historical data can be handled correctly.

#### 3. Balancing Before Splitting the Data

The code duplicates denied claims before creating the training and testing sets.

Because of this, the same claim can appear in both sets.

This can make the evaluation results look better than they actually are.

The test dataset also has an artificial denial rate of around 50%, while the original denial rate is only 11.5%.

**Fix:** Split the original dataset first. If balancing is needed, apply it only to the training data.

#### 4. Daily Review Limit Is Not Followed

The billing team can review only 150 claims per day.

However, the original code does not enforce this limit.

If the model holds more than 150 claims on a particular day, the billing team may not be able to review all of them.

**Fix:** Rank claims based on their predicted denial probability and select a maximum of 150 claims per processing day.

### High-Priority Issues

#### 5. Incorrect Data Sorting

The original code uses:

```python
data.sort_values("service_date")
```

But it does not save the sorted result.

So the model is not actually testing on the newest 20% of claims, even though the output says so.

**Fix:**

```python
data = data.sort_values("service_date")
```

Then divide the data into training, validation, and testing sets based on dates.

#### 6. Incorrect Recall Calculation

The original code calculates recall as:

```python
recall = tp / max(tp + fp, 1)
```

This is actually the formula for precision.

The correct formula is:

```python
recall = tp / max(tp + fn, 1)
```

Because of this mistake, the original model reports the same value for precision and recall.

**Fix:** Use Scikit-learn's built-in precision, recall, and F1 score functions.

#### 7. Threshold Is Not Used Correctly

The model selects 0.90 as the best threshold during evaluation.

But when predicting pending claims, it uses `model.predict()`, which applies the default threshold of 0.5.

Also, the threshold is selected using the test data itself, which makes the final evaluation less reliable.

**Fix:** Select the threshold using validation data and apply the chosen decision rule to the predicted probabilities.

#### 8. Submission Delay May Not Be Available

The model uses `days_to_submit` to identify late filing.

However, pending claims have not been submitted yet, so the actual submission delay may not be known.

**Fix:** Calculate the number of days from the service date to the current processing date, or use a confirmed planned submission date.

### Other Issues

#### 9. Missing Values

The code replaces all missing feature values with zero.

But missing information does not always mean zero, so this can affect predictions.

**Fix:** Handle missing values based on the meaning of each feature and use training data to determine appropriate replacements.

#### 10. Limited Model Evaluation

The original model mainly checks ROC AUC, precision, and recall.

For this business problem, I would also evaluate:

- Precision and recall within the 150-claim review limit
- How many actual denials are identified
- Whether predicted probabilities are reliable
- Cost of manual review compared with potential savings

#### 11. Synthetic Dataset

The dataset is generated using Python and does not represent all the factors involved in real medical billing.

Before deployment, the model should be tested using real historical claims.

## 4. Changes Made in the Corrected Model

I created `claim_denial_model_fixed.py` to address the main issues.

The changes include:

1. Removed `days_to_payment` and `patient_denial_rate` to avoid data leakage.
2. Used chronological training, validation, and testing splits.
3. Kept the original denial distribution instead of balancing the entire dataset.
4. Corrected the precision and recall calculations.
5. Used predicted probabilities for identifying high-risk claims.
6. Added a daily review limit of 150 claims.

The corrected version is only a demonstration using synthetic data.

It still assumes that `days_to_submit` is known when scoring claims. In a real system, this should be replaced with the actual filing age at the time of prediction.

Also, the dataset does not contain a separate processing date for pending claims, so the corrected example uses `service_date` to demonstrate daily grouping.

## 5. How to Run the Project

### Requirements

- Python 3.10+
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

Both Python files should be in the same folder because the corrected script uses the original data generator.

## 6. Is the Model Ready for Production?

**My decision: No, the current model is not ready for production.**

Even though the original model reports a ROC AUC of 0.917, the results are affected by data leakage and problems in the evaluation process.

The model also does not properly follow the daily review capacity.

Before deploying it, I would recommend:

- Removing data leakage
- Testing on properly separated historical data
- Using correct evaluation metrics
- Following the daily review limit
- Testing the model with real insurance claims
- Monitoring predictions and actual denial outcomes

I would also suggest running the model alongside the existing billing process before allowing it to make automatic hold decisions.

## 7. Conclusion

After reviewing the code, I understood that a high ROC AUC score does not always mean a model is reliable.

The way data is prepared, the features used for training, and the evaluation process are equally important.

For this problem, the model should not only predict claim denials but also help the billing team identify the most important claims within their daily review capacity.

The original model is a good starting point, but the identified issues need to be fixed before it can be used in production.

The corrected version addresses the main problems and provides a better starting point for further testing.

---

