# AI-ML-Engineer-technical-assessment

1. Execution and overview
I ran the supplied script with Python. Its output was:
Reported measure   |   Value
Closed / pending claims   |   10,000 / 500
Historical denial prevalence   |   11.5%
Test ROC AUC   |   0.917
Selected threshold   |   0.90
Reported precision / recall   |   0.928 / 0.928
Pending claims held   |   41 / 500
These figures are the script's output, not trustworthy evidence of deployable performance.
Walkthrough
• 1. `load_claims()` synthesizes 10,000 closed and 500 pending claims, patient identifiers, services, authorizations, submission delays and charges. A latent patient-risk variable, missing authorization for services that require it, and filing after 90 days determine denial probabilities. The generator then samples denials and generates `days_to_payment` conditional on the outcome. Pending labels and payment delays are hidden.
• 2. `add_features()` derives a late-filing flag and amount in thousands, computes each patient's average denial rate from the *entire* dataframe, and replaces missing feature values with zero.
• 3. `balance()` retains all paid claims and oversamples denied claims with replacement to match the paid count.
• 4. The program calls `sort_values('service_date')` without assigning its result; it then splits the oversampled, shuffled data by row position, 80%/20%.
• 5. Logistic regression learns coefficients for six variables. It can learn real signal from authorization and filing, but it also learns from post-outcome payment delay and target-derived patient denial history. Those are not valid submission-time predictors.
• 6. It selects the F1-maximizing threshold on the same test set used for final reporting. The custom recall function mistakenly computes precision again.
• 7. It scores pending claims using `model.predict()` (default 0.5 cutoff), ignores the selected threshold, and reports total holds without daily capacity control.
2. Findings ranked by severity
Priority   |   Finding and code location   |   Why it matters / remedy
P0 — Critical   |   Outcome leakage: `days_to_payment` is a feature (lines 9–16, 43–44).   |   Only known after payer response; it was explicitly generated from denial status. Closed-claim performance is inflated, while pending values are imputed to zero. Remove entirely.
P0 — Critical   |   Target leakage: `patient_denial_rate` is calculated across all rows before splitting (line 51).   |   It includes the current claim's label and future/test labels, including a patient's own target. This makes the strong coefficient (+10.83) misleading. Use strictly prior, already-resolved claims with time-aware history, or omit until available.
P0 — Critical   |   Contaminated evaluation: oversampling occurs before splitting (lines 56–60, 79–82).   |   Duplicated denied claims can appear in both sets; the test set has artificially ~50% denials rather than ~11.5%. ROC AUC and threshold metrics are unrepresentative; precision and calibration are especially distorted. Split chronologically first; resample training only if justified; keep validation/test natural.
P0 — Critical   |   Operational policy does not enforce 150 reviews/day (line 106).   |   A production batch could exceed staffing capacity. Rank eligible pending claims by risk separately for each day and hold at most 150/day.
P1 — High   |   Temporal split is not implemented (line 80).   |   `sort_values` does not modify `data` without assignment or `inplace=True`. The printed “newest 20%” is false. Sort before splitting and use disjoint train, validation, test time windows.
P1 — High   |   Recall formula is wrong (lines 63–69).   |   It divides TP by TP+FP rather than TP+FN; reported recall always equals precision, corrupting F1 and threshold selection. Use `sklearn.metrics.precision_score`, `recall_score`, `f1_score`.
P1 — High   |   Threshold chosen on test and not used in production (lines 87–106).   |   The reported test results are optimistically tuned, and `model.predict()` silently applies 0.5 instead of 0.90. Tune on validation; lock policy; evaluate once on untouched test; use `predict_proba` in deployment.
P1 — High   |   Potential submission-time availability issue: `days_to_submit` / `late_filing` (lines 31, 49).   |   Pending claims are described as *not yet sent*. Actual days until submission are not yet known. Use actual elapsed days as of scoring, or a verified planned submission date; do not use eventual actual submission delay.
P2 — Medium   |   Missingness is silently filled with zero (line 52).   |   Zero payment delay or zero historical denial rate is not necessarily a real observation; missingness is systematic between closed/pending groups. Remove unavailable variables; impute genuine missing values from training only, with indicators where appropriate.
P2 — Medium   |   No calibration, baseline, or business-impact assessment.   |   AUC alone does not establish that probabilities are meaningful or that review work is worthwhile. Report PR AUC, precision@capacity, recall@capacity, calibration/Brier score, number held per day and estimated avoidable denials.
P2 — Medium   |   Synthetic data limitations and deployment gaps.   |   Generated labels omit real payer/provider/coding effects and distribution drift. There is no model persistence, schema validation, monitoring, or audit trail. Validate on representative real historical claims and implement operational controls.
3. Recommended corrected approach
Data contract: At the decision instant, only use fields already known to the billing team. `needs_auth`, `prior_auth`, `service_type`, and `billed_amount` are plausible; filing age must be calculated from the decision date, not the eventual submission date. Exclude `days_to_payment` and full-dataset patient denial rate. The attached corrected example retains `late_filing` only as a synthetic proxy, assuming `days_to_submit` is known at the time of scoring. In real deployment this assumption must be replaced with an as-of-date calculation.
Evaluation: Split *closed* claims into chronological training (60%), validation (20%), and test (20%) partitions; keep class prevalence intact and avoid oversampling. Fit on training, choose any policy/threshold on validation, then report performance once on the untouched test. If historical claims can remain unresolved, use a label-availability cutoff and a temporal gap. Audit repeated-patient generalization separately.
Operational selection: Produce denial probabilities for pending claims, rank them *within each processing day*, and hold the highest-risk 150 at most. If an economically justified minimum risk threshold exists, hold only claims above it. In a real system, distinguish `service_date` from the submission-ready processing date; the supplied pending records do not include that date, so the demonstration explicitly uses service date as a grouping proxy rather than claiming it is the real daily queue.
Validation and monitoring: Compare against simple business rules (missing required authorization; >90-day filing age), measure precision@150 and captured denials@150 on retrospective daily batches, check probability calibration and group performance, and monitor drift, missing fields, queue size, and subsequent denial rates. Evaluate expected value: avoidable denial losses versus review cost and delay.
4. Release decision
Block the next-sprint production release. The current AUC of 0.917 is not a valid deployment estimate because the test set is contaminated and the features leak outcomes. The current 41 held claims are also not proof of a viable daily workflow. A release requires an as-of-time feature pipeline, leakage-free chronological holdout, capacity-aware policy, calibrated and business-relevant validation, and shadow testing against real submission-ready claims.
5. Included artifact
`claim_denial_model_fixed.py` is a runnable demonstration using the original synthetic data generator. It removes the most serious leakage, corrects evaluation, and enforces a daily top-150 policy. It is not a production-ready model; see the assumptions above.

