"""Leakage-reduced demonstration; NOT production-ready.

Uses the assessment's synthetic generator. `days_to_submit` is treated as
known for demonstration only; in production compute filing age as-of scoring.
The pending service_date is used as a daily grouping proxy because the
assessment supplies no actual submission-ready/processing date.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score, brier_score_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from claim_denial_model import load_claims

SEED = 7
CAPACITY = 150
FEATURES = ["prior_auth", "needs_auth", "late_filing", "billed_amount_k"]


def prepare(df):
    df = df.copy()
    df["late_filing"] = (df["days_to_submit"] > 90).astype(int)
    df["billed_amount_k"] = df["billed_amount"] / 1000
    if df[FEATURES].isna().any().any():
        raise ValueError("Missing submission-time features: investigate rather than silently fill")
    return df


def evaluate(name, y, probability, threshold):
    predicted = (probability >= threshold).astype(int)
    print(f"{name}: n={len(y)}, prevalence={np.mean(y):.3f}, "
          f"AUC={roc_auc_score(y, probability):.3f}, "
          f"AP={average_precision_score(y, probability):.3f}, "
          f"precision={precision_score(y, predicted, zero_division=0):.3f}, "
          f"recall={recall_score(y, predicted, zero_division=0):.3f}, "
          f"Brier={brier_score_loss(y, probability):.3f}")


def main():
    df = prepare(load_claims())
    closed = df.loc[df.status.eq("closed")].sort_values(["service_date", "claim_id"]).reset_index(drop=True)
    pending = df.loc[df.status.eq("pending")].copy()
    # Split chronologically without duplicating/resampling claims.
    n = len(closed)
    train = closed.iloc[:int(n * 0.6)]
    validation = closed.iloc[int(n * 0.6):int(n * 0.8)]
    test = closed.iloc[int(n * 0.8):]
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=SEED))
    model.fit(train[FEATURES], train.denied.astype(int))
    val_prob = model.predict_proba(validation[FEATURES])[:, 1]
    # Illustrative threshold chosen ONLY on validation; business utility should replace F1.
    thresholds = np.arange(0.05, 0.96, 0.01)
    from sklearn.metrics import f1_score
    threshold = max(thresholds, key=lambda t: f1_score(validation.denied.astype(int), val_prob >= t))
    print(f"Validation-selected threshold: {threshold:.2f}")
    evaluate("Validation", validation.denied.astype(int), val_prob, threshold)
    test_prob = model.predict_proba(test[FEATURES])[:, 1]
    evaluate("Untouched test", test.denied.astype(int), test_prob, threshold)
    pending["denial_probability"] = model.predict_proba(pending[FEATURES])[:, 1]
    pending["hold"] = False
    # For this toy dataset only, service_date stands in for a processing date.
    for day, group in pending.groupby("service_date"):
        eligible = group[group.denial_probability >= threshold]
        selected = eligible.nlargest(CAPACITY, "denial_probability").index
        pending.loc[selected, "hold"] = True
        print(f"Queue proxy {day.date()}: {len(group)} claims; {len(selected)} held (cap {CAPACITY})")
    print(f"Total held: {int(pending.hold.sum())} / {len(pending)}")
    pending[["claim_id", "service_date", "denial_probability", "hold"]].to_csv(
        "pending_claim_decisions.csv", index=False
    )
    print("Saved pending_claim_decisions.csv")


if __name__ == "__main__":
    main()
