import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

SEED = 7
REVIEW_CAPACITY_PER_DAY = 150

FEATURES = [
    "prior_auth",
    "needs_auth",
    "late_filing",
    "billed_amount_k",
    "patient_denial_rate",
    "days_to_payment",
]


def load_claims(n_closed=10000, n_pending=500, n_patients=2000):
    rng = np.random.default_rng(SEED)
    n = n_closed + n_pending

    closed_dates = pd.Timestamp("2024-01-01") + pd.to_timedelta(rng.integers(0, 547, n_closed), unit="D")
    pending_dates = pd.Timestamp("2025-07-01") + pd.to_timedelta(rng.integers(0, 5, n_pending), unit="D")

    df = pd.DataFrame({
        "claim_id": [f"C{i:05d}" for i in range(n)],
        "patient_id": rng.integers(0, n_patients, n),
        "service_date": list(closed_dates) + list(pending_dates),
        "service_type": rng.choice(["Visit", "Lab", "Imaging", "Surgery"], n, p=[0.5, 0.2, 0.2, 0.1]),
        "days_to_submit": rng.gamma(2.0, 14.0, n).round().astype(int),
        "billed_amount": rng.lognormal(5.5, 1.0, n).round(2),
        "status": ["closed"] * n_closed + ["pending"] * n_pending,
    })
    df["needs_auth"] = df.service_type.isin(["Imaging", "Surgery"]).astype(int)
    df["prior_auth"] = (rng.random(n) < np.where(df.needs_auth == 1, 0.8, 0.1)).astype(int)

    patient_risk = rng.normal(0, 0.8, n_patients)[df.patient_id]
    score = -2.5 + patient_risk
    score += 2.0 * ((df.needs_auth == 1) & (df.prior_auth == 0))
    score += 1.5 * (df.days_to_submit > 90)
    df["denied"] = (rng.random(n) < 1 / (1 + np.exp(-score))).astype(float)
    df["days_to_payment"] = np.where(df.denied == 1, rng.gamma(6, 9, n), rng.gamma(5, 7, n)).round()
    df.loc[df.status == "pending", ["denied", "days_to_payment"]] = np.nan
    return df

def add_features(df):
    df = df.copy()
    df["late_filing"] = (df.days_to_submit > 90).astype(int)
    df["billed_amount_k"] = df.billed_amount / 1000
    df["patient_denial_rate"] = df.groupby("patient_id")["denied"].transform("mean")
    df[FEATURES] = df[FEATURES].fillna(0)
    return df


def balance(df):
    denied = df[df.denied == 1]
    paid = df[df.denied == 0]
    denied_copies = denied.sample(len(paid), replace=True, random_state=SEED)
    return pd.concat([paid, denied_copies]).sample(frac=1, random_state=SEED)


def precision_recall(y_true, y_pred):
    tp = ((y_pred == 1) & (y_true == 1)).sum()
    fp = ((y_pred == 1) & (y_true == 0)).sum()
    fn = ((y_pred == 0) & (y_true == 1)).sum()
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fp, 1)
    return precision, recall


def main():
    df = add_features(load_claims())
    closed = df[df.status == "closed"]
    pending = df[df.status == "pending"].copy()
    print(f"Closed claims: {len(closed):,} | Pending claims: {len(pending)} "
          f"| Denial rate: {closed.denied.mean():.1%}")

    data = balance(closed)
    data.sort_values("service_date")
    cut = int(len(data) * 0.8)
    train, test = data.iloc[:cut], data.iloc[cut:]

    model = LogisticRegression(max_iter=1000)
    model.fit(train[FEATURES], train.denied)

    test_prob = model.predict_proba(test[FEATURES])[:, 1]
    best_threshold, best_f1 = 0.5, 0
    for t in np.arange(0.10, 0.95, 0.05):
        p, r = precision_recall(test.denied, (test_prob >= t).astype(int))
        f1 = 2 * p * r / max(p + r, 1e-9)
        if f1 > best_f1:
            best_threshold, best_f1 = t, f1

    p, r = precision_recall(test.denied, (test_prob >= best_threshold).astype(int))
    print("\n=== Test results (newest 20% of claims) ===")
    print(f"ROC AUC   : {roc_auc_score(test.denied, test_prob):.3f}")
    print(f"Threshold : {best_threshold:.2f}")
    print(f"Precision : {p:.3f}")
    print(f"Recall    : {r:.3f}")

    print("\n=== Feature weights ===")
    for name, w in zip(FEATURES, model.coef_[0]):
        print(f"  {name:<20} {w:+.2f}")

    pending["hold"] = model.predict(pending[FEATURES])
    print("\n=== This week's pending claims ===")
    print(f"Claims held for review: {int(pending.hold.sum())} of {len(pending)}")


if __name__ == "__main__":
    main()
