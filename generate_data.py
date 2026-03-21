import os
import numpy as np
import pandas as pd

np.random.seed(42)

FEATURE_COLS = [
    "Recency", "Frequency", "Monetary",
    "AvgOrderValue", "DaysSinceFirst", "Country_encoded",
]


def generate_batch(n: int, drift_factor: float = 0.0) -> pd.DataFrame:
    recency = np.random.exponential(
        scale=30 * (1 + drift_factor), size=n
    ).clip(0, 365)

    frequency = np.random.poisson(
        lam=max(0.5, 5 - 3 * drift_factor), size=n
    ).astype(float)

    monetary = np.random.lognormal(
        mean=5 + drift_factor, sigma=1.2, size=n
    ).clip(0, 50_000)

    avg_order_value = monetary / np.maximum(frequency, 1)

    days_since_first = np.random.uniform(10, 730, size=n)

    if drift_factor < 0.3:
        country = np.random.randint(0, 38, size=n)
    else:
        country = np.random.choice(
            range(38), size=n,
            p=np.array([0.5] + [0.5 / 37] * 37),
        )

    logit = (
        0.02 * frequency
        - 0.008 * recency
        + 0.00005 * monetary
        - 0.5
        - drift_factor * 0.5
    )
    score = 1 / (1 + np.exp(-logit))
    label = (score + np.random.normal(0, 0.1, n) > 0.5).astype(int)

    return pd.DataFrame({
        "Recency": recency,
        "Frequency": frequency,
        "Monetary": monetary,
        "AvgOrderValue": avg_order_value,
        "DaysSinceFirst": days_since_first,
        "Country_encoded": country,
        "prediction": score,
        "label": label,
    })


def main():
    os.makedirs("data/reference", exist_ok=True)
    os.makedirs("data/production", exist_ok=True)

    # Reference data
    ref = generate_batch(n=5_000, drift_factor=0.0)
    ref.to_parquet("data/reference/reference_data.parquet", index=False)
    print(f"Reference data saved: {len(ref):,} rows")

    # Production batches with escalating drift
    batch_config = [
        (0, 0.00),
        (1, 0.05),
        (2, 0.15),
        (3, 0.35),
        (4, 0.60),
        (5, 0.80),
    ]

    for idx, drift_factor in batch_config:
        batch = generate_batch(n=1_000, drift_factor=drift_factor)
        path = f"data/production/batch_{idx:03d}.parquet"
        batch.to_parquet(path, index=False)
        print(f"Batch {idx:03d} saved (drift={drift_factor:.2f}): {len(batch):,} rows")

    print("\nDone. Run run_monitoring.py to start monitoring.")


if __name__ == "__main__":
    main()