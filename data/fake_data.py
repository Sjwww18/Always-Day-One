# data/fake_data.py

import numpy as np
import pandas as pd

np.random.seed(42)


def make_fake_parquet(
    path: str,
    dates=("2020-01-02", "2020-01-03"),
    stocks=("A", "B", "C"),
    intervals=range(5),
    features=("f1", "f2", "f3"),
    label=("y",),
):
    rows = []
    
    for d in dates:
        for s in stocks:
            for i in intervals:
                row = {
                    "date": d,
                    "stock": s,
                    "interval": i,
                }
                # fake features
                for f in features:
                    row[f] = np.random.randn()
                # fake label
                for y in label:
                    row[y] = np.random.randn()
                rows.append(row)

    df = pd.DataFrame(rows)
    df.to_parquet(path, engine="pyarrow")
    print(f"Fake parquet saved to: {path}.")
    print(df.head())


if __name__ == "__main__":
    make_fake_parquet("data/fake_data.parquet")


# end of data/fake_data.py