import pandas as pd


def load_and_clean_data(path):
    df = pd.read_csv(path)
    df.replace("?", pd.NA, inplace=True)
    df.dropna(inplace=True)

    df = df.apply(pd.to_numeric)
    df["target"] = (df["target"] > 0).astype(int)

    return df
