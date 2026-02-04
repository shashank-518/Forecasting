import pandas as pd


class Preprocessor:

    def clean(self, df):
        df = df[["State", "Date", "Total"]].copy()

        df.columns = ["state", "date", "sales"]

        df["date"] = pd.to_datetime(df["date"], dayfirst=True)

        df = df.sort_values(["state", "date"])

        return df


    def weekly_series(self, df, state):

        ts = (
            df[df["state"] == state]
            .set_index("date")["sales"]
            .resample("W")
            .mean()
            .ffill()
            .bfill()
        )

        return ts
