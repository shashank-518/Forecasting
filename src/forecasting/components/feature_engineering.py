import pandas as pd


class FeatureEngineer:

    def create_features(self, ts):

        df = pd.DataFrame(ts)
        df.columns = ["sales"]

        # Lags
        df["lag_1"] = df["sales"].shift(1)
        df["lag_7"] = df["sales"].shift(7)
        df["lag_30"] = df["sales"].shift(30)

        # Rolling
        df["roll_mean_4"] = df["sales"].rolling(4).mean()
        df["roll_std_4"] = df["sales"].rolling(4).std()

        # Calendar
        df["month"] = df.index.month
        df["week"] = df.index.isocalendar().week
        df["quarter"] = df.index.quarter

        df.dropna(inplace=True)

        return df
