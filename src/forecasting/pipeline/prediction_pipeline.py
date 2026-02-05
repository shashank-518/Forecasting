import os
import joblib
import pandas as pd
import numpy as np

from prophet import Prophet
import xgboost as xgb
from tensorflow.keras.models import Sequential # type: ignore

from src.forecasting.components.data_ingestion import DataIngestion
from src.forecasting.components.preprocessing import Preprocessor
from src.forecasting.components.feature_engineering import FeatureEngineer


class PredictionPipeline:

    def __init__(
        self,
        data_path="data/raw/Forecasting Case- Study.xlsx",
        model_dir="artifacts/models",
        horizon=8
    ):
        self.data_path = data_path
        self.model_dir = model_dir
        self.horizon = horizon

        self.ingestion = DataIngestion(self.data_path)
        self.preprocessor = Preprocessor()
        self.fe = FeatureEngineer()



    def forecast(self, state):

        model = self._load_model(state)

        ts = self._get_series(state)

        last_date = ts.index[-1]

        future_dates = pd.date_range(
            last_date + pd.Timedelta(weeks=1),
            periods=self.horizon,
            freq="W"
        )

        model_name = self._get_model_name(model)
        preds = self._route_prediction(model, ts)

        return pd.DataFrame({
            "state": state,
            "model_used": model_name,
            "date": future_dates,
            "forecast": preds
        })

    

    def _load_model(self, state):

        path = os.path.join(self.model_dir, f"{state}.pkl")

        if not os.path.exists(path):
            raise FileNotFoundError(f"No model found for {state}")

        return joblib.load(path)


    def _get_series(self, state):

        df = self.ingestion.load_data()
        df = self.preprocessor.clean(df)

        return self.preprocessor.weekly_series(df, state)


    
    def _route_prediction(self, model, ts):

        if self._is_statsmodel(model):
            return self._predict_statsmodel(model)

        elif isinstance(model, Prophet):
            return self._predict_prophet(model)

        elif isinstance(model, xgb.XGBRegressor):
            return self._predict_xgb(model, ts)

        elif isinstance(model, Sequential):
            return self._predict_lstm(model, ts)

        else:
            raise ValueError("Unsupported model type")
        
    
    def _get_model_name(self, model):

        from prophet import Prophet
        import xgboost as xgb
        from tensorflow.keras.models import Sequential # type: ignore

        if model.__class__.__module__.startswith("statsmodels"):
            return "ARIMA/SARIMA"

        elif isinstance(model, Prophet):
            return "Prophet"

        elif isinstance(model, xgb.XGBRegressor):
            return "XGBoost"

        elif isinstance(model, Sequential):
            return "LSTM"

        else:
            return "Unknown"


    

    
    def _predict_statsmodel(self, model):
        return model.forecast(self.horizon)


    
    def _predict_prophet(self, model):

        future = model.make_future_dataframe(
            periods=self.horizon,
            freq="W"
        )

        fc = model.predict(future)

        return fc["yhat"].tail(self.horizon).values


    
    def _predict_xgb(self, model, ts):

        feat = self.fe.create_features(ts)

        last_row = feat.drop("sales", axis=1).iloc[-1:].copy()

        preds = []

        for _ in range(self.horizon):
            p = model.predict(last_row)[0]
            preds.append(p)

            last_row["lag_30"] = last_row["lag_7"]
            last_row["lag_7"] = last_row["lag_1"]
            last_row["lag_1"] = p

        return preds


   
    def _predict_lstm(self, model, ts):

        from sklearn.preprocessing import MinMaxScaler

        scaler = MinMaxScaler()

        scaled = scaler.fit_transform(ts.values.reshape(-1,1))

        last_seq = scaled[-8:]

        preds = []

        for _ in range(self.horizon):
            pred = model.predict(last_seq.reshape(1,8,1), verbose=0)
            preds.append(pred[0][0])
            last_seq = np.append(last_seq[1:], pred)

        return scaler.inverse_transform(
            np.array(preds).reshape(-1,1)
        ).flatten()


    
    def _is_statsmodel(self, model):
        return model.__class__.__module__.startswith("statsmodels")
