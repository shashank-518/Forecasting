import numpy as np
import xgboost as xgb

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet

from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense


class ModelTrainer:

    def __init__(self, feature_engineer):
        self.feature_engineer = feature_engineer


    # =====================================================
    # MAIN TRAIN FUNCTION
    # =====================================================
    def train_all_models(self, ts, forecast_horizon=8):

        train_ts = ts[:-forecast_horizon]
        test_ts  = ts[-forecast_horizon:]

        forecasts = {}
        trained_models = {}   # ✅ ADD THIS


        # =================================================
        # ARIMA
        # =================================================
        arima_model = ARIMA(train_ts, order=(2,1,2)).fit()
        forecasts["ARIMA"] = arima_model.forecast(forecast_horizon).values
        trained_models["ARIMA"] = arima_model


        # =================================================
        # SARIMA
        # =================================================
        sarima_model = SARIMAX(
            train_ts,
            order=(2,1,2),
            seasonal_order=(1,1,1,52)
        ).fit()

        forecasts["SARIMA"] = sarima_model.forecast(forecast_horizon).values
        trained_models["SARIMA"] = sarima_model


        # =================================================
        # PROPHET
        # =================================================
        prophet_model = Prophet(weekly_seasonality=True)

        p_df = ts.reset_index()
        p_df.columns = ["ds","y"]

        prophet_model.fit(p_df[:-forecast_horizon])

        future = prophet_model.make_future_dataframe(periods=forecast_horizon, freq="W")
        fc = prophet_model.predict(future)

        forecasts["Prophet"] = fc["yhat"].tail(forecast_horizon).values
        trained_models["Prophet"] = prophet_model


        # =================================================
        # XGBOOST
        # =================================================
        feat = self.feature_engineer.create_features(ts)

        X = feat.drop("sales", axis=1)
        y = feat["sales"]

        split = len(X)-forecast_horizon

        X_train, X_test = X[:split], X[split:]
        y_train = y[:split]

        xgb_model = xgb.XGBRegressor(n_estimators=400)
        xgb_model.fit(X_train, y_train)

        forecasts["XGB"] = xgb_model.predict(X_test)
        trained_models["XGB"] = xgb_model


        # =================================================
        # LSTM
        # =================================================
        scaler = MinMaxScaler()

        scaled = scaler.fit_transform(ts.values.reshape(-1,1))

        def seq(data, step=8):
            X,y=[],[]
            for i in range(len(data)-step):
                X.append(data[i:i+step])
                y.append(data[i+step])
            return np.array(X), np.array(y)

        X_seq, y_seq = seq(scaled)

        split = len(X_seq)-forecast_horizon

        X_train, X_test = X_seq[:split], X_seq[split:]
        y_train = y_seq[:split]

        lstm_model = Sequential([
            LSTM(64, activation='relu', input_shape=(8,1)),
            Dense(1)
        ])

        lstm_model.compile(optimizer="adam", loss="mse")
        lstm_model.fit(X_train, y_train, epochs=20, verbose=0)

        preds = lstm_model.predict(X_test)

        forecasts["LSTM"] = scaler.inverse_transform(preds).flatten()
        trained_models["LSTM"] = lstm_model


        # =================================================
        # RETURN ALL 3
        # =================================================
        return test_ts.values, forecasts, trained_models
