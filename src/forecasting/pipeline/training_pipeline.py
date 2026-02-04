import os
import pandas as pd

from forecasting.components.data_ingestion import DataIngestion
from forecasting.components.data_validation import DataValidation
from forecasting.components.preprocessing import Preprocessor
from forecasting.components.feature_engineering import FeatureEngineer
from forecasting.components.model_trainer import ModelTrainer
from forecasting.components.model_evaluator import ModelEvaluator
from forecasting.components.model_selector import ModelSelector


class TrainingPipeline:

    def __init__(
        self,
        data_path=r"data\raw\Forecasting Case- Study.xlsx",
        forecast_horizon=8
    ):

        self.data_path = data_path
        self.forecast_horizon = forecast_horizon

        # components
        self.ingestion = DataIngestion(self.data_path)
        self.validator = DataValidation()
        self.preprocessor = Preprocessor()
        self.fe = FeatureEngineer()
        self.trainer = ModelTrainer(self.fe)
        self.evaluator = ModelEvaluator()
        self.selector = ModelSelector(self.evaluator)


    def run(self):

        print("Starting Training Pipeline...")

        # ================================
        # Load
        # ================================
        df = self.ingestion.load_data()

        # ================================
        # Validate
        # ================================
        self.validator.validate(df)

        # ================================
        # Clean
        # ================================
        df = self.preprocessor.clean(df)

        states = df["state"].unique()

        summary_rows = []

        # ================================
        # Train per state
        # ================================
        for state in states:

            print(f"Training state: {state}")

            ts = self.preprocessor.weekly_series(df, state)

            # -----------------------------
            # Train models
            # -----------------------------
            y_true, forecasts, trained_models = self.trainer.train_all_models(
                ts,
                self.forecast_horizon
            )

            # -----------------------------
            # Select best + save pickle
            # -----------------------------
            best_model_name, metrics_dict = self.selector.select_and_save(
                state,
                y_true,
                forecasts,
                trained_models
            )

            # -----------------------------
            # Save metrics summary
            # -----------------------------
            best_metrics = metrics_dict[best_model_name]

            summary_rows.append([
                state,
                best_model_name,
                best_metrics["MAE"],
                best_metrics["RMSE"],
                best_metrics["MAPE"]
            ])


        # ================================
        # Save summary CSV
        # ================================
        os.makedirs("artifacts", exist_ok=True)

        summary_df = pd.DataFrame(
            summary_rows,
            columns=["state", "best_model", "MAE", "RMSE", "MAPE"]
        )

        summary_df.to_csv(
            "artifacts/best_model_per_state.csv",
            index=False
        )

        print("Training completed successfully ✅")
