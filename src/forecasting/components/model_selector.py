import os
import joblib
import numpy as np


class ModelSelector:

    def __init__(self, evaluator, save_dir="artifacts/models"):
        self.evaluator = evaluator
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)


    # =====================================================
    # MAIN FUNCTION
    # =====================================================
    def select_and_save(
        self,
        state,
        y_true,
        forecasts,
        trained_models
    ):
        """
        Parameters
        ----------
        state : str
        y_true : actual values
        forecasts : dict {model_name: predictions}
        trained_models : dict {model_name: fitted_model_object}
        """

        best_model_name = None
        best_score = np.inf
        metrics_dict = {}

        # ---------------- Evaluate all ----------------
        for name, preds in forecasts.items():

            metrics = self.evaluator.evaluate(y_true, preds)

            metrics_dict[name] = metrics

            if metrics["MAE"] < best_score:
                best_score = metrics["MAE"]
                best_model_name = name

        # ---------------- Save best ----------------
        best_model = trained_models[best_model_name]

        model_path = os.path.join(self.save_dir, f"{state}.pkl")

        joblib.dump(best_model, model_path)

        return best_model_name, metrics_dict
