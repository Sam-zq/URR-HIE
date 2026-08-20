"""PV forecasting metrics that remain meaningful when nighttime truth is zero."""

import numpy as np


def pv_metrics(prediction, target, capacity=1.0, daylight_mask=None,
               persistence=None):
    pred = np.asarray(prediction, dtype=np.float64)
    true = np.asarray(target, dtype=np.float64)
    if daylight_mask is None:
        mask = true > 1e-4
    else:
        mask = np.asarray(daylight_mask).astype(bool)
    error = pred - true
    daylight_error = error[mask]
    daylight_true = true[mask]
    result = {
        "mae_all": float(np.mean(np.abs(error))),
        "rmse_all": float(np.sqrt(np.mean(error ** 2))),
        "nmae_daylight": float(np.mean(np.abs(daylight_error)) / capacity),
        "nrmse_daylight": float(
            np.sqrt(np.mean(daylight_error ** 2)) / capacity
        ),
        "r2_daylight": float(
            1.0 - np.sum(daylight_error ** 2)
            / max(np.sum((daylight_true - daylight_true.mean()) ** 2), 1e-12)
        ),
        "negative_rate": float(np.mean(pred < 0.0)),
        "over_capacity_rate": float(np.mean(pred > capacity)),
    }
    if persistence is not None:
        base = np.asarray(persistence, dtype=np.float64)
        model_rmse = np.sqrt(np.mean(daylight_error ** 2))
        base_rmse = np.sqrt(np.mean((base[mask] - true[mask]) ** 2))
        result["forecast_skill_rmse"] = float(
            1.0 - model_rmse / max(base_rmse, 1e-12)
        )
    return result

