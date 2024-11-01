import json
import os
import pickle
import time
import numpy as np
import pandas as pd  # type: ignore
from lifelines import CoxPHFitter  # type: ignore
from statsmodels.stats.outliers_influence import (  # type: ignore
    variance_inflation_factor,
)

from .data.trial_datasets import load_trial_datasets
from .data.uk_biobank import load_biobank_data


def fit_all_cox_models(trial_datasets: dict[str, pd.DataFrame], robust: bool = False):
    regressor_colnames = [
        "age_at_recruitment",
        "sex_male",
        "pack_years",
        "quit_years_at_recruitment",
    ]
    save_dir = os.path.join("output", "cox_models", "robust" if robust else "standard")
    os.makedirs(save_dir, exist_ok=True)
    with open(
        os.path.join("output", "cox_models", "regressor_colnames.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(regressor_colnames, f)
    for source, dataset in trial_datasets.items():
        for histology in ["lung_adenocarcinoma", "lung_squamous_cell_carcinoma"]:
            print(
                f"\n{'-'*5}{'non-' if not robust else ''}robust - {source} - {histology}{'-'*5}"
            )
            start_time = time.time()
            cox_model = fit_cox_model(dataset, histology, regressor_colnames, robust)
            print(f"Time taken: {time.time() - start_time:.2f}s")
            print()
            with open(
                os.path.join(save_dir, f"{source}_{histology}_cox_model.pkl"), "wb"
            ) as f:
                pickle.dump(cox_model, f)


def fit_pollution_cox_model(dataset: pd.DataFrame, robust: bool = False) -> None:
    regressor_colnames = [
        "age_at_recruitment",
        "sex_male",
        "pack_years",
        "quit_years_at_recruitment",
        "scaled_airpollution_pm2point5",
    ]
    save_dir = os.path.join(
        "output", "pollution_cox_models", "robust" if robust else "standard"
    )
    os.makedirs(save_dir, exist_ok=True)
    with open(
        os.path.join("output", "pollution_cox_models", "regressor_colnames.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(regressor_colnames, f)
    for histology in ["lung_adenocarcinoma", "lung_squamous_cell_carcinoma"]:
        print(
            f"\n{'-'*5}{'non-' if not robust else ''}robust - pollution - {histology}{'-'*5}"
        )
        start_time = time.time()
        cox_model = fit_cox_model(dataset, histology, regressor_colnames, robust)
        print(f"Time taken: {time.time() - start_time:.2f}s")
        print()
        with open(
            os.path.join(save_dir, f"UK Biobank_{histology}_cox_model.pkl"), "wb"
        ) as f:
            pickle.dump(cox_model, f)


def fit_cox_model(
    dataset: pd.DataFrame,
    histology: str,
    regressor_colnames: list[str],
    robust: bool = False,
    print_summary: bool = True,
) -> CoxPHFitter:
    cph = CoxPHFitter()
    cph.fit(
        dataset,
        duration_col="years_to_censoring",
        event_col=histology,
        formula=" + ".join(regressor_colnames),
        fit_options={"step_size": 0.05},
        robust=robust,
    )
    if print_summary:
        cph.print_summary()
    return cph


def read_regressor_colnames(pollution: bool = False) -> list[str]:
    with open(
        os.path.join(
            "output",
            "cox_models" if not pollution else "pollution_cox_models",
            "regressor_colnames.json",
        ),
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


def read_cox_model(
    source: str, histology: str, robust: bool = False, pollution: bool = False
) -> CoxPHFitter:
    with open(
        os.path.join(
            "output",
            "cox_models" if not pollution else "pollution_cox_models",
            "robust" if robust else "standard",
            f"{source}_{histology}_cox_model.pkl",
        ),
        "rb",
    ) as f:
        return pickle.load(f)


def calculate_vif_values(
    dataset: pd.DataFrame, regressor_colnames: list[str]
) -> dict[str, float]:
    vif_values = {}
    for regressor_colname in regressor_colnames:
        vif_values[regressor_colname] = variance_inflation_factor(
            dataset[regressor_colnames].to_numpy(dtype=float),
            regressor_colnames.index(regressor_colname),
        )
    return vif_values


def calculate_condition_number(
    dataset: pd.DataFrame, regressor_colnames: list[str]
) -> float:
    return np.linalg.cond(dataset[regressor_colnames].to_numpy(dtype=float))


def record_collinearity_analysis(
    trial_datasets: dict[str, pd.DataFrame], regressor_colnames: list[str]
) -> None:
    vif = {
        source: calculate_vif_values(dataset, regressor_colnames)
        for source, dataset in trial_datasets.items()
    }
    condition = {
        source: calculate_condition_number(dataset, regressor_colnames)
        for source, dataset in trial_datasets.items()
    }
    with open(
        os.path.join("output", "cox_models", "collinearity_analysis.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump({"vif": vif, "condition": condition}, f)


def read_condition() -> dict[str, float]:
    with open(
        os.path.join("output", "cox_models", "collinearity_analysis.json"),
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)["condition"]


def read_vif() -> dict[str, dict[str, float]]:
    with open(
        os.path.join("output", "cox_models", "collinearity_analysis.json"),
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)["vif"]


if __name__ == "__main__":
    trial_datasets_ = load_trial_datasets()
    regressor_colnames_ = [
        "age_at_recruitment",
        "sex_male",
        "pack_years",
        "quit_years_at_recruitment",
    ]

    record_collinearity_analysis(trial_datasets_, regressor_colnames_)
    for robust_ in [False, True]:
        # fit_all_cox_models(trial_datasets_, robust_)
        fit_pollution_cox_model(
            load_biobank_data(relevant_columns_only=False)
            .assign(
                scaled_airpollution_pm2point5=lambda x: x["airpollution_pm2point5"]
                * 10,
            )
            .loc[lambda x: x["scaled_airpollution_pm2point5"].notnull()],
            robust_,
        )
