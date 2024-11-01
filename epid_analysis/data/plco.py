import numpy as np
import pandas as pd  # type: ignore


def load_plco_data(relevant_columns_only: bool = False) -> pd.DataFrame:
    """
    Load in the PLCO data, assigning relevant columns (including selection of
    histologies).
    Returns a DataFrame with columns:
        lung_adenocarcinoma,lung_squamous_cell_carcinoma,years_to_censoring,
        age_at_censoring,quit_years,quit_years_at_recruitment,age_at_recruitment,
        pack_years,smoking_intensity,sex_male,age_started_smoking,age_stopped_smoking
    """
    plco_df = (
        pd.read_csv("data/package-plco-1258/Lung/lung_data_mar22_d032222.csv")
        .assign(
            lung_histtype_name=lambda x: x.lung_histtype.replace(
                {  # lifted from the PLCO data dictionary
                    2: "Squamous Cell Carcinoma",
                    3: "Spindle Cell Carcinoma",
                    4: "Small Cell Carcinoma",
                    5: "Intermediate Cell Carcinoma",
                    7: "Adenocarcinoma",
                    8: "Acinar Adenocarcinoma",
                    9: "Papillary Adenocarcinoma",
                    10: "Bronchioalveolar Adenocarcinoma",
                    11: "Adenocarcinoma w/Mucus Formation",
                    12: "Large Cell Carcinoma",
                    13: "Giant Cell Carcinoma",
                    14: "Clear Cell Carcinoma",
                    15: "Adenosquamous Carcinoma",
                    18: "Adenoid Cystic Carcinoma",
                    31: "Carcinoma NOS (recoded)",
                    32: "Mixed small and non-small cell (recoded)",
                    33: "Neuroendocrine NOS (recoded)",
                }
            ),
            lung_adenocarcinoma=lambda x: (~x.lung_histtype_name.isna())
            & x.lung_histtype_name.str.contains("Adenocarcinoma"),
            lung_squamous_cell_carcinoma=lambda x: (~x.lung_histtype_name.isna())
            & (x.lung_histtype_name == "Squamous Cell Carcinoma"),
            pack_years=lambda x: x["pack_years"].fillna(0.0),
            smoking_intensity=lambda x: np.where(
                x["cig_years"] > 0, x["pack_years"] / x["cig_years"] * 20, 0.0
            ),
            years_to_censoring=lambda x: x["lung_exitage"] - x["age"],
            quit_years=lambda x: np.where(
                x["cig_stop"] > 0, x["cig_stop"] + x["years_to_censoring"], 0.0
            ),
            age_at_censoring=lambda x: x["age"] + x["years_to_censoring"],
            sex_male=lambda x: (x["sex"] == 1),
            age_started_smoking=lambda x: np.where(
                x["cig_years"] > 0, x["age"] - x["cig_stop"] - x["cig_years"], np.nan
            ),
            age_stopped_smoking=lambda x: np.where(
                x["cig_years"] > 0, x["age"] - x["cig_stop"], np.nan
            ),
            quit_years_at_recruitment=lambda x: np.where(
                x["cig_stop"] > 0, x["cig_stop"], 0.0
            ),
        )
        .rename({"age": "age_at_recruitment"}, axis=1)
    )
    if not relevant_columns_only:
        return plco_df
    return plco_df.loc[:, get_relevant_columns()]


def get_relevant_columns():
    return [
        "lung_adenocarcinoma",
        "lung_squamous_cell_carcinoma",
        "years_to_censoring",
        "age_at_censoring",
        "quit_years",
        "quit_years_at_recruitment",
        "age_at_recruitment",
        "pack_years",
        "smoking_intensity",
        "sex_male",
        "age_started_smoking",
        "age_stopped_smoking",
    ]
