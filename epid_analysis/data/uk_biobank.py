import json
import numpy as np
import pandas as pd  # type: ignore

from .plco import get_relevant_columns


BIOBANK_FOLLOWUP_CUTOFF = 9


def load_biobank_data(
    relevant_columns_only: bool = True,
    followup_cutoff: int | None = BIOBANK_FOLLOWUP_CUTOFF,
    restrict_to_plco_ages: bool = False,
    impute_missing: bool = False,
    include_tracheal: bool = False,
) -> pd.DataFrame:
    """
    Load in the UK Biobank data, assigning relevant columns (including selection of
    ICD codes for histologies).
    Returns a DataFrame with columns:
        lung_adenocarcinoma,lung_squamous_cell_carcinoma,years_to_censoring,
        age_at_censoring,quit_years,quit_years_at_recruitment,age_at_recruitment,
        pack_years,smoking_intensity,sex_male,age_started_smoking,age_stopped_smoking
    """
    icd_morphology_codes = get_icd_morphology_codes()
    site_codes = [
        get_icd_site_codes(site) for site in ["lung"] + (["trachea"] * include_tracheal)
    ]
    biobank_df = (
        pd.read_csv(f"data/TC_biobank/data/{'non' * (not impute_missing)}imputed.csv")
        .loc[
            lambda x: (
                ~x["time_to_censoring"].isna()
                & ~x["age_at_recruitment"].isna()
                & ~x["smoking_status"].isna()
                & ~(
                    (x["smoking_status"] != "Never")
                    & (x["age_started_smoking"].isna() | x["n_cig_per_day"].isna())
                )
                & ~(
                    (x["smoking_status"] == "Previous")
                    & (x["age_stopped_smoking"].isna())
                )
            ),
            :,
        ]
        .assign(
            lung_adenocarcinoma=lambda x: (
                x["type_of_cancer_icd10"].apply(
                    lambda y: any(
                        isinstance(y, str) and y.startswith(site) for site in site_codes
                    )
                )
                & x["histology_of_cancer_tumour"].apply(
                    lambda y: (y in icd_morphology_codes["lung_adenocarcinoma"])
                )
            ),
            lung_squamous_cell_carcinoma=lambda x: (
                x["type_of_cancer_icd10"].apply(
                    lambda y: any(
                        isinstance(y, str) and y.startswith(site) for site in site_codes
                    )
                )
                & x["histology_of_cancer_tumour"].apply(
                    lambda y: (
                        y in icd_morphology_codes["lung_squamous_cell_carcinoma"]
                    )
                )
            ),
            years_to_censoring=lambda x: (
                x["time_to_censoring"].str.split(" ").str[0].astype(float) / 365.25
            ),
            age_at_censoring=lambda x: (
                x["age_at_recruitment"] + x["years_to_censoring"]
            ),
            quit_years=lambda x: np.where(
                x["smoking_status"] == "Previous",
                x["age_at_censoring"] - x["age_stopped_smoking"],
                0,
            ),
            quit_years_at_recruitment=lambda x: np.where(
                (x["smoking_status"] == "Previous")
                & (x["age_at_recruitment"] - x["age_stopped_smoking"] > 0),
                x["age_at_recruitment"] - x["age_stopped_smoking"],
                0,
            ),
            smoking_duration=lambda x: (
                np.where(
                    x["smoking_status"] == "Previous",
                    x["age_stopped_smoking"] - x["age_started_smoking"],
                    np.where(
                        x["smoking_status"] == "Current",
                        x["age_at_censoring"] - x["age_started_smoking"],
                        0,
                    ),
                )
            ),
            pack_years=lambda x: np.where(
                (x["smoking_status"] == "Current")
                | (x["smoking_status"] == "Previous"),
                x["n_cig_per_day"] * x["smoking_duration"] / 20,
                0,
            ),
            smoking_intensity=lambda x: np.where(
                (x["smoking_status"] == "Current")
                | (x["smoking_status"] == "Previous"),
                x["n_cig_per_day"],
                0,
            ),
            sex_male=lambda x: x["sex"] == "Male",
        )
    )

    if followup_cutoff is not None:
        # remove any cancer events after cutoff, and censor those who have not had an
        # event by cutoff
        biobank_df = biobank_df.assign(
            lung_adenocarcinoma=lambda x: np.where(
                x["years_to_censoring"] > followup_cutoff,
                False,
                x["lung_adenocarcinoma"],
            ),
            lung_squamous_cell_carcinoma=lambda x: np.where(
                x["years_to_censoring"] > followup_cutoff,
                False,
                x["lung_squamous_cell_carcinoma"],
            ),
            years_to_censoring=lambda x: np.where(
                x["years_to_censoring"] > followup_cutoff,
                followup_cutoff,
                x["years_to_censoring"],
            ),
        )

    if restrict_to_plco_ages:
        biobank_df = biobank_df.loc[lambda x: x["age_at_recruitment"] >= 55, :]
        biobank_df = biobank_df.loc[lambda x: x["age_at_recruitment"] <= 74, :]

    if not relevant_columns_only:
        return biobank_df
    return biobank_df.loc[:, get_relevant_columns()]


def get_icd_morphology_codes() -> dict[str, list[int]]:
    """
    Load in ICD-O-3 morphology codes for LUAD and LUSC, acquired from the rare_cancers
    recode of SEER data: https://seer.cancer.gov/seerstat/variables/seer/raresiterecode
    """
    with open(
        "epid_analysis/data/ICD_histology_codes.json", "r", encoding="utf-8"
    ) as f:
        icd_codes = json.load(f)
    histology_codes = {}
    for histology, code_strings in icd_codes["morphology"].items():
        codes: list[int] = []
        for code_string in code_strings:
            if "-" in code_string:
                start, end = code_string.split("-")
                codes.extend(range(int(start), int(end) + 1))
            else:
                codes.append(int(code_string))
        histology_codes[histology] = codes
    return histology_codes


def get_icd_site_codes(site: str) -> str:
    with open(
        "epid_analysis/data/ICD_histology_codes.json", "r", encoding="utf-8"
    ) as f:
        icd_codes = json.load(f)
    return icd_codes["site"][site]
