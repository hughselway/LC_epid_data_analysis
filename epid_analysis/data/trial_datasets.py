import pandas as pd  # type: ignore
import numpy as np

from .uk_biobank import load_biobank_data, BIOBANK_FOLLOWUP_CUTOFF
from .plco import load_plco_data


def load_trial_datasets(
    include_combined: bool = True,
    include_smoking_status: bool = False,
    biobank_followup_cutoff: int | None = BIOBANK_FOLLOWUP_CUTOFF,
    restrict_to_plco_ages: bool = False,
) -> dict[str, pd.DataFrame]:
    datasets = {
        "UK Biobank": load_biobank_data(
            relevant_columns_only=True,
            followup_cutoff=biobank_followup_cutoff,
            restrict_to_plco_ages=restrict_to_plco_ages,
        ),
        "PLCO": load_plco_data(relevant_columns_only=True),
    }
    if include_combined:
        datasets["Combined"] = pd.concat(datasets.values()).reset_index(drop=True)
    if include_smoking_status:
        return {k: annotate_smoking_status(v) for k, v in datasets.items()}
    return datasets


def annotate_smoking_status(df: pd.DataFrame) -> pd.DataFrame:
    return df.assign(
        smoking_status=lambda x: np.where(
            x["pack_years"] == 0,
            "Never smoker",
            np.where(
                x["quit_years_at_recruitment"] <= 0, "Current smoker", "Ex-smoker"
            ),
        )
    )


def load_age_stratified_incidence_data(
    age_bin_edges: list[float], rate_scale: float = 1e5, include_combined: bool = True
) -> pd.DataFrame:
    """
    Load in the UK Biobank and PLCO data, stratify by age group and smoking status, and
    calculate the yearly incidence rate of LUAD and LUSC, calculated as the number of
    cases per 100,000 person-years spent during follow-up in the age group/smoking
    status category.
    Returns a DataFrame with columns:
        source, age_group, smoking_status, sex, person_years, histology, count,
        yearly_rate
    """
    age_bin_labels = get_age_bin_labels(age_bin_edges)
    age_stratified_incidence_data = (
        pd.concat(
            df.assign(source=source)
            for source, df in load_trial_datasets(
                include_combined=False, include_smoking_status=True
            ).items()
        )
        .reset_index(drop=True)
        .assign(
            age_group_at_censoring=lambda x: pd.cut(
                x["age_at_censoring"], bins=age_bin_edges, labels=age_bin_labels
            ),
            sex=lambda x: np.where(x["sex_male"] == 1, "Male", "Female"),
            years_in_age_group=lambda x: x.apply(
                lambda row: years_by_age_bin(
                    row["age_at_recruitment"], row["age_at_censoring"], age_bin_edges
                ),
                axis=1,
            ),
            age_group=lambda x: [age_bin_labels for _ in range(len(x))],
        )
        .explode(["years_in_age_group", "age_group"])
        .assign(
            **{
                cancer_type: lambda x, cancer_type=cancer_type: (
                    x[cancer_type] & (x["age_group_at_censoring"] == x["age_group"])
                )
                for cancer_type in [
                    "lung_adenocarcinoma",
                    "lung_squamous_cell_carcinoma",
                ]
            }
        )
        .groupby(["age_group", "smoking_status", "sex", "source"])
        .agg(
            lung_adenocarcinoma=("lung_adenocarcinoma", "sum"),
            lung_squamous_cell_carcinoma=("lung_squamous_cell_carcinoma", "sum"),
            person_years=("years_in_age_group", "sum"),
        )
        .reset_index()
        .melt(
            id_vars=["age_group", "smoking_status", "sex", "person_years", "source"],
            value_vars=["lung_adenocarcinoma", "lung_squamous_cell_carcinoma"],
            var_name="histology",
            value_name="count",
        )
        .loc[
            :,
            [
                "source",
                "age_group",
                "smoking_status",
                "sex",
                "person_years",
                "histology",
                "count",
            ],
        ]
    )
    if include_combined:
        combined_data = (
            age_stratified_incidence_data.groupby(
                ["age_group", "smoking_status", "sex", "histology"]
            )[["person_years", "count"]]
            .sum()
            .reset_index()
        ).assign(source="combined")

        age_stratified_incidence_data = (
            pd.concat([age_stratified_incidence_data, combined_data])
            .sort_values(["age_group", "smoking_status", "sex", "histology", "source"])
            .reset_index(drop=True)
            .loc[
                :,
                [
                    "source",
                    "age_group",
                    "smoking_status",
                    "sex",
                    "person_years",
                    "histology",
                    "count",
                ],
            ]
        )
    return age_stratified_incidence_data.assign(
        yearly_rate=lambda x: x.apply(
            lambda row: (
                row["count"] * rate_scale / row["person_years"]
                if row["person_years"] != 0
                else (np.nan if row["count"] == 0 else np.inf)
            ),
            axis=1,
        )
    )


def get_age_bin_labels(age_bin_edges: list[float]) -> list[str]:
    return [
        f"{age_bin_edges[i]}-{age_bin_edges[i+1]}"
        for i in range(len(age_bin_edges) - 1)
    ]


def years_by_age_bin(
    age_at_recruitment: float,
    age_at_censoring: float,
    age_bin_edges: list[float],
) -> list[float]:
    return [
        years_in_age_bin(
            age_at_recruitment, age_at_censoring, age_bin_edges, age_bin_index
        )
        for age_bin_index in range(len(age_bin_edges) - 1)
    ]


def years_in_age_bin(
    age_at_recruitment: float,
    age_at_censoring: float,
    age_bin_edges: list[float],
    age_bin_index: int,
) -> float:
    lower_edge = age_bin_edges[age_bin_index]
    upper_edge = age_bin_edges[age_bin_index + 1]
    if age_at_censoring <= lower_edge or age_at_recruitment >= upper_edge:
        return 0
    return min(age_at_censoring, upper_edge) - max(age_at_recruitment, lower_edge)
