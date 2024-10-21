"""
Functions to load and process BRFSS smoking survey data
"""

import json
import os
import pandas as pd  # type: ignore


def get_brfss_age_bins() -> list[tuple[int, int]]:
    return [(18, 24)] + [(i, i + 4) for i in range(25, 80, 5)] + [(80, 150)]


def load_brfss_annotated(
    age_bin_set_size: int = 1, pivot: bool = False
) -> pd.DataFrame:
    """
    Load BRFSS data, annotated with age groups and smoking status
    Returns a DataFrame with columns:
        if pivot:
            year, sex, age_group, current_smoker, former_smoker, never_smoker, total,
            ever_smoker
        else:
            year, sex, smoking_status, age_group, count
    """
    age_bins = get_brfss_age_bins()
    age_bin_sets = [
        age_bins[i : i + age_bin_set_size]
        for i in range(0, len(age_bins), age_bin_set_size)
    ]
    brfss_data = (
        import_brfss()
        .assign(age_bin_set=lambda df: df["age_bin"].astype(int) // age_bin_set_size)
        .groupby(["year", "sex", "age_bin_set", "smoking_status"])["count"]
        .sum()
        .reset_index()
        .assign(
            age_group=lambda df: df["age_bin_set"].map(
                {
                    i: (
                        f"{age_bin_set[0][0]}-{age_bin_set[-1][1]}"
                        if age_bin_set[-1][1] < 100
                        else f"{age_bin_set[0][0]}+"
                    )
                    for i, age_bin_set in enumerate(age_bin_sets)
                }
            )
        )
        .drop(columns=["age_bin_set"])
    )
    if not pivot:
        return brfss_data
    return (
        brfss_data.pivot_table(
            index=["year", "sex", "age_group"],
            columns="smoking_status",
            values="count",
            fill_value=0,
        )
        .reset_index()
        .rename_axis(None, axis=1)
        .rename(
            columns={
                status: status.lower().replace(" ", "_")
                for status in ["Never smoker", "Former smoker", "Current smoker"]
            }
        )
        .assign(
            total=lambda df: (
                df["current_smoker"] + df["former_smoker"] + df["never_smoker"]
            ),
            ever_smoker=lambda df: df["current_smoker"] + df["former_smoker"],
        )
    )


def import_brfss() -> pd.DataFrame:
    """
    Load in BRFSS data from raw data, or from a CSV file if it exists
    return cols:
    year, sex, age_bin, smoking_status, count
    """
    if os.path.exists("data/brfss/brfss_smoking_status.csv"):
        return pd.read_csv("data/brfss/brfss_smoking_status.csv")
    print("All-years CSV file does not exist; reading in data")
    parse_txt_brfss_to_csv()
    smoking_status_by_year = pd.DataFrame()
    for year in range(1988, 2021):
        this_year_brfss = pd.read_csv(f"data/brfss/brfss_{year}.csv", dtype=str)
        this_year_brfss = (
            this_year_brfss.loc[lambda df: df["SMOKE100"].isin(["1.0", "2.0"])]
            .assign(
                year=year,
                smoking_status=lambda df: df.apply(
                    lambda row: (
                        "Never smoker"
                        if row["SMOKE100"] == "2.0"
                        else (
                            "Former smoker"
                            if (
                                ("SMOKENOW" in row and row["SMOKENOW"] == "2.0")
                                or ("SMOKEDAY" in row and row["SMOKEDAY"] == "3.0")
                                or ("SMOKDAY2" in row and row["SMOKDAY2"] == "3.0")
                            )
                            else "Current smoker"
                        )
                    ),
                    axis=1,
                ),
                sex=lambda df: df.apply(
                    lambda row: (
                        row["SEX"]
                        if "SEX" in row
                        else (row["SEX1"] if "SEX1" in row else row["BIRTHSEX"])
                    ),
                    axis=1,
                ),
            )
            .loc[lambda df: df["sex"].isin(["1.0", "2.0"]), :]
            .loc[lambda df: df["_AGEG5YR"] != "14.0", :]  # Don't know/Refused/Missing
            .assign(sex=lambda df: df["sex"].replace({"1.0": "Male", "2.0": "Female"}))
            .rename(columns={"_AGEG5YR": "age_bin"})
            .groupby(["year", "sex", "age_bin", "smoking_status"])
            .size()
            .reset_index()
            .rename(columns={0: "count"})
        )
        smoking_status_by_year = pd.concat([smoking_status_by_year, this_year_brfss])
    smoking_status_by_year.to_csv("data/brfss/brfss_smoking_status.csv", index=False)
    return smoking_status_by_year


def parse_txt_brfss_to_csv() -> None:
    """
    Download and parse BRFSS data from CDC website to CSV files
    """
    brfss_urls = [
        f"https://www.cdc.gov/brfss/annual_data/{year}/files/"
        + (("CDBRFS" + str(year)[-2:]) if year < 2011 else f"LLCP{year}")
        + f"{'_' if year < 1990 else ''}XPT.{'zip' if year < 2011 else 'ZIP'}"
        for year in range(1988, 2021)
    ]
    all_columns: list[list[str]] = []
    os.makedirs("data/brfss", exist_ok=True)
    for url in brfss_urls:
        year = int(url.split("/")[5])
        if os.path.exists(f"data/brfss/brfss_{year}.csv"):
            continue
        print(f"reading {url}, year {year}")
        this_year = pd.read_sas(url, format="xport")
        all_columns.append(this_year.columns.tolist())
        try:
            this_year_data = this_year[all_columns[-1]]
            this_year_data.to_csv(f"data/brfss/brfss_{year}.csv", index=False)
        except KeyError as e:
            raise KeyError(
                f"Year {year}; all columns {this_year.columns.tolist()}"
            ) from e
    with open("data/brfss/possible_columns.json", "w", encoding="utf-8") as f:
        json.dump(all_columns, f)
