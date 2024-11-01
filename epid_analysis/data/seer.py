"""
Functions to import and process SEER incidence and population data.
"""

import os
import re
import pandas as pd  # type: ignore

SEX_VALUES = ["Male and Female", "Male", "Female"]
AGE_VALUES = list(range(0, 85)) + ["85+", "Unknown"]
YEAR_VALUES = {
    8: ["1975-2021"] + list(range(1975, 2022)),
    12: ["1992-2021"] + list(range(1992, 2022)),
    17: ["2000-2021"] + list(range(2000, 2022)),
}


def load_rates(
    age_groups: list[tuple[int, int]] | None = None,
    age_standardised: bool = True,
    by_sex: bool = True,
    rates_per: int = 100_000,
) -> pd.DataFrame:
    """
    Import SEER incidence data and population data, and calculate crude and standardised
    incidence rates of each of LUAD, LUSC and unspecified NSCLC, for each SEER registry
    and recode.
    Returns a DataFrame with columns:
        registry, recode, histology, year, sex (if by_sex), count, population, rate,
        standardised_rate (if age_standardised)
    """
    return pd.concat(
        load_rates_by_registry_recode(
            registry, recode, age_groups, age_standardised, by_sex, rates_per
        ).assign(registry=registry, recode=recode)
        for registry in [8, 12, 17]
        for recode in ["rare_cancers", "AYA"]
    ).reset_index(drop=True)[
        (
            [
                "registry",
                "recode",
                "histology",
                "year",
                "sex",
                "count",
                "population",
                "rate",
            ]
            if by_sex
            else [
                "registry",
                "recode",
                "histology",
                "year",
                "count",
                "population",
                "rate",
            ]
        )
        + (["standardised_rate"] if age_standardised else [])
    ]


def load_rates_by_registry_recode(
    registry: int,
    recode: str,
    age_groups: list[tuple[int, int]] | None = None,
    age_standardised: bool = True,
    by_sex: bool = True,
    rates_per: int = 100_000,
) -> pd.DataFrame:
    """
    Import SEER incidence data and population data, and calculate crude and standardised
    incidence rates of each of LUAD, LUSC and unspecified NSCLC, as defined by the given
    recode, for each year in the given SEER registry.
    Returns a DataFrame with columns:
        histology, year, sex, count, population, rate, age (if not age_standardised),
        standardised_rate (if age_standardised)
    """
    assert (
        not age_standardised or age_groups is None
    ), "cannot standardise rates by age group if age_groups is not None"
    crude_rates = (
        load_incidence(registry=registry, recode=recode, age_groups=age_groups)
        .merge(
            load_registry_population_data(
                seer_registries=registry, age_groups=age_groups
            ),
            on=["year", "age", "sex"],
        )
        .groupby(
            ["year", "sex", "age", "histology"]
            if by_sex
            else ["year", "age", "histology"]
        )[["count", "population"]]
        .sum()
        .assign(rate=lambda df: df["count"] / df["population"] * rates_per)
        .reset_index()
    )
    if not age_standardised:
        return crude_rates
    return standardise_rates(crude_rates, load_who_reference_population(), by_sex)[
        (
            ["histology", "year", "sex", "count", "population", "rate"]
            if by_sex
            else ["histology", "year", "count", "population", "rate"]
        )
        + (["standardised_rate"] if age_standardised else [])
    ]


def standardise_rates(
    crude_rates: pd.DataFrame, reference_population: pd.DataFrame, by_sex: bool
) -> pd.DataFrame:
    """
    Standardise incidence rates by age group
    """
    return (
        crude_rates.merge(reference_population, on="age")
        .drop(columns=["reference_population"])
        .loc[lambda df: ~df["reference_fraction"].isna() & ~df["rate"].isna()]
        .assign(standardised_rate=lambda df: df["rate"] * df["reference_fraction"])
        .groupby(["histology", "year", "sex"] if by_sex else ["histology", "year"])[
            ["standardised_rate", "count", "population"]
        ]
        .sum()
        .reset_index()
        .assign(rate=lambda df: df["count"] / df["population"] * 100_000)
    )


def load_incidence(
    registry: int, recode: str, age_groups: list[tuple[int, int]] | None = None
) -> pd.DataFrame:
    """
    Load incidence data from SEER registries 8, 12, or 17, as taken from the SEER*Stat
    database.
    Returns a DataFrame with columns:
        histology, year, age, sex, count
    """
    assert registry in [8, 12, 17], "registry must be 8, 12, or 17"
    assert recode in ["AYA", "rare_cancers"], "recode must be AYA or rare_cancers"
    assert age_groups is None or all(
        a < b for a, b in age_groups
    ), "age_groups must be a list of tuples with increasing values"
    assert age_groups is None or all(
        (a >= 0 and b <= 85) or (a <= 85 and b > 100) for a, b in age_groups
    ), "age_groups must be a list of tuples with values between 0 and 85"
    df = (
        pd.concat(
            pd.read_csv(
                f"data/seer/raw/{recode}/SEER_{registry}/{histology}.csv",
                thousands=",",
            ).assign(
                histology=histology,
                sex=lambda df: df["sex"].map(dict(zip(range(3), SEX_VALUES))),
                age_at_diagnosis=lambda df: df["age_at_diagnosis"].map(
                    dict(zip(range(87), AGE_VALUES))
                ),
                year_of_diagnosis=lambda df: df["year_of_diagnosis"].map(
                    dict(zip(range(len(YEAR_VALUES[registry])), YEAR_VALUES[registry]))
                ),
            )
            for histology in ["LUAD", "LUSC", "other"]
            if os.path.exists(f"data/seer/raw/{recode}/SEER_{registry}/{histology}.csv")
        )
        .reset_index(drop=True)
        .rename(columns={"year_of_diagnosis": "year", "age_at_diagnosis": "age"})
    )
    if age_groups is None:
        return df[["histology", "year", "age", "sex", "count"]]
    return (
        df.loc[~(df["age"] == "Unknown")]
        .assign(
            age=lambda df: df["age"]
            .apply(lambda x: x if x != "85+" else 85)
            .astype(int),
            age_bin=lambda df: df["age"].map(
                lambda x: next(
                    (i for i, (a, b) in enumerate(age_groups) if a <= x <= b), None
                )
            ),
        )
        .dropna(subset=["age_bin"])
        .assign(age_bin=lambda df: df["age_bin"].astype(int))
        .groupby(["histology", "year", "age_bin", "sex"])[["count"]]
        .sum()
        .reset_index()
        .assign(
            age=lambda df: df["age_bin"].apply(
                lambda x: f"{age_groups[x][0]}-{age_groups[x][1]}"
            )
        )
        .drop(columns="age_bin")[["histology", "year", "age", "sex", "count"]]
    )


def load_who_reference_population() -> pd.DataFrame:
    """
    Load in the population distribution by age group from the 2000 US Census, as
    standardised by the World Health Organisation.
    Returns a DataFrame with columns:
        age, reference_population, reference_fraction
    """
    reference_population = (
        pd.read_csv(
            "data/seer/raw/2000_US_population_WHO_standard.tsv", thousands=",", sep="\t"
        )
        .assign(
            age=lambda df: df["age"]
            .apply(
                lambda x: re.sub(r"^0", "", x)
                .replace(" years", "")
                .replace("85", "85+")
            )
            .apply(lambda x: int(x) if x != "85+" else x),
            fraction=lambda df: df["count"] / df["count"].sum(),
        )
        .rename(
            columns={"count": "reference_population", "fraction": "reference_fraction"}
        )
    )
    return reference_population


def load_population_data(
    age_groups: list[tuple[int, int]] | None = None
) -> pd.DataFrame:
    """
    Load population data within each SEER registry grouping, as taken from the SEER*Stat
    database.
    Returns a DataFrame with columns:
        registry, year, age, sex, population
    """
    return pd.concat(
        load_registry_population_data(
            seer_registries=registry, age_groups=age_groups
        ).assign(registry=registry)
        for registry in [8, 12, 17]
    ).reset_index(drop=True)[["registry", "year", "age", "sex", "population"]]


def load_registry_population_data(
    seer_registries: int | None = None, age_groups: list[tuple[int, int]] | None = None
) -> pd.DataFrame:
    """
    Load population data from the SEER registries, as taken from the SEER*Stat database.
    Returns a DataFrame with columns:
        year, age, sex, population
    """
    csv_filename = (
        "data/seer/processed_populations/populations_by_age_sex"
        + ("" if seer_registries is None else f"_seer_{seer_registries}_registries")
        + ".csv"
    )
    if os.path.exists(csv_filename):
        print(f"CSV file already exists; reading in data from {csv_filename}")
        raw_populations_data = pd.read_csv(csv_filename)
    else:
        print("CSV file does not exist; processing raw text file")
        raw_populations_data = pd.read_fwf(
            "data/seer/raw/us.1969_2022.singleages.adjusted.txt",
            header=None,
            widths=[4, 2, 2, 3, 2, 1, 1, 1, 2, 8],
        )
        raw_populations_data.columns = pd.Index(
            [
                "year",
                "state_postal",
                "state_FIPS",
                "county_FIPS",
                "registry",
                "race",
                "origin",
                "sex",
                "age",
                "population",
            ]
        )
        print("dtypes:", raw_populations_data.dtypes)
        if seer_registries is not None:
            assert seer_registries in [
                8,
                12,
                17,
            ], "seer_registries must be 8, 12, or 17"
            registries_per_seer_release = get_registries_per_seer_release()
            assert all(
                registry in raw_populations_data["registry"].unique()
                for registry in registries_per_seer_release[seer_registries]
            ), (
                "one or more registries not available in the given SEER release: "
                + ", ".join(
                    str(registry)
                    for registry in raw_populations_data["registry"].unique()
                )
            )
            raw_populations_data = raw_populations_data.loc[
                lambda df: df["registry"].isin(
                    registries_per_seer_release[seer_registries]
                )
            ]
        raw_populations_data = (
            raw_populations_data.groupby(["year", "age", "sex"])[["population"]]
            .sum()
            .reset_index()
            .assign(sex=lambda df: df["sex"].map({1: "Male", 2: "Female"}))
        )
        raw_populations_data.to_csv(csv_filename, index=False)
    if age_groups is None:
        # then set 85 to 85+
        return raw_populations_data.assign(
            age=lambda df: df["age"].apply(lambda x: x if x != 85 else "85+")
        )
    return (
        raw_populations_data.assign(
            age_bin=lambda df: df["age"].map(
                lambda x: next(
                    (i for i, (a, b) in enumerate(age_groups) if a <= x <= b), None
                )
            ),
        )
        .dropna(subset=["age_bin"])
        .assign(age_bin=lambda df: df["age_bin"].astype(int))
        .groupby(["year", "age_bin", "sex"])[["population"]]
        .sum()
        .reset_index()
        .assign(
            age=lambda df: df["age_bin"].apply(
                lambda x: f"{age_groups[x][0]}-{age_groups[x][1]}"
            )
        )
        .drop(columns="age_bin")
    )


def get_registries_per_seer_release() -> dict[int, list[int]]:
    """
    Get which SEER registries are available for each SEER release.
    from https://seer.cancer.gov/popdata/popdic.html
    and https://seer.cancer.gov/registries/terms.html
    """
    registries_first_available_per_seer_release = {
        8: [
            "02",  # Connecticut
            "27",  # Atlanta
            "01",  # San Francisco-Oakland
            "21",  # Hawaii
            "22",  # Iowa
            "23",  # New Mexico
            "25",  # Seattle-Puget Sound
            "26",  # Utah
        ],
        12: [
            "29",  # Alaska
            "37",  # Rural Georgia
            "31",  # San Jose-Monterey
            "35",  # Los Angeles
        ],
        17: [
            "47",  # Greater Georgia
            "41",  # Greater California
            "42",  # Kentucky
            "43",  # Louisiana
            "44",  # New Jersey
        ],
    }
    registries_per_seer_release = {
        8: registries_first_available_per_seer_release[8],
        12: registries_first_available_per_seer_release[8]
        + registries_first_available_per_seer_release[12],
        17: registries_first_available_per_seer_release[8]
        + registries_first_available_per_seer_release[12]
        + registries_first_available_per_seer_release[17],
    }
    return {
        key: [int(registry) for registry in registries]
        for key, registries in registries_per_seer_release.items()
    }
