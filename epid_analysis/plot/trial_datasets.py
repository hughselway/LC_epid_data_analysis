import os
import time
import numpy as np  # type: ignore
import matplotlib.pyplot as plt
import pandas as pd  # type: ignore
import seaborn as sns  # type: ignore

from .colours import get_histology_colour, get_smoking_status_colours, get_dataset_cmap

from ..data.trial_datasets import load_trial_datasets


def plot_baseline_hazard_estimation(
    dataset: dict[str, pd.DataFrame], restricted: bool = False
):
    fig, axes = plt.subplots(2, len(dataset), figsize=(5 * len(dataset), 8))

    for row, cumulative in enumerate([False, True]):
        for (dataset_name, df), axis in zip(
            dataset.items(), axes[row, :] if len(dataset) > 1 else [axes[row]]
        ):
            axis.plot(
                len(df) - df["years_to_censoring"].value_counts().sort_index().cumsum(),
                label="exposed to risk",
                color="black",
            )
            rates_axis = axis.twinx()
            for histology in ["lung_adenocarcinoma", "lung_squamous_cell_carcinoma"]:
                this_histology_relevant_df = (
                    df[
                        (df[histology] == 1)
                        & (
                            df["years_to_censoring"]
                            <= (
                                (9 if dataset_name == "UK Biobank" else 15)
                                if restricted
                                else 100
                            )
                        )
                    ]["years_to_censoring"]
                    .value_counts()
                    .sort_index()
                )
                number_of_events = (
                    this_histology_relevant_df.cumsum()
                    if cumulative
                    else this_histology_relevant_df
                )
                exposed_to_risk_at_indices = (
                    len(df)
                    - df["years_to_censoring"].value_counts().sort_index().cumsum()
                ).loc[number_of_events.index]
                rates_axis.plot(
                    number_of_events / exposed_to_risk_at_indices,
                    label=histology,
                    color=get_histology_colour(histology),
                )
            axis.set_xlabel("Years since recruitment")
            axis.set_ylabel("Number exposed to risk")
            rates_axis.set_ylabel(
                "Cancer rate in exposed" + (" (cumulative)" if cumulative else ""),
                rotation=270,
                labelpad=15,
            )
            axis.set_title(dataset_name)
            rates_axis.legend(loc="center left")
    fig.suptitle("Restricted to 10 followup years" if restricted else "Unrestricted")
    fig.tight_layout()
    save_dir = os.path.join("plots", "trial_datasets", "baseline_hazard_estimation")
    os.makedirs(save_dir, exist_ok=True)
    fig.savefig(
        os.path.join(save_dir, f"{'restricted' if restricted else 'unrestricted'}.pdf")
    )
    plt.close(fig)


def plot_regressors(
    datasets: dict[str, pd.DataFrame],
    regressor_colnames: list[str],
    regressor_bins: dict[str, list[int]],
) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(12, 8))

    continuous_reg_colnames = [
        colname
        for colname in regressor_colnames
        if ("sex" not in colname and "*" not in colname)
    ]

    for dataset_name, dataset_df in datasets.items():
        if dataset_name == "Combined":
            continue
        for i, colname in enumerate(continuous_reg_colnames):
            # instead of dataset colours, use deepest colour from dataset cmap
            dataset_cmap_name = get_dataset_cmap(dataset_name)
            dataset_cmap = plt.get_cmap(dataset_cmap_name)
            axes[i, i].hist(
                dataset_df[colname],
                color=dataset_cmap(0.9),
                alpha=0.5,
                label=dataset_name,
                bins=regressor_bins[colname],
            )
            axes[i, i].set_xlabel(colname)
            axes[i, i].set_ylabel("Participant count")
            for j, second_colname in enumerate(continuous_reg_colnames):
                if (i > j and dataset_name == "PLCO") or (
                    i < j and dataset_name == "UK Biobank"
                ):
                    axes[i, j].hist2d(
                        dataset_df[second_colname],
                        dataset_df[colname],
                        alpha=0.5,
                        cmap=get_dataset_cmap(dataset_name),
                        bins=[regressor_bins[second_colname], regressor_bins[colname]],
                    )
                    axes[i, j].set_xlabel(second_colname)
                    axes[i, j].set_ylabel(colname)
                    plt.colorbar(axes[i, j].collections[0], ax=axes[i, j])

                    correlation = (
                        dataset_df.loc[
                            (dataset_df[colname] > 0) & (dataset_df[second_colname] > 0)
                        ][[colname, second_colname]]
                        .corr()
                        .iloc[0, 1]
                    )
                    axes[i, j].annotate(
                        f"corr={correlation:.2f}",
                        xy=(0.05, 0.92),
                        xycoords="axes fraction",
                    )

    fig.tight_layout()
    save_dir = os.path.join("plots", "trial_datasets", "regressors")
    os.makedirs(save_dir, exist_ok=True)
    fig.savefig(os.path.join(save_dir, "without_legend.pdf"))

    ## Save again with legend
    axes[0, 0].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "with_legend.pdf"))
    plt.close(fig)

    ## Save legend separately
    legend_fig, legend_ax = plt.subplots(figsize=(1.7, 0.8))
    legend_ax.set_axis_off()
    legend_ax.legend(*axes[0, 0].get_legend_handles_labels(), loc="center", fontsize=10)
    legend_fig.tight_layout()
    legend_fig.savefig(f"{save_dir}/legend.pdf")
    plt.close(legend_fig)


def barplot_smoking_histories(trial_datasets: dict[str, pd.DataFrame]) -> None:
    fig, axes = plt.subplots(
        1, len(trial_datasets), figsize=(2.5 * len(trial_datasets), 1.4)
    )
    smoking_status_colours = get_smoking_status_colours()
    smoking_status_colours["Ex-smoker"] = smoking_status_colours.pop("Former smoker")
    for i, (ax, (dataset_name, dataset_df)) in enumerate(
        zip(axes, trial_datasets.items())
    ):
        smoking_status_counts = (
            dataset_df.assign(
                sex=lambda x: np.where(x["sex_male"] == 1, "Male", "Female")
            )
            .groupby(["sex"])["smoking_status"]
            .value_counts(normalize=True)
            .unstack()
            .fillna(0)
        )
        overall_smoking_status_counts = (
            dataset_df["smoking_status"]
            .value_counts(normalize=True)
            .reindex(smoking_status_counts.columns)
            .fillna(0)
        )
        pd.concat(
            [
                smoking_status_counts,
                pd.DataFrame({"Overall": overall_smoking_status_counts}).T,
            ]
        ).plot(
            kind="barh",
            stacked=True,
            color=smoking_status_colours,
            ax=ax,
            legend=False,
            width=0.75,
        )
        ax.set_title(dataset_name)
        for side in ["top", "right"]:
            ax.spines[side].set_visible(False)
        if i > 0:
            ax.set_ylabel("")
            ax.set_yticklabels([])
            ax.tick_params(axis="y", length=0)
            ax.spines["left"].set_visible(False)

    axes[2].legend(
        loc="center left",
        bbox_to_anchor=(1, 0.5),
        title="Smoking status",
    )
    fig.tight_layout()
    save_dir = os.path.join("plots", "trial_datasets")
    os.makedirs(save_dir, exist_ok=True)
    fig.savefig(os.path.join(save_dir, "smoking_histories_by_sex.pdf"))
    plt.close(fig)


def boxplot_params_by_sex(
    trial_datasets: dict[str, pd.DataFrame],
    cts_regressor_colnames: list[str],
    regressor_bins: dict[str, list[int]],
) -> None:
    fig, axes = plt.subplots(
        len(cts_regressor_colnames),
        len(trial_datasets),
        figsize=(2.5 * len(trial_datasets), 2 * len(cts_regressor_colnames)),
    )

    for col_ix, (dataset_name, dataset_df) in enumerate(trial_datasets.items()):
        for row_ix, colname in enumerate(cts_regressor_colnames):
            sns.violinplot(
                x=colname,
                y=dataset_df.assign(
                    sex=lambda x: np.where(x["sex_male"] == 1, "Male", "Female")
                )["sex"],
                data=dataset_df.loc[lambda df, colname_=colname: df[colname_] != 0],
                ax=axes[row_ix, col_ix],
                color=plt.get_cmap(get_dataset_cmap(dataset_name))(0.9),
                legend=False,
            )

            axes[row_ix, col_ix].set_xlabel(
                colname.replace("_", " ").replace("at recruitment", "")
            )
            axes[row_ix, col_ix].set_ylabel("")
            axes[row_ix, col_ix].set_xlim(
                min(regressor_bins[colname]), max(regressor_bins[colname])
            )
            for side in ["top", "right"]:
                axes[row_ix, col_ix].spines[side].set_visible(False)
            if col_ix > 0:
                axes[row_ix, col_ix].set_ylabel("")
                axes[row_ix, col_ix].set_yticklabels([])
                axes[row_ix, col_ix].tick_params(axis="y", length=0)
                axes[row_ix, col_ix].spines["left"].set_visible(False)
            if row_ix == 0:
                axes[row_ix, col_ix].set_title(dataset_name)

    fig.tight_layout()
    save_dir = os.path.join("plots", "trial_datasets")
    os.makedirs(save_dir, exist_ok=True)
    fig.savefig(os.path.join(save_dir, "cts_params_by_sex.pdf"))


def print_data_summary(trial_datasets: dict[str, pd.DataFrame]) -> None:
    for dataset_name, dataset_df in trial_datasets.items():
        print(f"\n{dataset_name}")
        print(dataset_df["lung_adenocarcinoma"].value_counts())
        print(dataset_df["lung_squamous_cell_carcinoma"].value_counts())
        print(f"Total participants: {len(dataset_df)}")


if __name__ == "__main__":
    start_time = time.time()
    trial_datasets_ = load_trial_datasets()
    print(f"Loaded datasets in {time.time() - start_time:.2f} seconds")

    print_data_summary(trial_datasets_)

    regressor_colnames_ = [
        "age_at_recruitment",
        "sex_male",
        "pack_years",
        "quit_years_at_recruitment",
    ]
    regressor_bins_ = {
        "age_at_recruitment": list(range(39, 75, 1)),
        "pack_years": list(range(1, 90, 3)),
        "quit_years_at_recruitment": list(range(1, 60, 2)),
    }
    boxplot_params_by_sex(
        trial_datasets_,
        [x for x in regressor_colnames_ if x != "sex_male"],
        regressor_bins_,
    )
    plot_regressors(trial_datasets_, regressor_colnames_, regressor_bins_)

    plot_baseline_hazard_estimation(trial_datasets_, restricted=False)
    plot_baseline_hazard_estimation(trial_datasets_, restricted=True)

    barplot_smoking_histories(load_trial_datasets(include_smoking_status=True))
