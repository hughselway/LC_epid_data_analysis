import os
import matplotlib.pyplot as plt
import seaborn as sns  # type: ignore

from .colours import get_smoking_status_colours
from .seer_populations import LINESTYLES_BY_SEX

from ..data.smoking_surveys import load_brfss_annotated


def lineplot_by_status_sex(proportion: bool = False):
    brfss_data = load_brfss_annotated(pivot=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    for sex, linestyle in LINESTYLES_BY_SEX.items():
        for status, colour in get_smoking_status_colours().items():
            status_colname = status.lower().replace(" ", "_")
            sns.lineplot(
                data=brfss_data.loc[brfss_data["sex"] == sex,]
                .groupby(["year"])[["total", status_colname]]
                .sum()
                .reset_index()
                .assign(
                    frequency=lambda df, status_colname_=status_colname: (
                        df[status_colname_] / df["total"]
                    )
                ),
                x="year",
                y="frequency" if proportion else status_colname,
                ax=ax,
                color=colour,
                linestyle=linestyle,
                legend=False,
            )
    ax.set_xlabel("Year")
    if proportion:
        ax.set_ylim(0, 1)
        ax.set_ylabel("Proportion of population")
        ax.set_yticks(ax.get_yticks().tolist())
        ax.set_yticklabels([f"{y:.0%}" for y in ax.get_yticks().tolist()])
    else:
        ax.set_ylabel("Count")
        ax.set_ylim(0, None)
    ax.legend(
        handles=[
            plt.Line2D(
                [0],
                [0],
                color=colour,
                label=f"{status} ({sex})",
                linestyle=linestyle,
            )
            for sex, linestyle in LINESTYLES_BY_SEX.items()
            for status, colour in get_smoking_status_colours().items()
        ],
        ncol=2 if proportion else 1,
        loc="upper center" if proportion else "upper left",
    )
    fig.tight_layout()
    save_dir = os.path.join("plots", "smoking_surveys")
    os.makedirs(save_dir, exist_ok=True)
    fig.savefig(f"{save_dir}/{'prop' if proportion else 'count'}_by_status_sex.pdf")
    plt.close(fig)


def ratio_lineplot_by_sex():
    brfss_data = load_brfss_annotated(pivot=True)
    fig, ax = plt.subplots(figsize=(5, 4))
    for sex, linestyle in LINESTYLES_BY_SEX.items():
        sns.lineplot(
            data=brfss_data.loc[brfss_data["sex"] == sex]
            .groupby(["year"])[["current_smoker", "former_smoker"]]
            .sum()
            .reset_index()
            .assign(ratio=lambda df: df["current_smoker"] / df["former_smoker"]),
            x="year",
            y="ratio",
            ax=ax,
            color="black",
            linestyle=linestyle,
            legend=False,
        )
    ax.set_xlabel("Year")
    ax.set_ylabel("Current smokers per former smokers")
    ax.set_ylim(0, None)
    ax.legend(
        handles=[
            plt.Line2D(
                [0],
                [0],
                color="black",
                label=f"Current smokers per former smokers ({sex})",
                linestyle=linestyle,
            )
            for sex, linestyle in LINESTYLES_BY_SEX.items()
        ],
        loc="lower center",
    )
    fig.tight_layout()
    save_dir = os.path.join("plots", "smoking_surveys")
    os.makedirs(save_dir, exist_ok=True)
    fig.savefig(f"{save_dir}/current_to_former_ratio.pdf")
    plt.close(fig)


def lineplot_by_age_sex(age_bin_set_size: int = 1):
    brfss_data = load_brfss_annotated(age_bin_set_size, pivot=True)
    fig = plt.figure(figsize=(8, 6))
    gs = fig.add_gridspec(2, 3, figure=fig, width_ratios=[1, 1, 0.05])
    axes = [
        [fig.add_subplot(gs[row, column]) for column in range(2)] for row in range(2)
    ]

    for row_idx, status in enumerate(["Current smoker", "Ever smoker"]):
        for col_idx, sex in enumerate(["Female", "Male"]):
            ax = axes[row_idx][col_idx]
            sns.lineplot(
                brfss_data.loc[
                    lambda x, sex_=sex: (x["sex"] == sex_) & (x["total"] > 1000),
                ].assign(
                    frequency=lambda df, status_=status: df[
                        status_.replace(" ", "_").lower()
                    ]
                    / df["total"]
                ),
                x="year",
                y="frequency",
                hue="age_group",
                ax=ax,
                palette="viridis",
                legend=False,
            )
            ax.set_title(f"{status} - {sex}")
            ax.set_ylabel("Proportion of respondents")
            ax.set_ylim(0, 1)
    fig.tight_layout()
    for row in axes:
        for ax in row:
            ax.set_yticks(ax.get_yticks())
            ax.set_yticklabels(["{:.0%}".format(x) for x in ax.get_yticks()])
    # add a legend
    legend_ax = fig.add_subplot(gs[:, 2])
    legend_ax.axis("off")
    legend_ax.legend(
        handles=[
            plt.Line2D(
                [0],
                [0],
                color=sns.color_palette(
                    "viridis", len(brfss_data["age_group"].unique())
                )[i],
                label=age_group,
            )
            for i, age_group in enumerate(brfss_data["age_group"].unique())
        ],
        title="Age group",
        loc="center",
    )

    fig.tight_layout()
    save_dir = os.path.join(
        "plots", "smoking_surveys", "by_age", f"age_bin_set_size_{age_bin_set_size}"
    )
    os.makedirs(save_dir, exist_ok=True)
    fig.savefig(f"{save_dir}/current_ex.pdf")


def total_lineplot_by_age_sex(age_bin_set_size: int = 1):
    brfss_data = load_brfss_annotated(age_bin_set_size, pivot=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
    for col_idx, sex in enumerate(["Male", "Female"]):
        sns.lineplot(
            brfss_data.loc[brfss_data["sex"] == sex],
            x="year",
            y="total",
            hue="age_group",
            ax=axes[col_idx],
            palette="viridis",
            legend=False,
        )
        axes[col_idx].set_xlabel("Year")
        axes[col_idx].set_ylabel("Total respondents")
    axes[1].legend(
        handles=[
            plt.Line2D(
                [0],
                [0],
                color=sns.color_palette(
                    "viridis", len(brfss_data["age_group"].unique())
                )[i],
                label=age_group,
            )
            for i, age_group in enumerate(brfss_data["age_group"].unique())
        ],
        title="Age group",
    )
    axes[1].set_ylim(0, None)
    fig.tight_layout()
    save_dir = os.path.join(
        "plots", "smoking_surveys", "by_age", f"age_bin_set_size_{age_bin_set_size}"
    )
    os.makedirs(save_dir, exist_ok=True)
    fig.savefig(f"{save_dir}/total_respondents.pdf")


if __name__ == "__main__":
    for proportion_ in [True, False]:
        lineplot_by_status_sex(proportion_)
    ratio_lineplot_by_sex()
    for age_bin_set_size_ in [1, 2, 4]:
        lineplot_by_age_sex(age_bin_set_size_)
        total_lineplot_by_age_sex(age_bin_set_size_)