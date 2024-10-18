import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns  # type: ignore

from .colours import get_histology_colours
from .seer_populations import LINESTYLES_BY_SEX, get_age_groups
from ..data.seer import load_rates, load_rates_by_registry_recode

ALPHA_BY_RECODE = {"rare_cancers": 1.0, "AYA": 0.7}
LINESTYLES_BY_REGISTRY = {8: "-", 12: "--", 17: ":"}


def lineplot_by_registry_recode_sex(
    age_standardised: bool = True,
    all_registries: bool = False,
    recode_plotting: str = "rare_cancers_only",
):
    rates = load_rates(age_standardised=age_standardised)

    assert recode_plotting in ["rare_cancers_only", "separate_plots", "same_plot"], (
        "recode_plotting must be one of 'rare_cancers_only', "
        "'separate_plots', 'same_plot'"
    )
    registries_to_plot = [8, 12, 17] if all_registries else [8]
    ncol = 2 if recode_plotting == "separate_plots" else 1
    fig, axes = plt.subplots(
        len(registries_to_plot),
        ncol,
        figsize=(4 + 3 * ncol, 4 * len(registries_to_plot)),
        sharex=True,
        sharey=True,
    )
    for registry, row in zip(
        registries_to_plot, axes if len(registries_to_plot) > 1 else [axes]
    ):
        for recode, ax in (
            zip(
                ["rare_cancers", "AYA"],
                [row, row] if (recode_plotting == "same_plot") else row,
            )
            if recode_plotting != "rare_cancers_only"
            else [("rare_cancers", row)]
        ):
            for histology in ["LUAD", "LUSC", "other"]:
                if (
                    len(
                        rates.loc[
                            (rates["registry"] == registry)
                            & (rates["recode"] == recode)
                            & (rates["histology"] == histology)
                        ]
                    )
                    == 0
                ):
                    continue
                for sex in ["Female", "Male"]:
                    sns.lineplot(
                        data=(
                            rates.loc[
                                (rates["registry"] == registry)
                                & (rates["recode"] == recode)
                                & (rates["histology"] == histology)
                                & (rates["sex"] == sex)
                            ]
                        ),
                        x="year",
                        y="standardised_rate" if age_standardised else "rate",
                        ax=ax,
                        linestyle=LINESTYLES_BY_SEX[sex],
                        color=get_histology_colours()[histology],
                        label=(
                            f"{histology} ({sex})"
                            + (f" ({recode})" if recode_plotting == "same_plot" else "")
                        ),
                        legend=False,
                        alpha=(
                            ALPHA_BY_RECODE[recode]
                            if recode_plotting == "same_plot"
                            else 1.0
                        ),
                    )
            ax.set_xlabel("Year of diagnosis")
            ax.set_ylabel(
                "Cases per 100,000" + " (age-standardised)" * age_standardised
            )
            ax.legend(
                title="Histology",
                bbox_to_anchor=(
                    (1.05, 1) if recode_plotting != "separate_plots" else None
                ),
            )
    for ax in axes.flatten() if all_registries or ncol > 1 else [axes]:
        ax.set_ylim(0, None)
    fig.tight_layout()
    save_dir = os.path.join(
        "plots",
        "seer",
        "age_standardised_rates" if age_standardised else "crude_rates",
        "by_registry" if all_registries else "SEER_8",
        "by_sex",
    )
    os.makedirs(save_dir, exist_ok=True)
    fig.savefig(
        f"{save_dir}/recode_{recode_plotting}.pdf",
    )
    plt.close(fig)


def lineplot_by_registry(age_standardised: bool = True, recode: str = "rare_cancers"):
    populations = load_rates(by_sex=False).loc[lambda df: df["recode"] == recode]
    fig, ax = plt.subplots(figsize=(6.25, 4))
    for registry, linestyle in LINESTYLES_BY_REGISTRY.items():
        sns.lineplot(
            data=populations.loc[populations["registry"] == registry],
            x="year",
            y="standardised_rate" if age_standardised else "rate",
            hue="histology",
            ax=ax,
            linestyle=linestyle,
            palette=get_histology_colours(),
        )
    ax.legend(
        handles=[
            plt.Line2D(
                [0],
                [0],
                color=get_histology_colours()[histology],
                label=f"{histology} (SEER {registry})",
                linestyle=linestyle,
            )
            for histology in populations["histology"].unique()
            for registry, linestyle in LINESTYLES_BY_REGISTRY.items()
        ],
        ncol=3,
        loc="lower center",
    )
    ax.set_ylim(0, None)
    ax.set_xlabel("Year of diagnosis")
    ax.set_ylabel("Cases per 100,000" + " (age-standardised)" * age_standardised)
    fig.tight_layout()
    save_dir = os.path.join(
        "plots",
        "seer",
        "age_standardised_rates" if age_standardised else "crude_rates",
        "by_registry",
    )
    os.makedirs(save_dir, exist_ok=True)
    fig.savefig(f"{save_dir}/recode_{recode}.pdf")
    plt.close(fig)


def lineplot_by_age_group(registry: int, age_bin_size: int = 5):
    age_bins = get_age_groups(age_bin_size)
    rates = load_rates_by_registry_recode(
        registry, "rare_cancers", age_bins, age_standardised=False
    )
    fig = plt.figure(figsize=(8, 6))
    gs = gridspec.GridSpec(2, 3, figure=fig, width_ratios=[1, 1, 0.05])
    axes = [
        [fig.add_subplot(gs[row, column]) for column in range(2)] for row in range(2)
    ]
    for row, histology in enumerate(["LUAD", "LUSC"]):
        for column, sex in enumerate(["Male", "Female"]):
            sns.lineplot(
                data=rates.loc[
                    (rates["histology"] == histology) & (rates["sex"] == sex)
                ],
                x="year",
                y="rate",
                hue="age",
                ax=axes[row][column],
                palette="viridis",
                legend=False,
            )
            axes[row][column].set_title(f"{histology} ({sex})")
            axes[row][column].set_ylim(0, None)
    legend_ax = fig.add_subplot(gs[:, 2])
    legend_ax.axis("off")
    legend_ax.legend(
        title="Age",
        handles=[
            plt.Line2D(
                [0],
                [0],
                color=plt.get_cmap("viridis")(i / len(age_bins)),
                label=f"{age[0]}-{age[1]}" if age[1] < 100 else f"{age[0]}+",
            )
            for i, age in enumerate(age_bins)
        ],
        loc="center",
    )
    # set y-axis limits to be the same for each row
    for row_axes in axes:
        for axis in row_axes:
            axis.set_ylim(0, max(axis.get_ylim()[1] for axis in row_axes))
    # remove x-axis labels from top row
    for axis in axes[0]:
        axis.set_xlabel("")
        axis.set_xticklabels([])
    # remove y-axis labels from right column
    for axis in [axes[0][1], axes[1][1]]:
        axis.set_ylabel("")
        axis.set_yticklabels([])
    fig.tight_layout()
    save_dir = os.path.join("plots", "seer", "populations", "by_age_group")
    os.makedirs(save_dir, exist_ok=True)
    fig.savefig(os.path.join(save_dir, f"registry_{registry}.pdf"))
    plt.close(fig)


if __name__ == "__main__":
    for age_standardised_ in [True, False]:
        for all_registries_ in [True, False]:
            for recode_plotting_ in [
                "rare_cancers_only",
                "separate_plots",
                "same_plot",
            ]:
                lineplot_by_registry_recode_sex(
                    age_standardised=age_standardised_,
                    all_registries=all_registries_,
                    recode_plotting=recode_plotting_,
                )
    for recode_ in ["rare_cancers", "AYA"]:
        lineplot_by_registry(age_standardised=True, recode=recode_)
        lineplot_by_registry(age_standardised=False, recode=recode_)
    for registry_ in [8, 12, 17]:
        lineplot_by_age_group(registry_, 10)
