import os
import math
import matplotlib.pyplot as plt
from matplotlib import animation
import pandas as pd  # type: ignore
import seaborn as sns  # type: ignore

from ..data.seer import load_population_data, load_registry_population_data

LINESTYLES_BY_SEX = {"Male": "--", "Female": "-"}
COLOURS_BY_REGISTRY = {8: "C0", 12: "C1", 17: "C2"}


def get_age_groups(bin_size: int) -> list[tuple[int, int]]:
    final_bin_edge = bin_size * math.floor(85 / bin_size)
    age_groups: list[tuple[int, int]] = [
        (i * bin_size, (i + 1) * bin_size - 1)
        for i in range(0, final_bin_edge // bin_size)
    ] + [(final_bin_edge, 150)]
    return age_groups


def lineplot_by_registry():
    populations = (
        load_population_data()
        .groupby(["registry", "year", "sex"])[["population"]]
        .sum()
        .reset_index()
    )
    fig, ax = plt.subplots()
    for sex, linestyle in LINESTYLES_BY_SEX.items():
        sns.lineplot(
            data=populations.loc[populations["sex"] == sex],
            x="year",
            y="population",
            hue="registry",
            ax=ax,
            linestyle=linestyle,
            palette=COLOURS_BY_REGISTRY,
        )
    ax.set_yticks(ax.get_yticks().tolist())
    ax.set_yticklabels([f"{int(y/1e6)}" for y in ax.get_yticks().tolist()])
    ax.set_ylabel("Population (millions)")
    ax.set_ylim(0, None)
    ax.set_title("Total population in SEER data")
    ax.legend(
        handles=[
            plt.Line2D(
                [0],
                [0],
                color=COLOURS_BY_REGISTRY[registry],
                label=f"Registry {registry} ({sex})",
                linestyle=linestyle,
            )
            for registry in populations["registry"].unique()
            for sex, linestyle in LINESTYLES_BY_SEX.items()
        ]
    )
    fig.tight_layout()
    save_dir = os.path.join("plots", "seer", "populations")
    os.makedirs(save_dir, exist_ok=True)
    fig.savefig(os.path.join(save_dir, "by_registry.pdf"))
    plt.close(fig)


def plot_population_pyramid(
    axis: plt.Axes,
    year_to_plot: int,
    dataset: pd.DataFrame,
    age_groups: list[tuple[int, int]],
    xlim: float = 1.3e6,
    colname_to_plot: str = "population",
):
    this_year_data = dataset.query(f"year == {year_to_plot}")

    legend_handles = []
    for sex, colour, sign in zip(
        ["Male", "Female"], ["tab:blue", "tab:orange"], [1, -1]
    ):
        this_year_data[this_year_data["sex"] == sex].assign(
            sex_adjusted_plot_column=lambda df, sign=sign: (df[colname_to_plot] * sign)
        )[["age", "sex_adjusted_plot_column"]].set_index("age").plot.barh(
            ax=axis, color=colour, legend=False, width=0.95
        )
        legend_handles.append(
            plt.Line2D([0, 0], [0, 0], linewidth=8, color=colour, label=sex)
        )
    axis.set_yticks(range(len(age_groups)))
    axis.set_yticklabels(
        [
            f"{a}-{b}" if 0 < b - a < 5 else f"{a}+" if b > 85 else f"{a}"
            for a, b in age_groups
        ]
    )
    axis.set_ylabel("Age")
    axis.set_xlabel(
        "Population" if colname_to_plot == "population" else "Rate per 100,000"
    )
    axis.set_title(
        "SEER data population distribution"
        if colname_to_plot == "population"
        else "SEER data incidence rate distribution"
    )
    axis.text(
        0.95,
        0.95 if colname_to_plot == "population" else 0.05,
        f"Year = {year_to_plot}",
        transform=axis.transAxes,
        ha="right",
        va="top",
    )
    axis.set_xlim(-xlim, xlim)
    axis.set_xticks(axis.get_xticks().tolist())
    axis.set_xticklabels(
        [
            (
                f"{abs(int(x/1e5))}00k"
                if colname_to_plot == "population"
                else f"{abs(int(x))}"
            )
            for x in axis.get_xticks().tolist()
        ]
    )
    axis.legend(
        handles=legend_handles,
        loc="upper left" if colname_to_plot == "population" else "lower left",
    )


def plot_animated_population_pyramid(registry: int | None, bin_size: int = 5):
    age_groups = get_age_groups(bin_size)
    populations = load_registry_population_data(registry, age_groups)
    fig, ax = plt.subplots()

    def animate(year):
        ax.clear()
        plot_population_pyramid(
            axis=ax,
            year_to_plot=year,
            dataset=populations,
            age_groups=age_groups,
            xlim=1.3e6 if registry in [8, 12] else 3e6 if registry else 1.5e7,
        )

    ani = animation.FuncAnimation(
        fig, animate, frames=populations["year"].unique(), interval=200
    )
    save_dir = os.path.join("plots", "seer", "populations")
    os.makedirs(save_dir, exist_ok=True)
    ani.save(
        os.path.join(
            save_dir,
            (f"population_SEER_{registry}.gif" if registry else "population_total.gif"),
        )
    )
    plt.close(fig)


def plot_interval_population_pyramids(registry: int | None, bin_size: int = 5):
    age_groups = get_age_groups(bin_size)
    for year in range(1970, 2020, 5):
        fig, ax = plt.subplots()
        plot_population_pyramid(
            axis=ax,
            year_to_plot=year,
            dataset=load_registry_population_data(registry, age_groups),
            age_groups=age_groups,
        )
        fig.tight_layout()
        save_dir = os.path.join(
            "plots",
            "seer",
            "populations",
            "interval_pyramids",
            f"SEER_{registry}" if registry else "total",
        )
        os.makedirs(save_dir, exist_ok=True)
        fig.savefig(os.path.join(save_dir, f"population_pyramid_{year}.pdf"))
        plt.close()


if __name__ == "__main__":
    lineplot_by_registry()
    for registry_ in [8, 12, 17, None]:
        plot_interval_population_pyramids(registry_)
        plot_animated_population_pyramid(registry_)
