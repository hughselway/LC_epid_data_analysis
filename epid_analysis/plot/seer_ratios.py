import os
import matplotlib.pyplot as plt
import seaborn as sns  # type: ignore

from .seer_populations import COLOURS_BY_REGISTRY
from .seer_rates import LINESTYLES_BY_SEX

from ..data.seer import load_rates


def ratio_lineplot_by_registry(
    all_registries: bool, plot_LUSC_per_LUAD_or_LUSC: bool = False
):
    ratios = calculate_LUSC_LUAD_ratio(all_registries)
    for standardised in [True, False]:
        fig, ax = plt.subplots(figsize=(6, 4))
        for sex, linestyle in LINESTYLES_BY_SEX.items():
            sns.lineplot(
                data=ratios.loc[
                    (ratios["standardised"] == standardised) & (ratios["sex"] == sex)
                ],
                x="year",
                y=(
                    "LUSCs_per_LUAD_or_LUSC"
                    if plot_LUSC_per_LUAD_or_LUSC
                    else "LUSCs_per_LUAD"
                ),
                hue="registry",
                ax=ax,
                linestyle=linestyle,
                palette=COLOURS_BY_REGISTRY,
            )
        ax.set_title(
            f"LUSC/LUAD {'or LUSC/' if plot_LUSC_per_LUAD_or_LUSC else ''}ratio"
            + (" by registry" if all_registries else "")
        )
        ax.set_ylabel(
            f"LUSC per LUAD{' or LUSC' if plot_LUSC_per_LUAD_or_LUSC else ''} "
            f"({'age standardised' if standardised else 'crude'})"
        )
        ax.set_ylim(0, 1 if plot_LUSC_per_LUAD_or_LUSC else None)
        if plot_LUSC_per_LUAD_or_LUSC:
            ax.set_yticks(ax.get_yticks().tolist())
            ax.set_yticklabels([f"{y:.0%}" for y in ax.get_yticks().tolist()])
        ax.legend(
            handles=[
                plt.Line2D(
                    [0],
                    [0],
                    color=COLOURS_BY_REGISTRY[registry],
                    label=f"Registry {registry} ({sex})",
                    linestyle=linestyle,
                )
                for registry in ratios["registry"].unique()
                for sex, linestyle in LINESTYLES_BY_SEX.items()
            ]
        )
        fig.tight_layout()
        save_dir = os.path.join(
            "plots", "seer", "ratios", "standardised" if standardised else "crude"
        )
        os.makedirs(save_dir, exist_ok=True)
        fig.savefig(
            f"{save_dir}/{'' if all_registries else 'SEER_8_'}LUSC_LUAD_"
            f"{'or_LUSC_' if plot_LUSC_per_LUAD_or_LUSC else ''}ratio.pdf"
        )
        plt.close(fig)


def calculate_LUSC_LUAD_ratio(all_registries: bool):
    rates = load_rates()
    if not all_registries:
        rates = rates.loc[rates["registry"] == 8]
    # AYA only provides LUAD and other, so can't calculate LUSC/LUAD ratio
    rates = rates.loc[rates["recode"] == "rare_cancers"]

    return (
        rates.melt(  # first melt the data to get standardised and non-standardised rates in separate columns
            id_vars=(["registry", "recode", "year", "sex", "histology"]),
            value_vars=["standardised_rate", "rate"],
            var_name="standardised",
            value_name="value",
        )
        .assign(standardised=lambda df: df["standardised"] == "standardised_rate")
        .pivot_table(
            index=(["registry", "recode", "year", "sex", "standardised"]),
            columns="histology",
            values="value",
        )
        .reset_index()
        .assign(
            LUSCs_per_LUAD=lambda df: df["LUSC"] / df["LUAD"],
            LUSCs_per_LUAD_or_LUSC=lambda df: df["LUSC"] / (df["LUAD"] + df["LUSC"]),
        )
    )


if __name__ == "__main__":
    for all_registries_ in [True, False]:
        for plot_LUSC_LUAD_or_LUSC_ in [True, False]:
            ratio_lineplot_by_registry(all_registries_, plot_LUSC_LUAD_or_LUSC_)
