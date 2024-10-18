import os
import matplotlib.pyplot as plt
import seaborn as sns  # type: ignore

from .seer_populations import COLOURS_BY_REGISTRY
from .seer_rates import LINESTYLES_BY_SEX

from ..data.seer import load_rates


def ratio_lineplot_by_registry(all_registries: bool):
    ratios = calculate_LUSC_LUAD_ratio(all_registries)
    for standardised in [True, False]:
        fig, ax = plt.subplots()
        for sex, linestyle in LINESTYLES_BY_SEX.items():
            sns.lineplot(
                data=ratios.loc[
                    (ratios["standardised"] == standardised) & (ratios["sex"] == sex)
                ],
                x="year",
                y="LUSCs_per_LUAD",
                hue="registry",
                ax=ax,
                linestyle=linestyle,
                palette=COLOURS_BY_REGISTRY,
            )
        ax.set_title("LUSC/LUAD ratio" + " by registry" if all_registries else "")
        ax.set_ylabel(
            f"LUSC per LUAD ({'age standardised' if standardised else 'crude'})"
        )
        ax.set_ylim(0, None)
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
            f"{save_dir}/{'' if all_registries else 'SEER_8_'}LUSC_LUAD_ratio.pdf"
        )
        plt.close(fig)


def calculate_LUSC_LUAD_ratio(
    all_registries: bool,
):
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
        ratio_lineplot_by_registry(all_registries_)
