import os
import numpy as np
import pandas as pd  # type: ignore
import matplotlib.pyplot as plt
from matplotlib import transforms
from .colours import get_dataset_cmap, get_histology_colour

from ..cox_model import (
    read_regressor_colnames,
    read_cox_model,
    read_condition,
    read_vif,
)
from ..data.trial_datasets import load_trial_datasets


def plot_hazard_ratios(
    source: str,
    robust: bool,
    include_sex: bool,
    include_age: bool,
    pollution: bool = False,
) -> None:
    regressor_colnames = read_regressor_colnames(pollution)

    fig, ax = plt.subplots(
        figsize=(5, 0.8 * (1 + 0.5 * (2 + include_sex + include_age + pollution)))
    )

    ## Manually track min and max x values for setting xlim
    xmin, xmax = 1.0, 1.0

    for histology, offset_value in zip(
        ["lung_adenocarcinoma", "lung_squamous_cell_carcinoma"], [-5, 5]
    ):
        cox_model = read_cox_model(source, histology, robust, pollution)
        colnames = [
            colname.replace(" * ", ":")
            for colname in regressor_colnames
            if (include_sex or "sex" not in colname)
            and (include_age or colname != "age_at_recruitment")
        ]
        cox_model_summary = cox_model.summary.loc[
            lambda df, colnames_=colnames: df.index.isin(colnames_)
        ]
        xmin = min(xmin, cox_model_summary["exp(coef) lower 95%"].min())
        xmax = max(xmax, cox_model_summary["exp(coef) upper 95%"].max())
        cox_model.plot(
            columns=colnames,
            hazard_ratios=True,
            ax=ax,
            label=get_histology_label(histology),
            c=get_histology_colour(histology),
            transform=ax.transData + get_histology_y_offset(offset_value, fig),
        )
    ax.set_xlim(xmin - 0.05 * (xmax - xmin), xmax + 0.05 * (xmax - xmin))
    y_expansion = (
        0.1 * (3 - include_age - include_sex) * (ax.get_ylim()[1] - ax.get_ylim()[0])
    )
    ax.set_ylim(ax.get_ylim()[0] - y_expansion, ax.get_ylim()[1] + y_expansion)
    ax.set_xticks(ax.get_xticks())
    ax.set_xticklabels(
        [
            f"{(100*float(label.get_text()) - 100):.1f}%"
            for label in ax.get_xticklabels()
        ]
    )
    ax.set_xlabel("Change in Hazard")

    assert "0.0%" in [label.get_text() for label in ax.xaxis.get_ticklabels()], str(
        [label.get_text() for label in ax.xaxis.get_ticklabels()]
    )
    zero_index = [
        index
        for index, label in enumerate(ax.xaxis.get_ticklabels())
        if label.get_text() == "0.0%"
    ][0]
    for index, label in enumerate(ax.xaxis.get_ticklabels()):
        if index % 2 != zero_index % 2:
            label.set_visible(False)
            # make other tick lines shorter
            ax.xaxis.get_ticklines()[2 * index].set_markersize(
                ax.xaxis.get_ticklines()[2 * index].get_markersize() / 2
            )

    ax.set_yticklabels(
        [
            label.get_text().replace("_", " ").replace("scaled ", "").title()
            for label in ax.get_yticklabels()
        ]
    )

    fig.tight_layout()
    save_dir = os.path.join(
        "plots", "cox_regression", "hazard_ratios", "pollution" * pollution, source
    )
    os.makedirs(save_dir, exist_ok=True)
    fig.savefig(
        os.path.join(
            save_dir,
            "_".join(
                [
                    x
                    for x, present in zip(
                        ["robust", "no_sex", "no_age"],
                        [robust, not include_sex, not include_age],
                    )
                    if present
                ]
                or ["default"]
            )
            + ".pdf",
        )
    )
    plt.close(fig)


def get_histology_y_offset(
    offset_value: float, fig: plt.Figure
) -> transforms.ScaledTranslation:
    """
    Get a y-axis offset for plotting hazard ratios for different histologies on the same
    axis.
    """
    return transforms.ScaledTranslation(0, offset_value / 72.0, fig.dpi_scale_trans)


def plot_baseline_hazard(source: str, robust: bool, cumulative: bool) -> None:
    fig, ax = plt.subplots(figsize=(4, 3))
    for histology in ["lung_adenocarcinoma", "lung_squamous_cell_carcinoma"]:
        cox_model = read_cox_model(source, histology, robust)
        baseline_hazard: pd.DataFrame = (
            cox_model.baseline_cumulative_hazard_
            if cumulative
            else cox_model.baseline_hazard_
        )
        ax.plot(
            baseline_hazard.index,
            baseline_hazard,
            label=get_histology_label(histology),
            color=get_histology_colour(histology),
        )
    ax.set_xlabel("Time from recruitment (years)")
    ax.set_ylabel(f"{'Cumulative ' if cumulative else ''}Baseline hazard")
    ax.legend()
    fig.tight_layout()
    save_dir = os.path.join("plots", "cox_regression", "baseline_hazard")
    os.makedirs(save_dir, exist_ok=True)
    fig.savefig(
        os.path.join(save_dir, f"{source}{'_cumulative' if cumulative else ''}.pdf")
    )
    plt.close(fig)


def get_histology_label(histology: str) -> str:
    return {"lung_adenocarcinoma": "LUAD", "lung_squamous_cell_carcinoma": "LUSC"}[
        histology
    ]


def plot_vif() -> None:
    vif_by_source = read_vif()
    fig, axes = plt.subplots(len(vif_by_source), 1, figsize=(5, 3.5))

    for i, source in enumerate(sorted(vif_by_source.keys(), reverse=True)):
        vif_by_regressor = vif_by_source[source]
        regressors = sorted(vif_by_regressor.keys(), reverse=True)
        axes[i].barh(
            regressors,
            [vif_by_regressor[key] for key in regressors],
            label=source,
            color=plt.get_cmap(get_dataset_cmap(source))(0.9),
        )
        axes[i].axvline(5, color="black", linestyle="--")
        if i < len(vif_by_source) - 1:
            axes[i].set_xticks([])
            axes[i].spines["bottom"].set_visible(False)
        else:
            axes[i].set_xlabel("Variance Inflation Factor")
        for side in ["top", "right"]:
            axes[i].spines[side].set_visible(False)

    fig.tight_layout()
    save_dir = os.path.join("plots", "cox_regression", "collinearity")
    os.makedirs(save_dir, exist_ok=True)
    fig.savefig(os.path.join(save_dir, "vif.pdf"))

    legend_handles = [
        plt.Rectangle(
            (0, 0),
            1,
            1,
            color=plt.get_cmap(get_dataset_cmap(source))(0.9),
            label=source,
        )
        for source in vif_by_source.keys()
    ]
    axes[0].legend(
        legend_handles,
        vif_by_source.keys(),
        bbox_to_anchor=(0.93, 0.5),
        loc="center right",
    )

    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "vif_with_legend.pdf"))
    plt.close(fig)

    legend_fig, legend_ax = plt.subplots(figsize=(1.7, 0.8))
    legend_ax.set_axis_off()
    legend_ax.legend(handles=legend_handles)
    legend_fig.tight_layout()
    legend_fig.savefig(f"{save_dir}/legend.pdf")
    plt.close(legend_fig)


def plot_condition() -> None:
    condition = read_condition()
    # this one's just source -> condition, so one h bar plot
    fig, ax = plt.subplots(figsize=(2.5, 3.5))
    sources = sorted(condition.keys())
    ax.barh(
        sources,
        [condition[key] for key in sources],
        color=[plt.get_cmap(get_dataset_cmap(source))(0.9) for source in sources],
    )
    ax.set_xlabel("Condition number")
    ax.set_yticklabels([])
    ax.tick_params(axis="y", length=0)
    for side in ["top", "right", "left"]:
        ax.spines[side].set_visible(False)

    fig.tight_layout()
    save_dir = os.path.join("plots", "cox_regression", "collinearity")
    os.makedirs(save_dir, exist_ok=True)
    fig.savefig(os.path.join(save_dir, "condition.pdf"))
    plt.close(fig)


def get_example_smoking_histories() -> list[tuple[int, int | None]]:
    return [(18, 52), (18, None)]


def plot_example_risk(
    source: str,
    robust: bool,
    include_age: bool,
    age_started_smoking: int,
    age_stopped_smoking: int | None,
) -> None:
    regressor_colnames = read_regressor_colnames()
    assert {"age_at_recruitment", "pack_years", "quit_years_at_recruitment"}.issubset(
        set(regressor_colnames)
    ), f"Regressor colnames: {regressor_colnames}"

    age_linspace = np.linspace(45, 70, 100)
    fig, ax = plt.subplots(figsize=(4, 3))

    for histology in ["lung_adenocarcinoma", "lung_squamous_cell_carcinoma"]:
        cox_model = read_cox_model(source, histology, robust)
        age_specific_hazard = np.exp(
            cox_model.params_["age_at_recruitment"] * age_linspace
        )
        smoking_specific_hazard = calculate_smoking_hazard(
            age_started_smoking, age_stopped_smoking, age_linspace, cox_model.params_
        )
        ax.plot(
            age_linspace,
            (
                (smoking_specific_hazard * age_specific_hazard)
                if include_age
                else smoking_specific_hazard
            )
            - 1,
            label=get_histology_label(histology),
            color=get_histology_colour(histology),
        )
    if age_stopped_smoking is not None:
        ax.axvline(
            age_stopped_smoking,
            color="black",
            linestyle="--",
            label="Stopped smoking",
        )
    ax.set_xlabel("Age")
    ax.set_ylabel("Yearly hazard" if include_age else "Hazard increase due to smoking")
    ax.set_ylim(0, None)
    ax.set_yticks(ax.get_yticks())
    ax.set_yticklabels(
        [
            f"{float(label.get_text().replace('−', '-')):.1%}"
            for label in ax.get_yticklabels()
        ]
    )
    fig.tight_layout()
    save_dir = os.path.join(
        "plots",
        "cox_regression",
        "smoking_risk" if not include_age else "smoking_risk_with_age",
        source,
        "robust" if robust else "standard",
    )
    os.makedirs(save_dir, exist_ok=True)
    fig.savefig(
        os.path.join(
            save_dir,
            f"{age_started_smoking}{'_' + str(age_stopped_smoking) if age_stopped_smoking is not None else ''}.pdf",
        )
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        os.path.join(
            save_dir,
            f"{age_started_smoking}{'_' + str(age_stopped_smoking) if age_stopped_smoking is not None else ''}_with_legend.pdf",
        )
    )
    plt.close(fig)


def calculate_smoking_hazard(
    age_started_smoking: int,
    age_stopped_smoking: int | None,
    age_linspace: np.ndarray,
    cox_model_coeffs: pd.Series,
) -> np.ndarray:
    return np.array(
        list(
            map(
                lambda t: np.exp(
                    (
                        0
                        if age_started_smoking is None
                        else (
                            ((t - age_started_smoking) * cox_model_coeffs["pack_years"])
                            if age_stopped_smoking is None or t < age_stopped_smoking
                            else (
                                (age_stopped_smoking - age_started_smoking)
                                * cox_model_coeffs["pack_years"]
                                + (t - age_stopped_smoking)
                                * cox_model_coeffs["quit_years_at_recruitment"]
                            )
                        )
                    )
                ),
                age_linspace,
            )
        )
    )


if __name__ == "__main__":
    for source_ in ["UK Biobank", "PLCO", "Combined"]:
        for robust_ in [False, True]:
            for include_sex_, include_age_ in [
                (True, True),
                (False, False),
                (False, True),
            ]:
                plot_hazard_ratios(source_, robust_, include_sex_, include_age_)
                if source_ == "UK Biobank":
                    plot_hazard_ratios(
                        source_, robust_, include_sex_, include_age_, pollution=True
                    )
            for cumulative_ in [False, True]:
                plot_baseline_hazard(source_, robust_, cumulative_)
            for (
                age_started_smoking_,
                age_stopped_smoking_,
            ) in get_example_smoking_histories():
                for include_age_ in [True, False]:
                    plot_example_risk(
                        source=source_,
                        robust=robust_,
                        include_age=include_age_,
                        age_started_smoking=age_started_smoking_,
                        age_stopped_smoking=age_stopped_smoking_,
                    )
    plot_vif()
    plot_condition()
