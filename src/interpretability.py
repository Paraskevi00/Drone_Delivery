import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def get_feature_names(fitted_pipeline):
    """Return output feature names from the preprocessing step."""
    return fitted_pipeline.named_steps["preprocessing"].get_feature_names_out()


def assign_feature_group(feature_name: str) -> str:
    """Assign each transformed feature to a broader interpretable group."""
    if feature_name.startswith("landing_zone_"):
        return "Landing zone"
    if feature_name in ["route_length", "direction_changes"]:
        return "Route complexity"
    if feature_name == "distance_km":
        return "Distance"
    if feature_name == "payload_kg":
        return "Payload"
    if feature_name in ["wind_avg", "wind_gust"] or feature_name.startswith("wind_direction_"):
        return "Wind"
    if feature_name in ["mean_altitude", "altitude_changes"]:
        return "Altitude profile"
    if feature_name.startswith("drone_model_"):
        return "Drone model"
    if feature_name.startswith("operator_tag_"):
        return "Operator strategy"
    if feature_name == "slot_15min":
        return "Time slot"
    return "Other"


def logistic_regression_coefficients(fitted_pipeline) -> pd.DataFrame:
    """Create a coefficient and odds-ratio table for a fitted Logistic Regression pipeline."""
    feature_names = get_feature_names(fitted_pipeline)
    classifier = fitted_pipeline.named_steps["classifier"]

    coefficients = classifier.coef_[0]
    odds_ratios = np.exp(coefficients)

    coef_df = pd.DataFrame(
        {
            "feature": feature_names,
            "coefficient": coefficients,
            "odds_ratio": odds_ratios,
            "abs_coefficient": np.abs(coefficients),
        }
    )

    coef_df["feature_group"] = coef_df["feature"].apply(assign_feature_group)
    coef_df["is_landing_zone"] = coef_df["feature"].str.startswith("landing_zone_")

    return coef_df.sort_values("abs_coefficient", ascending=False)


def plot_top_logistic_coefficients(
    coef_df,
    n: int = 20,
    exclude_landing_zone: bool = True,
    save_path=None,
):
    """Plot strongest positive and negative coefficients.

    Positive coefficients increase the log-odds of on-time delivery.
    Negative coefficients decrease the log-odds of on-time delivery.
    """
    plot_source = coef_df.copy()

    if exclude_landing_zone:
        plot_source = plot_source[~plot_source["is_landing_zone"]].copy()

    top_positive = plot_source.sort_values("coefficient", ascending=False).head(n)
    top_negative = plot_source.sort_values("coefficient", ascending=True).head(n)

    plot_df = pd.concat([top_negative, top_positive]).drop_duplicates("feature")
    plot_df = plot_df.sort_values("coefficient")

    colors = np.where(plot_df["coefficient"] >= 0, "tab:green", "tab:red")

    fig, ax = plt.subplots(figsize=(9, max(6, 0.35 * len(plot_df))))
    ax.barh(plot_df["feature"], plot_df["coefficient"], color=colors)
    ax.axvline(0, color="black", linestyle="--", linewidth=1)

    ax.set_xlabel("Logistic Regression coefficient")
    title_suffix = "excluding landing-zone dummy variables" if exclude_landing_zone else "including all features"
    ax.set_title(f"Top positive and negative coefficients ({title_suffix})")

    ax.text(
        0.01,
        0.01,
        "Green: increases predicted on-time probability | Red: decreases predicted on-time probability",
        transform=ax.transAxes,
        fontsize=9,
        va="bottom",
    )

    plt.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def grouped_coefficient_importance(coef_df) -> pd.DataFrame:
    """Aggregate absolute coefficients by broader feature group.

    Mean and max absolute coefficients are more comparable than sums when groups have
    very different numbers of encoded columns.
    """
    grouped = (
        coef_df.groupby("feature_group")
        .agg(
            n_features=("feature", "count"),
            mean_abs_coefficient=("abs_coefficient", "mean"),
            max_abs_coefficient=("abs_coefficient", "max"),
        )
        .sort_values("mean_abs_coefficient", ascending=False)
        .reset_index()
    )

    return grouped


def plot_grouped_coefficient_importance(
    grouped_df,
    metric: str = "mean_abs_coefficient",
    save_path=None,
):
    """Plot grouped coefficient importance using mean or max absolute coefficient."""
    if metric not in grouped_df.columns:
        raise ValueError(f"{metric} is not available in grouped_df.")

    plot_df = grouped_df.sort_values(metric, ascending=True)

    fig, ax = plt.subplots(figsize=(8, max(4.5, 0.45 * len(plot_df))))
    ax.barh(plot_df["feature_group"], plot_df[metric])
    ax.set_xlabel(metric.replace("_", " "))
    ax.set_title("Grouped coefficient importance")
    plt.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()
