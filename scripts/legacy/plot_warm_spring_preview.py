import os
# =========================================================
# FIG. S3(c) — WARM-SPRING SPATIAL-TRANSLATION TEST
# Standalone script: run from Anaconda Prompt
# =========================================================

import faulthandler
faulthandler.enable()

from pathlib import Path
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg


def main():
    # -----------------------------------------------------
    # Paths
    # -----------------------------------------------------
    project = Path(os.environ["PBMMAE_PROJECT_ROOT"]).expanduser().resolve()

    spring_dir = (
        project
        / "03_outputs"
        / "external_validation"
        / "warm_spring_domain_validation"
    )

    output_dir = (
        project
        / "03_outputs"
        / "figures"
        / "FigS3_panels"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    dist_file = (
        spring_dir
        / "Paper1_warm_spring_translation_concentration_distribution.csv"
    )

    summary_file = (
        spring_dir
        / "Paper1_warm_spring_domain_validation_summary.csv"
    )

    # -----------------------------------------------------
    # Load saved results
    # -----------------------------------------------------
    print("1. Loading saved results...", flush=True)

    dist = pd.read_csv(dist_file)
    summary = pd.read_csv(summary_file)

    row = summary.iloc[0]

    observed = float(row["Observed_concentration_fraction"])
    stored_p = float(row["Spatial_translation_p_omnibus"])
    resolved = int(row["Resolved_springs"])
    tied = int(row["Tied_springs"])
    domain = str(row["Observed_dominant_domain"])

    observed_count = int(round(observed * resolved))

    # -----------------------------------------------------
    # Build the distribution used by the original test.
    #
    # The input CSV contains admissible non-original
    # translations. Include configurations with at least
    # as many resolved springs as observed.
    # -----------------------------------------------------
    null = dist.loc[
        dist["N_resolved"] >= resolved
    ].copy()

    if null.empty:
        raise ValueError("No admissible translations were found.")

    freq = (
        null.groupby(
            "Concentration_fraction",
            as_index=False
        )["Translation_count"]
        .sum()
        .sort_values("Concentration_fraction")
        .reset_index(drop=True)
    )

    # -----------------------------------------------------
    # Verify the empirical probability.
    # The added unit includes the original configuration.
    # -----------------------------------------------------
    n_translations = int(freq["Translation_count"].sum())

    n_extreme = int(
        freq.loc[
            freq["Concentration_fraction"] >= observed,
            "Translation_count"
        ].sum()
    )

    calculated_p = (n_extreme + 1) / (n_translations + 1)

    print("2. Statistical check:", flush=True)
    print(
        f"   Admissible non-original translations: {n_translations}",
        flush=True
    )
    print(
        "   Translations at least as concentrated as observed: "
        f"{n_extreme}",
        flush=True
    )
    print(f"   Resolved springs: {resolved}", flush=True)
    print(f"   Tied springs: {tied}", flush=True)
    print(f"   Observed concentration: {observed:.6f}", flush=True)
    print(f"   Recomputed p: {calculated_p:.12f}", flush=True)
    print(f"   Stored p: {stored_p:.12f}", flush=True)

    if not np.isclose(
        calculated_p,
        stored_p,
        rtol=0,
        atol=1e-6
    ):
        raise ValueError(
            "The plotted distribution does not reproduce the stored "
            "probability. Check that the distribution CSV contains "
            "all admissible non-original translations and matches "
            "the summary file."
        )

    # -----------------------------------------------------
    # Create figure with a noninteractive renderer
    # -----------------------------------------------------
    print("3. Creating figure...", flush=True)

    fig = Figure(figsize=(8.6, 6.8), dpi=150)
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)

    ax.bar(
        freq["Concentration_fraction"],
        freq["Translation_count"],
        width=0.025,
        color="#4477AA",
        edgecolor="white",
        linewidth=0.6,
        zorder=3
    )

    # Dashed line marks the observed concentration.
    ax.axvline(
        observed,
        color="#AA3333",
        linestyle="--",
        linewidth=1.8,
        zorder=4
    )

    ymax = float(freq["Translation_count"].max())
    ax.set_ylim(0, ymax * 1.40)

    # -----------------------------------------------------
    # Observed-statistic annotation
    # Opaque background keeps the dashed line behind text.
    # -----------------------------------------------------
    ax.text(
        0.98,
        0.96,
        (
            f"Observed: {observed_count}/{resolved} "
            f"resolved springs in {domain}\n"
            f"Concentration = {observed:.2f}\n"
            f"Spatial translation p = {calculated_p:.3f}"
        ),
        transform=ax.transAxes,
        fontsize=11.2,
        fontweight="bold",
        ha="right",
        va="top",
        bbox=dict(
            facecolor="white",
            edgecolor="0.75",
            alpha=1.0,
            pad=5
        ),
        zorder=5
    )

    # -----------------------------------------------------
    # Panel label
    # -----------------------------------------------------
    ax.text(
        0.015,
        0.985,
        "(c)",
        transform=ax.transAxes,
        fontsize=16,
        fontweight="bold",
        ha="left",
        va="top"
    )

    # -----------------------------------------------------
    # Axes
    # -----------------------------------------------------
    ax.set_xlabel(
        "Maximum same-domain concentration fraction",
        fontsize=14,
        fontweight="bold"
    )

    ax.set_ylabel(
        "Number of spatial translations",
        fontsize=14,
        fontweight="bold"
    )

    ticks = freq["Concentration_fraction"].to_numpy()

    ax.set_xticks(ticks)
    ax.set_xticklabels(
        [f"{value:.2f}" for value in ticks],
        fontsize=11,
        fontweight="bold",
        rotation=45,
        ha="right"
    )

    ax.set_xlim(
        min(ticks.min(), observed) - 0.04,
        max(ticks.max(), observed) + 0.04
    )

    ax.tick_params(
        axis="y",
        labelsize=12,
        width=1.2,
        length=5
    )

    for tick in ax.get_yticklabels():
        tick.set_fontweight("bold")

    # -----------------------------------------------------
    # Grid and frame
    # -----------------------------------------------------
    ax.set_axisbelow(True)

    ax.grid(
        axis="y",
        linestyle="--",
        alpha=0.22
    )

    for spine in ax.spines.values():
        spine.set_linewidth(1.0)

    print("4. Applying layout and rendering...", flush=True)

    fig.tight_layout()
    canvas.draw()

    # -----------------------------------------------------
    # Save publication-quality figures
    # -----------------------------------------------------
    print("5. Saving PNG and PDF...", flush=True)

    stem = "Paper1_FigS3c_WarmSpring_SpatialTranslation"

    for extension in ("png", "pdf"):
        output_path = output_dir / f"{stem}.{extension}"

        fig.savefig(
            output_path,
            dpi=600,
            bbox_inches="tight",
            facecolor="white"
        )

        print(f"   Saved: {output_path}", flush=True)

    fig.clear()

    print("\nExport completed successfully.", flush=True)


if __name__ == "__main__":
    main()
