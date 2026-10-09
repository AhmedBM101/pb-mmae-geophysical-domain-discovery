import os
# ============================================================
# Paper 1 — Final External-Validation Synthesis
# LARGE-FONT VERSION
# ============================================================

import os

os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"

from pathlib import Path
import textwrap

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(os.environ["PBMMAE_PROJECT_ROOT"]).expanduser().resolve()

EXTERNAL_ROOT = (
    PROJECT_ROOT /
    "03_outputs" /
    "external_validation"
)

GEOLOGY_ROOT = (
    EXTERNAL_ROOT /
    "geology_domain_validation"
)

GPI_ROOT = (
    EXTERNAL_ROOT /
    "GPI_domain_validation"
)

SPRING_ROOT = (
    EXTERNAL_ROOT /
    "warm_spring_domain_validation"
)

OUTPUT_ROOT = (
    EXTERNAL_ROOT /
    "final_synthesis"
)

OUTPUT_ROOT.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 2. INPUT FILES
# ============================================================

DOMAIN_MASTER = (
    EXTERNAL_ROOT /
    "Paper1_PBMMAE_domain_external_validation_MASTER.csv"
)

SPRING_MASTER = (
    EXTERNAL_ROOT /
    "Paper1_warm_spring_external_validation_MASTER.csv"
)

GEOLOGY_SUMMARY = (
    GEOLOGY_ROOT /
    "Paper1_geology_domain_association_summary.csv"
)

GEOLOGY_RESIDUALS = (
    GEOLOGY_ROOT /
    "Paper1_geology_domain_adjusted_standardized_residuals.csv"
)

GEOLOGY_ASSOCIATIONS = (
    GEOLOGY_ROOT /
    "Paper1_geology_domain_significant_residual_associations.csv"
)

GPI_SUMMARY = (
    GPI_ROOT /
    "Paper1_GPI_domain_global_test_summary.csv"
)

GPI_DESCRIPTIVE = (
    GPI_ROOT /
    "Paper1_GPI_by_domain_descriptive_statistics.csv"
)

GPI_PAIRWISE = (
    GPI_ROOT /
    "Paper1_GPI_pairwise_MannWhitney_Holm_CliffsDelta.csv"
)

SPRING_SUMMARY = (
    SPRING_ROOT /
    "Paper1_warm_spring_domain_validation_summary.csv"
)


# ============================================================
# 3. FILE CHECK
# ============================================================

required_files = [
    DOMAIN_MASTER,
    SPRING_MASTER,
    GEOLOGY_SUMMARY,
    GEOLOGY_RESIDUALS,
    GEOLOGY_ASSOCIATIONS,
    GPI_SUMMARY,
    GPI_DESCRIPTIVE,
    GPI_PAIRWISE,
    SPRING_SUMMARY
]


for path in required_files:

    if not path.exists():

        raise FileNotFoundError(
            path
        )


print("=" * 120)
print("PAPER 1 — FINAL EXTERNAL-VALIDATION SYNTHESIS")
print("=" * 120)


# ============================================================
# 4. LOAD DATA
# ============================================================

domain_df = pd.read_csv(
    DOMAIN_MASTER
)

spring_df = pd.read_csv(
    SPRING_MASTER
)

geology_summary = pd.read_csv(
    GEOLOGY_SUMMARY
)

geology_residuals = pd.read_csv(
    GEOLOGY_RESIDUALS,
    index_col=0
)

geology_assoc = pd.read_csv(
    GEOLOGY_ASSOCIATIONS
)

gpi_summary = pd.read_csv(
    GPI_SUMMARY
)

gpi_desc = pd.read_csv(
    GPI_DESCRIPTIVE
)

gpi_pairwise = pd.read_csv(
    GPI_PAIRWISE
)

spring_summary = pd.read_csv(
    SPRING_SUMMARY
)


DOMAIN_ORDER = [
    "M0",
    "M1",
    "M2",
    "M3"
]


geo = geology_summary.iloc[0]
gpi = gpi_summary.iloc[0]
spr = spring_summary.iloc[0]


# ============================================================
# 5. CONSOLIDATED VALIDATION TABLE
# ============================================================

consolidated_df = pd.DataFrame(
    [
        {
            "Validation_source":
                "Withheld geological units",

            "Role":
                "External geological consistency",

            "N":
                int(
                    geo["N_cells"]
                ),

            "Primary_statistic":
                "Cramer's V",

            "Statistic_value":
                float(
                    geo["Cramers_V"]
                ),

            "Test_statistic":
                (
                    f"Chi-square={geo['Chi_square']:.3f}; "
                    f"df={int(geo['Chi_square_df'])}"
                ),

            "Primary_p":
                float(
                    geo["Permutation_p"]
                ),

            "P_method":
                "10,000 label permutations",

            "Main_result":
                (
                    "Substantial geological association "
                    "between withheld lithostratigraphy "
                    "and frozen PB-MMAE domains."
                )
        },

        {
            "Validation_source":
                "Withheld continuous AHP-GPI",

            "Role":
                "Knowledge-driven benchmark comparison",

            "N":
                int(
                    gpi["N_cells"]
                ),

            "Primary_statistic":
                "Epsilon-squared",

            "Statistic_value":
                float(
                    gpi["Epsilon_squared"]
                ),

            "Test_statistic":
                (
                    f"Kruskal H={gpi['Kruskal_H']:.3f}; "
                    f"df={int(gpi['Kruskal_df'])}"
                ),

            "Primary_p":
                float(
                    gpi["Permutation_p"]
                ),

            "P_method":
                "10,000 label permutations",

            "Main_result":
                (
                    "GPI distributions differ among "
                    "latent domains; M0 approximately "
                    "M1 > M2 > M3."
                )
        },

        {
            "Validation_source":
                "Warm-spring manifestations",

            "Role":
                "Independent surface-manifestation comparison",

            "N":
                int(
                    spr["Total_springs"]
                ),

            "Primary_statistic":
                "Spatial translation concentration",

            "Statistic_value":
                float(
                    spr[
                        "Observed_concentration_fraction"
                    ]
                ),

            "Test_statistic":
                (
                    f"{int(spr['Observed_max_same_domain'])}/"
                    f"{int(spr['Resolved_springs'])} "
                    f"resolved springs in "
                    f"{spr['Observed_dominant_domain']}"
                ),

            "Primary_p":
                float(
                    spr[
                        "Spatial_translation_p_omnibus"
                    ]
                ),

            "P_method":
                "Exhaustive spatial translation",

            "Main_result":
                (
                    "Observed spring concentration is "
                    "descriptively strong but not "
                    "significant after preserving spring "
                    "spatial geometry."
                )
        }
    ]
)


consolidated_file = (
    OUTPUT_ROOT /
    "Paper1_FINAL_external_validation_summary.csv"
)


consolidated_df.to_csv(
    consolidated_file,
    index=False
)


# ============================================================
# 6. DOMAIN-LEVEL SYNTHESIS
# ============================================================

gpi_desc = (
    gpi_desc
    .set_index(
        "PBMMAE_domain"
    )
    .reindex(
        DOMAIN_ORDER
    )
)


spring_resolved = spring_df[
    spring_df[
        "PBMMAE_domain"
    ].notna()
].copy()


spring_counts = (

    spring_resolved[
        "PBMMAE_domain"
    ]

    .value_counts()

    .reindex(
        DOMAIN_ORDER,
        fill_value=0
    )
)


domain_counts = (

    domain_df[
        "PBMMAE_domain"
    ]

    .value_counts()

    .reindex(
        DOMAIN_ORDER,
        fill_value=0
    )
)


domain_synthesis_records = []


for domain in DOMAIN_ORDER:

    positive_geo = geology_assoc[
        (
            geology_assoc[
                "PBMMAE_domain"
            ] == domain
        )
        &
        (
            geology_assoc[
                "Adjusted_standardized_residual"
            ] > 1.96
        )
    ].copy()


    positive_geo = positive_geo.sort_values(
        "Adjusted_standardized_residual",
        ascending=False
    )


    top_units = positive_geo[
        "Geology_unit"
    ].head(
        3
    ).tolist()


    if len(
        top_units
    ) == 0:

        geology_text = (
            "No unit with adjusted residual > 1.96"
        )

    else:

        geology_text = "; ".join(
            top_units
        )


    domain_synthesis_records.append(
        {
            "PBMMAE_domain":
                domain,

            "Resolved_cells":
                int(
                    domain_counts[
                        domain
                    ]
                ),

            "Resolved_fraction":
                float(
                    domain_counts[
                        domain
                    ]
                    /
                    domain_counts.sum()
                ),

            "Mean_GPI":
                float(
                    gpi_desc.loc[
                        domain,
                        "Mean"
                    ]
                ),

            "Median_GPI":
                float(
                    gpi_desc.loc[
                        domain,
                        "Median"
                    ]
                ),

            "Resolved_warm_springs":
                int(
                    spring_counts[
                        domain
                    ]
                ),

            "Top_geological_enrichments":
                geology_text
        }
    )


domain_synthesis_df = pd.DataFrame(
    domain_synthesis_records
)


domain_synthesis_file = (
    OUTPUT_ROOT /
    "Paper1_FINAL_domain_external_synthesis.csv"
)


domain_synthesis_df.to_csv(
    domain_synthesis_file,
    index=False
)


# ============================================================
# 7. LARGE-FONT FIGURE CONFIGURATION
# ============================================================

plt.rcParams.update(
    {
        "font.size":
            18,

        "axes.titlesize":
            20,

        "axes.labelsize":
            18,

        "xtick.labelsize":
            16,

        "ytick.labelsize":
            16,

        "legend.fontsize":
            15,

        "figure.dpi":
            150,

        "savefig.dpi":
            600,

        "font.family":
            "DejaVu Sans"
    }
)


# Larger canvas to accommodate larger text.
fig = plt.figure(
    figsize=(
        22.0,
        15.0
    )
)


gs = fig.add_gridspec(
    nrows=2,
    ncols=2,

    width_ratios=[
        1.55,
        1.0
    ],

    height_ratios=[
        1.0,
        1.0
    ],

    hspace=0.42,
    wspace=0.36
)


ax_a = fig.add_subplot(
    gs[
        :,
        0
    ]
)

ax_b = fig.add_subplot(
    gs[
        0,
        1
    ]
)

ax_c = fig.add_subplot(
    gs[
        1,
        1
    ]
)


# ============================================================
# 8. PANEL A — GEOLOGICAL RESIDUAL HEATMAP
# ============================================================

geology_residuals = geology_residuals.reindex(
    DOMAIN_ORDER
)


geo_values = geology_residuals.to_numpy(
    dtype=float
)


max_abs = max(
    3.0,
    np.nanmax(
        np.abs(
            geo_values
        )
    )
)


im = ax_a.imshow(
    geo_values,
    aspect="auto",
    cmap="RdBu_r",
    vmin=-max_abs,
    vmax=max_abs
)


wrapped_labels = [
    "\n".join(
        textwrap.wrap(
            str(label),
            width=15
        )
    )
    for label
    in geology_residuals.columns
]


ax_a.set_xticks(
    np.arange(
        len(
            geology_residuals.columns
        )
    )
)


ax_a.set_xticklabels(
    wrapped_labels,
    rotation=90,
    ha="center",
    va="top",
    fontsize=14
)


ax_a.set_yticks(
    np.arange(
        len(
            DOMAIN_ORDER
        )
    )
)


ax_a.set_yticklabels(
    DOMAIN_ORDER,
    fontsize=17
)


ax_a.set_xlabel(
    "Withheld geological unit",
    fontsize=19,
    labelpad=16
)


ax_a.set_ylabel(
    "PB-MMAE domain",
    fontsize=19
)


ax_a.set_title(
    "(a) Geological correspondence: adjusted standardized residuals",
    loc="left",
    fontweight="bold",
    fontsize=20,
    pad=14
)


# Larger cell values.
for i in range(
    geo_values.shape[
        0
    ]
):

    for j in range(
        geo_values.shape[
            1
        ]
    ):

        value = geo_values[
            i,
            j
        ]

        marker = (
            "*"
            if abs(
                value
            ) >= 1.96
            else
            ""
        )


        ax_a.text(
            j,
            i,
            f"{value:.1f}{marker}",
            ha="center",
            va="center",
            fontsize=12.5
        )


cbar = fig.colorbar(
    im,
    ax=ax_a,
    fraction=0.032,
    pad=0.025
)


cbar.set_label(
    "Adjusted standardized residual",
    fontsize=17
)


cbar.ax.tick_params(
    labelsize=14
)


ax_a.text(
    0.0,
    -0.23,
    (
        f"Cramer's V = "
        f"{geo['Cramers_V']:.3f}; "
        f"permutation p = "
        f"{geo['Permutation_p']:.4f}. "
        "* |adjusted residual| ≥ 1.96."
    ),
    transform=ax_a.transAxes,
    ha="left",
    va="top",
    fontsize=15
)


# ============================================================
# 9. PANEL B — GPI DISTRIBUTIONS
# ============================================================

gpi_groups = [
    domain_df.loc[
        domain_df[
            "PBMMAE_domain"
        ] == domain,
        "GPI_continuous"
    ]
    .dropna()
    .to_numpy()

    for domain
    in DOMAIN_ORDER
]


ax_b.boxplot(
    gpi_groups,
    tick_labels=DOMAIN_ORDER,
    showfliers=False,
    widths=0.58
)


means = [
    np.mean(
        values
    )
    for values
    in gpi_groups
]


ax_b.scatter(
    np.arange(
        1,
        5
    ),
    means,
    marker="D",
    s=70,
    label="Mean",
    zorder=3
)


ax_b.set_xlabel(
    "PB-MMAE domain",
    fontsize=18
)


ax_b.set_ylabel(
    "Continuous AHP-GPI",
    fontsize=18
)


ax_b.set_title(
    "(b) Withheld GPI distributions",
    loc="left",
    fontweight="bold",
    fontsize=20,
    pad=14
)


ax_b.tick_params(
    axis="both",
    labelsize=16
)


ax_b.legend(
    frameon=False,
    loc="upper right",
    fontsize=15
)


ax_b.grid(
    axis="y",
    alpha=0.25
)


ax_b.text(
    0.03,
    0.05,
    (
        f"Kruskal-Wallis H = "
        f"{gpi['Kruskal_H']:.2f}\n"
        f"epsilon-squared = "
        f"{gpi['Epsilon_squared']:.3f}\n"
        f"permutation p = "
        f"{gpi['Permutation_p']:.4f}"
    ),
    transform=ax_b.transAxes,
    ha="left",
    va="bottom",
    fontsize=15
)


# ============================================================
# 10. PANEL C — SPRING OCCUPANCY VS BACKGROUND
# ============================================================

background_fraction = (
    domain_counts
    /
    domain_counts.sum()
)


spring_fraction = (
    spring_counts
    /
    spring_counts.sum()
)


x = np.arange(
    len(
        DOMAIN_ORDER
    )
)


width = 0.36


ax_c.bar(
    x - width / 2,
    background_fraction.values,
    width,
    label="Resolved domain background"
)


ax_c.bar(
    x + width / 2,
    spring_fraction.values,
    width,
    label="Resolved warm springs"
)


ax_c.set_xticks(
    x
)


ax_c.set_xticklabels(
    DOMAIN_ORDER,
    fontsize=16
)


ax_c.set_xlabel(
    "PB-MMAE domain",
    fontsize=18
)


ax_c.set_ylabel(
    "Fraction",
    fontsize=18
)


ax_c.set_ylim(
    0,
    1.08
)


ax_c.set_title(
    "(c) Warm-spring domain occupancy",
    loc="left",
    fontweight="bold",
    fontsize=20,
    pad=14
)


ax_c.tick_params(
    axis="y",
    labelsize=16
)


# Legend moved to upper right.
ax_c.legend(
    frameon=False,
    loc="upper right",
    fontsize=14
)


ax_c.grid(
    axis="y",
    alpha=0.25
)


# Annotation moved lower to avoid overlap.
ax_c.text(
    0.03,
    0.78,
    (
        f"5/5 resolved springs in M2\n"
        f"1 spring tied/unresolved\n"
        f"spatial-translation p = "
        f"{spr['Spatial_translation_p_omnibus']:.3f}"
    ),
    transform=ax_c.transAxes,
    ha="left",
    va="top",
    fontsize=15
)


# ============================================================
# 11. OVERALL FIGURE TITLE
# ============================================================

fig.suptitle(
    "External post hoc evaluation of the frozen four-domain PB-MMAE solution",
    fontsize=24,
    fontweight="bold",
    y=0.995
)


# ============================================================
# 12. SAVE FIGURE
# ============================================================

png_file = (
    OUTPUT_ROOT /
    "Paper1_Fig_external_validation_synthesis_600dpi.png"
)


pdf_file = (
    OUTPUT_ROOT /
    "Paper1_Fig_external_validation_synthesis.pdf"
)


fig.savefig(
    png_file,
    dpi=600,
    bbox_inches="tight"
)


fig.savefig(
    pdf_file,
    bbox_inches="tight"
)


plt.close(
    fig
)


# ============================================================
# 13. NUMERICAL LOCK
# ============================================================

summary_text = f"""
FINAL EXTERNAL-VALIDATION NUMERICAL LOCK
========================================

GEOLOGY
-------
N = {int(geo['N_cells'])}
Geological units = {int(geo['N_geology_units'])}
Chi-square = {geo['Chi_square']:.6f}
df = {int(geo['Chi_square_df'])}
Cramer's V = {geo['Cramers_V']:.6f}
Asymptotic p = {geo['Chi_square_asymptotic_p']:.8e}
10,000-permutation p = {geo['Permutation_p']:.8f}

GPI
---
N = {int(gpi['N_cells'])}
Kruskal-Wallis H = {gpi['Kruskal_H']:.6f}
df = {int(gpi['Kruskal_df'])}
epsilon-squared = {gpi['Epsilon_squared']:.6f}
Asymptotic p = {gpi['Kruskal_asymptotic_p']:.8e}
10,000-permutation p = {gpi['Permutation_p']:.8f}

Domain means:
M0 = {gpi_desc.loc['M0', 'Mean']:.6f}
M1 = {gpi_desc.loc['M1', 'Mean']:.6f}
M2 = {gpi_desc.loc['M2', 'Mean']:.6f}
M3 = {gpi_desc.loc['M3', 'Mean']:.6f}

WARM SPRINGS
------------
Total springs = {int(spr['Total_springs'])}
Resolved = {int(spr['Resolved_springs'])}
Tied/unresolved = {int(spr['Tied_springs'])}
Observed dominant domain = {spr['Observed_dominant_domain']}
Observed maximum same-domain count =
{int(spr['Observed_max_same_domain'])}/{int(spr['Resolved_springs'])}

Finite-population p, all five in M2 =
{spr['Exact_p_all5_M2']:.8f}

Finite-population omnibus p =
{spr['Exact_p_all5_same_any_domain']:.8f}

Primary spatial-translation p =
{spr['Spatial_translation_p_omnibus']:.8f}

M2-specific spatial-translation p =
{spr['Spatial_translation_p_M2_specific']:.8f}


INTERPRETATION LOCK
-------------------
1. Geology:
   The frozen PB-MMAE domains exhibit substantial post hoc
   correspondence with withheld lithostratigraphy.

2. GPI:
   PB-MMAE domains differ substantially in the withheld
   continuous AHP-GPI benchmark, with the approximate
   ordering M0 ≈ M1 > M2 > M3.

3. Warm springs:
   Five of five resolved springs occur in M2, but this
   concentration is not significant after preserving the
   observed spring geometry through spatial translation.

4. The GPI comparison is NOT independent geothermal ground
   truth because GPI and PB-MMAE share underlying geophysical
   information sources.

5. The warm-spring result must remain cautious because only
   six mapped manifestations are available.
"""


summary_txt_file = (
    OUTPUT_ROOT /
    "Paper1_FINAL_external_validation_numerical_lock.txt"
)


with open(
    summary_txt_file,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        summary_text
    )


# ============================================================
# 14. PRINT FINAL RESULTS
# ============================================================

print(
    "\n" +
    "=" * 120
)

print(
    "CONSOLIDATED EXTERNAL-VALIDATION RESULTS"
)

print(
    "=" * 120
)


print(
    consolidated_df[
        [
            "Validation_source",
            "Primary_statistic",
            "Statistic_value",
            "Primary_p",
            "P_method"
        ]
    ]
    .round(
        6
    )
    .to_string(
        index=False
    )
)


print(
    "\n" +
    "=" * 120
)

print(
    "DOMAIN-LEVEL EXTERNAL SYNTHESIS"
)

print(
    "=" * 120
)


print(
    domain_synthesis_df
    .round(
        4
    )
    .to_string(
        index=False
    )
)


print(
    "\nFiles created:"
)


for path in [
    consolidated_file,
    domain_synthesis_file,
    png_file,
    pdf_file,
    summary_txt_file
]:

    print(
        path
    )


print(
    "\nExternal-validation synthesis COMPLETE."
)

print(
    "=" * 120
)