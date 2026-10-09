import os
# ============================================================
# Paper 1 — FINAL QUANTITATIVE RESULTS LOCK
#
# Purpose
# -------
# Consolidate the frozen numerical results from:
#
#   1. Data / spatial CV
#   2. Baseline models
#   3. Primary 13x13 PB-MMAE
#   4. 9x9 spatial-scale sensitivity
#   5. Ablations A, B and C
#   6. Four-domain hierarchical consensus
#   7. Physical characterization
#   8. External post hoc evaluation
#
# IMPORTANT
# ---------
# This script DOES NOT train, cluster, tune or re-estimate
# anything. It records the locked results already obtained.
# ============================================================

from pathlib import Path

import pandas as pd


# ============================================================
# 1. PATHS
# ============================================================

PROJECT_ROOT = Path(os.environ["PBMMAE_PROJECT_ROOT"]).expanduser().resolve()

OUTPUT_ROOT = (
    PROJECT_ROOT /
    "03_outputs" /
    "final_results_lock"
)

OUTPUT_ROOT.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 2. MASTER DATA / CV LOCK
# ============================================================

data_cv_records = [

    {
        "Section": "Master grid",
        "Metric": "CRS",
        "Value": "EPSG:32632",
        "Notes": "Immutable Paper 1 analytical grid"
    },

    {
        "Section": "Master grid",
        "Metric": "Grid dimensions",
        "Value": "69 x 42",
        "Notes": "2898 total cells"
    },

    {
        "Section": "Master grid",
        "Metric": "Cell size",
        "Value": "4000 m",
        "Notes": "4-km analytical resolution"
    },

    {
        "Section": "Data cube",
        "Metric": "Primary input channels",
        "Value": 14,
        "Notes": "Six physics families"
    },

    {
        "Section": "Data cube",
        "Metric": "Common valid cells",
        "Value": 2718,
        "Notes": "93.79% of master grid"
    },

    {
        "Section": "Spatial CV",
        "Metric": "Number of folds",
        "Value": 5,
        "Notes": "Contiguous west-to-east folds"
    },

    {
        "Section": "Spatial CV",
        "Metric": "Primary patch size",
        "Value": "13 x 13",
        "Notes": "52 x 52 km spatial context"
    },

    {
        "Section": "Spatial CV",
        "Metric": "Primary global patch centers",
        "Value": 1477,
        "Notes": "13x13 valid patch centers"
    },

    {
        "Section": "Spatial CV",
        "Metric": "9x9 global patch centers",
        "Value": 1857,
        "Notes": "Spatial-scale sensitivity"
    },

    {
        "Section": "Spatial CV",
        "Metric": "17x17 global patch centers",
        "Value": 1129,
        "Notes": "Excluded from primary analysis because Fold 3 retained only 112 training patches"
    }
]


data_cv_df = pd.DataFrame(
    data_cv_records
)


# ============================================================
# 3. BASELINE MODEL LOCK
# ============================================================

baseline_records = [

    {
        "Method": "PCA + k-means",
        "Preferred_k": 3,
        "Init_ARI": 0.9308,
        "Init_NMI": 0.9344,
        "Silhouette": 0.1350,
        "DBI": 2.2330,
        "CH": None,
        "Validation_empty_clusters": "F4 and F5",
        "Notes": "Highest initialization stability among baselines"
    },

    {
        "Method": "NMF + k-means",
        "Preferred_k": 2,
        "Init_ARI": 0.7219,
        "Init_NMI": 0.7552,
        "Silhouette": 0.5651,
        "DBI": 0.6797,
        "CH": 828.09,
        "Validation_empty_clusters": "None",
        "Notes": "Strongest internal separation metrics"
    },

    {
        "Method": "SOM",
        "Preferred_k": 2,
        "Init_ARI": 0.8626,
        "Init_NMI": 0.8035,
        "Silhouette": 0.2439,
        "DBI": 2.0343,
        "CH": None,
        "Validation_empty_clusters": "F5",
        "Notes": "Moderate stability; QE validation/train ratio = 1.2718"
    },

    {
        "Method": "Conventional CAE + k-means",
        "Preferred_k": 2,
        "Init_ARI": 0.4187,
        "Init_NMI": 0.3917,
        "Silhouette": 0.1732,
        "DBI": 2.8052,
        "CH": None,
        "Validation_empty_clusters": "None at 9x9",
        "Notes": "Best CAE scale = 9x9; train MSE 0.0887, validation MSE 0.6014, ratio 6.77"
    }
]


baseline_df = pd.DataFrame(
    baseline_records
)


# ============================================================
# 4. PRIMARY PB-MMAE 13x13 LOCK
# ============================================================

primary_records = [

    {
        "Metric": "Architecture parameters",
        "Value": 206190,
        "Notes": "14-channel PB-MMAE"
    },

    {
        "Metric": "Spatial masking fraction",
        "Value": 0.402367,
        "Notes": "68/169 positions masked"
    },

    {
        "Metric": "Whole-modality masking probability",
        "Value": 0.30,
        "Notes": "At most one whole family masked per sample"
    },

    {
        "Metric": "Mean fused validation/train SD ratio",
        "Value": 0.6432,
        "Notes": "Outer folds F1,F2,F4,F5"
    },

    {
        "Metric": "SD of fused validation/train SD ratio",
        "Value": 0.0536,
        "Notes": "Outer-fold variation"
    },

    {
        "Metric": "Fine clustering k",
        "Value": 9,
        "Notes": "Development-selected fine latent partition"
    },

    {
        "Metric": "Mean ARI",
        "Value": 0.4857,
        "Notes": "Outer-fold k=9 consensus"
    },

    {
        "Metric": "Mean NMI",
        "Value": 0.6515,
        "Notes": "Outer-fold k=9 consensus"
    },

    {
        "Metric": "PAC",
        "Value": 0.2324,
        "Notes": "Outer-fold k=9 consensus"
    },

    {
        "Metric": "Within-consensus score",
        "Value": 0.6065,
        "Notes": "Outer-fold k=9 consensus"
    },

    {
        "Metric": "Silhouette",
        "Value": 0.1446,
        "Notes": "Outer-fold k=9"
    },

    {
        "Metric": "DBI",
        "Value": 2.1060,
        "Notes": "Outer-fold k=9"
    },

    {
        "Metric": "Mean validation occupancy",
        "Value": 0.6111,
        "Notes": "Fraction of k=9 clusters represented in validation"
    },

    {
        "Metric": "Total empty validation clusters",
        "Value": 14,
        "Notes": "Across four outer folds"
    },

    {
        "Metric": "Mean validation dominant-cluster fraction",
        "Value": 0.4656,
        "Notes": "Outer-fold k=9"
    },

    {
        "Metric": "Worst validation dominant-cluster fraction",
        "Value": 0.5327,
        "Notes": "Outer-fold k=9"
    },

    {
        "Metric": "Fine global mean agreement",
        "Value": 0.653,
        "Notes": "Aligned k=9 outer-fold predictions"
    },

    {
        "Metric": "Fine global complete agreement",
        "Value": 0.227,
        "Notes": "Approximately 22.7%"
    },

    {
        "Metric": "Fine tied cells",
        "Value": 440,
        "Notes": "k=9 fine consensus"
    }
]


primary_df = pd.DataFrame(
    primary_records
)


# ============================================================
# 5. 9x9 SPATIAL-SCALE SENSITIVITY
# ============================================================

sensitivity_records = [

    {
        "Metric": "Mean fused validation/train SD ratio",
        "Value": 0.8094
    },

    {
        "Metric": "SD fused validation/train SD ratio",
        "Value": 0.0420
    },

    {
        "Metric": "Mean ARI",
        "Value": 0.4678
    },

    {
        "Metric": "Mean NMI",
        "Value": 0.6315
    },

    {
        "Metric": "PAC",
        "Value": 0.2399
    },

    {
        "Metric": "Within-consensus score",
        "Value": 0.5874
    },

    {
        "Metric": "Silhouette",
        "Value": 0.0902
    },

    {
        "Metric": "DBI",
        "Value": 2.5691
    },

    {
        "Metric": "Validation occupancy",
        "Value": 0.8889
    },

    {
        "Metric": "Total empty validation clusters",
        "Value": 4
    },

    {
        "Metric": "Mean validation dominant-cluster fraction",
        "Value": 0.4109
    },

    {
        "Metric": "Worst validation dominant-cluster fraction",
        "Value": 0.4735
    },

    {
        "Metric": "Fine global mean agreement",
        "Value": 0.651
    },

    {
        "Metric": "Fine complete agreement fraction",
        "Value": 0.211
    },

    {
        "Metric": "Fine tied fraction",
        "Value": 0.257
    },

    {
        "Metric": "4-meta mean agreement",
        "Value": 0.800
    },

    {
        "Metric": "4-meta full agreement",
        "Value": 0.496
    },

    {
        "Metric": "4-meta tie fraction",
        "Value": 0.108
    },

    {
        "Metric": "4-meta pairwise accuracy",
        "Value": 0.626
    }
]


sensitivity_df = pd.DataFrame(
    sensitivity_records
)


# ============================================================
# 6. ABLATION LOCK
# ============================================================

ablation_records = [

    {
        "Experiment": "Full PB-MMAE",
        "Representation_ratio": 0.6432,
        "ARI": 0.4857,
        "NMI": 0.6515,
        "PAC": 0.2324,
        "Within_consensus": 0.6065,
        "Silhouette": 0.1446,
        "DBI": 2.1060,
        "Validation_occupancy": 0.6111,
        "Empty_validation_clusters": 14,
        "Mean_val_dominance": 0.4656,
        "Interpretation": "Preferred frozen model"
    },

    {
        "Experiment": "Ablation A — no physics-balanced loss",
        "Representation_ratio": 0.6348,
        "ARI": 0.4751,
        "NMI": 0.6457,
        "PAC": 0.2354,
        "Within_consensus": 0.5928,
        "Silhouette": 0.1427,
        "DBI": 2.1190,
        "Validation_occupancy": 0.5833,
        "Empty_validation_clusters": 15,
        "Mean_val_dominance": 0.4583,
        "Interpretation": "Modest but consistent deterioration; retain physics-balanced loss"
    },

    {
        "Experiment": "Ablation B — no whole-modality masking",
        "Representation_ratio": 0.7796,
        "ARI": 0.4386,
        "NMI": 0.6165,
        "PAC": 0.2546,
        "Within_consensus": 0.5699,
        "Silhouette": 0.1307,
        "DBI": 2.2726,
        "Validation_occupancy": 0.5833,
        "Empty_validation_clusters": 15,
        "Mean_val_dominance": 0.6220,
        "Interpretation": "Higher representation transfer but poorer clustering stability/balance; retain whole-modality masking"
    },

    {
        "Experiment": "Ablation C — remove MAG_LD",
        "Representation_ratio": 0.6406,
        "ARI": 0.4770,
        "NMI": 0.6452,
        "PAC": 0.2338,
        "Within_consensus": 0.5940,
        "Silhouette": 0.1407,
        "DBI": 2.1220,
        "Validation_occupancy": 0.8611,
        "Empty_validation_clusters": 5,
        "Mean_val_dominance": 0.3882,
        "Interpretation": "Improves fine occupancy but weakens reproducibility of four-domain hierarchy; retain MAG_LD"
    }
]


ablation_df = pd.DataFrame(
    ablation_records
)


# ============================================================
# 7. FOUR-DOMAIN HIERARCHY LOCK
# ============================================================

hierarchy_records = [

    {
        "Configuration": "Primary 13x13",
        "Meta_domains": 4,
        "Mean_agreement": 0.823,
        "Full_agreement": 0.517,
        "Tie_fraction": 0.064,
        "Pairwise_accuracy": 0.627,
        "Notes": "Clear 4-to-5 stability break"
    },

    {
        "Configuration": "9x9 sensitivity",
        "Meta_domains": 4,
        "Mean_agreement": 0.800,
        "Full_agreement": 0.496,
        "Tie_fraction": 0.108,
        "Pairwise_accuracy": 0.626,
        "Notes": "Independent support for four-domain hierarchy"
    },

    {
        "Configuration": "Remove MAG_LD",
        "Meta_domains": 4,
        "Mean_agreement": 0.799,
        "Full_agreement": 0.477,
        "Tie_fraction": 0.115,
        "Pairwise_accuracy": 0.610,
        "Notes": "Weaker four-domain hierarchy"
    }
]


hierarchy_df = pd.DataFrame(
    hierarchy_records
)


# ============================================================
# 8. FINAL FOUR-DOMAIN CONSENSUS LOCK
# ============================================================

domain_consensus_records = [

    {
        "Domain": "M0",
        "Resolved_cells": 356,
        "Resolved_fraction": 0.2576,
        "Mean_consensus_agreement": 0.853,
        "Physical_signature": (
            "Low RTE-TMI; shallower CPD tendency; "
            "elevated MAG_LD and ID; lower terrain"
        )
    },

    {
        "Domain": "M1",
        "Resolved_cells": 157,
        "Resolved_fraction": 0.1136,
        "Mean_consensus_agreement": 0.795,
        "Physical_signature": (
            "Shallowest CPD tendency; K enriched; "
            "positive magnetic/gravity expression; "
            "lower structural density"
        )
    },

    {
        "Domain": "M2",
        "Resolved_cells": 697,
        "Resolved_fraction": 0.5043,
        "Mean_consensus_agreement": 0.868,
        "Physical_signature": (
            "Lower CBG; elevated GRAV_LD; modestly deeper CPD; "
            "moderately elevated terrain; broad background domain"
        )
    },

    {
        "Domain": "M3",
        "Resolved_cells": 172,
        "Resolved_fraction": 0.1245,
        "Mean_consensus_agreement": 0.834,
        "Physical_signature": (
            "Deep CPD; eTh/eU enriched; lower K; "
            "reduced DEM lineament density"
        )
    }
]


domain_consensus_df = pd.DataFrame(
    domain_consensus_records
)


consensus_global_records = [

    {
        "Metric": "13x13 patch centers",
        "Value": 1477
    },

    {
        "Metric": "Resolved four-domain cells",
        "Value": 1382
    },

    {
        "Metric": "Tied/unresolved cells",
        "Value": 95
    },

    {
        "Metric": "Tied fraction",
        "Value": 0.0643
    },

    {
        "Metric": "Mean consensus agreement",
        "Value": 0.8233
    },

    {
        "Metric": "Median consensus agreement",
        "Value": 1.0
    },

    {
        "Metric": "Mean model support",
        "Value": 2.85
    }
]


consensus_global_df = pd.DataFrame(
    consensus_global_records
)


# ============================================================
# 9. PHYSICAL-SEPARATION LOCK
#
# Epsilon-squared values from post hoc characterization
# of the frozen four-domain solution.
# ============================================================

physical_separation_records = [

    {
        "Variable": "CPD",
        "Epsilon_squared": 0.301,
        "Relative_importance": "Strongest"
    },

    {
        "Variable": "RTE-TMI",
        "Epsilon_squared": 0.221,
        "Relative_importance": "Strong"
    },

    {
        "Variable": "CBG",
        "Epsilon_squared": 0.172,
        "Relative_importance": "Strong"
    },

    {
        "Variable": "MAG_LD",
        "Epsilon_squared": 0.102,
        "Relative_importance": "Moderate"
    },

    {
        "Variable": "DEM",
        "Epsilon_squared": 0.102,
        "Relative_importance": "Moderate"
    }
]


physical_separation_df = pd.DataFrame(
    physical_separation_records
)


# ============================================================
# 10. EXTERNAL VALIDATION LOCK
# ============================================================

external_records = [

    {
        "Validation_source": "Geological units",
        "N": 1382,
        "Statistic": "Cramer's V",
        "Statistic_value": 0.395260,
        "Test": "Chi-square",
        "Test_statistic": 647.731201,
        "df": 39,
        "Permutation_p": 0.00009999,
        "Interpretation": (
            "Substantial post hoc geological correspondence "
            "with withheld lithostratigraphy"
        )
    },

    {
        "Validation_source": "Continuous AHP-GPI",
        "N": 1382,
        "Statistic": "Epsilon-squared",
        "Statistic_value": 0.144170,
        "Test": "Kruskal-Wallis H",
        "Test_statistic": 201.666491,
        "df": 3,
        "Permutation_p": 0.00009999,
        "Interpretation": (
            "Meaningful GPI differentiation; "
            "M0 approximately M1 > M2 > M3"
        )
    },

    {
        "Validation_source": "Warm springs",
        "N": 6,
        "Statistic": "Spatial concentration fraction",
        "Statistic_value": 1.0,
        "Test": "Spatial translation",
        "Test_statistic": None,
        "df": None,
        "Permutation_p": 0.17265193,
        "Interpretation": (
            "5/5 resolved springs in M2, but concentration "
            "is not spatially significant"
        )
    }
]


external_df = pd.DataFrame(
    external_records
)


# ============================================================
# 11. KEY MODEL DECISIONS
# ============================================================

decision_records = [

    {
        "Decision": "Primary spatial scale",
        "Frozen_choice": "13x13",
        "Reason": (
            "Pre-specified primary scale; 52-km context and "
            "more reproducible four-domain hierarchy"
        )
    },

    {
        "Decision": "Fine latent clustering",
        "Frozen_choice": "k=9",
        "Reason": (
            "Development-selected fine partition; retained "
            "as subdomain structure, not final basin taxonomy"
        )
    },

    {
        "Decision": "Basin-scale taxonomy",
        "Frozen_choice": "4 meta-domains",
        "Reason": (
            "Clear 4-to-5 reproducibility break and independent "
            "9x9 support"
        )
    },

    {
        "Decision": "Physics-balanced reconstruction loss",
        "Frozen_choice": "Retain",
        "Reason": (
            "Modest but consistent robustness benefit in ablation A"
        )
    },

    {
        "Decision": "Whole-modality masking",
        "Frozen_choice": "Retain, p=0.30",
        "Reason": (
            "Improves domain stability and balance despite lower "
            "latent variance transfer"
        )
    },

    {
        "Decision": "MAG_LD",
        "Frozen_choice": "Retain",
        "Reason": (
            "Improves reproducibility of the four-domain hierarchy "
            "despite lower fine-cluster geographic occupancy"
        )
    },

    {
        "Decision": "Geology in training",
        "Frozen_choice": "Excluded",
        "Reason": "Withheld for external post hoc interpretation"
    },

    {
        "Decision": "Warm springs in training",
        "Frozen_choice": "Excluded",
        "Reason": "Withheld independent manifestation dataset"
    },

    {
        "Decision": "AHP-GPI in training",
        "Frozen_choice": "Excluded",
        "Reason": (
            "Withheld benchmark; not treated as independent "
            "ground truth because of shared geophysical evidence"
        )
    }
]


decision_df = pd.DataFrame(
    decision_records
)


# ============================================================
# 12. WRITE CSV TABLES
# ============================================================

tables = {

    "Paper1_FINAL_lock_data_and_spatial_CV.csv":
        data_cv_df,

    "Paper1_FINAL_lock_baseline_models.csv":
        baseline_df,

    "Paper1_FINAL_lock_primary_PBMMAE_13x13.csv":
        primary_df,

    "Paper1_FINAL_lock_9x9_sensitivity.csv":
        sensitivity_df,

    "Paper1_FINAL_lock_ablations.csv":
        ablation_df,

    "Paper1_FINAL_lock_hierarchical_stability.csv":
        hierarchy_df,

    "Paper1_FINAL_lock_domain_consensus.csv":
        domain_consensus_df,

    "Paper1_FINAL_lock_consensus_global.csv":
        consensus_global_df,

    "Paper1_FINAL_lock_physical_separation.csv":
        physical_separation_df,

    "Paper1_FINAL_lock_external_validation.csv":
        external_df,

    "Paper1_FINAL_lock_model_decisions.csv":
        decision_df
}


for filename, dataframe in tables.items():

    output_file = (
        OUTPUT_ROOT /
        filename
    )

    dataframe.to_csv(
        output_file,
        index=False
    )


# ============================================================
# 13. COMPACT MANUSCRIPT RESULTS TABLE
# ============================================================

manuscript_summary_records = [

    {
        "Result_block": "Primary PB-MMAE representation",
        "Key_result": (
            "13x13 outer-fold mean VAL/TRAIN latent SD ratio"
        ),
        "Value": "0.6432 ± 0.0536"
    },

    {
        "Result_block": "Fine k=9 stability",
        "Key_result": "ARI / NMI / PAC",
        "Value": "0.4857 / 0.6515 / 0.2324"
    },

    {
        "Result_block": "Fine k=9 separation",
        "Key_result": "Silhouette / DBI",
        "Value": "0.1446 / 2.1060"
    },

    {
        "Result_block": "Four-domain hierarchy",
        "Key_result": (
            "Mean agreement / full agreement / tie fraction"
        ),
        "Value": "0.823 / 0.517 / 0.064"
    },

    {
        "Result_block": "Final consensus",
        "Key_result": (
            "Resolved / tied cells; mean agreement"
        ),
        "Value": "1382 / 95; 0.8233"
    },

    {
        "Result_block": "9x9 sensitivity",
        "Key_result": (
            "4-meta mean agreement / full agreement"
        ),
        "Value": "0.800 / 0.496"
    },

    {
        "Result_block": "Ablation A",
        "Key_result": "ARI / NMI",
        "Value": "0.4751 / 0.6457"
    },

    {
        "Result_block": "Ablation B",
        "Key_result": "ARI / NMI",
        "Value": "0.4386 / 0.6165"
    },

    {
        "Result_block": "Ablation C",
        "Key_result": (
            "4-meta mean agreement / full agreement"
        ),
        "Value": "0.799 / 0.477"
    },

    {
        "Result_block": "Strongest physical separator",
        "Key_result": "CPD epsilon-squared",
        "Value": "0.301"
    },

    {
        "Result_block": "Geological correspondence",
        "Key_result": "Cramer's V; permutation p",
        "Value": "0.3953; 0.0001"
    },

    {
        "Result_block": "GPI correspondence",
        "Key_result": "epsilon-squared; permutation p",
        "Value": "0.1442; 0.0001"
    },

    {
        "Result_block": "Warm-spring spatial test",
        "Key_result": (
            "5/5 resolved springs in M2; spatial p"
        ),
        "Value": "5/5; 0.1727"
    }
]


manuscript_summary_df = pd.DataFrame(
    manuscript_summary_records
)


manuscript_summary_file = (
    OUTPUT_ROOT /
    "Paper1_FINAL_manuscript_results_summary.csv"
)


manuscript_summary_df.to_csv(
    manuscript_summary_file,
    index=False
)


# ============================================================
# 14. TEXT NUMERICAL LOCK
# ============================================================

lock_text = """
======================================================================
PAPER 1 — FINAL QUANTITATIVE RESULTS LOCK
======================================================================

STUDY / DATA
------------
Master analytical grid:
    EPSG:32632
    69 x 42 cells
    4 km cell size
    2898 total cells

14-channel common-valid data:
    2718 cells
    93.79% master-grid coverage

Primary model scale:
    13 x 13 cells
    52 x 52 km spatial context

Primary valid patch centers:
    1477


PRIMARY PB-MMAE
---------------
Architecture:
    206,190 trainable parameters

Spatial mask:
    68 / 169 cells
    fraction = 0.402367

Whole-modality masking:
    p = 0.30

Outer-fold representation transfer:
    mean VAL/TRAIN SD ratio = 0.6432
    SD = 0.0536

Fine latent clustering:
    k = 9

Outer-fold fine-domain metrics:
    ARI = 0.4857
    NMI = 0.6515
    PAC = 0.2324
    within-consensus = 0.6065
    silhouette = 0.1446
    DBI = 2.1060
    validation occupancy = 0.6111
    total empty validation clusters = 14
    mean validation dominance = 0.4656

Aligned fine-domain consensus:
    mean agreement = 0.653
    complete agreement ≈ 22.7%
    tied cells = 440


FOUR-DOMAIN HIERARCHY
---------------------
Primary 13x13:
    mean agreement = 0.823
    full agreement = 0.517
    tie fraction = 0.064
    pairwise accuracy = 0.627

9x9 sensitivity:
    mean agreement = 0.800
    full agreement = 0.496
    tie fraction = 0.108
    pairwise accuracy = 0.626

Four domains are therefore retained as the basin-scale taxonomy.


FINAL FOUR-DOMAIN CONSENSUS
---------------------------
Patch centers = 1477

Resolved:
    1382

Tied/unresolved:
    95
    6.43%

Mean agreement:
    0.8233

Median agreement:
    1.000

Mean model support:
    2.85

Domain occupancy:
    M0 = 356 cells = 25.76%
    M1 = 157 cells = 11.36%
    M2 = 697 cells = 50.43%
    M3 = 172 cells = 12.45%


PHYSICAL CHARACTERIZATION
-------------------------
Strongest four-domain separators:

    CPD       epsilon² = 0.301
    RTE-TMI   epsilon² = 0.221
    CBG       epsilon² = 0.172
    MAG_LD    epsilon² = 0.102
    DEM       epsilon² = 0.102

Domain signatures:

M0:
    low RTE-TMI
    shallow CPD tendency
    elevated MAG_LD and ID
    lower terrain

M1:
    shallowest CPD tendency
    K enriched
    positive magnetic/gravity expression
    lower structural density

M2:
    lower CBG
    elevated GRAV_LD
    modestly deeper CPD
    broad background domain

M3:
    deep CPD
    eTh/eU enriched
    lower K
    reduced DEM lineament density


BASELINES
---------
PCA + k-means:
    k = 3
    initialization ARI = 0.9308
    initialization NMI = 0.9344
    silhouette = 0.1350
    DBI = 2.233
    validation empty clusters in F4 and F5

NMF + k-means:
    k = 2
    initialization ARI = 0.7219
    initialization NMI = 0.7552
    silhouette = 0.5651
    DBI = 0.6797
    CH = 828.09
    no empty validation clusters

SOM:
    k = 2
    initialization ARI = 0.8626
    initialization NMI = 0.8035
    silhouette = 0.2439
    DBI = 2.0343
    empty validation cluster in F5

Conventional CAE:
    preferred scale = 9x9
    initialization ARI = 0.4187
    initialization NMI = 0.3917
    train MSE = 0.0887
    validation MSE = 0.6014
    validation/train ratio = 6.77
    silhouette = 0.1732
    DBI = 2.8052


ABLATIONS
---------
A — remove physics-balanced loss:
    representation ratio = 0.6348
    ARI = 0.4751
    NMI = 0.6457
    PAC = 0.2354
    within-consensus = 0.5928
    silhouette = 0.1427
    DBI = 2.1190

Decision:
    retain physics-balanced loss


B — remove whole-modality masking:
    representation ratio = 0.7796
    ARI = 0.4386
    NMI = 0.6165
    PAC = 0.2546
    within-consensus = 0.5699
    silhouette = 0.1307
    DBI = 2.2726
    mean validation dominance = 0.6220

Decision:
    retain whole-modality masking


C — remove MAG_LD:
    representation ratio = 0.6406
    ARI = 0.4770
    NMI = 0.6452
    PAC = 0.2338
    within-consensus = 0.5940
    silhouette = 0.1407
    DBI = 2.1220
    validation occupancy = 0.8611

Four-meta-domain stability:
    mean agreement = 0.799
    full agreement = 0.477
    tie fraction = 0.115
    pairwise accuracy = 0.610

Decision:
    retain MAG_LD because the scientific endpoint is the
    reproducibility of the basin-scale domain hierarchy.


EXTERNAL POST HOC EVALUATION
----------------------------

Geology:
    N = 1382
    geological units = 14
    chi-square = 647.731201
    df = 39
    Cramer's V = 0.395260
    permutation p = 0.00009999

Interpretation:
    substantial geological correspondence with withheld
    lithostratigraphy


Continuous AHP-GPI:
    N = 1382
    Kruskal-Wallis H = 201.666491
    df = 3
    epsilon² = 0.144170
    permutation p = 0.00009999

Mean GPI:
    M0 = 3.055714
    M1 = 3.041905
    M2 = 2.799689
    M3 = 2.148924

Approximate ordering:
    M0 ≈ M1 > M2 > M3

IMPORTANT:
    GPI is a withheld knowledge-driven benchmark.
    It is not independent geothermal ground truth.


Warm springs:
    total = 6
    resolved = 5
    tied/unresolved = 1

Observed:
    5 / 5 resolved springs occur in M2

Simple finite-population probability:
    all five in M2 = 0.03239852

Primary spatial translation:
    p = 0.17265193

M2-specific spatial translation:
    p = 0.33563536

Interpretation:
    descriptive concentration in M2, but no significant
    warm-spring concentration after preserving spatial geometry.


FINAL SCIENTIFIC LOCK
---------------------
1. The preferred primary model is the 14-channel,
   physics-balanced PB-MMAE using 13x13 patches.

2. k=9 is retained as the fine latent subdomain partition.

3. Four reproducible meta-domains are retained as the
   basin-scale geophysical taxonomy.

4. The 9x9 experiment independently supports the
   four-domain hierarchy.

5. Physics-balanced loss, whole-modality masking and MAG_LD
   remain in the preferred architecture based on ablation
   evidence.

6. The four domains are physically differentiated primarily
   by thermal, magnetic, gravity and structural properties.

7. The domains exhibit substantial correspondence with
   withheld geology.

8. The domains show meaningful but non-independent
   correspondence with AHP-GPI.

9. Warm-spring clustering in M2 is descriptive and is not
   significant under the spatial-translation null.

10. PB-MMAE should therefore be presented as a label-free
    multi-physics domain-discovery framework, not as a
    supervised geothermal-prospectivity predictor.
======================================================================
"""


lock_txt_file = (
    OUTPUT_ROOT /
    "Paper1_FINAL_quantitative_results_lock.txt"
)


with open(
    lock_txt_file,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        lock_text
    )


# ============================================================
# 15. PRINT OUTPUTS
# ============================================================

print(
    "=" * 120
)

print(
    "PAPER 1 — FINAL QUANTITATIVE RESULTS LOCK"
)

print(
    "=" * 120
)


print(
    manuscript_summary_df.to_string(
        index=False
    )
)


print(
    "\nFiles written to:"
)

print(
    OUTPUT_ROOT
)


print(
    "\nCreated:"
)


for filename in tables.keys():

    print(
        "  ",
        filename
    )


print(
    "  ",
    manuscript_summary_file.name
)

print(
    "  ",
    lock_txt_file.name
)


print(
    "\nFINAL QUANTITATIVE RESULTS LOCK COMPLETE."
)

print(
    "=" * 120
)