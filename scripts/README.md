# Workflow scripts

`legacy/` contains the frozen scripts used to produce the PB-MMAE paper
results. Their only path change is replacement of the former personal Windows
root with the `PBMMAE_PROJECT_ROOT` environment variable. Run them through
`pbmmae run` rather than directly, so stages execute in dependency order.

`assemble_k9_geographic_labels.py` is a small bridge added to the package. It
joins saved k=9 consensus arrays to same-order patch-centre metadata, validates
the one-to-one Cell_ID correspondence, and supplies the alignment stage.

Scripts write only to ignored `03_outputs/` locations. Do not add restricted
source data, model checkpoints, or credentials.
