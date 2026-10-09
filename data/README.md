# Data access, provenance, and redistribution policy

This directory documents the PB-MMAE inputs without redistributing data whose licences, permissions, or access conditions prohibit public release. Do not commit raw grids, licensed layers, derived rasters, trained checkpoints, personal information, or credentials.

## Canonical 14-channel analytical stack

The frozen channel order is defined in `src/pbmmae/config.py` and must be retained for a paper-reproduction run.

| Family | Channels |
| --- | --- |
| Magnetic | `RTE_TMI` |
| Gravity | `CBG`, `CBG_RES`, `HGM` |
| Radiometric | `K`, `eTh`, `eU` |
| Thermal | `CPD` |
| Structural | `MAG_LD`, `GRAV_LD`, `DEM_LD`, `ID` |
| Terrain | `DEM`, `SLOPE` |

## Source access and redistribution

- **Airborne magnetic and radiometric data:** obtain authorised NGSA products through the relevant NGSA access route. These source grids are not redistributed here.
- **Complete Bouguer gravity anomaly:** obtain the WGM2012 product from its official provider and comply with its cited terms of use.
- **Terrain elevation:** obtain the SRTM 1-arc-second DEM from the official USGS distribution route and cite the product in derivative work.
- **Geology, geothermal-prospectivity, and warm-spring information:** these layers are withheld external evaluation evidence. Use only versions that you are authorised to access and redistribute.
- **Derived CPD, lineament, residual, and harmonised products:** generate locally from authorised inputs with the documented workflow. They are not bundled in this repository.

## Local layout

```text
data/
  raw/        Original authorised source files; never commit
  interim/    Temporary conversions; never commit
  processed/  Harmonised analysis-ready arrays; do not commit without clearance
  external/   Withheld evaluation and reference layers; do not commit without clearance
```

## Reproduction boundary

A lawful reproduction requires the authorised local input tree referenced by `paths.project_root` in `configs/paper1.yaml`. The public code and configuration allow inspection of the analysis design and data-free software checks; numerical replication of the manuscript results requires the original lawful inputs.
