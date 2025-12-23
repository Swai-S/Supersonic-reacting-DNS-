# Supersonic 2D Laminar DNS of Transverse Hydrogen Jet in Crossflow

# Case Description
This repository contains a supersonic 2D laminar reacting flow case of hydrogen jet injected transversely into an air crossflow, simulated using multicomponentFluid in OpenFOAM v13.  

# Software Requirements
- OpenFOAM v13 from openfoam.org
- Linux environment (tested on Ubuntu)

# Shock tube comparison
- Script: `scripts/compare_shocktube_exact.py`
- Input data:
  - `data/shocktube/fluid/line_0p007.xy`
  - `data/shocktube/multicomponent/line_0p007.xy`

## How to Run

# Run
./Allrun

## How to clean
./Allclean

# or manually:
rm -rf [0-9]* log.*
