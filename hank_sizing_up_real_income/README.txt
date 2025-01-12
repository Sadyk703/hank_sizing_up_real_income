
Author: Sodik Umurzakov

Intro

This repository contains Python scripts to replicate the simulation results from Auclert et al. (2021), revisited in August 2024. Below is a guide to the primary files and their purposes:

Usage

1. Begin by running baseline_main.py to replicate the benchmark results.
2. Use iMPC.py to replicate plots from the paper and analyze intertemporal MPCs.
3. Run quant_main.py to explore extended model features, including Phillips curves and currency mismatch dynamics.



Main files

1. baseline_main.py
   This script replicates the simulation results for the benchmark models: RA, TA and HA models under both complete and incomplete markets. It integrates structural parameter calibrations and saves simulation outputs in the results folder.

2. iMPC.py
   This file focuses on calibrating intertemporal MPC (iMPC) for six models. It specifically computes values for the first column and subsequent columns of the M matrix.

3. quant_main.py
   This script extends the benchmark models by incorporating additional features:\
   - Phillips curves: Includes domestic price inflation, imported goods inflation, and inflation for home goods perceived by foreigners.\
   - Delayed substitution and consumption dynamics: Captures consumption responses over time.\
   - Currency mismatch: Models balance sheet effects stemming from exchange rate fluctuations.

Supporting files and folders\

1. calib_params
   Contains calibration tables based on the Mexican devaluation case study (Burstein and Gopinath, 2015). These tables are used to parameterize all models.

2. models
   Includes all model functions linked in the main scripts. These functions replicate closely with the structures and methods detailed in the paper.

3. results
   This folder stores the output of simulations for each model block.