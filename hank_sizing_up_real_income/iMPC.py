#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 2 16:52:07 2025

@author: Sodik Umurzakov
"""

import numpy as np
import os
import matplotlib.pyplot as plt
import sequence_jacobian as sj

# Local functions
import models.baseline_model as functions
from plots_functions import plot_intertemporal_mpcs, plot_mpc_columns_scaled
# Calibrated parameters
import calib_params.mpc_calibration as params

# Set structural parameters
calibrated_parameters = params.calibrated_parameters


# Create models - RA, TA and HA
het_agent_household = functions.households.add_hetinputs([functions.create_grids, 
                                                          functions.individual_income])

RA_model = sj.create_model([functions.UIP, functions.aggregate_income, functions.household_represent_agent, 
                            functions.domestic_demand, functions.foreign_demand, 
                            functions.balance_of_payment, functions.phillips_curve_wage, functions.taylor_rule, 
                            functions.market_clearing_cond], name="RA model")

TA_model = sj.create_model([functions.UIP, functions.aggregate_income, functions.household_two_agent_inc,
                            functions.domestic_demand, functions.foreign_demand, 
                            functions.balance_of_payment,functions.phillips_curve_wage, functions.taylor_rule, 
                            functions.market_clearing_cond], name="TA model")


HA_model = sj.create_model([functions.UIP, functions.aggregate_income, het_agent_household, 
                            functions.domestic_demand, functions.foreign_demand, 
                            functions.balance_of_payment,  functions.phillips_curve_wage, functions.taylor_rule,
                            functions.market_clearing_cond], name="HA Model")


calibrated_parameters['r_expected'] = calibrated_parameters['r_star']
calibrated_parameters['beta'] = 0.958
steady_states_ha = HA_model.steady_state(calibrated_parameters, dissolve=['UIP'])
steady_states_ra = RA_model.steady_state(calibrated_parameters, dissolve=['household_represent_agent', 'UIP'])
steady_states_ta = TA_model.steady_state(calibrated_parameters, dissolve=['household_two_agent_inc', 'UIP'])

quarters = 30
mpc_results = functions.compute_mpc_for_models(functions.household_represent_agent, functions.household_two_agent_inc, HA_model, steady_states_ra, steady_states_ta, steady_states_ha)

# Create the q' vector for iMPC of each model
q = (np.array([calibrated_parameters['beta'] ** i for i in range(quarters)]))

# Complete Markets
# converting float to int
iMPC_ra_cm = mpc_results['RA']['M']
iMPC_ra_cm[np.abs(iMPC_ra_cm) < 1e-15] = 0

# Create a 30x30 identity matrix
identity_matrix = np.eye(quarters)
# Set the first column to zeros
identity_matrix[:, 0] = 0

#from page A-17 of Auclert et al. (2021), revisited Aug 2024
iMPC_ta_cm = calibrated_parameters['lambd'] * identity_matrix

# Calibrate the parameters such that to recieve iMPC of 0.10
calibrated_parameters['min_a'] = 0.5
calibrated_parameters['max_a'] = 5
calibrated_parameters['n_e'] = 30
calibrated_parameters['n_a'] = 30
calibrated_parameters['markup_ss'] = 1.02

e_grid_mpc, Pi_pmc, a_grid_mpc = functions.create_grids(calibrated_parameters['rho_e'], calibrated_parameters['sd_e'], 
                       calibrated_parameters['n_e'], calibrated_parameters['min_a'], 
                       calibrated_parameters['max_a'], calibrated_parameters['n_a'])

mps_ha = mpc_results['HA']['M']
diag_D = np.diag(Pi_pmc)
diag_D_1 = np.diag(diag_D)
a_grid_mps_diag = np.diag(a_grid_mpc)
M_tilde = mps_ha @ diag_D_1 @ a_grid_mps_diag

# iMPC_ha_cm = (1/calibrated_parameters['markup_ss'])*mpc_results['HA']['M']+ (1/calibrated_parameters['markup_ss'])*M_tilde
iMPC_ha_cm = M_tilde

# Incomplete markets
# Initialize the matrix
calibrated_parameters['beta']=0.9899
iMPC_ra_im = np.zeros((quarters, quarters))
# Fill the matrix with the pattern
for i in range(quarters):
    for j in range(quarters):
        if j >= i:
            #from page A-14 of Auclert et al. (2021), revisited Aug 2024
            iMPC_ra_im[i, j] = (1 - calibrated_parameters['beta']) * (calibrated_parameters['beta'] ** (j - i))

iMPC_ra_im = np.tile(iMPC_ra_im[0, :], (quarters, 1))
calibrated_parameters['lambd'] = 0.093
iMPC_ta_im = (1-calibrated_parameters['lambd'])*iMPC_ra_im + calibrated_parameters['lambd']* np.eye(30)
# iMPC_ha_im = (1/calibrated_parameters['markup_ss'])*mpc_results['HA']['M'] + ((1-1/calibrated_parameters['markup_ss']) * (mpc_results['HA']['Mr']))*q
iMPC_ha_im = mpc_results['HA']['M']
iMPC_ha_im_r = mpc_results['HA']['Mr']

# Plot results
# Figure 1: Intertemporal MPCs (first column of M) in six calibrated models of 
# Auclert et al. (2021), revisited Aug 2024
plot_intertemporal_mpcs(iMPC_ra_cm[:, 0], iMPC_ta_cm[:, 0], iMPC_ha_cm[:, 0],
                        iMPC_ra_im[:, 0], iMPC_ta_im[:, 0], iMPC_ha_im[:, 0])

# Save the plot in the folder with results
output_path = os.path.join("results/iMPC_results", "iMPC_1st_col_M.eps")
plt.savefig(output_path, dpi=300, bbox_inches="tight")
# Display the plot
plt.show()



# Figure A.1: Intertemporal MPCs (columns of M) in six calibrated models of 
# Auclert et al. (2021), revisited Aug 2024

plot_mpc_columns_scaled(iMPC_ra_cm, iMPC_ta_cm, iMPC_ha_cm, iMPC_ra_im, iMPC_ta_im, iMPC_ha_im)

output_path = os.path.join("results/iMPC_results", "iMPC_cols_M.eps")
plt.savefig(output_path, dpi=300, bbox_inches="tight")
# Display the plot
plt.show()