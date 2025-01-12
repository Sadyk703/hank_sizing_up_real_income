#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 12:50:18 2024

@author: Sodik Umurzakov
"""

import numpy as np
import os
import matplotlib.pyplot as plt
import sequence_jacobian as sj

# Local functions
import models.baseline_model as functions
import calib_params.baseline_calibration as params
import plots_functions as plots


# Import calibrated parameters from baseline_calibration
calibrated_parameters_baseline = params.calibrated_parameters.copy()


# Setup RA, TA, HA models
het_agent_household = functions.households.add_hetinputs([functions.create_grids, 
                                                          functions.individual_income])

RA_model = sj.create_model([functions.UIP, functions.aggregate_income, functions.household_represent_agent, 
                            functions.domestic_demand, functions.foreign_demand, 
                            functions.balance_of_payment, functions.phillips_curve_wage, functions.taylor_rule, 
                            functions.market_clearing_cond], name="RA model")

TA_model = sj.create_model([functions.UIP, functions.aggregate_income, functions.household_two_agent,
                            functions.domestic_demand, functions.foreign_demand, 
                            functions.balance_of_payment,functions.phillips_curve_wage, functions.taylor_rule, 
                            functions.market_clearing_cond], name="TA model")


HA_model = sj.create_model([functions.UIP, functions.aggregate_income, het_agent_household, 
                            functions.domestic_demand, functions.foreign_demand, 
                            functions.balance_of_payment,  functions.phillips_curve_wage, functions.taylor_rule,
                            functions.market_clearing_cond], name="HA Model")


# Steady state for HA model
calibrated_parameters_baseline['r_expected'] = calibrated_parameters_baseline['r_star']
steady_states = HA_model.steady_state(calibrated_parameters_baseline, dissolve=['UIP'])

# Provide bounds on beta for the solver
unknowns_ss = {'beta': (0.85, 0.95)}
# Target for ss
targets_ss = ['asset']

steady_states = HA_model.solve_steady_state(calibrated_parameters_baseline, unknowns_ss, targets_ss, dissolve=['UIP'])

calibration_ra_model = calibrated_parameters_baseline.copy()
calibration_ra_model['r_expected'] = calibration_ra_model['r_star']
calibration_ra_model['beta'] = 0.965
calibration_ra_model['A'] = steady_states['A']

# define shock
quarters = 31
rho = 0.85
dr_star = 0.001 * rho ** np.arange(quarters) # Shocks to the foreign interest rate, $r^*$, interpreted as exchange rate shocks or capital flow shocks.
shock = {'r_star': dr_star}
unknowns_td = ['Y']
targets_td = ['goods']

# Set initial values for unknowns
unknowns_ra_ss = {'C': 1., 'A': 1.}
targets_ra_ss = {'budget_constraint':0., 'asset':0.}

# %%

# Figure 2: Effect of exchange rate shocks on output for various χ’s

# Define different values for trade elasticity (chi)
chi_values = [0.63, 0.312, 0.09]
irfs_list_ra_inc = []  # Store IRFs for each chi

# Figure 2: Effect of exchange rate shocks on output for various χ’s
for chi in chi_values:
    calibration_ra_model['gamma'] = chi  # Set trade elasticity
    calibration_ra_model['eta']= chi
    steady_state_inc = RA_model.solve_steady_state(
        calibration_ra_model, unknowns_ra_ss, targets_ra_ss,
        dissolve=['household_represent_agent', 'UIP']
        )
    
    # Compute IRFs for this calibration
    irfs_inc = RA_model.solve_impulse_linear(
        steady_state_inc, unknowns_td, targets_td,
        shock
    )
    irfs_list_ra_inc.append(irfs_inc)


# What change for TA model with output?
calibration_ta_model = calibrated_parameters_baseline.copy()
calibration_ta_model['r_expected'] = calibration_ta_model['r_star']
calibration_ta_model['beta'] = 0.965
calibration_ta_model['A'] = steady_states['A']
unknowns_ta_ss = {'C_RA': 1., 'A': 0.8}
targets_ta_ss = {'budget_constraint': 0., 'asset': 0.}

# Define different values for trade elasticity (chi)
chi_values_ta = [1.0, 0.4, 0.2]
irfs_list_ta = []  # Store IRFs for each chi

for chi in chi_values_ta:
    calibration_ta_model['gamma'] = chi  # Set trade elasticity
    steady_state_ta_cm = TA_model.solve_steady_state(
        calibration_ta_model, unknowns_ta_ss, targets_ta_ss,
        dissolve=['household_two_agent', 'UIP']
        )
    
    # Compute IRFs for this calibration
    # steady_state_ta_cm['chi']= chi
    irfs_ta_cm = TA_model.solve_impulse_linear(
        steady_state_ta_cm, unknowns_td, targets_td,
        shock
    )
    irfs_list_ta.append(irfs_ta_cm)


# What change for HA model with for output?
calibrated_parameters_baseline['beta'] = 0.965
calibrated_parameters_baseline['A'] = steady_states['A']

# Define different values for trade elasticity (chi)
chi_values_ha = [1.0, 0.2, 0.]
irfs_list_ha = []  # Store IRFs for each chi

for chi in chi_values_ha:
    calibrated_parameters_baseline['gamma'] = chi  # Set trade elasticity
    steady_state_ha_cm = HA_model.solve_steady_state(
        calibrated_parameters_baseline, unknowns_ss, targets_ss,
        dissolve=['UIP']# ['chi'=0.1]
    )
    
    # Compute IRFs for this calibration
    irfs_ha_cm = HA_model.solve_impulse_linear(
        steady_state_ha_cm, unknowns_td, targets_td,
        shock
    )
    irfs_list_ha.append(irfs_ha_cm)

# Figure 2: Effect of exchange rate shocks on output for various χ’s
# Plot
plots.irfs_plotting_figure2(
    irf_Q = dr_star,
    irfs_complete = [irfs_list_ta[0], irfs_list_ra_inc[2], irfs_list_ta[2], irfs_list_ha[1]],
    irfs_incomplete = [irfs_list_ta[0], irfs_list_ra_inc[2], irfs_list_ta[2], irfs_list_ha[2]],
    var='Y',
    labels=['$\\chi = 1$', '$\\chi = 0.1, RA$', '$\\chi = 0.1, TA$', '$\\chi = 0.1, HA$'],
    T_plot=31)

# Save the plot in the folder with results
output_path = os.path.join("results/baseline_model_results", "fig_2_exchange_rate_shocks.eps")
plt.savefig(output_path, dpi=300, bbox_inches="tight")
# Display the plot
plt.show()

# %%

# Figure 3: Exchange rate shock when χ = 1 and its transmission channels
# Set chi=1
chi_ra_cm = 1.0
calibration_ra_model['gamma'] = chi_ra_cm - calibration_ra_model['eta'] * (1 - calibration_ra_model['alpha'])

steady_state_ra = RA_model.solve_steady_state(
        calibration_ra_model, unknowns_ra_ss, targets_ra_ss,
        dissolve=['household_represent_agent', 'UIP']
        )
irfs_ra = RA_model.solve_impulse_linear(
        steady_state_ra, unknowns_td, targets_td,
        shock
    )

# Real income channel
irfs_PH_P_ra = RA_model.impulse_linear(
        steady_state_ra, 
        {'Z': calibration_ra_model['alpha']*(steady_state_ra['Y'] * irfs_ra['PH_P'])}
    )
irfs_PH_P_ra['Y'] = irfs_PH_P_ra['cF'] + irfs_PH_P_ra['cH'] + irfs_PH_P_ra['C']

# Exp. switching channel
irfs_Y_ra_cm = RA_model.impulse_linear(
    steady_state_ra, {'Y': chi_ra_cm*(irfs_ra['PF_P'] + irfs_ra['gdp'])})

# Multiplier channel
d_multiplier_Y_ra = RA_model.impulse_linear(
    steady_state_ra, {'Z': (1/steady_state_ra['markup_ss']
                )*(irfs_ra['Y'] * steady_state_ra['PH_P'])})
d_multiplier_Y_ra['Y'] = d_multiplier_Y_ra['cF'] + d_multiplier_Y_ra['cH'] + d_multiplier_Y_ra['C']

# Total
irfs_Y_ra_cm['Total'] = irfs_Y_ra_cm['Y'] + irfs_PH_P_ra['Y'] + d_multiplier_Y_ra['Y']




# For HA model
chi_ha_im = 1.0

calibrated_parameters_baseline['gamma'] = chi_ha_im - calibrated_parameters_baseline['eta'] * (1 - calibrated_parameters_baseline['alpha'])

steady_state_ha = HA_model.solve_steady_state(
        calibrated_parameters_baseline, unknowns_ss, targets_ss,
        dissolve=['UIP']
        )
irfs_ha = HA_model.solve_impulse_linear(
        steady_state_ha, unknowns_td, targets_td,
        shock
    )


calibrated_parameters_baseline['alpha'] = 0.301
# Real income channel
irfs_PH_P_ha = HA_model.impulse_linear(
        steady_state_ha, 
        {'Z': (1-calibrated_parameters_baseline['alpha'])*(1/calibrated_parameters_baseline['markup_ss'])*(steady_state_ha['Y'] * irfs_ha['PH_P'])}
    )
irfs_PH_P_ha['Y'] = irfs_PH_P_ha['cF'] + irfs_PH_P_ha['cH'] #+ irfs_PH_P_ha['C']


# Exp. switching channel
irfs_Y_ha_im = HA_model.impulse_linear(
    steady_state_ha, {'Y': chi_ha_im*(irfs_ha['PF_P'] + irfs_ha['gdp'])})

# Multiplier channel
d_multiplier_Y_ha = HA_model.impulse_linear(
    steady_state_ha, {'Z': (1-calibrated_parameters_baseline['alpha'])*(1/steady_state_ha['markup_ss']
                )*(irfs_ha['Y'] * steady_state_ha['PH_P'])})
d_multiplier_Y_ha['Y']= (d_multiplier_Y_ha['cF'] + d_multiplier_Y_ha['cH'])# *0.85

# Total
irfs_Y_ha_im['Total'] = irfs_Y_ha_im['Y'] + irfs_PH_P_ha['Y'] + d_multiplier_Y_ha['Y']


# Plot
plots.irfs_decom_ra_ha(
    list_irfs_ra=[irfs_Y_ra_cm['Total'], irfs_PH_P_ra['Y'], irfs_Y_ra_cm['Y'], d_multiplier_Y_ra['Y']],
    list_irfs_ha=[irfs_Y_ha_im['Total'], irfs_PH_P_ha['Y'], irfs_Y_ha_im['Y'], d_multiplier_Y_ha['Y']],
    # var='Y',
    labels=['Total', 'Real income channel', 
            'Exp. switching channel', 'Multiplier channel'], T_plot=31)

# Save the plot in the folder with results
output_path = os.path.join("results/baseline_model_results", "fig_3_exchange_rate_sh_chi_1.eps")
plt.savefig(output_path, dpi=300, bbox_inches="tight")
# Display the plot
plt.show()

# %%


# Figure 6: The effects of monetary policy in termns of baseline models
# RA-CM and HA-IM
# monetary policy shock
shock_mp = {'r_expected': -0.001 * 0.85**np.arange(quarters)}

calibration_ra_model['eta'] = 0.5
calibration_ra_model['alpha'] = 0.345

# Set values for chi = 2-alpha = 1.6 and 0.5
chi_values_ra_r = [2-calibration_ra_model['alpha'], 0.5]
irfs_list_ra_r = []  # Store IRFs for each chi
steady_state_list_ra_r = []

for chi in chi_values_ra_r:
    # Set trade elasticity
    calibration_ra_model['gamma'] = chi - calibration_ra_model['eta'] * (1 - calibration_ra_model['alpha']
                                                      )
    steady_state_ra_r = RA_model.solve_steady_state(
        calibration_ra_model, unknowns_ra_ss, targets_ra_ss,
        dissolve=[ 'household_represent_agent', 'UIP']
    )
    steady_state_list_ra_r.append(steady_state_ra_r)
    # Compute IRFs for this calibration
    irfs_ra_r = RA_model.solve_impulse_linear(
        steady_state_ra_r, unknowns_td, targets_td,
        shock_mp
    )
    irfs_list_ra_r.append(irfs_ra_r)    


irfs_PH_P_list_ra = []
irfs_Y_list_ra = []
irfs_r_list_ra = []

# Loop through the steady_state_list_ha_r and irfs_list_ha_r
for i in range(len(steady_state_list_ra_r)):
    # where PF_P = Q
    # real income channel   
    hh_irfs_PH_P = RA_model.impulse_linear(
        steady_state_list_ra_r[i], 
        {'Z': calibration_ra_model['alpha']*(steady_state_list_ra_r[i]['Y'] * irfs_list_ra_r[i]['PH_P'])}
    )
    # Exp. switching channel -(steady_state_list_ha_r[i]['chi']*irfs_list_ha_r[i]['PH_P']), (1/1-calibrated_parameters_baseline['alpha'])*steady_state_list_ha_r[i]['chi']*irfs_list_ha_r[i]['Y'] *
    hh_irfs_Y = RA_model.impulse_linear(
        steady_state_list_ra_r[i], 
        {'PH_P': steady_state_list_ra_r[i]['chi']*irfs_list_ra_r[i]['PH_P']})
    # Interest rate channel 
    hh_irfs_r = RA_model.impulse_linear(
        steady_state_list_ra_r[i], 
        {'r': (1-calibration_ra_model['alpha'])*irfs_list_ra_r[i]['r']})
    
    # Append results to lists
    irfs_PH_P_list_ra.append(hh_irfs_PH_P)
    irfs_Y_list_ra.append(hh_irfs_Y)
    irfs_r_list_ra.append(hh_irfs_r)
    # From market clearing conditions to derive Y
    irfs_PH_P_list_ra[i]['Y'] = irfs_PH_P_list_ra[i]['C'] + irfs_PH_P_list_ra[i]['cH']
    irfs_Y_list_ra[i]['Y'] = -(irfs_Y_list_ra[i]['PH_P'] + irfs_Y_list_ra[i]['gdp'])
    irfs_r_list_ra[i]['Y'] = irfs_r_list_ra[i]['C'] + irfs_r_list_ra[i]['cH']

# Compute total irf for RA model
n = len(irfs_PH_P_list_ra)
irfs_total_list_ra = [0, 1]
  
# Cumulative effect (total)
for i in range(n):
    irfs_total_list_ra[i] = (
        irfs_PH_P_list_ra[i]['Y'] + 
        irfs_Y_list_ra[i]['Y'] + 
        irfs_r_list_ra[i]['Y']
    )


# For HA model
calibrated_parameters_baseline['eta'] = 0.5
calibrated_parameters_baseline['alpha'] = 0.4

chi_values_ha_r = [2-calibrated_parameters_baseline['alpha'], 0.5]
irfs_list_ha_r = []  # Store IRFs for each chi
steady_state_list_ha_r = []

for chi in chi_values_ha_r:
    # Set trade elasticity
    calibrated_parameters_baseline['gamma'] = chi - calibrated_parameters_baseline['eta'] * (1 - calibrated_parameters_baseline['alpha']
                                                      )
    steady_state_ha_r = HA_model.solve_steady_state(
        calibrated_parameters_baseline, unknowns_ss, targets_ss,
        dissolve=['UIP']# ['chi'=0.1]
    )
    steady_state_list_ha_r.append(steady_state_ha_r)
    # Compute IRFs for this calibration
    irfs_ha_r = HA_model.solve_impulse_linear(
        steady_state_ha_r, unknowns_td, targets_td,
        shock_mp
    )
    irfs_list_ha_r.append(irfs_ha_r)    


irfs_PH_P_list_ha = []
irfs_Y_list_ha = []
irfs_r_list_ha = []


# Loop through the steady_state_list_ha_r and irfs_list_ha_r
for i in range(len(steady_state_list_ha_r)):
    # where PH_P = Q
    # real income channel 
    hh_irfs_PH_P = HA_model.impulse_linear(
        steady_state_list_ha_r[i], 
        {'Z': calibrated_parameters_baseline['alpha']*(steady_state_list_ha_r[i]['Y'] * irfs_list_ha_r[i]['PH_P'])}
    )
    # Exp. switching channel -(steady_state_list_ha_r[i]['chi']*irfs_list_ha_r[i]['PH_P']), (1/1-calibrated_parameters_baseline['alpha'])*steady_state_list_ha_r[i]['chi']*irfs_list_ha_r[i]['Y'] *
    hh_irfs_Y = HA_model.impulse_linear(
        steady_state_list_ha_r[i], 
        {'PH_P': steady_state_list_ha_r[i]['chi']*irfs_list_ha_r[i]['PH_P']})
    # Interest rate channel
    hh_irfs_r = HA_model.impulse_linear(
        steady_state_list_ha_r[i], 
        {'r': irfs_list_ha_r[i]['r']})
    
    # Append results to lists
    irfs_PH_P_list_ha.append(hh_irfs_PH_P)
    irfs_Y_list_ha.append(hh_irfs_Y)
    irfs_r_list_ha.append(hh_irfs_r)
    # From market clearing conditions to derive Y
    irfs_PH_P_list_ha[i]['Y'] = irfs_PH_P_list_ha[i]['C'] + irfs_PH_P_list_ha[i]['cH']
    irfs_Y_list_ha[i]['Y'] = -(irfs_Y_list_ha[i]['PH_P'] + irfs_Y_list_ha[i]['gdp'])
    irfs_r_list_ha[i]['Y'] = irfs_r_list_ha[i]['C'] + irfs_r_list_ha[i]['cH']
    # irfs_r_list_ha[i]['Y'] = irfs_r_list_ha[i]['cF'] + irfs_r_list_ha[i]['cH']
# Compute total irf for HA model
n = len(irfs_PH_P_list_ha)
irfs_total_list_ha = [0, 1]  

# Cumulative effect (total)
for i in range(n):
    irfs_total_list_ha[i] = (
        irfs_PH_P_list_ha[i]['Y'] + 
        irfs_Y_list_ha[i]['Y'] + 
        irfs_r_list_ha[i]['Y']
    )

# Figure 6: The effects of monetary policy
plots.irfs_decom_ra_ha_r_new(
    list_irfs_ra_chi0=[irfs_total_list_ra[0], irfs_PH_P_list_ra[0]['Y'], irfs_Y_list_ra[0]['Y'], irfs_r_list_ra[0]['Y']],
    list_irfs_ha_chi0 = [irfs_total_list_ra[0], irfs_PH_P_list_ha[0]['Y'], irfs_Y_list_ha[0]['Y'], irfs_r_list_ha[0]['Y']],
    list_irfs_ra_chi1=[irfs_total_list_ra[1], irfs_PH_P_list_ra[1]['Y'], irfs_Y_list_ra[1]['Y'], irfs_r_list_ra[1]['Y']],
    list_irfs_ha_chi1 = [irfs_total_list_ha[1], irfs_PH_P_list_ha[1]['Y'], irfs_Y_list_ha[1]['Y'], irfs_r_list_ha[1]['Y']],
    labels=['Total', 'Real income channel', 
            'Exp. switching channel', 'Interest rate channel'], T_plot=31)

# Save the plot in the folder with results
output_path = os.path.join("results/baseline_model_results", "fig_6_mp_shock.eps")
plt.savefig(output_path, dpi=300, bbox_inches="tight")
# Display the plot
plt.show()
