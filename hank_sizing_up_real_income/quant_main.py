#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 11 14:43:22 2025

@author: Sodik Umurzakov
"""

import numpy as np
import os
import matplotlib.pyplot as plt
# Manual functions (local functions) 
import plots_functions as plots
import models.quant_model as mod
# SJ packages
import sequence_jacobian as sj

# Figure 8: Contractionary depreciations (Auclert et al. (2022), revisited
# Aug 2024)

# Quantitative model (section 5)
exogenous_quant=['ishock','rstar','Z']
unknowns_quant=['y','pi','w','r']
targets_quant=['goods_clearing','pires','real_wage','fisher']
blocks_quant = [mod.hh_HA, mod.hh_outputs_ds, mod.foreign_c_ds, mod.xrule, 
                mod.xrule_foreign, mod.rsimple, mod.income_quant, mod.longbonds, 
                mod.revaluation_quant, mod.profitcenters_quant, mod.UIP_quant, 
                mod.assets_market, mod.goods_market, mod.CA_quant, 
                mod.unions, mod.pi_to_P, mod.nkpc, mod.nkpc_X, mod.nkpc_I, 
                mod.cpi, mod.prices, mod.eq_quant, mod.taylor, mod.fiscal]
mod_quant = sj.create_model(blocks_quant, name="Quantitative Model")



# Import calibration for quantitative model
import calib_params.quant_calibration as q_calibration

ss = q_calibration.calibrate_ss()
T = 400

calib_list = {}
calib_list['baseline'] = {'markup_ss': 1.0434272960924276,'cbarF': 0,'alpha': 0.4,'zeta_e': 0}
calib_list['quant'] = {'markup_ss': 1.0406422322881439,'cbarF': 0.0854397736545839,'alpha': 0.3439472041842487,'zeta_e': -0.1955188498403125}
calib_list['theta_share'] = 0.9763


ss, G, shock, irf = {}, {}, {}, {}
m_list = ['realrate', 'taylor']

# Steady state
ss['taylor'] = q_calibration.calibrate_ss(calib = 'quant', calib_list = calib_list, eta = 4, gamma = 4, theta_share = calib_list['theta_share'], theta_w = 0.938, theta_X = 0.66, theta_p=0.66, phi_i = 0.8, phi_pinext = 1.5)
ss['realrate'] = q_calibration.calibrate_ss(calib = 'quant', calib_list = calib_list, eta = 4, gamma = 4, theta_share = calib_list['theta_share'], theta_w = 0.938, theta_X = 0.66, theta_p=0.66)

# Compute model jacobian
for m in m_list: 
    G[m] = mod_quant.solve_jacobian(ss[m], unknowns_quant, targets_quant, exogenous_quant, T=T)

# Shocks
shock, dQ = mod.rshock(0.01,0.85,ss['realrate'],T,'rstar',Q0=1)

# Compute IRF
va={1:'y', 2:'C', 3:'netexports',4:'nfa',5:'atw_n',6:'div_tot',7:'Q',8:'r'}
for m in m_list: 
    irf[m]={}
    for i,k in va.items():
        irf[m][k]=G[m][k]['rstar'] @ shock 

# Plot
# Figure 8: Contractionary depreciations (Auclert et al. (2022), revisited
# Aug 2024)
plots.plot_irf_comparison(
    irf=irf,  # Dictionary containing IRFs for 'taylor' and 'realrate' rules
    va=va,    # Dictionary mapping subplot indices to variable keys
    Tplot=31  # Number of quarters to plot
)

output_path = os.path.join("results/quant_model_results", "fig_8_contractionary_depr.eps")
plt.savefig(output_path, dpi=300, bbox_inches="tight")
# Display the plot
plt.show()

# Figure 9: Policies to deal with contractionary depreciations 
# (Auclert et al. (2022), revisited
# Aug 2024)


# When does the real income channel matter?
# Model steady state and jacobians
ss = q_calibration.calibrate_ss(calib = 'quant', calib_list = calib_list, eta = 4, gamma = 4, theta_share = calib_list['theta_share'], theta_w = 0.938, theta_X = 0.66, theta_p=0.66)
G_fighting = mod_quant.solve_jacobian(ss, unknowns_quant, targets_quant, exogenous_quant, T=T)
shock, dQ = mod.rshock(1,0.85,ss,T,'rstar',Q0=1)
J_yrstar = G_fighting['y']['rstar']
J_yr = G_fighting['y']['ishock']
J_Qrstar = G_fighting['Q']['rstar']
J_Qr = G_fighting['Q']['ishock']

# Fighting the depreciation
dr = - np.linalg.inv(J_Qr)@J_Qrstar

# Fighting the contraction
dr2 = - np.linalg.inv(J_yr)@J_yrstar

# Plot
plots.plot_irf_decomposition(
    shock=shock,
    dr=dr,
    dr2=dr2,
    J_Qrstar=J_Qrstar,
    J_Qr=J_Qr,
    J_yrstar=J_yrstar,
    J_yr=J_yr
)

output_path = os.path.join("results/quant_model_results", "fig_9_fighting_depr.eps")
plt.savefig(output_path, dpi=300, bbox_inches="tight")
# Display the plot
plt.show()


