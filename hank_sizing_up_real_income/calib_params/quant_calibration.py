#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 03 15:58:52 2025

@author: Sodik Umurzakov
This function created by using the tutorials from NBER HANK workshop 2022 
and by communication with the authors of the paper.
"""

import numpy as np
import scipy.optimize as opt
np.seterr(divide='ignore', invalid='ignore')

# Manual functions (local functions) 
import models.quant_model as mod

# SJ packages
import sequence_jacobian as sj
from sequence_jacobian.utilities import discretize

calib = False

def calibrate_ss(calib = '', calib_list = [], eta = 1, gamma=1, eis = 1, alpha = 0.4, alphastar=0.4, markup_ss= 1.031, r=0.01, 
                  sigma_e = 0.8830957229473172, rho_e = 0.9118856622848671, nE = 7, 
                  cbarF = 0, zeta_e = 0, 
                  eps_dcp = 1, pcX_home = 1, theta_X = 1e-6, 
                  theta_share = 1e-6, theta_w = 0.938, theta_p = 1e-6, theta_I = 1e-6, 
                  realrate = 1, phi_piHH = 0, phi_pi = 0, phi_pinext = 1, phi_i = 0,
                  f_FY = 0, BbarY = 0, rho_B = 0.8, lbda = 1, D = 18, foreign_owned = 0,
                  beta_min=0.90, beta_max=0.99, amin = 0, verbose=True):
    '''

    Calibrate steady-state parameters for a quant HA model.

    Parameters:
    -----------
    calib : Calibration mode. Default is ''.
    calib_list : Dictionary of calibration options. Default is None.
    eta, gamma, eis, alpha, alphastar : Elasticity, substitution, and preference parameters.
    markup_ss : Steady-state markup.
    r : Real interest rate.
    sigma_e, rho_e : Standard deviation and persistence of idiosyncratic income shocks.
    nE : Number of income grid points.
    cbarF, zeta_e, eps_dcp, pcX_home : Model parameters.
    theta_X, theta_share, theta_w, theta_p, theta_I : Calvo parameters.
    realrate, phi_piHH, phi_pi, phi_pinext, phi_i : Policy rule parameters.
    f_FY, BbarY, rho_B, lbda, D, foreign_owned : Fiscal and external balance parameters.
    beta_min, beta_max : Bounds on discount factor.
    amin : Minimum asset level.
    verbose : Display verbose output. Default is True.

    Returns:
    --------
    ss : Steady-state calibration results.

    '''
    
    # Structural parameters
    ss = {'r': r,                         # real interest rate
          'eis': eis,                     # EIS
          'gamma': gamma,                 # Elasticity of foreign demand
          'eta': eta,                     # Elasticity of substitution between goods
          'alpha': alpha,                 # Home bias - consumption share on foreign good
          'alphastar': alphastar,         # Home bias abroad - consumption share on foreign good
          'frisch': 1/2,                  # labor supply elasticity
          'cbarF': cbarF,                 # substistance level on foreign good (necessity)
          'zeta_e': zeta_e,               # degree of gamma heterogeneity
          'phi_piHH': phi_piHH,           # Taylor rule parameter on PPI
          'phi_pi': phi_pi,               # Taylor rule parameter on CPI
          'phi_pinext': phi_pinext,       # Taylor rule parameter on future CPI
          'phi_i': phi_i,                 # Taylor rule parameter for inertia
          'realrate': realrate,           # Real rate rule or Taylor rule
          'markup_ss': markup_ss,         # Firms markup
          'theta_w': theta_w,             # Calvo parameter for wages
          'theta_p': theta_p,             # Calvo parameter for prices
          'theta_I': theta_I,             # Calvo parameter for imports
          'theta_X': theta_X,             # Calvo parameter for exports
          'theta_share': theta_share,     # Calvo parameter for consumption shares
          'lamba': eta/(1-alpha),         # Other parameter for consumption shares
          'lamba_F': gamma/alphastar,     # Other parameter for consumption shares
          'nfa': 0,                       # steady state nfa
          'f_FY': f_FY,                   # real assets invested in foreign assets as a share of quarterly gdp
          'foreign_owned': foreign_owned, # are firms owned by foreigners?
          'BbarY': BbarY,                 # Gross government debt held in foreign assets as a share of quarterly gdp
          'rho_B': rho_B,                 # AR(1) coefficient on public debt
          'lbda': lbda,                   # Progressivity of taxation: 1 is lump-sum; 0 is proportional to labor income
          'eps_dcp':eps_dcp,              # degree of pcp-dcp (1 if pcp; 0 if dcp)
          'pcX_home':pcX_home}            # Home profit center or not

    # Select calibration
    if calib != '':
        for vv in ['markup_ss','cbarF','alpha','zeta_e']:
            ss[vv] = calib_list[calib][vv]

    # Rescale standard deviation of income for non-homothetic model to make sure that everyone can afford the subsistance level and the cross-sectional deviation of income is the same
    ss.update({'nA' : 150, 'sigma_e': sigma_e, 'rho_e': rho_e, 'amin': amin, 'amax': 4*100, 'nE': nE})
    e, pi, _ = discretize.markov_rouwenhorst(rho=rho_e, sigma=sigma_e, N=nE)                                  # note: here sigma_e is the standard deviation of logs (not of the innovation)
    sigma_ref = np.sqrt(pi@(e**2)-(pi@e)**2)
    def res(sigma):
        e, pi, _ = discretize.markov_rouwenhorst(rho=rho_e, sigma=sigma, N=nE)
        e = e*(1-ss['markup_ss']*ss['cbarF']) + ss['markup_ss']*ss['cbarF']
        return sigma_ref - np.sqrt(pi@(e**2)-(pi@e)**2)
    sigma_e_rescaled = opt.brentq(res, 0.1, 1.5, xtol=1e-12, rtol = 1e-12)

    # Computational parameters
    ss['a_grid'] = discretize.agrid(amax=ss['amax'], n=ss['nA'], amin=ss['amin'])
    ss['e_grid'], ss['pi_e'], ss['Pi'] = discretize.markov_rouwenhorst(rho=ss['rho_e'], sigma=sigma_e_rescaled, N=ss['nE'])
    ss['e_grid'] = ss['e_grid']*(1-ss['markup_ss']*ss['cbarF']) + ss['markup_ss']*ss['cbarF']               # adjust grids for non-homotheticity
    ss.update({'beta_min': beta_min, 'beta_max': beta_max})
    ss['n_beta'] = 1
    ss['n_exog'] = ss['n_beta'] * nE

    # Normalizations
    ss.update({'Q': 1, 'Z': 1, 'y': 1, 'rpost_shock': 0, 'ishock': 0, 'Transfer': 0, 'piw': 0, 'P': 1, 'B': 0})
    ss['M'] = ss['n_beta']*np.tile(ss['e_grid']**(1-ss['lbda'])/((ss['e_grid']**(1-ss['lbda']))@ss['pi_e']),(ss['nA'],1)).T

    # From UIP
    ss.update({'rstar': ss['r'], 'rstar_out': ss['r'], 'phh': 1, 'phf': 1, 'pfh': 1, 'dividend_X': 0})

    # Solve steady state
    blocks_ss = [mod.hh_HA, mod.hh_outputs, mod.rsimple, mod.income, mod.revaluation, mod.assets_market]
    unknowns_ss = {'beta': (beta_min,beta_max)}
    targets_ss= ['assets_clearing']
    mod_ss = sj.create_model(blocks_ss, name="SS Model")
    ss = mod_ss.solve_steady_state(ss, unknowns_ss, targets_ss)
    ss['a'], ss['c'], ss['coh'] = ss.internals['hh_HA']['a'], ss.internals['hh_HA']['c'], ss.internals['hh_HA']['coh']
    
    # Delayed substitution
    ss['x'] = 1-ss['alpha']
    ss['xstar'] = ss['x']
    ss['x_F'] = ss['alphastar']
    ss['xstar_F'] = ss['x_F']
    ss['beta_star'] = 1/(1+ss['r'])

    # Consumption statistics
    ss['cHi'] = (1-ss['alpha'])*ss['c']
    ss['cFi'] = ss['cbarF'] + ss['alpha']*ss['c']
    ss['share_F'] = ss['cFi']/(ss['cFi']+ss['cHi'])
    ss['cHstar'] = (ss['y']-ss['cH'])
    ss['Cstar'] = ss['cHstar']/ss['alphastar']
    ss['CT'] = ss['cH'] + ss['cF']
    ss['chi'] = (1-ss['alpha'])*ss['eta'] + ss['gamma']*ss['eps_dcp']
    ss['netexports'] = ss['y'] - ss['cF'] - ss['cH'] 
    ss['w'] = 1/ss['markup_ss']
    ss['Css'] = ss['CT']
    ss['div_tot'] = ss['dividend']

    # Nominal variables
    ss.update({'pi':ss['piw'], 'piHH':ss['piw'], 'PHH':1, 'piHF':ss['piw'], 'PHF':1, 'piFH':ss['piw'], 'PFH':1, 'piout': ss['piw'], 'piw_target': ss['piw']})
    ss['kappa_w'] = (1-ss['theta_w'])*(1-ss['beta']*ss['theta_w'])/ss['theta_w']
    ss['kappa_p'] = (1-ss['theta_p'])*(1-ss['beta']*ss['theta_p'])/ss['theta_p']
    ss['kappa_I'] = (1-ss['theta_I'])*(1-ss['beta']*ss['theta_I'])/ss['theta_I']
    ss['kappa_X'] = (1-ss['theta_X'])*(1-ss['beta']*ss['theta_X'])/ss['theta_X']
    ss['vphi'] = ss['atw'] * 1/ss['markup_ss'] * ss['CT']**(-1/ss['eis']) * ss['n']**(-1/ss['frisch'])
    ss['E'] = ss['Q']*ss['P']
    ss['i'] = ss['r'] + ss['pi']
    ss['rss'] = ss['r']
    ss['ires'] = 0
   
    # Compute incidence for calibration of gamma heterogeneity
    ss['incidence_i'] = np.tile(1+ss['zeta_e']*np.log(ss['e_grid'])-ss['zeta_e']*np.sum(ss['pi_e']*np.log(ss['e_grid'])*ss['e_grid']),(ss['nA'],1)).T
    
    # Balance sheet effects
    ss['rpost_F'] = ss['r']
    ss['rpost_H'] = ss['r']
    ss['f_firm'] = ss['j']/ss['A']
    ss['a_F'] = ss['f_FY']*ss['y']
    ss['a_H'] = ss['nfa']-ss['a_F']
    ss['f_F'] = ss['a_F']/ss['A']
    ss['Bbar'] = ss['BbarY']*ss['y']
    ss['delta'] = (1+ss['rstar'])*(1-1/D)
    ss['q'] = 1/(1 + ss['rstar'] - ss['delta'])
    ss['qH'] = ss['q']

    return ss
