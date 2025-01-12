#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 22:45:05 2024

@author: Sodik Umurzakov
"""

import numpy as np
import sequence_jacobian as sj

# %%
"RA model"
# Set up RA model for one agent
@sj.solved(unknowns={'C':1, 'A':1},
           targets=["euler", "budget_constraint"])
def household_represent_agent(C, A, Z, sigma, beta, r):
    """
    This function solves for the representative agent's consumption (C) and asset holdings (A) 
    using the Euler equation and the budget constraint.

    Parameters
    ----------
    C: Household consumption.
    A: Household asset holdings.
    Z: Household income.
    sigma: Intertemporal Elasticity of Substitution, which determines the household's willingness 
        to smooth consumption over time.
    beta: Discount factor
    r: Real interest rate

    Returns
    -------
    euler: Euler equation
    budget_constraint: Budget constraint
    """
    euler = (beta * (1 + r(+1)))**(-sigma) * C(+1) - C
    budget_constraint = (1 + r) * A(-1) + Z - C - A
    return euler, budget_constraint


# %%

"TA model"


# Set up two agent economy model
@sj.solved(unknowns={'C_RA': 1, 'A': 1},
           targets=["euler", "budget_constraint"])
def household_two_agent(C_RA, A, Z, sigma, beta, r, lambd):
    """
    Solves the household problem in a two-agent economy model with infinitely-lived (Ricardian) agents 
    and hand-to-mouth (HtM) agents. Computes the Euler equation, budget constraint, and aggregate consumption.

    Parameters
    ----------
    C_RA: Consumption of the Ricardian agent (infinitely-lived household).
    A: Asset holdings of the Ricardian agent.
    Z: Income or endowment received by the hand-to-mouth agent.
    sigma:  Intertemporal Elasticity of Substitution (EIS).
    beta: Discount factor, representing the Ricardian agent's preference for future utility.
    r: Interest rate on assets.
    lambd: Share of hand-to-mouth agents in the economy.

    Returns
    -------
    euler: Residual of the Euler equation for Ricardian agents. It must equal zero in equilibrium.
    budget_constraint: budget constraint for RA. It ensures that income, consumption, 
        and savings balance in equilibrium.
    C_hand_to_mouse: Consumption of hand-to-mouth agents, directly tied to their income (`Z`).
    C: Aggregate consumption in the economy, weighted by the proportions of RA and HtM agents.
    """
    euler = (beta * (1 + r(+1))) ** (-sigma) * C_RA(+1) - C_RA  # Euler equation for Ricardian agents
    C_hand_to_mouse = Z  # Consumption of hand-to-mouth agents
    C = (1 - lambd) * C_RA + lambd * C_hand_to_mouse  # Aggregate consumption
    budget_constraint = (1 + r) * A(-1) + Z - C - A  # Budget constraint for Ricardian agents
    return euler, budget_constraint, C_hand_to_mouse, C


@sj.solved(unknowns={'C_RA': 1, 'A': 1},
           targets=["euler", "budget_constraint"]) 
def household_two_agent_inc(C_RA, A, Z, sigma, beta, r, lambd, markup_ss):
    """
    Solves the household problem in a two-agent economy model with infinitely-lived (Ricardian) agents 
    and hand-to-mouth (HtM) agents. Computes the Euler equation, budget constraint, and aggregate consumption.

    Parameters
    ----------
    C_RA: Consumption of the Ricardian agent (infinitely-lived household).
    A: Asset holdings of the Ricardian agent.
    Z: Income or endowment received by the hand-to-mouth agent.
    sigma:  Intertemporal Elasticity of Substitution (EIS).
    beta: Discount factor, representing the Ricardian agent's preference for future utility.
    r: Interest rate on assets.
    lambd: Share of hand-to-mouth agents in the economy.

    Returns
    -------
    euler: Residual of the Euler equation for Ricardian agents. It must equal zero in equilibrium.
    budget_constraint: budget constraint for RA. It ensures that income, consumption, 
        and savings balance in equilibrium.
    C_hand_to_mouse: Consumption of hand-to-mouth agents, directly tied to their income (`Z`).
    C: Aggregate consumption in the economy, weighted by the proportions of RA and HtM agents.
    """
    euler = (beta * (1 + r(+1))) ** (-sigma) * C_RA(+1) - C_RA  # Euler equation for Ricardian agents
    C_hand_to_mouse = Z  # Consumption of hand-to-mouth agents
    C = (1 - lambd*markup_ss) * C_RA + lambd*markup_ss * C_hand_to_mouse  # Aggregate consumption
    budget_constraint = (1 + r) * A(-1) + Z - C - A  # Budget constraint for Ricardian agents
    return euler, budget_constraint, C_hand_to_mouse, C


# %%

"HA model"

# Initialize households for initial point of 
# value function for further backward iteration
def initial_households(a_grid, z, r, sigma):
    """
    This function initializes the value of assets for households based on cash-on-hand, idiosyncratic income shocks,
    real interest rates and intertemporal preferences.

    Parameters:
    a_grid: A grid of asset levels available to households.
    z: A grid of income idiosyncratic shocks affecting households.
    r: The real interest rate determining returns on assets.
    sigma:  Intertemporal Elasticity of Substitution, which governs households' willingness 
                   to adjust consumption over time.

    Returns:
    Va: The marginal value of assets, computed as a function of cash-on-hand 
                           and intertemporal preferences.
    """
    # Total resources available to households
    cash_on_hand = (1 + r) * a_grid[np.newaxis, :] + z[:, np.newaxis]
    
    # Value of the assets based on marginal utility
    Va = (1 + r) * (0.1 * cash_on_hand) ** (-1 / sigma)
    
    return Va


# Set up backward step by linking 
# to the SJ for the HA model solution block
@sj.het(exogenous='Pi',
        policy='a',  
        backward='Va',  
        backward_init=initial_households)

def households(Va_p, a_grid, z, r, beta, sigma):
    """
    This function solves the household optimization problem to determine the value of assets (Va),
    optimal savings (a) and consumption (c) based on the given parameters.

    Parameters:
    Va_p: Marginal value of assets on tomorrow's grid (V'(a') for the next period).
    a_grid: Grid of asset levels representing possible savings or borrowing levels.
    z: idiosyncratic income shocks affecting households.
    r: Real interest rate determining the return on assets.
    beta: Discount factor representing the degree to which households value future utility.
    sigma: Intertemporal Elasticity Substitution, which determines how households 
                   smooth consumption across time.

    Returns:
    Va: Updated marginal value of assets (V'(a)) for the current period.
    a: Optimal asset policy, how much households save or borrow.
    c: Optimal consumption for the current period.
    
    Steps:
    1. Computes the marginal utility of consumption on tomorrow's grid (u'(c')) using the discount factor and Va_p.
    2. Calculates tomorrow's consumption (c') using the inverse of the marginal utility function.
    3. Computes total cash-on-hand (coh) for the current period, which is the sum of income and returns on assets.
    4. Uses interpolation to determine the optimal asset policy (a) by matching cash-on-hand with c' + a'.
    5. Imposes a borrowing constraint to ensure households cannot hold assets below the minimum grid level.
    6. Calculates current consumption (c) as the difference between cash-on-hand and optimal savings (a).
    7. Updates the marginal value of assets (V'(a)) using the marginal utility of consumption.
    """
    ut_c_tomorrows_grid = beta * Va_p  
    c_tomorrows_grid = ut_c_tomorrows_grid ** (-sigma)
    cash_on_hand = (1 + r) * a_grid[np.newaxis, :] + z[:, np.newaxis]  
    a = sj.interpolate.interpolate_y(c_tomorrows_grid + a_grid, cash_on_hand, a_grid)  
    sj.misc.setmin(a, a_grid[0])  
    c = cash_on_hand - a  
    Va = (1 + r) * c ** (-1 / sigma)  
    return Va, a, c


def create_grids(rho_e, sd_e, n_e, min_a, max_a, n_a):
    """
    This fucntion creates grids for income states, assets and the transition matrix for income shocks.

    Parameters:
    rho_e: Persistence of income shocks (autocorrelation coefficient).
    sd_e: Standard deviation of income shocks.
    n_e: Number of discrete income states.
    min_a: Minimum asset level (e.g., borrowing limit).
    max_a: Maximum asset level (e.g., maximum savings).
    n_a: Number of grid points for the asset grid.

    Returns:
    e_grid: Discrete grid of income states.
    Pi: Transition matrix (Markov) for income states, describing probabilities of moving
                     between income levels.
    a_grid: Discrete grid of asset levels for savings and borrowing choices.
    """
    e_grid, _, Pi = sj.grids.markov_rouwenhorst(rho_e, sd_e, n_e)
    a_grid = sj.grids.asset_grid(min_a, max_a, n_a)
    return e_grid, Pi, a_grid


def individual_income(Z, e_grid):
    """
    This function calculates household-specific income based on the aggregate income level and 
    the idiosyncratic income grid.

    Parameters:
    - Z : Aggregate income level.
    - e_grid: Grid of idiosyncratic income states calculated using proportions of aggregate income.

    Returns:
    - Individual household income for each state in the income grid.
    """
    z = Z * e_grid
    return z

# %% 
"MPCs for RA, TA and HA models"

def compute_mpc_for_models(ra_model, ta_model, ha_model, steady_states_ra, steady_states_ta, steady_states_ha, quarters=30):
    """
    Compute the MPC matrices for three models: RA, TA and HA.

    Parameters:
    ra_model: The RA model object.
    ta_model: The TA model object.
    ha_model: The HA model object.
    steady_states_ra: Steady-state results for the RA model.
    steady_states_ta: Steady-state results for the TA model.
    steady_states_ha: Steady-state results for the HA model.
    quarters: Number of quarters to compute MPCs for (default is 30).

    Returns:
    mpc_results: A dictionary containing MPC matrices (M and Mr) for each model.
    """
    # Compute Jacobians for each model
    J_ra = ra_model.jacobian(steady_states_ra, ['Z', 'r'], T=quarters)
    J_ta = ta_model.jacobian(steady_states_ta, ['Z', 'r'], T=quarters)
    J_ha = ha_model['households'].jacobian(steady_states_ha, ['Z', 'r'], T=quarters)

    # Extract MPC matrices
    M_ra = J_ra['C']['Z']
    Mr_ra = J_ra['C']['r']

    M_ta = J_ta['C']['Z']
    Mr_ta = J_ta['C']['r']

    M_ha = J_ha['C']['Z']
    Mr_ha = J_ha['C']['r']

    # Store results in a dictionary
    mpc_results = {
        'RA': {'M': M_ra, 'Mr': Mr_ra},
        'TA': {'M': M_ta, 'Mr': Mr_ta},
        'HA': {'M': M_ha, 'Mr': Mr_ha}
    }

    return mpc_results


# %% Other blocks

" Other simple model blocks "
@sj.simple
def domestic_demand(C, PF_P, PH_P, eta, alpha):
    """
    This function computes the domestic demand for home and foreign goods based on total consumption, real prices 
    and preferences.

    Parameters:
    C: Total domestic consumption.
    PF_P: Real price of the foreign good in terms of domestic consumption good units.
    PH_P: Real price of the home good in terms of domestic consumption good units.
    eta: Elasticity of substitution between home and foreign goods.
    alpha: Share of preference for foreign goods (1 - alpha is the preference for home goods).

    Returns:
    cH: Domestic demand for the home good.
    cF: Domestic demand for the foreign good.

    Notes:
    Domestic consumption is split between home and foreign goods according to preferences and relative prices.
    """
    cH = (1 - alpha) * PH_P ** (-eta) * C  # Demand for home goods
    cF = alpha * PF_P ** (-eta) * C        # Demand for foreign goods
    return cH, cF

# Foreign demand for the goods
@sj.simple
def foreign_demand(PH_star, alpha_star, gamma, C_star):
    """
    This function computes the foreign demand for the home good based on foreign consumption, relative prices, 
    and preferences.

    Parameters:
    PH_star: Real price of the home good abroad, in terms of foreign consumption good units.
    alpha_star: Foreign preference for the home good (1 - alpha_star is the preference for foreign goods).
    gamma: Elasticity of foreign demand for the home good.
    C_star Total foreign consumption.

    Returns:
    cH_star: Foreign demand for the home good.
   """
    cH_star = alpha_star * PH_star ** (-gamma) * C_star  # Foreign demand for home goods
    return cH_star


#  Uncovered Interest Parity (UIP) condition
@sj.solved(unknowns={'Q': (0.01, 3.)}, targets=['uip'])
def UIP(Q, r_expected, r_star, eta, alpha, gamma):
    """
    This function solves for the real exchange rate (Q) using the Uncovered Interest Parity (UIP) condition
    and calculates related price levels and trade elasticity.

    Parameters:
    - Q: Real exchange rate (to be solved for), representing the relative price of foreign
                 goods in terms of domestic goods.
    - r_expected: Anticipated real domestic interest rate.
    - r_star: Real foreign interest rate.
    - eta: Elasticity of substitution between home and foreign goods.
    - alpha: Preference for foreign goods (1 - alpha is the preference for home goods).
    - gamma: Elasticity of foreign demand for home goods.

    Returns:
    - uip: Residual of the UIP condition. When solved, it ensures that UIP holds.
    - PH_star: Price of home goods in foreign markets, expressed in terms of Q.
    - PH_P: Price of home goods in the domestic market, expressed in terms of Q.
    - PF_P: Price of foreign goods in the domestic market, equivalent to Q.
    - chi: Trade elasticity, which reflects the sensitivity of trade flows to exchange rate
                   changes.
    """
    # Recursive equation for UIP
    uip = 1 + r_expected - (1 + r_star) * Q(1) / Q

    # Price of home goods abroad in terms of Q
    PH_star = ((Q ** (eta - 1) - alpha) / (1 - alpha)) ** (1 / (1 - eta))

    # Price of home goods domestically in terms of Q
    PH_P = ((1 - alpha * Q ** (1 - eta)) / (1 - alpha)) ** (1 / (1 - eta))

    # Price of foreign goods domestically in terms of Q
    PF_P = Q

    # Trade elasticity
    chi = eta * (1 - alpha) + gamma
    # chi=0.1

    return uip, PH_star, PH_P, PF_P, chi


# Aggregate income
@sj.solved(unknowns={'J': (0.001, 15.)}, targets=['valuation_cond'])

def aggregate_income(Y, PH_P, J, r_expected, markup_ss):
    """
    This function computes key aggregate economic variables such as aggregate income, dividends, GDP, 
    valuation conditions and interest rates in HA model.

    Parameters
    ----------
    Y: Real output (production) of the economy.
    PH_P: Real price of the home good in terms of domestic consumption units.
    J: Valuation of assets at the beginning of the period.
    r_expected: Anticipated real interest rate.
    markup_ss: Steady-state markup of prices over marginal costs.

    Returns
    -------
    j: End-of-period asset valuation.
    valuation_cond: Valuation condition ensuring equilibrium pricing of assets.
    GDP: Nominal GDP, adjusted for purchasing power parity (PPP).
    div: Real dividend from production, representing income distributed to asset holders.
    Z: Real aggregate labor income, calculated as a share of production.
    r: Ex post real interest rate, including revaluation effects.
    """
    # Real aggregate labor income
    Z = 1 / markup_ss * PH_P * Y
    
    # Real dividend from production
    div = (1 - 1 / markup_ss) * PH_P * Y   
    
    # Nominal PPP-adjusted GDP
    gdp = PH_P * Y
    
    # Valuation condition to price the asset
    valuation_cond = div + J(1) / (1 + r_expected) - J  # J = beginning of period valuation
    j = J(1) / (1 + r_expected)  # j = end of period valuation
    
    # Ex post interest rate including revaluation
    r = J / j(-1) - 1
    return j, valuation_cond, gdp, div, Z, r



# Balance of payments
@sj.solved(unknowns={'nfa': (-5., 5.)}, targets=['nfa_cond'])
def balance_of_payment(nfa, PH_P, cH_star, PF_P, cF, r_expected):
    """
    Solves for the net foreign assets (nfa) equilibrium using the balance of payments condition 
    and calculates the net exports (NX).

    Parameters
    ----------
    nfa: Net foreign assets at the beginning of the period, bounded between -5 and 5.
    PH_P: Real price of the home good in domestic consumption units.
    cH_star: Foreign demand for the home good.
    PF_P: Real price of the foreign good in domestic consumption units.
    cF: Domestic demand for the foreign good.
    r_expected: Anticipated real interest rate.

    Returns
    -------
    NX : float
        Net exports, representing the trade balance (exports minus imports).
    nfa_cond : float
        Balance of payments condition, ensuring equilibrium in the net foreign asset position.
    """
    # Net exports: value of exports minus value of imports
    NX = PH_P * cH_star - PF_P * cF
    
    # Balance of payments condition
    nfa_cond = NX + (1 + r_expected(-1)) * nfa(-1) - nfa  
    
    return NX, nfa_cond


# Market Clearing condition
@sj.simple
def market_clearing_cond(Y, cH, cH_star, A, nfa, j):
    """
    Computes the market clearing conditions for goods and assets in the economy.

    Parameters
    ----------
    Y: Total output (production) of the economy.
    cH: Domestic demand for home goods.
    cH_star: Foreign demand for home goods (exports).
    A: Total assets in the economy.
    nfa: Net foreign assets, representing the economy's external financial position.
    j: End-of-period valuation of domestic assets.

    Returns
    -------
    goods: Goods market clearing condition, calculated as the difference between total demand 
        (domestic and foreign) and total output.
    asset: Asset market clearing condition, calculated as the difference between total assets 
        and the sum of net foreign assets and domestic asset valuations.
    """
    goods = cH + cH_star - Y
    asset = A - nfa - j
    return goods, asset


# Phillips Curve with wage rigidities
@sj.simple
def phillips_curve_wage(pi, Y, X, C, theta_w, v_phi, frisch, markup_ss, sigma, beta):
    """
    Computes the Phillips Curve residual and wage inflation.

    Parameters
    ----------
    pi: Inflation rate (percentage change in the price level).
    Y: Output level (e.g., real GDP).
    X: Real wage or nominal wage divided by the price level.
    C: Consumption level.
    theta_w: Wage stickiness parameter (probability of not adjusting wages in a given period).
    v_phi: Parameter for labor disutility in the household's utility function (related to labor supply elasticity).
    frisch: Frisch elasticity of labor supply (inverse of v_phi).
    markup_ss: Steady-state markup in the labor market.
    sigma:  Intertemporal Elasticity of Substitution in consumption.
    beta: Discount factor, representing time preference for future utility.

    Returns
    -------
    pi_w_res: Residual of the Phillips Curve equation for wage inflation. 
    pi_w: Wage inflation.
    """
    pi_w = pi + X - X(-1)  # Wage inflation
    kappa_w = (1 - theta_w) * (1 - beta * theta_w) / theta_w  # Slope of the wage Phillips Curve
    pi_w_res = (kappa_w * (v_phi * (Y/X)**(1/frisch) - 1/markup_ss * X * C**(-1/sigma)) 
                + beta * pi_w(1) - pi_w)  # Residual of the wage Phillips Curve
    return pi_w_res, pi_w

# Taylor rule for monetary policy decition
@sj.simple
def taylor_rule(pi, i_policy_shock, r_steady_state, phi_pi):
    """
    This function computes the nominal interest rate (i) based on the Taylor rule and the expected real interest rate (r_expected).

    Parameters
    ----------
    pi: Current inflation rate.
    i_policy_shock: Exogenous policy shock to the nominal interest rate.
    r_steady_state: Real interest rate in the long-run.
    phi_pi: Taylor rule inflation coefficient, determining how aggressively monetary policy reacts to inflation deviations.

    Returns
    -------
    i: Nominal interest rate, determined by the Taylor rule.
    r_expected : Expected real interest rate
    
    """
    i = r_steady_state + phi_pi * pi + i_policy_shock  # Taylor rule for nominal interest rate
    r_expected = i - pi(1)  # Expected real interest rate
    return i, r_expected


# Taylor rule for monetary policy decition (from updated paper)
@sj.simple
def taylor_rule_updated(i, rho_m, pi, i_policy_shock, r_steady_state, phi_pi):
    """
    This function computes the nominal interest rate (i) based on the Taylor rule and the expected real interest rate (r_expected).

    Parameters
    ----------
    pi: Current inflation rate.
    i_policy_shock: Exogenous policy shock to the nominal interest rate.
    r_steady_state: Real interest rate in the long-run.
    phi_pi: Taylor rule inflation coefficient, determining how aggressively monetary policy reacts to inflation deviations.

    Returns
    -------
    i: Nominal interest rate, determined by the Taylor rule.
    r_expected : Expected real interest rate
    
    """
    i_updated = rho_m*i(-1) + (1-rho_m)*(r_steady_state + phi_pi * pi) + i_policy_shock  # Taylor rule for nominal interest rate
    r_expected_updated = i_updated - pi(1)  # Expected real interest rate
    return i_updated, r_expected_updated



def household_represent_agent_incomplete(Y, PH_P, Z, div, A, sigma, beta, r):
    """
    Solves for the representative agent's consumption (C) and asset holdings (A) 
    using the Euler equation and the budget constraint.

    Parameters
    ----------
    C: Household consumption.
    A: Household asset holdings.
    Z: Household income or endowment.
    sigma: Intertemporal Elasticity of Substitution
    beta: Discount factor
    r: Real interest rate

    Returns
    -------
    euler: 
    budget_constraint:
    """
    C = PH_P * Y
    euler = (beta * (1 + r(+1)))**(-sigma) * C(+1) - C 

    budget_constraint = (1 + r) * A(-1) + Z + div - C - A
    return euler, budget_constraint, C

@sj.simple
def domestic_demand_non_homothetic(C_bar, C, PF_P, PH_P, eta, alpha):
    """
    This function computes the domestic demand for home and foreign goods based on total consumption, real prices 
    and preferences.

    Parameters:
    C: Total domestic consumption.
    C_bar: Non-homothetic preferences coefficient
    PF_P: Real price of the foreign good in terms of domestic consumption good units.
    PH_P: Real price of the home good in terms of domestic consumption good units.
    eta: Elasticity of substitution between home and foreign goods.
    alpha: Share of preference for foreign goods (1 - alpha is the preference for home goods).

    Returns:
    cH: Domestic demand for the home good.
    cF: Domestic demand for the foreign good.
    """
    cH = (1 - alpha) * PH_P ** (-eta) * C  # Demand for home goods
    cF = alpha * PF_P ** (-eta) * C  + C_bar      # Demand for foreign goods
    return cH, cF

# %%

# Phillips Curve for prices of the domestic goods
@sj.simple
def phillips_curve_domestic(pi, Z, X, theta_H, markup_ss, sigma, r, beta):
    """
    Computes the residual and inflation for domestic goods using the Phillips Curve.

    Parameters
    ----------
    pi: Inflation rate for domestic goods.
    Z: Output or production of domestic goods.
    X: Real wage.
    theta_H: Price stickiness for domestic goods.
    markup_ss: Steady-state markup.
    sigma: Intertemporal elasticity of substitution.
    r: Real interest rate.
    beta: Discount factor.

    Returns
    -------
    pi_H_res: domestic Phillips Curve.
    pi_H: Inflation for domestic goods.
    """ 
    pi_H = pi + X - X(-1)
    kappa_H = (1 - theta_H) * (1 - (1 / (1 + r)) * theta_H) / theta_H
    pi_H_res = kappa_H * (markup_ss * (Z * X) - 1) + (1 / (1 + r)) * pi_H(1) - pi_H
    return pi_H_res, pi_H

@sj.simple
def phillips_curve_foreign(pi, X, theta_F, markup_ss, sigma, r, beta):
    """
    Computes the residual and inflation for imported goods using the Phillips Curve.

    Parameters
    ----------
    pi: Inflation rate for imported goods.
    X: Real wage.
    theta_F: Price stickiness for imported goods.
    markup_ss: Steady-state markup.
    sigma: Intertemporal elasticity of substitution.
    r: Real interest rate.
    beta: Discount factor.

    Returns
    -------
    pi_F_res: foreign Phillips Curve.
    pi_F: Inflation for imported goods.
    """ 
    pi_F = pi + X - X(-1)
    kappa_F = 0  # No slope for this example
    pi_F_res = kappa_F * (markup_ss - 1) + (1 / (1 + r)) * pi_F(1) - pi_F
    return pi_F_res, pi_F

@sj.simple
def phillips_curve_star(pi, X, theta_H_star, PH_star, markup_ss, sigma, r, beta):
    """
    Computes the residual and inflation for foreign home goods using the Phillips Curve.

    Parameters
    ----------
    pi: Inflation rate for foreign home goods.
    X: Real wage.
    theta_H_star: Price stickiness for foreign home goods.
    PH_star: Price level of foreign home goods.
    markup_ss: Steady-state markup.
    sigma: Intertemporal elasticity of substitution.
    r: Real interest rate.
    beta: Discount factor.

    Returns
    -------
    pi_H_res_star: foreign home goods Phillips Curve.
    pi_H_star: Inflation for foreign home goods.
    """ 
    pi_H_star = pi + X - X(-1)
    kappa_H_star = (1 - theta_H_star) * (1 - (1 / (1 + r)) * theta_H_star) / theta_H_star
    pi_H_res_star = kappa_H_star * (markup_ss / PH_star - 1) + (1 / (1 + r)) * pi_H_star(1) - pi_H_star
    return pi_H_res_star, pi_H_star

