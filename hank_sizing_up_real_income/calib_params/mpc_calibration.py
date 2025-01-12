#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 16:52:07 2025

@author: Sodik Umurzakov
"""

# Dictionary of calibrated parameters for the model with detailed descriptions
calibrated_parameters = {
    # Preferences and Utility
    'sigma': 1.,  # Elasticity of intertemporal substitution (EIS)
    'frisch': 0.5,  # Frisch elasticity of labor supply

    # Preferences for Goods
    'alpha': 0.4,  # Home bias in domestic consumption (share of foreign goods in consumption)
    'alpha_star': 0.4,  # Home bias for foreign consumers (share of domestic goods in foreign consumption)

    # Elasticities
    'eta': 0.5,  # Elasticity of substitution between goods
    'gamma': 0.5,  # Elasticity of foreign demand

    # Firm Behavior and Pricing
    'markup_ss': 1.043,  # Steady-state markup of prices over marginal cost
    'theta_w': 0.938,  # Wage stickiness parameter (Calvo parameter for wages)

    # Household Asset Grid
    'min_a': 0.,  # Minimum asset level
    'max_a': 400,  # Maximum asset level
    'n_a': 200,  # Number of grid points for asset levels

    # Idiosyncratic Income Shocks
    'rho_e': 0.912,  # Persistence of idiosyncratic productivity shocks
    'sd_e': 0.883,  # Standard deviation of productivity shocks
    'n_e': 7,  # Number of discrete income states in the Markov chain

    # Macro Aggregates
    'Y': 1.,  # Aggregate output (normalized to 1)
    'A': 1,  # Aggregate assets
    'C': 1,  # Aggregate consumption
    'C_RA': 1.,  # Representative agent's consumption (used in RA model)
    'C_star': 1,  # Foreign consumption (normalized)
    'nfa': 0,  # Net foreign assets (steady state assumed to be 0)

    # Real Exchange Rate and Prices
    'Q': 1,  # Real exchange rate (steady state normalized to 1)
    'X': 1.,  # Export quantity (normalized steady state value)

    # Monetary Policy
    'r_star': 0.02,  # Foreign interest rate (steady state value)
    'r_steady_state': 0.02,  # Domestic steady-state real interest rate
    'pi': 0.,  # Inflation rate (steady state assumed to be 0)
    'i_policy_shock': 0.,  # Monetary policy shock (default to no shock)
    'phi_pi': 1.5,  # Taylor rule coefficient on inflation (monetary policy rule)

    # Labor Supply
    'v_phi': 0.8,  # Disutility of labor parameter

    # Hand-to-Mouth Agents
    'lambd': 0.1  # Share of hand-to-mouth agents in the economy
}

