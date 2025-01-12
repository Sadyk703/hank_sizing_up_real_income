#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 28 21:36:40 2024

@author: Sodik Umurzakov
"""
import numpy as np
import matplotlib.pyplot as plt



# For figure 2 from baseline model
def irfs_decom_ra_ha(
    list_irfs_ra, list_irfs_ha, labels=[" "], ylabel=r"Percent of s.s.", T_plot=30, figsize=(12, 6), x_min=0
):
    """
    Plot Impulse Response Functions (IRFs) for RA and HA models side-by-side with adjusted line styles and colors.

    Parameters:
    list_irfs_ra: List of dictionaries containing IRFs for the RA model.
    list_irfs_ha: List of dictionaries containing IRFs for the HA model.
    var: Variable name to plot (e.g., 'Y').
    labels: List of labels for each IRF.
    ylabel: Label for the y-axis.
    T_plot: Number of periods to plot.
    figsize: Size of the overall figure.
    """
    if len(list_irfs_ra) != len(labels) or len(list_irfs_ha) != len(labels):
        labels = [" "] * max(len(list_irfs_ra), len(list_irfs_ha))

    fig, axes = plt.subplots(1, 2, figsize=figsize, sharex=True, sharey=True)

    # Define styles to match the legend
    line_styles = ['-', '--', '--', '--']  # Solid, Dashed, Dash-Dot, Dotted
    colors = ['k', 'g', 'r', 'b']  # Black, Green, Red, Blue

    # Plot for RA model
    ax = axes[0]
    for j, irf in enumerate(list_irfs_ra):
        ax.plot(
            np.arange(T_plot),
            100*irf[:T_plot],
            line_styles[j % len(line_styles)],
            color=colors[j % len(colors)],
            label=labels[j]
        )
    ax.set_title("RA model - complete markets", fontsize=12)
    ax.set_xlabel("Quarters", fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.axhline(0, color="k", linestyle="--", linewidth=0.75)  # Add horizontal line at 0
    ax.grid(False)
    ax.legend(fontsize=8, loc='best', frameon=False)  # Add legend
    
    # Plot for HA model
    ax = axes[1]
    for j, irf in enumerate(list_irfs_ha):
        ax.plot(
            np.arange(T_plot),
            100*irf[:T_plot],
            line_styles[j % len(line_styles)],
            color=colors[j % len(colors)],
            label=labels[j]
        )
    ax.set_title("HA model", fontsize=12)
    ax.set_xlabel("Quarters", fontsize=10)
    ax.axhline(0, color="k", linestyle="--", linewidth=0.75)  # Add horizontal line at 0
    ax.grid(False)
    ax.legend(fontsize=8, loc='best', frameon=False)  # Add legend
    ax.set_xlim(x_min, None)  # Ensure x-axis starts at `x_min`
    plt.tight_layout()  # Adjust layout for better appearance
    # plt.show()

 
#  Plotting of figure 1 - iMPC
def plot_intertemporal_mpcs(M_complete_ra, M_complete_ta, M_complete_ha,
                            M_incomplete_ra, M_incomplete_ta, M_incomplete_ha,
                            quarters=30, x_min=0):
    """
    Plots intertemporal MPCs for six calibrated models (RA, TA, HA) under complete and incomplete markets.

    Parameters:
    M_complete_ra: First column of the MPC matrix for RA model under complete markets.
    M_complete_ta: First column of the MPC matrix for TA model under complete markets.
    M_complete_ha: First column of the MPC matrix for HA model under complete markets.
    M_incomplete_ra: First column of the MPC matrix for RA model under incomplete markets.
    M_incomplete_ta: First column of the MPC matrix for TA model under incomplete markets.
    M_incomplete_ha: First column of the MPC matrix for HA model under incomplete markets.
    quarters: Number of quarters to plot.
    """
    time = np.arange(quarters)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    y_max = 0.11  # Set the maximum y-axis value for uniform scaling
    y_min = -0.01  # Set the minimum y-axis value for uniform scaling

    # Left Panel: Complete markets
    axes[0].set_title("iMPCs with complete markets")
    axes[0].plot(time, M_complete_ra[:quarters], label="RA", linestyle="-.", color="red")
    axes[0].plot(time, M_complete_ta[:quarters], label="TA", linestyle="--", color="blue")
    axes[0].plot(time, M_complete_ha[:quarters], label="HA", linestyle="-", color="green")
    axes[0].set_xlabel("Quarters")
    axes[0].set_ylabel("$∂ C_t / ∂ Z_0$ (% of Y_ss)")
    axes[0].axhline(0, color="k", linestyle="--", linewidth=0.75)
    axes[0].set_xlim(x_min, None)
    axes[0].set_ylim(y_min, y_max)
    axes[0].legend()
    axes[0].grid(False)

    # Right Panel: Incomplete markets
    axes[1].set_title("iMPCs with incomplete markets")
    axes[1].plot(time, M_incomplete_ra[:quarters], label="RA", linestyle="-.", color="red")
    axes[1].plot(time, M_incomplete_ta[:quarters], label="TA", linestyle="--", color="blue")
    axes[1].plot(time, M_incomplete_ha[:quarters], label="HA", linestyle="-", color="green")
    axes[1].set_xlabel("Quarters")
    axes[1].set_ylabel("$∂ C_t / ∂ Z_0$ (% of Y_ss)")
    axes[1].axhline(0, color="k", linestyle="--", linewidth=0.75)
    axes[1].set_xlim(x_min, None)
    axes[1].set_ylim(y_min, y_max)
    axes[1].legend()
    axes[1].grid(False)

    plt.tight_layout()
    # plt.show()

# iMPC for figure 2
def plot_mpc_columns_scaled(M_ra_cm, M_ta_cm, M_ha_cm, M_ra_im, M_ta_im, M_ha_im, quarters=30, x_min=0):
    """
    Generate plots for intertemporal MPCs (columns of M) for six calibrated models with uniform scaling.

    Parameters:
    M_ra_cm: Jacobian matrix for RA-CM model.
    M_ta_cm: Jacobian matrix for TA-CM model.
    M_ha_cm: Jacobian matrix for HA-CM model.
    M_ra_im: Jacobian matrix for RA-IM model.
    M_ta_im: Jacobian matrix for TA-IM model.
    M_ha_im: Jacobian matrix for HA-IM model.
    quarters: Number of quarters to plot.
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    columns_to_plot = [0, 10, 20]  # Columns (s) to plot
    labels = ["s=0", "s=10", "s=20"]
    line_styles = ['-', '--', '-.']
    colors = ['g', 'r', 'b']

    model_titles = ["RA-CM", "TA-CM", "HA-CM", "RA-IM", "TA-IM", "HA-IM"]
    models = [M_ra_cm, M_ta_cm, M_ha_cm, M_ra_im, M_ta_im, M_ha_im]  # CM and IM for each model

    y_max = 0.11  # Set the maximum y-axis value for uniform scaling
    y_min = -0.01  # Set the minimum y-axis value for uniform scaling

    for i, ax in enumerate(axes.flat):
        model_title = model_titles[i]
        model_M = models[i]

        for col, label, linestyle, color in zip(columns_to_plot, labels, line_styles, colors):
            ax.plot(
                np.arange(quarters),
                model_M[:, col][:quarters],
                linestyle=linestyle,
                color=color,
                label=label,
            )

        ax.set_title(model_title, fontsize=12)
        ax.set_xlabel("Quarters", fontsize=10)
        ax.set_ylabel(r"$\partial C_t / \partial Z_s$", fontsize=10)
        ax.axhline(0, color="k", linestyle="--", linewidth=0.75)
        ax.legend(fontsize=8, loc="upper right", frameon=False)
        ax.set_xlim(x_min, None)
        ax.set_ylim(y_min, y_max)  # Apply uniform scaling to all subplots
        ax.grid(False)

    plt.tight_layout()
    # plt.show()


    
# Figure 2: Effect of exchange rate shocks on output for various χ’s
def irfs_plotting_figure2(irf_Q, irfs_complete, irfs_incomplete, var, labels=[" "], ylabel=r"Percent of s.s.", T_plot=30, figsize=(18, 6), x_min=0):
    """
    Plot the effect of exchange rate shocks on real exchange rate and output under different market structures.

    Parameters:
    irf_Q: Impulse Response Function (IRF) for Real Exchange Rate Q.
    irfs_complete: List of IRFs for Output Y under complete markets.
    irfs_incomplete: List of IRFs for Output Y under incomplete markets.
    ylabel: Label for the y-axis.
    T_plot: Number of periods to plot.
    figsize: Size of the overall figure.
    x_min: Minimum value for the x-axis.
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize, sharex=True, sharey=False)

    line_styles = ['-', '--', '-.', ':']
    colors = ['black', 'red', 'blue', 'green']
    labels_complete = [r"$\chi=1$", r"$\chi=0.1, RA$", r"$\chi=0.1, TA$", r"$\chi=0.1, HA$"]
    labels_incomplete = [r"$\chi=1$", r"$\chi=0.1, RA$", r"$\chi=0.1, TA$", r"$\chi=0.1, HA$"]

    # Panel 1: Real exchange rate Q
    axes[0].plot(np.arange(T_plot), 100 * irf_Q, linestyle='-', color='black', label='Real Exchange Rate Q')
    axes[0].set_title("Real exchange rate Q", fontsize=12)
    axes[0].set_xlabel("Quarters", fontsize=10)
    axes[0].set_ylabel(ylabel, fontsize=10)
    axes[0].axhline(0, color="k", linestyle="--", linewidth=0.75)
    axes[0].grid(False)
    axes[0].set_xlim(x_min, None)
    axes[0].legend(fontsize=8, loc='best', frameon=False)

    # Panel 2: Output Y, complete markets
    for j, irf in enumerate(irfs_complete):
        axes[1].plot(
            np.arange(T_plot),
            100 * irf[var][:T_plot],
            linestyle=line_styles[j % len(line_styles)],
            color=colors[j % len(colors)],
            label=labels_complete[j]
        )
    axes[1].set_title("Output Y, complete markets", fontsize=12)
    axes[1].set_xlabel("Quarters", fontsize=10)
    axes[1].axhline(0, color="k", linestyle="--", linewidth=0.75)
    axes[1].grid(False)
    axes[1].set_xlim(x_min, None)
    axes[1].legend(fontsize=8, loc='best', frameon=False)

    # Panel 3: Output Y, incomplete markets
    for j, irf in enumerate(irfs_incomplete):
        axes[2].plot(
            np.arange(T_plot),
            100 * irf[var][:T_plot],
            linestyle=line_styles[j % len(line_styles)],
            color=colors[j % len(colors)],
            label=labels_incomplete[j]
        )
    axes[2].set_title("Output Y, incomplete markets", fontsize=12)
    axes[2].set_xlabel("Quarters", fontsize=10)
    axes[2].axhline(0, color="k", linestyle="--", linewidth=0.75)
    axes[2].grid(False)
    axes[2].set_xlim(x_min, None)
    axes[2].legend(fontsize=8, loc='best', frameon=False)

    plt.tight_layout()
    # plt.show()

# Plotting of figure 6 
def irfs_decom_ra_ha_r_new(
    list_irfs_ra_chi0, list_irfs_ha_chi0, list_irfs_ra_chi1, list_irfs_ha_chi1, labels, ylabel=r"Percent of s.s.", T_plot=30, figsize=(12, 10), x_min=0
):
    """
    Plot Impulse Response Functions (IRFs) for RA and HA models with decomposition.

    Parameters:
    list_irfs_ra: List of lists containing decomposed IRFs for the RA model.
    list_irfs_ha_chi0: List of lists containing decomposed IRFs for the HA model (χ = 2 - α).
    list_irfs_ha_chi1: List of lists containing decomposed IRFs for the HA model (χ = 0.5).
    labels: List of labels for each IRF component.
    ylabel: Label for the y-axis (default is "Percent of s.s.").
    T_plot: Number of periods to plot (default is 30).
    figsize: Size of the overall figure (default is (18, 6)).
    x_min: Minimum x-axis value (default is 0).
    """
    if len(list_irfs_ra_chi0) != len(labels) or  len(list_irfs_ha_chi0) != len(labels) or len(list_irfs_ra_chi1) != len(labels) or len(list_irfs_ha_chi1) != len(labels):
        raise ValueError("Mismatch between number of IRFs and labels")

    # Create a subplot with 2 rows and 2 columns
    fig, axes = plt.subplots(2, 2, figsize=figsize, sharex=True)

    # Define styles for the lines
    line_styles = ['-', '--', '--', '--']  # Solid for total, dashed for components
    colors = ['k', 'g', 'r', 'y']  # Black for total, others for decomposition

    # Titles for the four panels
    titles = [
        r"Output, complete markets RA ($\chi = 2 - \alpha$)",
        r"Output, HA ($\chi = 2 - \alpha$)",
        r"Output, complete markets RA ($\chi = 0.5$)",
        r"Output, HA ($\chi = 0.5$)"
    ]
    models = [list_irfs_ra_chi0, list_irfs_ha_chi0, list_irfs_ra_chi1, list_irfs_ha_chi1]

    # Set individual y-axis limits for each plot
    y_limits = [
        (-0.5, 2.0),  # Output, complete markets RA ($\chi = 2 - \alpha$)
        (-0.5, 2.0),  # Output, HA ($\chi = 2 - \alpha$)
        (-0.4, 1.0),  # Output, complete markets RA ($\chi = 0.5$)
        (-0.4, 1.0)   # Output, HA ($\chi = 0.5$)
    ]

    # Plot each panel
    for i, ax in enumerate(axes.flat):
        for j, irf in enumerate(models[i]):
            ax.plot(
                np.arange(T_plot),
                100 * irf[:T_plot],  # Scale IRFs to percentage points
                line_styles[j % len(line_styles)],
                color=colors[j % len(colors)],
                label=labels[j]
            )
        # Add title and formatting
        ax.set_title(titles[i], fontsize=12)
        ax.set_xlabel("Quarters", fontsize=10)
        ax.axhline(0, color="k", linestyle="--", linewidth=0.75)  # Horizontal line at 0
        ax.grid(False)
        ax.legend(fontsize=8, loc='best', frameon=False)  # Legend
        ax.set_xlim(x_min, None)
        ax.set_ylim(*y_limits[i])  # Apply individual y-axis limits

    # Set y-axis label for the first column
    axes[0, 0].set_ylabel(ylabel, fontsize=10)
    axes[1, 0].set_ylabel(ylabel, fontsize=10)

    # Adjust layout for better appearance
    plt.tight_layout()
    # plt.show()

# for plotting figure 9
def plot_irf_comparison(irf, va, Tplot=31):
    """
    Plot impulse response comparisons for various variables under Taylor and Real rate rules.

    Parameters
    ----------
    irf: Dictionary containing impulse responses for different rules (e.g., 'taylor', 'realrate').
    va: Dictionary mapping subplot indices to variable keys in the `irf` dictionary.
    Tplot: Number of quarters to plot. Default is 31.
    """
    titles = [
        'Output', 'Consumption', 'Net exports', 'NFA', 
        'Real wage income', 'Dividends', 'Real exchange rate', 'Real interest rate'
    ]
    ylabels = [
        'Percent of $Y_{ss}$', 'Percent of $Y_{ss}$', 'Percent of $Y_{ss}$', 'Percent of $Y_{ss}$',
        'Percent of $Y_{ss}$', 'Percent of $Y_{ss}$', 'Percent of s.s.', 'Percent'
    ]
    
    plt.figure(figsize=(13, 6))

    for i, k in va.items():
        plt.subplot(2, 4, i)
        plt.title(titles[i - 1])
        
        # Plot Taylor rule and Real rate rule IRFs
        plt.plot(100 * irf['taylor'][k][:Tplot], color='#008C00', label=r'Taylor rule')
        plt.plot(100 * irf['realrate'][k][:Tplot], color='#0000A0', linestyle='--', label=r'Real rate rule')
        plt.axhline(y=0, color='#808080', linestyle=':')

        # Set specific y-axis limits and labels
        if k == 'y': 
            plt.legend(framealpha=0, loc='upper left')
            plt.ylim([-0.15, 0.2])
        if i >= 5: 
            plt.xlabel('Quarters')
        if i in [1, 5, 7, 8]: 
            plt.ylabel(ylabels[i - 1])
        if k == 'C': 
            plt.ylim([-0.25, 0.1])
        if k == 'netexports': 
            plt.ylim([-0.6, 0.2])
        if k == 'nfa': 
            plt.ylim([-2.5, 0.1])
        if k == 'atw_n': 
            plt.ylim([-0.8, 0.2])
        if k == 'r': 
            plt.ylim([-0.02, 0.04])

    plt.tight_layout()
    # plt.show()

# for plotting figure 10
def plot_irf_decomposition(shock, dr, dr2, J_Qrstar, J_Qr, J_yrstar, J_yr, Tplot=31):
    """
    Plots the impulse response decomposition for real rate, exchange rate, and output.

    Parameters
    ----------
    shock : Initial shock vector.
    dr: Response of the real interest rate to fight depreciation.
    dr2: Response of the real interest rate to fight recession.
    J_Qrstar: Jacobian for the exchange rate's response to the foreign interest rate shock.
    J_Qr: Jacobian for the exchange rate's response to the real interest rate.
    J_yrstar: Jacobian for the output's response to the foreign interest rate shock.
    J_yr: Jacobian for the output's response to the real interest rate.
    Tplot: Number of periods to plot. Default is 31.
    """
    plt.figure(figsize=(12, 4))
    
    # Panel 1: Real rate and foreign interest rates
    plt.subplot(1, 3, 1)
    plt.title('Real rate $r$ and foreign interest rates $i^*$')
    plt.plot(shock[:Tplot], color='#008C00', label='Initial shock to $i^*$')
    plt.plot((dr @ shock)[:Tplot], color='#A00000', linestyle='--', label='$r$ to fight depreciation')
    plt.plot((dr2 @ shock)[:Tplot], color='#0000A0', linestyle='dashdot', label='$r$ to fight recession')
    plt.axhline(y=0, color='#808080', linestyle=':')
    plt.legend(framealpha=0)
    plt.xlabel('Quarters')
    plt.ylabel('Percent')
    plt.ylim(-0.06, 0.16)
    
    # Panel 2: Real exchange rate
    plt.subplot(1, 3, 2)
    plt.title('Real exchange rate $Q$')
    plt.plot((J_Qrstar @ shock)[:Tplot], color='#008C00', label='Initial path of $Q$')
    plt.plot((J_Qrstar @ shock + J_Qr @ dr @ shock)[:Tplot], color='#A00000', linestyle='--', label='$Q$ to fight depreciation')
    plt.plot((J_Qrstar @ shock + J_Qr @ dr2 @ shock)[:Tplot], color='#0000A0', linestyle='dashdot', label='$Q$ to fight recession')
    plt.axhline(y=0, color='#808080', linestyle=':')
    plt.ylabel('Percent of s.s.')
    plt.xlabel('Quarters')
    plt.ylim(-0.1, 1.5)
    plt.legend(framealpha=0)
    
    # Panel 3: Output
    plt.subplot(1, 3, 3)
    plt.title('Output $Y$')
    plt.plot((J_yrstar @ shock)[:Tplot], color='#008C00', label='Initial path of $Y$')
    plt.plot((J_yrstar @ shock + J_yr @ dr @ shock)[:Tplot], color='#A00000', linestyle='--', label='$Y$ to fight depreciation')
    plt.plot((J_yrstar @ shock + J_yr @ dr2 @ shock)[:Tplot], color='#0000A0', linestyle='dashdot', label='$Y$ to fight recession')
    plt.axhline(y=0, color='#808080', linestyle=':')
    plt.xlabel('Quarters')
    plt.ylim(-0.5, 0.4)
    plt.legend(framealpha=0)
    
    plt.tight_layout()
    # plt.show()


# %%


def irfs_plotting(list_irfs, varss, labels=[" "], ylabel=r"Percent of s.s.", T_plot=30, figsize=(12, 6)):
    """
    Plot IRFs.

    Parameters:
    list_irfs: List of dictionaries containing IRFs for each variable.
    varss: List of variable names to plot.
    labels: List of labels for each IRF.
    ylabel: Label for the y-axis.
    T_plot: Number of periods to plot.
    figsize: Size of the overall figure.
    """
    if len(list_irfs) != len(labels):
        labels = [" "] * len(list_irfs)

    n_var = len(varss)  # Number of variables to plot
    fig, axes = plt.subplots(1, n_var, figsize=figsize, sharex=True)

    if n_var == 1:
        axes = [axes]  # Ensure axes is iterable if only one variable

    line_styles = ['--', '-.', '-']  # Line styles for different series
    colors = ['r', 'b', 'g']  # Colors for different series

    for i, var in enumerate(varss):
        ax = axes[i]
        # Plot all IRFs for the variable
        for j, irf in enumerate(list_irfs):
            ax.plot(
                np.arange(T_plot),
                100 * irf[var][:T_plot],
                line_styles[j % len(line_styles)],
                color=colors[j % len(colors)],
                label=labels[j]
            )

        ax.set_title(var, fontsize=12)
        ax.set_xlabel("Quarters", fontsize=10)
        if i == 0:
            ax.set_ylabel(ylabel, fontsize=10)

        ax.axhline(0, color="k", linestyle="--", linewidth=0.75)  # Add horizontal line at 0
        ax.grid(False)
        ax.legend(fontsize=8, loc='best', frameon=False)  # Add legend

    plt.tight_layout()  # Adjust layout for better appearance
    # plt.show()



def irfs_plotting_new(list_irfs_ra, list_irfs_ha, var, labels=[" "], ylabel=r"Percent of s.s.", T_plot=30, figsize=(12, 6), x_min=0):
    """
    Plot IRFs for RA and HA models side-by-side.

    Parameters:
    list_irfs_ra: List of dictionaries containing IRFs for the RA model.
    list_irfs_ha: List of dictionaries containing IRFs for the HA model.
    var: Variable name to plot (e.g., 'Y').
    labels: List of labels for each IRF.
    ylabel: Label for the y-axis.
    T_plot: Number of periods to plot.
    figsize: Size of the overall figure.
    """
    if len(list_irfs_ra) != len(labels) or len(list_irfs_ha) != len(labels):
        labels = [" "] * max(len(list_irfs_ra), len(list_irfs_ha))

    fig, axes = plt.subplots(1, 2, figsize=figsize, sharex=True, sharey=True)

    line_styles = ['--', '-.', '-']  # Line styles for different series
    colors = ['r', 'b', 'g']  # Colors for different series

    # Plot for RA model
    ax = axes[0]
    for j, irf in enumerate(list_irfs_ra):
        ax.plot(
            np.arange(T_plot),
            100 * irf[var][:T_plot],
            line_styles[j % len(line_styles)],
            color=colors[j % len(colors)],
            label=labels[j]
        )
    ax.set_title("Output, RA", fontsize=12)
    ax.set_xlabel("Quarters", fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.axhline(0, color="k", linestyle="--", linewidth=0.75)  # Add horizontal line at 0
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)  # Add dashed grid lines
    ax.legend(fontsize=8, loc='best', frameon=False)  # Add legend

    # Plot for HA model
    ax = axes[1]
    for j, irf in enumerate(list_irfs_ha):
        ax.plot(
            np.arange(T_plot),
            100 * irf[var][:T_plot],
            line_styles[j % len(line_styles)],
            color=colors[j % len(colors)],
            label=labels[j]
        )
    ax.set_title("Output, HA", fontsize=12)
    ax.set_xlabel("Quarters", fontsize=10)
    ax.axhline(0, color="k", linestyle="--", linewidth=0.75)  # Add horizontal line at 0
    
    ax.grid(False)
    ax.legend(fontsize=8, loc='best', frameon=False)  # Add legend
    ax.set_xlim(x_min, None)  # Ensure y-axis starts at `y_min` (default 0)
    plt.tight_layout()  # Adjust layout for better appearance
    # plt.show()

def irfs_plotting_new_ra(
    list_irfs_ra, varss, labels=[" "], ylabel=r"Percent of s.s.", T_plot=30, figsize=(12, 6), title_map=None, x_min=0
):
    """
    Plot Impulse Response Functions (IRFs) for RA models with multiple variables.

    Parameters:
    - list_irfs_ra: List of dictionaries containing IRFs for the RA model.
    - varss: List of variable names to plot (e.g., ['Y', 'C']).
    - labels: List of labels for each IRF.
    - ylabel: Label for the y-axis.
    - T_plot: Number of periods to plot.
    - figsize: Size of the overall figure.
    - title_map: Optional dictionary to map variable names to descriptive titles.
    - y_min: Minimum value for the y-axis (default is 0 for starting from horizontal axis).
    """
    if len(list_irfs_ra) != len(labels):
        labels = [" "] * len(list_irfs_ra)

    fig, axes = plt.subplots(1, len(varss), figsize=figsize, sharex=True)

    line_styles = ['--', '-.', '-']  # Line styles for different series
    colors = ['r', 'b', 'g']  # Colors for different series

    for i, var in enumerate(varss):
        ax = axes[i]
        for j, irf in enumerate(list_irfs_ra):
            ax.plot(
                np.arange(T_plot),
                100*irf[var][:T_plot],
                line_styles[j % len(line_styles)],
                color=colors[j % len(colors)],
                label=labels[j]
            )
        title = title_map.get(var, var) if title_map else var  # Use the title map if provided
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("Quarters", fontsize=10)
        if i == 0:
            ax.set_ylabel(ylabel, fontsize=10)
        ax.axhline(0, color="k", linestyle="--", linewidth=0.75)  # Add horizontal line at 0
        ax.set_xlim(x_min, None)  # Ensure y-axis starts at `y_min` (default 0)
        ax.grid(False)
        ax.legend(fontsize=8, loc='best', frameon=False)  # Add legend

    plt.tight_layout()  # Adjust layout for better appearance
    # plt.show()


def irfs_plotting_new_three(list_irfs_ra, list_irfs_ta, list_irfs_ha, var, labels=[" "], ylabel=r"Percent of s.s.", T_plot=30, figsize=(18, 6), x_min=0):
    """
    Plot Impulse Response Functions (IRFs) for RA, TA, and HA models side-by-side.

    Parameters:
    - list_irfs_ra: List of dictionaries containing IRFs for the RA model.
    - list_irfs_ta: List of dictionaries containing IRFs for the TA model.
    - list_irfs_ha: List of dictionaries containing IRFs for the HA model.
    - var: Variable name to plot (e.g., 'Y').
    - labels: List of labels for each IRF.
    - ylabel: Label for the y-axis.
    - T_plot: Number of periods to plot.
    - figsize: Size of the overall figure.
    """
    if len(list_irfs_ra) != len(labels) or len(list_irfs_ta) != len(labels) or len(list_irfs_ha) != len(labels):
        labels = [" "] * max(len(list_irfs_ra), len(list_irfs_ta), len(list_irfs_ha))

    fig, axes = plt.subplots(1, 3, figsize=figsize, sharex=True, sharey=True)

    line_styles = ['--', '-.', '-']  # Line styles for different series
    colors = ['r', 'b', 'g']  # Colors for different series

    # Plot for RA model
    ax = axes[0]
    for j, irf in enumerate(list_irfs_ra):
        ax.plot(
            np.arange(T_plot),
            100 * irf[var][:T_plot],
            line_styles[j % len(line_styles)],
            color=colors[j % len(colors)],
            label=labels[j]
        )
    ax.set_title("Output, RA", fontsize=12)
    ax.set_xlabel("Quarters", fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.axhline(0, color="k", linestyle="--", linewidth=0.75)  # Add horizontal line at 0
    ax.set_xlim(x_min, None)  # Ensure y-axis starts at `y_min` (default 0)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)  # Add dashed grid lines
    ax.legend(fontsize=8, loc='best', frameon=False)  # Add legend

    # Plot for TA model
    ax = axes[1]
    for j, irf in enumerate(list_irfs_ta):
        ax.plot(
            np.arange(T_plot),
            100 * irf[var][:T_plot],
            line_styles[j % len(line_styles)],
            color=colors[j % len(colors)],
            label=labels[j]
        )
    ax.set_title("Output, TA", fontsize=12)
    ax.set_xlabel("Quarters", fontsize=10)
    ax.axhline(0, color="k", linestyle="--", linewidth=0.75)  # Add horizontal line at 0
    ax.grid(False)
    ax.legend(fontsize=8, loc='best', frameon=False)  # Add legend

    # Plot for HA model
    ax = axes[2]
    for j, irf in enumerate(list_irfs_ha):
        ax.plot(
            np.arange(T_plot),
            100 * irf[var][:T_plot],
            line_styles[j % len(line_styles)],
            color=colors[j % len(colors)],
            label=labels[j]
        )
    ax.set_title("Output, HA", fontsize=12)
    ax.set_xlabel("Quarters", fontsize=10)
    ax.axhline(0, color="k", linestyle="--", linewidth=0.75)  # Add horizontal line at 0
    ax.set_xlim(x_min, None)  # Ensure y-axis starts at `y_min` (default 0)
    ax.grid(False)
    ax.legend(fontsize=8, loc='best', frameon=False)  # Add legend

    plt.tight_layout()  # Adjust layout for better appearance
    # plt.show()

def irfs_plotting_decom(
    list_irfs_ra, list_irfs_ra_in, list_irfs_ha, var, labels=[" "], ylabel=r"Percent of s.s.", T_plot=30, figsize=(12, 6), x_min=0
):
    """
    Plot Impulse Response Functions (IRFs) for RA and HA models side-by-side with adjusted line styles and colors.

    Parameters:
    - list_irfs_ra: List of dictionaries containing IRFs for the RA model.
    - list_irfs_ha: List of dictionaries containing IRFs for the HA model.
    - var: Variable name to plot (e.g., 'Y').
    - labels: List of labels for each IRF.
    - ylabel: Label for the y-axis.
    - T_plot: Number of periods to plot.
    - figsize: Size of the overall figure.
    """
    if len(list_irfs_ra) != len(labels) or len(list_irfs_ra_in) != len(labels) or len(list_irfs_ha) != len(labels):
        labels = [" "] * max(len(list_irfs_ra), len(list_irfs_ra_in), len(list_irfs_ha))

    fig, axes = plt.subplots(1, 3, figsize=figsize, sharex=True, sharey=True)

    # Define styles to match the legend
    line_styles = ['-', '--', '--', '--']  # Solid, Dashed, Dash-Dot, Dotted
    colors = ['k', 'g', 'r', 'b']  # Black, Green, Red, Blue

    # Plot for RA model
    ax = axes[0]
    for j, irf in enumerate(list_irfs_ra):
        ax.plot(
            np.arange(T_plot),
            100 * irf[var][:T_plot],
            line_styles[j % len(line_styles)],
            color=colors[j % len(colors)],
            label=labels[j]
        )
    ax.set_title("RA model - complete markets", fontsize=12)
    ax.set_xlabel("Quarters", fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.axhline(0, color="k", linestyle="--", linewidth=0.75)  # Add horizontal line at 0
    ax.grid(False)
    ax.legend(fontsize=8, loc='best', frameon=False)  # Add legend

    # Plot for RA incomp model
    ax = axes[1]
    for j, irf in enumerate(list_irfs_ra_in):
        ax.plot(
            np.arange(T_plot),
            100 * irf[var][:T_plot],
            line_styles[j % len(line_styles)],
            color=colors[j % len(colors)],
            label=labels[j]
        )
    ax.set_title("RA model - incomplete markets", fontsize=12)
    ax.set_xlabel("Quarters", fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.axhline(0, color="k", linestyle="--", linewidth=0.75)  # Add horizontal line at 0
    ax.grid(False)
    ax.legend(fontsize=8, loc='best', frameon=False)  # Add legend


    # Plot for HA model
    ax = axes[2]
    for j, irf in enumerate(list_irfs_ha):
        ax.plot(
            np.arange(T_plot),
            100 * irf[var][:T_plot],
            line_styles[j % len(line_styles)],
            color=colors[j % len(colors)],
            label=labels[j]
        )
    ax.set_title("HA model", fontsize=12)
    ax.set_xlabel("Quarters", fontsize=10)
    ax.axhline(0, color="k", linestyle="--", linewidth=0.75)  # Add horizontal line at 0
    ax.grid(False)
    ax.legend(fontsize=8, loc='best', frameon=False)  # Add legend
    ax.set_xlim(x_min, None)  # Ensure x-axis starts at `x_min`
    plt.tight_layout()  # Adjust layout for better appearance
    # plt.show()

