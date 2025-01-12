#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 27 21:23:05 2024

@author: Sodik Umurzakov
"""

import numpy as np
import sequence_jacobian as sj
from sequence_jacobian.utilities import discretize, interpolate, optimized_routines
from sequence_jacobian.blocks.auxiliary_blocks import jacobiandict_block
import scipy.optimize as opt
from sequence_jacobian.blocks.support.het_support import CombinedTransition, ForwardShockableTransition


def hh_init(rpost, w, n, cbarF, eis, a_grid, e_grid, M, Transfer):
    """ initialize guess for policy function iteration """
    Tf = - cbarF
    coh = (1 + rpost) * a_grid + w * n * e_grid[:, np.newaxis] + Tf + M*Transfer
    Va = (1 + rpost) * (0.2 * coh) ** (-1 / eis)
    return Va, coh

@sj.het(exogenous='Pi', policy='a', backward='Va',backward_init=hh_init)
def hh_HA(Va_p, a_grid, e_grid, coh, rsub_ha, beta, eis):

    """
    Single backward iteration step using endogenous gridpoint method for households with CRRA utility.
    """

    # Solve HH problem
    uc_nextgrid = (beta) * Va_p
    c_nextgrid = uc_nextgrid ** (-eis)
    a = interpolate.interpolate_y(c_nextgrid + a_grid, coh, a_grid)
    optimized_routines.setmin(a, a_grid[0])
    c = coh - a
    Va = (1 + rsub_ha) * c ** (-1 / eis)
    
    # Compute uce
    uce = c ** (-1 / eis) * e_grid[:,np.newaxis]
    
    return Va, a, c, uce

def hetinput(e_grid, w, n, rpost, rsub, M, Transfer, pfh, cbarF, markup_ss, a_grid, zeta_e, pi_e):
    """ Define inputs that go into household's problem """
    atw_rel = markup_ss*w*n
    Tf = - pfh*cbarF
    incrisk1 = (e_grid[:, np.newaxis])**(zeta_e*np.log(atw_rel))/ np.vdot(e_grid ** (1+zeta_e*np.log(atw_rel)), pi_e)
    labinc = e_grid[:, np.newaxis]*incrisk1*w*n
    coh = (1 + rpost) * a_grid + labinc + Tf + M*Transfer
    rsub_ha = rsub
    return coh, rsub_ha

# attach input to HA block
hh_HA = hh_HA.add_hetinputs([hetinput])

""" Baseline model """

# sectoral demand
@sj.simple
def hh_outputs(C, pfh, phh, eta, alpha, cbarF):
    cH = (1-alpha) * phh**(-eta) * C
    cF = cbarF + alpha * pfh**(-eta) * C
    CT = cH + cF
    return cH, cF, CT

# foreign demand
@sj.simple
def foreign_c(phf,  alphastar, gamma, Cstar, eps_dcp):
    cHstar = alphastar*phf**(-gamma*eps_dcp) * Cstar
    return cHstar

@sj.solved(unknowns={'Q': (0.1,2)}, targets=['uip'], solver="brentq")
def UIP(ruip, Q, rstar, eta, alpha):
    uip = Q / Q(+1) * (1+ruip) - (1+rstar)
    rstar_out = rstar
    if eta == 1:
        phf = Q**(-1/(1-alpha))
        phh = Q**(-alpha/(1-alpha))
        pfh = Q
    else: 
        phf = ((Q**(eta-1) - alpha)/(1-alpha))**(1/(1-eta))
        phh = ((1-alpha*Q**(1-eta))/(1-alpha))**(1/(1-eta))
        pfh = Q
    return uip, phf, phh, pfh, rstar_out

@sj.solved(unknowns={'J': 15., 'j': 15.}, targets=['Jres','jres'], solver="broyden_custom")
def income(y,Z,phh,J,j,rinc,dividend_X,pcX_home,markup_ss):
    dividend = (1-1/markup_ss)*y*phh
    div_tot = dividend
    if pcX_home == 1: div_tot += dividend_X
    Jres = div_tot + J(1) / (1 + rinc) - J
    jres = J(1) / (1 + rinc) - j
    n = y/Z
    w = Z*phh/markup_ss
    atw_n = w*n
    atw = atw_n/n
    return jres, Jres, atw_n, dividend, atw, w, n

@sj.simple
def revaluation(J,j,rpost_shock):
    rpost = J/j(-1) - 1 + rpost_shock
    return rpost
    
@sj.simple
def rsimple(r):
    rsub = r(-1)
    rinc = r
    ruip = r
    return rsub, rinc, ruip

@sj.simple
def profitcenters(Q,phh,cHstar,eps_dcp):
    dividend_X = (Q**(1-eps_dcp)*phh**(eps_dcp) - phh)*cHstar
    return dividend_X

@sj.solved(unknowns={'nfa': (-2,2)}, targets=['nfares'], solver="brentq")
def CA(nfa, pfh, phh, y, cF, cH, rpost, dividend_X, pcX_home):
    div_tot = 0
    if pcX_home == 1: div_tot += dividend_X
    nfares = phh*y - pfh*cF - phh*cH + div_tot + rpost*nfa(-1) + nfa(-1) - nfa
    netexports = phh*y - pfh*cF - phh*cH + div_tot
    return nfares, netexports

@sj.solved(unknowns={'piw': (-2,2)}, targets=['piwres'], solver="brentq")
def unions(n, UCE, CT, atw, piw, kappa_w, markup_ss, beta, frisch, eis, vphi):
    #Cstar = UCE ** (-eis)
    Cstar = CT
    piwres = piw - beta*piw(1) - kappa_w*markup_ss*(vphi*n**(1/frisch) * Cstar**(1/eis) / atw - 1/markup_ss)
    return piwres

@sj.simple
def goods_market(y, cH, cHstar):
    goods_clearing = cH + cHstar - y
    return goods_clearing

@sj.simple
def assets_market(A, nfa, j, B):
    assets_clearing = A - nfa - j - B
    return assets_clearing

""" Delayed substitution """

@sj.solved(unknowns={'xstar': 1,'x': 1}, targets=['xstarres','xres'], solver="brentq")
def xrule(phh,xstar,x,eta,alpha,theta_share,beta):
    xstarres = xstar-xstar.ss + (1-alpha)*eta*(1-beta*theta_share)*(phh-phh.ss) - beta*theta_share*(xstar(1)-xstar.ss)
    xres = x - (1-theta_share)*xstar - theta_share*x(-1)
    return xstarres, xres

@sj.solved(unknowns={'xstar_F': 1,'x_F': 1}, targets=['xstarres_F','xres_F'], solver="brentq")
def xrule_foreign(phf,xstar_F,x_F,gamma,alphastar,theta_share,beta_star):
    xstarres_F = xstar_F-xstar_F.ss + alphastar*gamma*(1-beta_star*theta_share)*(phf-phf.ss) - beta_star*theta_share*(xstar_F(1)-xstar_F.ss)
    xres_F = x_F - (1-theta_share)*xstar_F - theta_share*x_F(-1)
    return xstarres_F, xres_F

def x_to_xF(x,eta,alpha):
    if eta != 1:
        xF = (1-(1-alpha)**(1/eta)*x**(1-1/eta))**(eta/(eta-1))*alpha**(1/(1-eta))
    else:
        xF = alpha*(1-alpha)**((1-alpha)/alpha)*x**(-(1-alpha)/alpha)
    return xF

@sj.simple
def hh_outputs_ds(C, x, alpha, eta, cbarF):
    cH = x*C
    cF = cbarF + x_to_xF(x,eta,alpha)*C
    CT = cH + cF
    return cH, cF, CT

@sj.simple
def foreign_c_ds(x_F, Cstar):
    cHstar = x_F * Cstar
    return cHstar

""" Quantitative model """

@sj.solved(unknowns={'Q': (0.1,2)}, targets=['uip'], solver="brentq")
def UIP_quant(ruip, Q, P, rstar):
    uip = Q / Q(+1) * (1+ruip) - (1+rstar)
    E = Q*P
    return uip, E

@sj.solved(unknowns={'J': 15.,'j': 15.}, targets=['Jres','jres'], solver="broyden_custom")
def income_quant(y,w,Z,phh,J,j,rinc,dividend_X,pcX_home):
    n = y/Z
    dividend = y*phh - w*n
    div_tot = dividend
    if pcX_home == 1: div_tot += dividend_X
    atw_n = w * n
    atw = w
    Jres = div_tot + J(1) / (1 + rinc) - J
    jres = J(1) / (1 + rinc) - j
    return jres, Jres, atw, n, atw_n, dividend, div_tot

@sj.simple
def revaluation_quant(q,qH,J,j,i,E,pi,rstar,rsub,delta,rpost_shock,f_firm,f_F,foreign_owned):
    rpost_firm = J/j(-1) - 1 + rpost_shock
    rpost_F = (1 + delta * q)*E/(q(-1)*E(-1)*(1+pi)) - 1     # foreign long bonds
    rpost_H = (1 + delta * qH)/(qH(-1)*(1+pi)) - 1           # local long bonds
    rpost1 = f_firm*rpost_firm + f_F*rpost_F + (1-f_firm-f_F)*rpost_H
    rpost = (1-foreign_owned)*rpost1 + foreign_owned*rsub
    return rpost, rpost_F, rpost_H, rpost_firm

@sj.solved(unknowns={'i': 0}, targets=['ires'], solver="brentq")
def taylor(i,pi,piHH,phi_pi,phi_piHH,phi_pinext,phi_i,rss,ishock,realrate):
    if realrate == 0: 
        istar = rss + phi_piHH*piHH + phi_pi*pi
        ires = (1-phi_i)*istar + phi_i*i(-1) - i + ishock
    else:
        istar = rss + phi_pinext*pi(1)
        ires = (1-phi_i)*istar + phi_i*i(-1) - i + ishock
    return ires

@sj.solved(unknowns={'B':0}, targets=['Bres'], solver="brentq")
def fiscal(B, rinc, rpost_F, rpost_H, Bbar, rho_B):
    Bres = rho_B*(B(-1)-(rpost_F - rpost_H)*Bbar) - B
    Transfer = B - (1+rinc(-1))*B(-1) + (rpost_F - rpost_H)*Bbar
    return Transfer, Bres

@sj.solved(unknowns={'q': (1,25),'qH': (1,25)}, targets=['qres','qHres'], solver="brentq")
def longbonds(q, qH, rstar, i, delta):
    qres = q - (1 + delta * q(+1))/(1 + rstar)
    qHres = qH - (1 + delta * qH(+1))/(1 + i)
    return qres, qHres

@sj.solved(unknowns={'nfa': (-2,2)}, targets=['nfares'], solver="brentq")
def CA_quant(nfa, pfh, phh, y, cF, cH, rpost, rinc, dividend_X, pcX_home, rpost_F, rpost_H, a_F, a_H, Bbar):
    div_tot = 0
    if pcX_home == 1: div_tot += dividend_X
    nfares = phh*y - pfh*cF - phh*cH + div_tot + rpost(-1)*nfa(-1) + nfa(-1) - nfa + a_F*(rpost_F-rinc(-1)) + a_H*(rpost_H-rinc(-1)) + Bbar*(rpost_F - rpost_H)
    netexports = phh*y - phh*cH - pfh*cF
    return nfares, netexports

@sj.simple
def profitcenters_quant(Q,phh,phf,cHstar):
    dividend_X = (Q*phf-phh)*cHstar
    return dividend_X

@sj.solved(unknowns={'P':1}, targets=['Pres'], solver="brentq")
def pi_to_P(P, pi):
    Pres = P - P(-1) - pi
    return Pres

@sj.solved(unknowns={'piFH': 0,'PFH': 1}, targets=['piFHres','PFHres'], solver="broyden_custom")
def nkpc_I(piFH, PFH, rinc, E, kappa_I):
    PFHres = PFH(-1) + piFH - PFH
    piFHres = kappa_I * (E/PFH - 1) + piFH(+1) / (1 + rinc) - piFH
    return piFHres, PFHres

@sj.solved(unknowns={'piHH': 0,'PHH': 1}, targets=['piHHres','PHHres'], solver="broyden_custom")
def nkpc(piHH, PHH, w, Z, P, rinc, markup_ss, kappa_p):
    real_mc = w * P / (Z * PHH)
    PHHres = PHH(-1) + piHH - PHH
    piHHres = piHH - piHH(+1)/(1 + rinc) - kappa_p * markup_ss * (real_mc - 1/markup_ss)
    return piHHres, PHHres, real_mc

@sj.solved(unknowns={'piHF': 0,'PHF': 1}, targets=['piHFres','PHFres'], solver="broyden_custom")
def nkpc_X(piHF, PHF, PHH, rinc, E, kappa_X):
    PHFres = PHF(-1) + piHF - PHF
    piHFres = kappa_X * (PHH/(E*PHF)-1) + piHF(+1) / (1 + rinc) - piHF
    return piHFres, PHFres

@sj.simple
def cpi(piHH,piFH,alpha):
    piout = (1-alpha)*piHH + alpha*piFH
    return piout

@sj.simple
def prices(P,PHH,PHF,PFH):
    phh = PHH/P
    pfh = PFH/P                  # price of foreign goods in home currency
    phf = PHF                    # price of home goods in foreign currency
    return phh, phf, pfh

@sj.simple
def eq_quant(pi, piout, piw, w, r, i):
    real_wage = piw - pi - (w - w(-1))
    pires = piout - pi
    fisher = r + piout(1) - i
    return real_wage, pires, fisher

""" Impulse responses """

def rshock(sd,rho,ss,T,shock_type,Q0=False):

    """Compute path of shocks"""

    # Compute path for dQ
    dr = rho**np.arange(T)
    dQ = (1/(1+ss['r'])*np.triu(np.ones((T,T)), k=0)@dr)[0]

    # Adjust path of r to hit specific value of dQ at t=0
    if Q0 == False: 

        if shock_type == 'rstar':
            dr = sd*dr
            dQ = 1/(1+ss['r'])*np.triu(np.ones((T,T)), k=0)@dr
            
        elif shock_type == 'r':
            dr = sd*dr
            dQ = - 1/(1+ss['r'])*np.triu(np.ones((T,T)), k=0)@dr

    else:

        if shock_type == 'rstar':
            dr = sd*dr*Q0/(1/(1+ss['r'])*np.triu(np.ones((T,T)), k=0)@dr)[0]
            dQ = 1/(1+ss['r'])*np.triu(np.ones((T,T)), k=0)@dr
            
        elif shock_type == 'r':
            dr = sd*dr*Q0/(1/(1+ss['r'])*np.triu(np.ones((T,T)), k=0)@dr)[0]
            dQ = - 1/(1+ss['r'])*np.triu(np.ones((T,T)), k=0)@dr

    return dr, dQ
