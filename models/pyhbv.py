#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 27 10:34:17 2018

@author: Georgy Ayzel
"""

import numpy as np
from numba import jit

@jit
def _tf(maxbas):
    ''' 
    Transfer function weight generator 
    
    source: https://github.com/UNESCO-IHE/HILAB-HBV/
    '''
    wi = []
    for x in range(1, maxbas+1):
        if x <= (maxbas)/2.0:
            # Growing transfer
            wi.append((x)/(maxbas+2.0))
        else:
            # Receding transfer
            wi.append(1.0 - (x+1)/(maxbas+2.0))
    
    #Normalise weights
    wi = np.array(wi)/np.sum(wi)
    return wi

@jit
def _routing(q, maxbas=1):
    """
    Runoff routing using triangular weighting function
    
    source: https://github.com/UNESCO-IHE/HILAB-HBV/
    """
    maxbas = int(np.round(maxbas,0))
    
    # get the weights
    w = _tf(maxbas)
    
    # rout the discharge signal
    q_r = np.zeros_like(q, dtype='float64')
    q_temp = q
    for w_i in w:
        q_r += q_temp*w_i
        q_temp = np.insert(q_temp, 0, 0.0)[:-1]

    return q_r


@jit
def _SS2(I,C,D):
    '''
    Values of the S curve (cumulative HU curve) of GR unit hydrograph UH2
    Inputs:
       C: time constant
       D: exponent
       I: time-step
    Outputs:
       SS2: Values of the S curve for I
    '''
    FI = I+1
    if FI <= 0: SS2 = 0
    elif FI <= C: SS2 = 0.5*(FI/C)**D
    elif C < FI <= 2*C: SS2 = 1 - 0.5*(2 - FI/C)**D
    else: SS2 = 1
    return SS2


@jit
def _UH2(C, D, NH):
    '''
    C Computation of ordinates of GR unit hydrograph UH1 using successive differences on the S curve SS1
    C Inputs:
    C    C: time constant
    C    D: exponent
    C Outputs:
    C    OrdUH1: NH ordinates of discrete hydrograph
    '''
    OrdUH2 = np.zeros(2*NH)
    for i in range(2*NH):
        OrdUH2[i] = _SS2(i, C, D)-_SS2(i-1, C, D)
    return OrdUH2


@jit
def _routing_UH(q, X4):

    # parameter for unit hydrograph length
    NH = 20

    # Unit hydrograph states holders
    StUH2 = np.zeros(2*NH)
    # Computation of UH ordinates
    OrdUH2 = _UH2(X4, 2.5, NH)
    
    Q_routed = np.zeros(len(q))
    
    for i in range(len(q)):

        PRHU2 = q[i]
        
        # convolution of unit hydrograph UH2
        for k in range(int( max(1, min(2*NH-1, 2*int(X4+1))) )):
            StUH2[k] = StUH2[k+1] + OrdUH2[k] * PRHU2
        StUH2[2*NH-1] = OrdUH2[2*NH-1] * PRHU2

        # runoff from direct branch QD
        QD = max(0, StUH2[0])
        
        Q_routed[i] = QD

    return np.array(Q_routed)


@jit
def hbv(Temp, Prec, Evap, params):
    
    """
    Input:
    1. Meteorological forcing
        'T'- mean daily temperature (Celsium degrees)
        'P'- mean daily precipitation (mm/day)    
        'PET' - mean daily potential evaporation (mm/day)
    2. list of model parameters:
        HBV params:
        # BETA   - parameter that determines the relative contribution to runoff from rain or snowmelt
        #          [1, 6]
        # FC     - maximum soil moisture storage
        #          [50, 700]
        # K0     - recession coefficient for surface soil box (upper part of SUZ)
        #          [0.05, 0.99]
        # K1     - recession coefficient for upper groudwater box (main part of SUZ)
        #          [0.01, 0.8]
        # K2     - recession coefficient for lower groudwater box (whole SLZ)
        #          [0.001, 0.15]
        # LP     - Threshold for reduction of evaporation (SM/FC)
        #          [0.3, 1]
        # MAXBAS - routing parameter
        #          [1, 7]
        # PERC   - percolation from soil to upper groundwater box
        #          [0, 6]
        # UZL    - threshold parameter for groundwater boxes runoff (mm)
        #          [0, 100]
        # TT     - Temperature which separate rain and snow fraction of precipitation
        #          [-2.5, 2.5]
        # CFMAX  - Snow melting rate (mm/day per Celsius degree)
        #          [0.5, 5]
        # SFCF   - SnowFall Correction Factor
        #          [1, 1.5]
        # CFR    - Refreezing coefficient
        #          [0, 0.1] (usually 0.05)
        # CWH    - Fraction (portion) of meltwater and rainfall which retain in snowpack (water holding capacity)
        #          [0, 0.2] (usually 0.1)
    """
    
    # 1. parameters initialization
    parBETA, parFC, parK0, parK1, parK2, parLP, parMAXBAS,\
    parPERC, parUZL, parTT, parCFMAX, parSFCF, parCFR, parCWH = params
    
    # 2. initialize boxes and initial conditions
    # snowpack box
    SNOWPACK = np.zeros(len(Prec))
    # meltwater box
    MELTWATER = np.zeros(len(Prec))
    # soil moisture box
    SM = np.zeros(len(Prec))
    # soil upper zone box
    SUZ = np.zeros(len(Prec))
    # soil lower zone box
    SLZ = np.zeros(len(Prec))
    # actual evaporation
    ETact = np.zeros(len(Prec))
    # simulated runoff box
    Qsim = np.zeros(len(Prec))
    
    # 3. meteorological forcing preprocessing
    # precipitation separation
    # if T < parTT: SNOW, else RAIN
    RAIN = np.where(Temp  > parTT, Prec, 0)
    SNOW = np.where(Temp <= parTT, Prec, 0)
    # snow correction factor
    SNOW = parSFCF * SNOW
    
    # 4. The main cycle of calculations
    for t in range(1, len(Qsim)):

        # 4.1 Snow routine
        # how snowpack forms
        SNOWPACK[t] = SNOWPACK[t-1] + SNOW[t]
        # how snowpack melts
        # day-degree simple melting
        melt = parCFMAX * (Temp[t] - parTT)
        # control melting
        if melt<0: melt = 0
        melt = min(melt, SNOWPACK[t])
        # how meltwater box forms
        MELTWATER[t] = MELTWATER[t-1] + melt
        # snowpack after melting
        SNOWPACK[t] = SNOWPACK[t] - melt
        # refreezing accounting
        refreezing = parCFR * parCFMAX * (parTT - Temp[t])
        # control refreezing
        if refreezing < 0: refreezing = 0
        refreezing = min(refreezing, MELTWATER[t])
        # snowpack after refreezing
        SNOWPACK[t] = SNOWPACK[t] + refreezing
        # meltwater after refreezing
        MELTWATER[t] = MELTWATER[t] - refreezing
        # recharge to soil
        tosoil = MELTWATER[t] - (parCWH * SNOWPACK[t]);
        # control recharge to soil
        if tosoil < 0: tosoil = 0
        # meltwater after recharge to soil
        MELTWATER[t] = MELTWATER[t] - tosoil

        # 4.2 Soil and evaporation routine
        # soil wetness calculation
        soil_wetness = (SM[t-1] / parFC)**parBETA
        # control soil wetness (should be in [0, 1])
        if soil_wetness < 0: soil_wetness = 0
        if soil_wetness > 1: soil_wetness = 1
        # soil recharge
        recharge = (RAIN[t] + tosoil) * soil_wetness
        # soil moisture update
        SM[t] = SM[t-1] + RAIN[t] + tosoil - recharge
        # excess of water calculation
        excess = SM[t] - parFC
        # control excess
        if excess < 0: excess = 0
        # soil moisture update
        SM[t] = SM[t] - excess

        # evaporation accounting
        evapfactor = SM[t] / (parLP * parFC)
        # control evapfactor in range [0, 1]
        if evapfactor < 0: evapfactor = 0
        if evapfactor > 1: evapfactor = 1
        # calculate actual evaporation
        ETact[t] = Evap[t] * evapfactor
        # control actual evaporation
        ETact[t] = min(SM[t], ETact[t])

        # last soil moisture updating
        SM[t] = SM[t] - ETact[t]

        # 4.3 Groundwater routine
        # upper groudwater box
        SUZ[t] = SUZ[t-1] + recharge + excess
        # percolation control
        perc = min(SUZ[t], parPERC)
        # update upper groudwater box
        SUZ[t] = SUZ[t] - perc
        # runoff from the highest part of upper grondwater box (surface runoff)
        Q0 = parK0 * max(SUZ[t] - parUZL, 0)
        # update upper groudwater box
        SUZ[t] = SUZ[t] - Q0
        # runoff from the middle part of upper groundwater box
        Q1 = parK1 * SUZ[t]
        # update upper groudwater box
        SUZ[t] = SUZ[t] - Q1
        # calculate lower groundwater box
        SLZ[t] = SLZ[t-1] + perc
        # runoff from lower groundwater box
        Q2 = parK2 * SLZ[t]
        # update lower groundwater box
        SLZ[t] = SLZ[t] - Q2

        # Total runoff calculation
        Qsim[t] = Q0 + Q1 + Q2
    
    # 6. Scale effect accounting
    #Qsim_smoothed = Qsim
    # delay and smoothing simulated hydrograph
    # Triangular routing
    #Qsim_smoothed = _routing(Qsim, parMAXBAS)

    # Routing with Unit Hydrograph (GR4J-like)
    Qsim_smoothed = _routing_UH(Qsim, parMAXBAS)

    # 7. Sanity check
    # get rid of occasional negative values 
    #Qsim_smoothed = np.where(Qsim_smoothed != np.nan , Qsim_smoothed, 0)
    #Qsim_smoothed = np.where(Qsim_smoothed > 0, Qsim_smoothed, 0)
    
    return Qsim_smoothed, SNOWPACK, MELTWATER, SM, SUZ, SLZ, ETact

def hbv_bounds():
    '''
    source: Beck et al. (2016). Global-scale regionalization of hydrologic model parameters. 
    Water Resources Research. https://doi.org/10.1002/2015WR018247
    # BETA   - parameter that determines the relative contribution to runoff from rain or snowmelt
    #          [1, 6]
    # FC     - maximum soil moisture storage
    #          [50, 700]
    # K0     - recession coefficient for surface soil box (upper part of SUZ)
    #          [0.05, 0.99]
    # K1     - recession coefficient for upper groudwater box (main part of SUZ)
    #          [0.01, 0.8]
    # K2     - recession coefficient for lower groudwater box (whole SLZ)
    #          [0.001, 0.15]
    # LP     - Threshold for reduction of evaporation (SM/FC)
    #          [0.3, 1]
    # MAXBAS - routing parameter
    #          [1, 7]
    # PERC   - percolation from soil to upper groundwater box
    #          [0, 6]
    # UZL    - threshold parameter for groundwater boxes runoff (mm)
    #          [0, 100]
    # TT     - Temperature which separate rain and snow fraction of precipitation
    #          [-2.5, 2.5]
    # CFMAX  - Snow melting rate (mm/day per Celsius degree)
    #          [0.5, 5]
    # SFCF   - SnowFall Correction Factor
    #          [1, 1.5]
    # CFR    - Refreezing coefficient
    #          [0, 0.1] (usually 0.05)
    # CWH    - Fraction (portion) of meltwater and rainfall which retain in snowpack (water holding capacity)
    #          [0, 0.2] (usually 0.1)
    '''
    bnds = ((1, 6), (0.01, 1500), (0.01, 0.99), (0.01, 0.99), (0.001, 0.30), (0.1, 1), (1, 30),\
            (0.01, 10), (0.01, 100), (-2.5, 2.5), (0.5, 5), (0.5, 1.5), (0.001, 0.1), (0.001, 0.2))
    return bnds
    