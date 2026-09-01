"""Coefficient-level form of the neutron Koning--Delaroche map."""

from __future__ import annotations
import numpy as np

COEFFICIENT_NAMES = (
    "E_F constant", "E_F A slope", "V1 constant", "V1 asymmetry", "V1 A slope",
    "V2 constant", "V2 A slope", "V3 constant", "V3 A slope", "V cubic",
    "rV constant", "rV A^(-1/3)", "aV constant", "aV A slope",
    "W1 constant", "W1 A slope", "W2 constant", "W2 A slope",
    "D1 constant", "D1 asymmetry", "D2 constant", "D2 sigmoid amplitude",
    "D2 sigmoid center", "D2 sigmoid width", "D denominator",
    "rD constant", "rD A^(1/3)", "aD constant", "aD A slope",
    "Vso constant", "Vso A slope", "Vso E slope",
    "rso constant", "rso A^(-1/3)", "aso constant",
    "Wso strength", "Wso denominator",
)

KD_COEFFICIENTS = np.asarray([
    -11.2814, 0.02646, 59.30, 21.0, 0.024, 0.007228, 1.48e-6,
    1.994e-5, 2.0e-8, 7.0e-9, 1.3039, 0.4054, 0.6778, 1.487e-4,
    12.195, 0.0167, 73.55, 0.0795, 16.0, 16.0, 0.0180, 0.003802,
    156.0, 8.0, 11.5, 1.3424, 0.01585, 0.5446, 1.656e-4,
    5.922, 0.0030, 0.0040, 1.1854, 0.647, 0.59, -3.1, 160.0,
])


def kd_parameters_from_coefficients(a, z, energy, log_shifts=None):
    """Evaluate the 15 local optical parameters from 37 global coefficients."""
    shifts = np.zeros(37) if log_shifts is None else np.asarray(log_shifts, float)
    if shifts.shape != (37,):
        raise ValueError("log_shifts must have shape (37,)")
    b = KD_COEFFICIENTS * np.exp(shifts)
    a, z, energy = np.broadcast_arrays(a, z, energy)
    a, z, energy = a.astype(float), z.astype(float), energy.astype(float)
    delta = (a - 2*z) / a
    ef = b[0] + b[1]*a
    de, a13 = energy - ef, a**(1/3)
    v1 = b[2] - b[3]*delta - b[4]*a
    v2, v3 = b[5] - b[6]*a, b[7] - b[8]*a
    vv = v1*(1 - v2*de + v3*de**2 - b[9]*de**3)
    rv = (b[10] - b[11]/a13)*a13
    av = b[12] - b[13]*a
    w1, w2 = b[14] + b[15]*a, b[16] + b[17]*a
    wv = w1*de**2/(de**2 + w2**2)
    d1 = b[18] - b[19]*delta
    d2 = b[20] + b[21]/(1 + np.exp((a-b[22])/b[23]))
    wd = d1*de**2/(de**2+b[24]**2)*np.exp(-d2*de)
    rd = (b[25]-b[26]*a13)*a13
    ad = b[27]-b[28]*a
    vso = (b[29]+b[30]*a)*np.exp(-b[31]*de)
    rso = (b[32]-b[33]/a13)*a13
    aso = np.broadcast_to(b[34], a.shape)
    wso = b[35]*de**2/(de**2+b[36]**2)
    return np.stack([vv,rv,av,wv,rv,av,wd,rd,ad,vso,rso,aso,wso,rso,aso], axis=-1)

