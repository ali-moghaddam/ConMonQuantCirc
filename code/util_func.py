#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 17:41:22 2023

@author: Ali Moghaddam
"""

import numpy as np
import scipy.sparse as sp
import scipy.linalg as la, scipy.sparse as sp



def Two_Qubit_Unitary_U1(phi: np.ndarray) -> np.ndarray:
    """
    Calculates the U(1) charge-conserving unitary for a two-qubit gate (with 6 parameter overall).

    Args:
        phi (list[float]): The list of 5 phase and 1 angle parameters for the gate

    Returns:
        np.ndarray: A 4x4 numpy array representing the unitary matrix for the two-qubit gate.
    """
    x = np.cos(phi[5])
    y = np.sin(phi[5])

    unitary_matrix = np.array([
        [np.exp(1j * phi[0]), 0, 0, 0],
        [0, np.exp(1j * (phi[2] + phi[3])) * x, np.exp(1j * (phi[2] + phi[4])) * y, 0],
        [0, -np.exp(1j * (phi[2] - phi[4])) * y, np.exp(1j * (phi[2] - phi[3])) * x, 0],
        [0, 0, 0, np.exp(1j * phi[1])]
    ],dtype = complex)

    return unitary_matrix