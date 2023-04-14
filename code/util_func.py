#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 17:41:22 2023

@author: Ali Moghaddam
"""

import numpy as np
import scipy.sparse as sp
import scipy.linalg as la, scipy.sparse as sp



def TwoQubit_Uni_U1(phi: np.ndarray) -> np.ndarray:
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


def Two_Qubit_Gate_on_Circuit(psi: np.ndarray, U: np.ndarray, m: int) -> np.ndarray:
    
    """
    Applies a two-qubit gate to a many-qubit state psi.

    Args:
        psi (np.ndarray): The state to apply the gate to. A 2^N dimensional numpy array (N = number of qubits).
        U (np.ndarray): The two-qubit unitary gate which is going to be applied. A 4x4 numpy array.
        m (int): the index of the first qubit in two neighboring qubits on which U is going to be applied.

    Returns:
        psi (np.ndarray): np.ndarray: The updated state after applying the gate. 
    """

    dim_qubits_on_left = 2 ** m     # Hilbert-space dimension of m qubits sitting before the two qubits on which U is applied
    psi = psi.reshape(dim_qubits_on_left, 4, -1).transpose(1, 0, 2).reshape(4, -1)
    psi = np.matmul(U, psi)
    psi = psi.reshape(4, dim_qubits_on_left, -1).transpose(1, 0, 2)

    return psi