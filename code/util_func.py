#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 10 17:41:22 2023

@author: Ali Moghaddam
"""

import numpy as np
import scipy.sparse as sp
import scipy.linalg as la, scipy.sparse as sp
from typing import Tuple



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


def Neel_state_DPS(N: int) -> np.ndarray:
    """
    Generates the direct product state (DPS) corresponding to a Neel state of N qubits, 
    a state where the spins are alternating up and down. 
    The state is represented as a 2^N-dimensional vector (a numpy array).
    
    Args:
    N: An integer indicating the number of qubits.

    Returns:
    A numpy array representing the Neel state of N qubits.
    """
    
    count = np.sum([2 ** (2 * m + 1) for m in range(N//2)])
    psi = np.zeros(2 ** N, dtype=complex)
    psi[count] = 1.
    return psi

def measure_single_qubit_sz(psi: np.ndarray, m: int) -> Tuple[np.ndarray, float, int]:
    """Perform a random single-qubit measurement on the specified qubit and update the state accordingly.

    Args:
        m (int): The index of the qubit to measure.
        psi (np.ndarray): The state vector of the system.

    Returns:
        psi (np.ndarray): The updated state vector after the measurement.
        prob_0 (float): The probability of measuring the qubit at site m in the |0> state.
        measurement_outcome (int): The outcome of the single-qubit measurement (0 or 1).

    """

    dim_qubits_on_left = 2 ** m     # Hilbert-space dimension of m qubits sitting before the two qubits on which U is applied
    psi = psi.reshape(dim_qubits_on_left, 2, -1).transpose(1, 0, 2).reshape(2, -1)
    psi_0 = psi[0, :]
    prob_0 = np.dot(psi_0, np.conjugate(psi_0)).real  


    if np.random.random() < prob_0:
        psi[1, :] = 0.
        psi = psi / np.sqrt(prob_0)
        measurement_outcome = 0
    else:
        psi[0, :] = 0.
        prob_1 = (lambda p: p if p >= 1e-10 else 1e-10)(1. - prob_0)
        psi = psi / np.sqrt(prob_1)
        measurement_outcome = 1

    psi = psi.reshape(2, dim_qubits_on_left, -1).transpose(1, 0, 2)

    return psi, prob_0, measurement_outcome
