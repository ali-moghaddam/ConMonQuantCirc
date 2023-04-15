#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 11:51:39 2023

@author: Ali Moghaddam
"""


import numpy as np
import pylab as py
import scipy.sparse as sp
import scipy.linalg as la, scipy.sparse as sp
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import time
from datetime import datetime
from util_func import TwoQubit_Uni_U1 , Two_Qubit_Gate_on_Circuit, Neel_state_DPS, measure_single_qubit_sz

TwoQubitUni = TwoQubit_Uni_U1
Neel = Neel_state_DPS


class QuantumTrajectoryDynamics:
    
    """ 
        Create and evolves a new manybody state of N qubits.  N := "num_qubits" or "circuit length"
        Evolution consits of (a) two-qubit unitary gates which are successively applied to neighboring qubits.
                         and (b) projective measurements of single qubits

    Args:

        num_qubits: An integer indicating the number of qubits or in other words the length of quantum circuit
        
        psi : manybody state of N qubits. A 2^N-dimensional numpy array

    """

    def __init__(self, num_qubits: int, psi: np.ndarray):
        self.num_qubits = num_qubits
        self.psi = psi

    def even_time_unitary(self, phi_cte = np.zeros((6,1)), randomness = 1) -> np.ndarray:
        """ Applies two-qubit unitary gates successively to neighboring qubits in even time steps.

        Args:
            phi_cte: A constant (6-component) set of unitary phases. Default all 0.
            randomness: A float indicating the amount of randomness in unitary phases. Default is 1.

        Returns:
            psi: The updated manybody state of N qubits. A 2^N-dimensional numpy array.

        """

        num_TwoQubitGate = self.num_qubits//2
        phi = phi_cte + randomness * np.pi * np.random.rand(6, num_TwoQubitGate)
        
        for m in range(num_TwoQubitGate):
            self.psi = Two_Qubit_Gate_on_Circuit(self.psi, TwoQubitUni(phi[:, m]), 2 * m)       # This function applies two-qubit gate on circuit
        


    def odd_time_unitary(self, phi_cte = np.zeros((6,1)), randomness = 1., PBC = True) -> np.ndarray:
        """ Applies two-qubit unitary gates successively to neighboring qubits in odd time steps.

        Args:
            phi_cte: A constant (6-component) set of unitary phases. Default all 0.
            randomness: A float indicating the amount of randomness in unitary phases. Default is 1.
            PBC: A boolean indicating if we have periodic boundary condiction (PBC) or not.

        Returns:
            psi: The updated manybody state of N qubits. A 2^N-dimensional numpy array.

        """
        
        if PBC:        
            self.psi = self.psi.reshape(2 , 2**(self.num_qubits-1)).transpose(1, 0)
            self.even_time_unitary(phi_cte, randomness)
            self.psi = self.psi.reshape(2**(self.num_qubits-1), 2).transpose(1, 0)
        
        else:
            num_TwoQubitGate = self.num_qubits//2 - 1    # The number of gates for OBC case.
            phi = phi_cte + randomness * np.pi * np.random.rand(6, num_TwoQubitGate)

            for m in range(num_TwoQubitGate):
                self.psi = Two_Qubit_Gate_on_Circuit(self.psi, TwoQubitUni(phi[:, m]), 2 * m + 1)

    def measurement_layer(self, measurement_rate: float = 0.) -> np.ndarray:
        """ Applies projective measurements of single qubits.

        Args:
            measurement_rate: A float indicating the probability of measuring each qubit. Default is 0. for no measurement at all. 

        Returns:
            psi: The updated manybody state of N qubits. A 2^N-dimensional numpy array.

        """
        
        for m in range(self.num_qubits):
            if np.random.rand() < measurement_rate:
                self.psi = measure_single_qubit_sz(self.psi, m)    # This function applies projective measurement on single qubit


    


