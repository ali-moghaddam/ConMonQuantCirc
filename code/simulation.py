#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Thu Apr 15 11:51:39 2023

@author: Ali Moghaddam
"""



import matplotlib.pyplot as plt
from utils import QuantumTrajectoryDynamics
from utils import *

def run(circuit_length, circuit_depth, phi_cte, gate_randomness, measurement_rate):

    EE = []
    initial_state = Neel(circuit_length)
    state = QuantumTrajectoryDynamics(circuit_length, initial_state, 0, {}, {})

    for _ in range(circuit_depth):
        state.even_time_unitary(phi_cte, gate_randomness)
        state.timestep += 1
        state.measurement_layer(measurement_rate)
        state.timestep += 1
        state.odd_time_unitary(phi_cte, gate_randomness, PBC = True)
        state.timestep += 1
        state.measurement_layer(measurement_rate)
        state.timestep += 1

        S = state.entanglement_entropy([circuit_length//2,circuit_length - circuit_length//2])
        EE.append(S)
    
    return EE


for length in [6,8,10,12]:
    EE = run(length, 20, 0., 1., 0.9)
    plt.plot(EE)
plt.savefig('EE9.pdf')
