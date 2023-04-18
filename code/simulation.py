#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Thu Apr 15 11:51:39 2023

@author: Ali Moghaddam
"""



import matplotlib.pyplot as plt
from utils import *

def run(circuit_length, circuit_depth, phi_cte, gate_randomness, measurement_rate, PBC):

    EE = []
    initial_state = Neel(circuit_length)
    #state = QuantumTrajectoryDynamics(circuit_length, initial_state, 0, {}, {})
    state = ConservedTrajectoryDynamics(circuit_length, initial_state, 0, {}, {})


    for _ in range(circuit_depth):
        state.even_time_unitary(phi_cte, gate_randomness)
        state.timestep += 1
        
        S = state.entanglement_entropy([circuit_length//2,circuit_length - circuit_length//2])
        EE.append(S)

        state.measurement_layer(measurement_rate)
        state.timestep += 1

        S = state.entanglement_entropy([circuit_length//2,circuit_length - circuit_length//2])
        EE.append(S)

        state.odd_time_unitary(phi_cte, gate_randomness, PBC)
        state.timestep += 1
        
        S = state.entanglement_entropy([circuit_length//2,circuit_length - circuit_length//2])
        EE.append(S)

        state.measurement_layer(measurement_rate)
        state.timestep += 1

        S = state.entanglement_entropy([circuit_length//2,circuit_length - circuit_length//2])
        EE.append(S)
    
    return EE

lengths = [13,14,15,16]
for length in lengths:
    EE = run(length, 50, 0., 1., 0., False if length//2 == 0 else True)
    plt.plot(EE/np.log(2.), label = f'L = {length}')
plt.legend()
plt.savefig(f'EE{lengths}-OBC-consv.pdf')
