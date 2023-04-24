#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Thu Apr 15 11:51:39 2023

@author: Ali Moghaddam
"""



import matplotlib.pyplot as plt
from utils import *


def run(circuit_length, circuit_depth, phi_cte, gate_randomness, measurement_rate, measure_strength, PBC):

    EE = []
    fluctuation = []

    initial_state = Neel(circuit_length)
    #state = QuantumTrajectoryDynamics(circuit_length, initial_state, 0, {}, {})
    state = ConservedTrajectoryDynamics(circuit_length, initial_state, 0, {}, {})


    for _ in range(circuit_depth):

        S, varSz = state.subsys_entanglemententropy_and_spinvariance([circuit_length//2,circuit_length - circuit_length//2])
        EE.append(S)
        fluctuation.append(varSz)

        state.even_time_unitary(phi_cte, gate_randomness)
        state.timestep += 1
        
        # S, varSz = state.subsys_entanglemententropy_and_spinvariance([circuit_length//2,circuit_length - circuit_length//2])
        # EE.append(S)
        # fluctuation.append(varSz)

        state.measurement_layer(measurement_rate, measure_strength)
        state.timestep += 1

        # S, varSz = state.subsys_entanglemententropy_and_spinvariance([circuit_length//2,circuit_length - circuit_length//2])
        # EE.append(S)
        # fluctuation.append(varSz)

        state.odd_time_unitary(phi_cte, gate_randomness, PBC)
        state.timestep += 1
        
        # S, varSz = state.subsys_entanglemententropy_and_spinvariance([circuit_length//2,circuit_length - circuit_length//2])
        # EE.append(S)
        # fluctuation.append(varSz)

        state.measurement_layer(measurement_rate, measure_strength)
        state.timestep += 1

    
    return np.array(EE), np.array(fluctuation)


# fig, ax = plt.subplots(1, 1, figsize=(10, 10))
fig, ((a1,a2)) = plt.subplots(ncols=1, nrows=2, figsize=(18,20), sharex=True) 

num_realizations = 5
lengths = [13,14,15,16]
for length in lengths:
    EE_avg = 0. 
    fluctuation_avg = 0.
    for i in range(num_realizations):
        EE, fluctuation = run(length, 50, 0., 1., 0.2, .3,False if length//2 == 0 else True)
        EE_avg += EE
        fluctuation_avg += fluctuation
    a1.plot(EE_avg/num_realizations, label = f'L = {length}')
    a2.plot(fluctuation_avg/num_realizations, label = f'L = {length}')
plt.legend()
plt.savefig(f'EE{lengths}-OBC-consv-p-0.2-lambdaa-.3.pdf')
