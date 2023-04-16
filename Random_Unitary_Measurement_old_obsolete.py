#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Updated on Thu Nov 16 09:10:39 2022

@author: Ali Moghaddam
"""


###################################################

# THIS CODE IS FOR PURELY REAL UNITARY EVOLUTION MEANING THAT ALL OF "U" AND "PSI" ELEMENTS ARE ALWAYS REAL.  #

# THIS CODE uses direct evaluation of manybody wavefunction coefficients under two-gate unitaries.           #

###################################################

import time
from datetime import datetime
import random
import numpy as np
import matplotlib.pyplot as plt

#from scipy.stats import unitary_group

# weak = 1.
# ###################################################

# def even_time_steps(N,theta,psi):
    
#     #beta = 0.05*np.pi*(np.random.random() -0.5)
    
#     for m in range(N//2):   # Note that N should be an even number anyhow.
           
#         randomness= 1.
#         theta=0.*np.pi+randomness*np.pi*np.random.random()
#         phi_1=0.*np.pi+randomness*np.pi*np.random.random()
#         phi_2=0.*np.pi+randomness*np.pi*np.random.random()
#         phi_3=0.0+.0*1.57*np.random.random()
#         phi_4=0.0+.0*1.57*np.random.random()
#         phi_5=0.0+.0*1.57*np.random.random()
    
#         x = np.cos(theta)
#         y = np.sin(theta)

#         # TwoQubit_small_mixing = np.array([[np.cos(beta), np.sin(beta), 0, 0],
#         #                              [-np.sin(beta),np.cos(beta),0,0],
#         #                              [0,0,1.,0],
#         #                              [0, 0, 0, 1.]])
        
#         TwoQubit_Unitary = np.array([[np.exp(1.j*phi_1), 0, 0, 0],
#                                      [0,np.exp(1.j*(phi_3+phi_4))*x,np.exp(1.j*(phi_3+phi_5))*y,0],
#                                      [0,-np.exp(1.j*(phi_3-phi_5))*y,np.exp(1.j*(phi_3-phi_4))*x,0],
#                                      [0, 0, 0, np.exp(1.j*phi_2)]])
        
#         # TwoQubit_Unitary = unitary_group.rvs(4)
        
#         #TwoQubit_Unitary = TwoQubit_Unitary @ TwoQubit_small_mixing


#         L_dim = 2**(2*m)
#         R_dim = 2**(N-2*m-2)
#         rest_dim = 2**(N-2)
#         psi = psi.reshape(L_dim,4,R_dim)
#         psi = psi.transpose(1,0,2)
#         psi = psi.reshape(4,rest_dim)
        
#         new_psi = np.matmul(TwoQubit_Unitary , psi)
#         new_psi = new_psi.reshape(4,L_dim,R_dim)
#         new_psi = new_psi.transpose(1,0,2)
        
#         psi = new_psi
        
#     return psi


# def odd_time_steps(N,theta,psi):
#     # for temporal-only random unitary gates

#     for m in range(N//2-1):   # Note that N should be an even number anyhow.

#         randomness= weak
#         theta=0.*np.pi+randomness*np.pi*np.random.random()
#         phi_1=0.*np.pi+0.*np.pi*np.random.random()
#         phi_2=0.*np.pi+0.*np.pi*np.random.random()
#         phi_3=0.0+.0*1.57*np.random.random()
#         phi_4=0.0+.0*1.57*np.random.random()
#         phi_5=0.0+.0*1.57*np.random.random()

#         x = np.cos(theta)
#         y = np.sin(theta)


#         TwoQubit_Unitary = np.array([[np.exp(1.j*phi_1), 0, 0, 0],
#                                   [0,np.exp(1.j*(phi_3+phi_4))*x,np.exp(1.j*(phi_3+phi_5))*y,0],
#                                   [0,-np.exp(1.j*(phi_3-phi_5))*y,np.exp(1.j*(phi_3-phi_4))*x,0],
#                                   [0, 0, 0, np.exp(1.j*phi_2)]])


#         L_dim = 2**(1+2*m)
#         R_dim = 2**(N-2*m-3)
#         rest_dim = 2**(N-2)
#         psi = psi.reshape(L_dim,4,R_dim)
#         psi = psi.transpose(1,0,2)
#         psi = psi.reshape(4,rest_dim)

#         new_psi = TwoQubit_Unitary @ psi
#         new_psi = new_psi.reshape(4,L_dim,R_dim)
#         new_psi = new_psi.transpose(1,0,2)

#         psi = new_psi



#     return psi


# # def odd_time_steps(N,theta,psi):
# #     # for temporal-only random unitary gates
    
# #     for m in range(N//2-1):   # Note that N should be an even number anyhow.
               
# #         randomness= 1.
# #         theta=0.*np.pi+randomness*np.pi*np.random.random()
# #         phi_1=0.*np.pi+randomness*np.pi*np.random.random()
# #         phi_2=0.*np.pi+randomness*np.pi*np.random.random()
# #         phi_3=0.0+.0*1.57*np.random.random()
# #         phi_4=0.0+.0*1.57*np.random.random()
# #         phi_5=0.0+.0*1.57*np.random.random()
    
# #         x = np.cos(theta)
# #         y = np.sin(theta)

    
# #         TwoQubit_Unitary = np.array([[np.exp(1.j*phi_1), 0, 0, 0],
# #                                   [0,np.exp(1.j*(phi_3+phi_4))*x,np.exp(1.j*(phi_3+phi_5))*y,0],
# #                                   [0,-np.exp(1.j*(phi_3-phi_5))*y,np.exp(1.j*(phi_3-phi_4))*x,0],
# #                                   [0, 0, 0, np.exp(1.j*phi_2)]])
        
        
# #         L_dim = 2**(1+2*m)
# #         R_dim = 2**(N-2*m-3)
# #         rest_dim = 2**(N-2)
# #         psi = psi.reshape(L_dim,4,R_dim)
# #         psi = psi.transpose(1,0,2)
# #         psi = psi.reshape(4,rest_dim)
        
# #         new_psi = np.matmul(TwoQubit_Unitary , psi  )
# #         new_psi = new_psi.reshape(4,L_dim,R_dim)
# #         new_psi = new_psi.transpose(1,0,2)

# #         psi = new_psi


        
# #    return psi


# #def odd_time_steps(N_site,theta,psi):
# #
# #    N = N_site
# #    psi = psi.reshape(2,2**(N-1))
# #    psi = np.transpose(psi)
# #
# #    psi = even_time_steps(N_site,theta,psi)
# #
# #    psi = psi.reshape(2**(N-1),2)
# #    psi = np.transpose(psi)
# #
# #    return psi
        


def red_den_mat(N,N_sub,MPS_to_DPS):
    
    N_site = N
    Dim_subsys = 2**(N_sub)
    N_prime =  (N_site - N_sub)
    Dim_subsys_prime =  2**N_prime
    
    N_prime_L =  0*(N_site - N_sub)//2
    N_prime_R =  N_site - N_sub - N_prime_L
    DPS_to_Kron = MPS_to_DPS.reshape(2**N_prime_L,2**N_sub,2**N_prime_R)
    
    ket=(DPS_to_Kron.transpose(1,0,2)).reshape(Dim_subsys, Dim_subsys_prime)
    #bra=(np.conjugate(DPS_to_Kron.transpose(0,2,1))).reshape(Dim_subsys_prime, Dim_subsys)
    
    rho_R = np.matmul( ket , np.conjugate(np.transpose(ket)) ) #bra

    return rho_R


# =============================================================================
# desity matrix of a subssytem of disjoint subsystems L and R
# =============================================================================

def red_den_mat_disjoint(N,cut_a,cut_b,cut_c,cut_d,MPS_to_DPS):
    
    Dim_prime_L = 2**(cut_a)
    Dim_sub_L = 2**(cut_b-cut_a)
    Dim_prime_center = 2**(cut_c-cut_b)
    Dim_sub_R = 2**(cut_d-cut_c)
    Dim_prime_R = 2**(N-cut_d)
    
    Dim_sub =  Dim_sub_L * Dim_sub_L
    Dim_prime = Dim_prime_L * Dim_prime_R * Dim_prime_center

    DPS_to_Kron = MPS_to_DPS.reshape(Dim_prime_L,Dim_sub_L,Dim_prime_center,Dim_sub_R,Dim_prime_R)
    ket = (DPS_to_Kron.transpose(1,3,0,2,4)).reshape(Dim_sub,Dim_prime)
    
    rho_sub = np.matmul( ket , np.conjugate(np.transpose(ket)) )

    return rho_sub


# =============================================================================
# desity matrix of a subssytem of subsystems at arbitrary position
# =============================================================================

def red_den_mat_single(N,cut_a,cut_b,MPS_to_DPS):
    
    Dim_prime_L = 2**(cut_a)
    Dim_sub = 2**(cut_b-cut_a)
    Dim_prime_R = 2**(N-cut_b)
    
    Dim_prime = Dim_prime_L * Dim_prime_R

    DPS_to_Kron = MPS_to_DPS.reshape(Dim_prime_L,Dim_sub,Dim_prime_R)
    ket = (DPS_to_Kron.transpose(1,0,2)).reshape(Dim_sub,Dim_prime)
    
    rho_sub = np.matmul( ket , np.conjugate(np.transpose(ket)) )

    return rho_sub



# =============================================================================
# Improved way of finding fluctuations - v9Nov2022
# =============================================================================


def expval_subsys_Sz(N,subsys,psi):

    spin = 0.
    
    for m in subsys:
        psi = psi.reshape(2**m,2,2**(N-m-1))
        ket = (psi.transpose(1,0,2)).reshape(2,2**(N-1))
        
        spin += np.sum(abs(ket[0,:])**2.-abs(ket[1,:])**2.)
                
    return spin


def expval_subsys_Sz2(N,subsys,psi):
    
    spin_fluc = 0.
    N_sub = len(subsys)

    for m in subsys:
        for mp in subsys:
            if  (m < mp):
                psi = psi.reshape(2**m,2,2**(mp-m-1),2,2**(N-mp-1))
                ket = (psi.transpose(1,3,0,2,4)).reshape(2,2,2**(N-2))
                spin_fluc += np.sum(abs(ket[0,0,:])**2.+abs(ket[1,1,:])**2. \
                                    - abs(ket[1,0,:])**2.  - abs(ket[0,1,:])**2.)
                
    return N_sub+2.*spin_fluc




###################################################

# def psi_Neel(N):
#     count = 0
#     for m in range(N//2):
#         count = count + 2**(2*m+1)
#     psi = np.zeros(2**N,dtype=np.complex128)
#     psi[count]=1.
#     #psi[0]=1.
#     return psi




###################################################



def measurement_single_random_a(m,N,psi,t):

    dim_L = 2**(m)
    dim_R = 2**(N-m-1)
    psi = psi.reshape(dim_L,2,dim_R)
    ket = (psi.transpose(1,0,2)).reshape(2,2**(N-1))
    

    prob_up = np.dot(ket[1,:],np.conjugate(ket[1,:]))  #np.sum(abs(ket[0,:])**2.)
    prob_down = np.dot(ket[0,:],np.conjugate(ket[0,:]))
    
    #print(psi.reshape(2**N))
    if (np.random.random() < prob_down):
        
        ket[1,:] = 0.
        ket = ket/np.sqrt(prob_down)
    else:
        
        ket[0,:] = 0.
        ket = ket/np.sqrt(prob_up)
        
    ket = ket.reshape(2,dim_L,dim_R)
    ket = (ket.transpose(1,0,2)).reshape(2**N)
  
    return ket








# ###################################################

def measurement_single_random_kim(m,N,psi,t):

    dim_L = 2**(m)
    dim_R = 2**(N-m-1)
    psi = psi.reshape(dim_L,2,dim_R)
    ket = (psi.transpose(1,0,2)).reshape(2,2**(N-1))
    

    prob_up = np.dot(ket[1,:],np.conjugate(ket[1,:]))  #np.sum(abs(ket[0,:])**2.)
    prob_down = np.dot(ket[0,:],np.conjugate(ket[0,:]))
    
    #print(psi.reshape(2**N))
    if (0.5 < prob_down):
        
        ket[1,:] = 0.
        ket = ket/np.sqrt(prob_down)
    else:
        
        ket[0,:] = 0.
        ket = ket/np.sqrt(prob_up)
  
    psi = ket
    #print(psi.reshape(2**N))
    return psi


           
    
# ###################################################



def measurement_rank_2(m,N,psi,t):
    #psi = psi_Neel(N)
    singlepsi = np.zeros(2,dtype=np.complex128)
    L_dim = 2**(m)
    R_dim = 2**(N-m-2)
    psi = psi.reshape(L_dim,4,R_dim)
    for i in range(L_dim):
        for j in range(R_dim):
            singlepsi[0] = singlepsi[0] + abs(psi[i,0,j])**2.  + abs(psi[i,3,j])**2.
            singlepsi[1] = singlepsi[1] + abs(psi[i,1,j])**2.  + abs(psi[i,2,j])**2.
    prob_para = singlepsi[0]
    prob_anti = singlepsi[1]
    
    if (np.random.random() < prob_anti):
  
        for i in range(L_dim):
            for j in range(R_dim):
                psi[i,0,j] = 0.
                psi[i,3,j] = 0.

                psi[i,1,j] = 1.*psi[i,1,j]
                psi[i,2,j] = 1.*psi[i,2,j]

        psi = psi/np.sqrt(prob_anti)
    else:
        for i in range(L_dim):
            for j in range(R_dim):
                
                psi[i,0,j] = 1.*psi[i,0,j]
                psi[i,3,j] = 1.*psi[i,3,j]

                psi[i,1,j] = 0.
                psi[i,2,j] = 0.
                

        psi = psi/np.sqrt(prob_para)
        
    psi_test = psi.reshape(2**N)
    psi_norm = np.sum(abs(psi_test)**2.)
    if abs(psi_norm-1.)>.001:
        print(psi_norm,'danger')
    # print(t, prob_up,prob_down)

    return psi
        

###################################################

def EE_fluc_calc(N,T_f,t,N_sub,psi):
    
    

    N_prime_L =  0*(N - N_sub)//2
    subsys = range(N_prime_L, N_prime_L+N_sub)
       
    SSz = expval_subsys_Sz(N, subsys, psi)
    SSz2 = expval_subsys_Sz2(N, subsys, psi)
    
    SpinFluc = SSz2 - SSz**2.
    tot_spin_z = SSz
    
    # SpinFluc=0.
    # tot_spin_z=0.
    
    rho_R = red_den_mat(N, N_sub,psi)
    
    lambdas = np.linalg.eigvalsh(rho_R)
    SvN = 0.
    for i in range(len(lambdas)):
        if  lambdas[i]> 10.**-8:
            SvN = SvN - lambdas[i] * np.log(lambdas[i])
            # SvN = SvN + lambdas[i]**2.

    EE = SvN.real # -(np.log(SvN)).real #
    
    return tot_spin_z,SpinFluc,EE

###################################################



###################################################

def fluc_mutual(N,T_f,t,cut_a,cut_b,cut_c,cut_d,MPS_to_DPS):
          

    subsys_A = range(cut_a, cut_b)
    SSz_A = expval_subsys_Sz(N, subsys_A, MPS_to_DPS)
    SSz2_A = expval_subsys_Sz2(N, subsys_A, MPS_to_DPS)

    subsys_B = range(cut_c, cut_d)
    SSz_B = expval_subsys_Sz(N, subsys_B, MPS_to_DPS)
    SSz2_B = expval_subsys_Sz2(N, subsys_B, MPS_to_DPS)

    subsys_AB = np.append(subsys_A , subsys_B)
    SSz_AB = SSz_A + SSz_B #expval_subsys_Sz(N, subsys_AB, MPS_to_DPS)
    SSz2_AB = expval_subsys_Sz2(N, subsys_AB, MPS_to_DPS)
    
    SpinFluc = SSz2_A - SSz_A**2. + SSz2_B - SSz_B**2. - (SSz2_AB - SSz_AB**2.)
    # SpinFluc = (SSz2_AB - SSz_AB**2.)

    tot_spin_z = SSz_AB
    
    return SpinFluc




###################################################




###################################################

def EE_mutual(N,T_f,t,cut_a,cut_b,cut_c,cut_d,MPS_to_DPS):
       
    
    rho_LR = red_den_mat_disjoint(N,cut_a,cut_b,cut_c,cut_d,MPS_to_DPS)
    rho_L = red_den_mat_single(N, cut_a, cut_b, MPS_to_DPS)
    rho_R = red_den_mat_single(N, cut_c, cut_d, MPS_to_DPS)


    
    def entropy_from_rho(rho):
        lambdas = np.linalg.eigvalsh(rho)
        EE = 0.
        for i in range(len(lambdas)):
            if  lambdas[i]> 10.**-8:
                EE = EE - lambdas[i] * np.log(lambdas[i])
                # EE = EE + lambdas[i]**2.
        return EE.real #-(np.log(EE)).real #
    
    EE_LR = entropy_from_rho(rho_LR)
    EE_L = entropy_from_rho(rho_L)
    EE_R = entropy_from_rho(rho_R)


    return EE_L + EE_R -EE_LR

###################################################






###################################################
def run(theta,N,N_sub,time_step,t_steady,T_f,measure_prop,cut_a,cut_b,cut_c,cut_d):
    
    measurement_single_random = measurement_single_random_a
    
    t=0
    
    mat_tot_spin_z = []
    mat_SpinFluc = []
    mat_EE = []
    mat_time_step = []
    mat_mutu_info = []
    num_measured = []
    
    mat_mutu_fluc = []
    
    MPS_to_DPS = psi_Neel(N)


    # cut_a,cut_b,cut_c,cut_d =1,3,8,10 # 1,3,5,7 #1,6,11,16 # 1,4,7,10 #  1,5,9,13 #  2,6,10,14 #

    while t<T_f:
        
# =============================================================================
# runs unitaries of even time steps
# =============================================================================

        
        # print(MPS_to_DPS.reshape(2**N))
        
        MPS_to_DPS = even_time_steps(N,theta,MPS_to_DPS)
        
        # print(MPS_to_DPS.reshape(2**N))

        
        t += 1
        
        
       ## for range(T_i_record, T_f, time_step p):

        if (t % time_step == 0 and t > t_steady):
            tot_spin_z,SpinFluc,EE =  EE_fluc_calc(N, T_f, t, N_sub, MPS_to_DPS)
            mutu_info = EE_mutual(N,T_f,t,cut_a,cut_b,cut_c,cut_d,MPS_to_DPS)
            mat_tot_spin_z = np.append(mat_tot_spin_z,tot_spin_z)
            mat_SpinFluc = np.append(mat_SpinFluc,SpinFluc)
            mat_EE = np.append(mat_EE,EE)
            mat_mutu_info =  np.append(mat_mutu_info,mutu_info)
            mat_time_step = np.append(mat_time_step,t)

            mutu_fluc = fluc_mutual(N,T_f,t,cut_a,cut_b,cut_c,cut_d,MPS_to_DPS)
            mat_mutu_fluc = np.append(mat_mutu_fluc,mutu_fluc)


# =============================================================================
# runs measurements after even time steps
# =============================================================================
    
            
        counter = 0
        
        for i in range(N):
            if np.random.random() < measure_prop:
                MPS_to_DPS = measurement_single_random(i,N,MPS_to_DPS,t)
                counter += 1
        num_measured = np.append(num_measured,counter)
            
        

        # MPS_to_DPS = measurement_single_random(0,N,MPS_to_DPS,t)
        # MPS_to_DPS = measurement_single_random(1,N,MPS_to_DPS,t)

        # print(MPS_to_DPS.reshape(2**N))


        counter +=2
        
        t += 1
                
        if (t % time_step == 0 and t > t_steady):
            tot_spin_z,SpinFluc,EE = EE_fluc_calc(N, T_f, t, N_sub, MPS_to_DPS)
            mutu_info = EE_mutual(N,T_f,t,cut_a,cut_b,cut_c,cut_d,MPS_to_DPS)#
            mat_tot_spin_z = np.append(mat_tot_spin_z,tot_spin_z)
            mat_SpinFluc = np.append(mat_SpinFluc,SpinFluc)
            mat_EE = np.append(mat_EE,EE)
            mat_mutu_info =  np.append(mat_mutu_info,mutu_info)
            mat_time_step = np.append(mat_time_step,t)
        
            mutu_fluc = fluc_mutual(N,T_f,t,cut_a,cut_b,cut_c,cut_d,MPS_to_DPS)
            mat_mutu_fluc = np.append(mat_mutu_fluc,mutu_fluc)
# =============================================================================
# runs unitaries of odd time steps
# =============================================================================
        
        
        MPS_to_DPS = odd_time_steps(N,theta,MPS_to_DPS)   #  _otherway
        
        t += 1
        
        if (t % time_step == 0 and t > t_steady):
            tot_spin_z,SpinFluc,EE = EE_fluc_calc(N, T_f, t, N_sub, MPS_to_DPS)
            mutu_info = EE_mutual(N,T_f,t,cut_a,cut_b,cut_c,cut_d,MPS_to_DPS) #
            mat_tot_spin_z = np.append(mat_tot_spin_z,tot_spin_z)
            mat_SpinFluc = np.append(mat_SpinFluc,SpinFluc)
            mat_EE = np.append(mat_EE,EE)
            mat_mutu_info =  np.append(mat_mutu_info,mutu_info)
            mat_time_step = np.append(mat_time_step,t)

            mutu_fluc = fluc_mutual(N,T_f,t,cut_a,cut_b,cut_c,cut_d,MPS_to_DPS)
            mat_mutu_fluc = np.append(mat_mutu_fluc,mutu_fluc)
            

# =============================================================================
# runs measurements after odd time steps
# =============================================================================

    

        counter = 0
        for i in range(N):
            if np.random.random() < measure_prop:
                MPS_to_DPS = measurement_single_random(i,N,MPS_to_DPS,t)
                counter += 1
        num_measured = np.append(num_measured,counter)


        # MPS_to_DPS = measurement_single_random(0,N,MPS_to_DPS,t)
        # MPS_to_DPS = measurement_single_random(1,N,MPS_to_DPS,t)
        # counter +=2
        
        t += 1
        
        
        if (t % time_step == 0 and t > t_steady):
            tot_spin_z,SpinFluc,EE = EE_fluc_calc(N, T_f, t, N_sub, MPS_to_DPS)
            mutu_info = EE_mutual(N,T_f,t,cut_a,cut_b,cut_c,cut_d,MPS_to_DPS) #
            mat_tot_spin_z = np.append(mat_tot_spin_z,tot_spin_z)
            mat_SpinFluc = np.append(mat_SpinFluc,SpinFluc)
            mat_EE = np.append(mat_EE,EE)
            mat_mutu_info =  np.append(mat_mutu_info,mutu_info)
            mat_time_step = np.append(mat_time_step,t)

            mutu_fluc = fluc_mutual(N,T_f,t,cut_a,cut_b,cut_c,cut_d,MPS_to_DPS)
            mat_mutu_fluc = np.append(mat_mutu_fluc,mutu_fluc)
    
        
    return mat_tot_spin_z, mat_SpinFluc, mat_EE,mat_time_step,counter,mat_mutu_info,num_measured,mat_mutu_fluc
   
    
   
# =============================================================================
# run for different realization and gather the results for fluctuations, EE, mutu_info
# =============================================================================
   

   



def data_gather(theta, N, N_sub, time_step, t_steady, T_f, measure_prop, num_realize, cut_a, cut_b, cut_c, cut_d):
    S_z_full = []
    S_fluc_full = []
    EE_full = []
    mutu_info_full = []
    num_measured_mat = []
    mutu_fluc_full = []

    for realization_index in range(num_realize):
        mat_tot_spin_z, mat_SpinFluc, mat_EE, mat_time_step, counter, mat_mutu_info, num_measured, mat_mutu_fluc = run(theta, N, N_sub, time_step, t_steady, T_f, measure_prop, cut_a, cut_b, cut_c, cut_d)
        num_measured_mat = np.append(num_measured_mat, num_measured)
        S_z_full = np.append(S_z_full, mat_tot_spin_z)
        S_fluc_full = np.append(S_fluc_full, mat_SpinFluc)
        EE_full = np.append(EE_full, mat_EE)
        mutu_info_full = np.append(mutu_info_full, mat_mutu_info)
        mutu_fluc_full = np.append(mutu_fluc_full, mat_mutu_fluc)
    
    steps = len(S_z_full)//num_realize
    S_z_full = S_z_full.reshape((num_realize, steps))
    S_fluc_full = S_fluc_full.reshape((num_realize, steps))
    mutu_info_full = mutu_info_full.reshape((num_realize, steps))
    EE_full = EE_full.reshape((num_realize, steps))
    mutu_fluc_full = mutu_fluc_full.reshape((num_realize, steps))

    return steps, mat_time_step, S_z_full, S_fluc_full, EE_full, mutu_info_full, num_measured_mat, mutu_fluc_full







def steady_vs_measurement(N):

    
    theta = 0.0*np.pi+0.0001
#    N=12
    N_sub=2 #N//2
    T_f=800
    t_steady = 400
    time_step = 1
    num_realize = 1
    
    
#    measure = []
#    svN = []
#    fluc = []
#    mutu = []
#    mutu_F = []
    

    distance = N-4
    cut_a,cut_b,cut_c,cut_d = 0,2,N-2,N #0,N//4,N//2,3*N//4 # 1,2,N-2,N-1 # 0,1,11,12 #3,4,11,12 #4,7,10,13 #5,7,9,11 #6,7,8,9 #4,5,6,7 #3,5,7,9  #1,4,7,10 #1,3,3+distance,5+distance
    
    def current_milli_time():
        return round(time.time() * 1000)

    t1 = current_milli_time()

    
    for measure_prop in np.arange(0.0, 0.5, 0.01):
        measure_prop += 0.0000000
        t2 = current_milli_time()- t1

        #print(measure_prop,t2)
        
        steps, mat_time_step, S_z_full, S_fluc_full, EE_full, mutu_info_full, num_measured_mat, mutu_fluc_full = data_gather(theta, N, N_sub, time_step, t_steady, T_f, measure_prop, num_realize, cut_a, cut_b, cut_c, cut_d)
        
        initial_time = mat_time_step[0]
        for timestep in list(mat_time_step):
            time_index=round(timestep-initial_time)
            for realization_index in range(num_realize):
                _EE_ = EE_full[realization_index,time_index]
                _fluc_ = S_fluc_full[realization_index,time_index]
                _mutu_info_ = mutu_info_full[realization_index,time_index]
                _mutu_fluc_ = mutu_fluc_full[realization_index,time_index]
                print(measure_prop, timestep, realization_index, _EE_, _fluc_, _mutu_info_, _mutu_fluc_)
        
        
#        steady_svN = np.sum(EE_full[:,:])/np.size(EE_full)
#        steady_fluc = np.sum(S_fluc_full[:,:])/np.size(EE_full)
#        steady_mutu = np.sum(mutu_info_full[:,:])/np.size(EE_full)
#        steady_mutu_F = np.sum((mutu_fluc_full[:,:]))/np.size(EE_full)
#        svN = np.append(svN , steady_svN )
#        fluc = np.append(fluc , steady_fluc )
#        mutu = np.append(mutu , steady_mutu )
#        measure = np.append(measure, measure_prop)
#        mutu_F = np.append(mutu_F , steady_mutu_F )
    
#    np.savetxt("EE-var-MI-MF-OBC-halfcycle-measuremens-N"+str(N)+"time"+str(current_milli_time())+".dat",np.c_[measure,svN,fluc,mutu,mutu_F], delimiter="\t", fmt='%2.7f')

    


def time_evolution(T_f,N):

    
    theta = 0.0*np.pi+0.0001
    N_sub=N//2
    t_steady = 0
    time_step = 1
    num_realize = 1
    
    
    cut_a,cut_b,cut_c,cut_d =0,N//4,N//2,3*N//4 # 1,2,N-2,N-1 # 0,1,11,12 #3,4,11,12 #4,7,10,13 #5,7,9,11 #6,7,8,9 #4,5,6,7 #3,5,7,9  #1,4,7,10 #1,3,3+distance,5+distance
    
    
    p=0.

    
        
    steps,mat_time_step,S_z_full,S_fluc_full,EE_full,mutu_info_full,num_measured_mat,mutu_fluc_full = \
    data_gather(theta,N,N_sub,time_step,t_steady,T_f,p,num_realize,cut_a,cut_b,cut_c,cut_d)

    svN = np.sum(EE_full,axis=0 )/(num_realize)
    fluc = np.sum(S_fluc_full,axis=0 )/(num_realize)
    mutu = np.sum(mutu_info_full,axis=0 )/(num_realize)
    mutu_F = np.sum(mutu_fluc_full,axis=0 )/(num_realize)
        
    np.savetxt("time-evolution-PBC-EE-var-MI-MF-PBC-halfcycle-measuremens-N"+str(N)+"-p-"+str(p)+"-.dat",np.c_[mat_time_step,svN,fluc,mutu,mutu_F], delimiter="\t", fmt='%2.7f')
        
        



    fig, ((a1,a3),(a2,a4)) = plt.subplots(nrows=2,ncols=2, figsize=(18,20), sharex=True)
        
    a1.plot(mat_time_step,EE_full[0,:],'-',ms=5 ,color=(0.7,0.0,0.),lw=2,label=r'$\epsilon_d=0.5$')
    a2.plot(mat_time_step,S_fluc_full[0,:],'-',ms=5 ,color=(0.,0.7,0.),lw=2,label=r'$\epsilon_d=0.5$')
    a3.plot(mat_time_step,mutu_info_full[0,:],'-',ms=5 ,color=(0.,0.0,0.7),lw=2,label=r'$\epsilon_d=0.5$')
    a4.plot(mat_time_step,mutu_fluc_full[0,:],'-',ms=5 ,color=(0.,0.0,0.7),lw=2,label=r'$\epsilon_d=0.5$')

    
    # a1.scatter(measure,svN,color=(0.7,0.0,0.))
    # a2.scatter(measure,fluc,color=(0.,0.7,0.))
    # a3.scatter(measure,mutu,color=(0.,0.0,0.7))
    # a4.scatter(measure,mutu_F,color=(0.,0.0,0.7))
    


    # a1.set_xlabel(r"$time$")
    # a2.set_xlabel(r"$time$")
    # a3.set_xlabel(r"$time$")


    # a1.set_ylabel(r"$S_{vN}$")
    # a2.set_ylabel(r"$var\: S_z$")
    # a3.set_ylabel(r"$I_{1,3}$")
    
    plt.savefig("time-ecolution-PBC-EE-var-MI-MF-PBC-halfcycle-measuremens-halfsubsys-quartersubsys-N"+str(N)+"-p-"+str(p)+"weak"+str(weak)+".pdf")

N=24
T_f = 800
#steady_vs_measurement(N)
time_evolution(T_f,N)
    

