#!/usr/bin/python

from ase.io import read, write
from ase.db import connect
import numpy as np
import os, sys
import argparse
sys.path.append('/Users/vijays/Documents/tools/scripts')
from useful_functions import get_vibrational_energy, get_sigma_wf, get_order, \
        energy_surface_charge, fit_to_curve, get_taylor_energy, energy_avg_wf
import matplotlib.pyplot as plt
from ase import units
plt.rcParams["figure.figsize"] = (12,16.1)


pccdb = connect('Au211_pcc_symmetric.db')
linmodpb = connect('Au211_linmodpb_symmetric.db')
referdb = connect('references_RPBE_vasp_600.db')

states = ['slab', 'CO_top', 'CO_bridge']

# Getting the gas phase energies

COgstat = referdb.get(implicit=False, formula='CO', functional='RPBE', \
                        pw_cutoff=600.0, paw='SSSP_efficiency_pseudos')

COg =  get_vibrational_energy(COgstat, method='ideal_gas', geometry='linear', \
        symmetrynumber=1)

# References dict
refer_dict = {
        'CO_top':COg['E'], 
        'CO_bridge':COg['E'], 

              }

all_objects_pcc = {}
all_objects_linmodpb = {}

# Create a list of atoms objects for plotting
for i in range(len(states)):
    temp_list_storage = []
    for row in pccdb.select(states = states[i]):
        temp_list_storage.append(row)
    all_objects_pcc[states[i]] = temp_list_storage
    temp_list_storage = []
    for row in linmodpb.select(states = states[i]):
        temp_list_storage.append(row)
    all_objects_linmodpb[states[i]] = temp_list_storage

for i in range(len(states)):
    # Plotting curves for each state
    # 1. Charge vs WF
    print(states[i])
    
    plt.figure()
    stat_pcc = all_objects_pcc[states[i]]
    stat_linmodpb = all_objects_linmodpb[states[i]]
    sigma_pcc, wf_pcc, wf0_pcc = get_sigma_wf(stat_pcc, 1)
    sigma_linmodpb, wf_linmodpb, wf0_linmodpb = get_sigma_wf(stat_linmodpb, 1)
    p_sigma_wf_pcc, fit_sigma_wf_pcc = fit_to_curve(wf_pcc-wf0_pcc, sigma_pcc, 1)
    p_sigma_wf_linmodpb, fit_sigma_wf_linmodpb = fit_to_curve(wf_linmodpb-wf0_linmodpb, sigma_linmodpb, 1)
    plt.plot(wf_pcc-wf0_pcc, sigma_pcc, 'ro', label='pcc')
    plt.plot(wf_pcc-wf0_pcc, p_sigma_wf_pcc(wf_pcc-wf0_pcc), 'r--')
    plt.plot(wf_linmodpb-wf0_linmodpb, sigma_linmodpb, 'bo', label='linmodpb')
    plt.plot(wf_linmodpb-wf0_linmodpb, p_sigma_wf_linmodpb(wf_linmodpb-wf0_linmodpb), 'b--')
    plt.xlabel(r'$\Phi - \Phi_{0} [eV] $')
    plt.ylabel(r'$\sigma [ \mu C / cm^{2} ] $')
    plt.annotate(r'$%1.1d \mu F / cm ^{2}$'%fit_sigma_wf_pcc[0], \
            xy = (wf_pcc[0]-wf0_pcc, sigma_pcc[0]), color='r')
    plt.annotate(r'$%1.1d \mu F / cm ^{2}$'%fit_sigma_wf_linmodpb[0], \
            xy = (wf_linmodpb[0]-wf0_linmodpb, sigma_linmodpb[0]), color='b')
    plt.title(states[i])
    plt.grid(False)
    plt.legend(loc='best')
    plt.savefig(states[i] + '_sigma_wf.png')
    plt.close()

    plt.figure()
    # 2. Taylor series for of ind state energy
    calculated_pcc, taylor_pcc = get_taylor_energy(stat_pcc)
    calculated_linmodpb, taylor_linmodpb = get_taylor_energy(stat_linmodpb)
    calc_charge_pcc, calc_energy_pcc = calculated_pcc
    calc_charge_linmodpb, calc_energy_linmodpb = calculated_linmodpb
    taylor_charge_pcc, taylor_energy_pcc = taylor_pcc
    taylor_charge_linmodpb, taylor_energy_linmodpb = taylor_linmodpb
    p_taylor_calc_pcc, fit_taylor_calc_pcc = fit_to_curve(calc_charge_pcc, calc_energy_pcc, 3)
    p_taylor_calc_linmodpb, fit_taylor_calc_linmodpb = fit_to_curve(calc_charge_linmodpb, calc_energy_linmodpb, 3)

    plt.grid(False)
    plt.plot(calc_charge_pcc, calc_energy_pcc, 'ro', label='Calculated pcc')
    plt.plot(calc_charge_linmodpb, calc_energy_linmodpb, 'bo', label='Calculated linmodpb')
    plt.plot(taylor_charge_pcc, taylor_energy_pcc, 'r--', label='Taylor series expansion pcc')
    plt.plot(taylor_charge_linmodpb, taylor_energy_linmodpb, 'b--', label='Taylor series expansion linmodpb')
    plt.legend(loc='best')
    plt.title(states[i])
    plt.ylabel('E [eV]')
    plt.xlabel('q [e]')
    plt.savefig(states[i] + '_taylor_energy.png')
    plt.close()

   
    if states[i] != 'slab':
        is_db_data_pcc = all_objects_pcc['slab']
        fs_db_data_pcc = all_objects_pcc[states[i]]
        is_data_pcc, fs_data_pcc = get_order(is_db_data_pcc, fs_db_data_pcc)
        sigma_pcc, rel_energy_pcc = energy_surface_charge(is_data_pcc, fs_data_pcc)
        avgwf_pcc, rel_energy_pcc = energy_avg_wf(is_data_pcc, fs_data_pcc)
        binding_energy_pcc = rel_energy_pcc - refer_dict[states[i]]

        is_db_data_linmodpb = all_objects_linmodpb['slab']
        fs_db_data_linmodpb = all_objects_linmodpb[states[i]]
        is_data_linmodpb, fs_data_linmodpb = get_order(is_db_data_linmodpb, fs_db_data_linmodpb)
        sigma_linmodpb, rel_energy_linmodpb = energy_surface_charge(is_data_linmodpb, fs_data_linmodpb)
        avgwf_linmodpb, rel_energy_linmodpb = energy_avg_wf(is_data_linmodpb, fs_data_linmodpb)
        binding_energy_linmodpb = ( rel_energy_linmodpb - 2 * refer_dict[states[i]] ) / 2

        # Getting fits
        p_sigma_deltaE_pcc, fit_sigma_deltaE_pcc = fit_to_curve(sigma_pcc, binding_energy_pcc, 2)
        p_avgwf_deltaE_pcc, fit_avgwf_deltaE_pcc = fit_to_curve(avgwf_pcc, binding_energy_pcc, 2)
        p_sigma_deltaE_linmodpb, fit_sigma_deltaE_linmodpb = fit_to_curve(sigma_linmodpb, binding_energy_linmodpb, 2)
        p_avgwf_deltaE_linmodpb, fit_avgwf_deltaE_linmodpb = fit_to_curve(avgwf_linmodpb, binding_energy_linmodpb, 2)

        # Plotting the Energy vs sigma curve
        plt.figure()
        plt.plot(sigma_pcc, binding_energy_pcc, 'ro', label='pcc')
        range_sigma_pcc = np.arange(min(sigma_pcc), max(sigma_pcc))
        plt.plot(range_sigma_pcc, p_sigma_deltaE_pcc(range_sigma_pcc), '--r')
        plt.plot(sigma_linmodpb, binding_energy_linmodpb, 'bo', label='linmodpb')
        range_sigma_linmodpb = np.arange(min(sigma_linmodpb), max(sigma_linmodpb))
        plt.plot(range_sigma_linmodpb, p_sigma_deltaE_linmodpb(range_sigma_linmodpb), '--b')
        plt.grid(False)

        plt.ylabel(r'$\Delta E [eV]$')
        plt.title(states[i])
        plt.annotate(r'$%1.4f x^{2} + %1.4f x + %1.4f$'%tuple(fit_sigma_deltaE_pcc), \
                xy = (range_sigma_pcc[0], p_sigma_deltaE_pcc(range_sigma_pcc[0])), \
                color='r')
        plt.annotate(r'$%1.4f x^{2} + %1.4f x + %1.4f$'%tuple(fit_sigma_deltaE_linmodpb), \
                xy = (range_sigma_linmodpb[0], p_sigma_deltaE_linmodpb(range_sigma_linmodpb[0])), 
                color='b')
        plt.xlabel(r'$\sigma [ \mu C / cm^{2} ]$')
        plt.legend(loc='best')
        plt.savefig(states[i] + '_sigma_deltaE.png')
        plt.close()
            
        plt.figure()
        plt.plot(avgwf_pcc, binding_energy_pcc, 'ro', label='pcc')
        range_avgwf_pcc = np.arange(min(avgwf_pcc), max(avgwf_pcc))
        plt.plot(avgwf_linmodpb, binding_energy_linmodpb, 'bo', label='linmodpb')
        range_avgwf_linmodpb = np.arange(min(avgwf_linmodpb), max(avgwf_linmodpb))
        #plt.plot(range_avgwf, p_avgwf_deltaE(range_avgwf), '--k') 
        plt.ylabel(r'$\Delta E [eV]$')
        plt.title(states[i])
        plt.xlabel(r'$\frac{\Phi_{1} + \Phi_{2}}{2} [eV]$')
        plt.savefig(states[i] + '_avgwf_deltaE.png')
        plt.close() 

        



        



    

