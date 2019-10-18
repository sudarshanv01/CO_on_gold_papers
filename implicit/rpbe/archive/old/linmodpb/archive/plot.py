#!/usr/bin/python

from ase.io import read, write
from ase.db import connect
import numpy as np
import os, sys
import argparse

sys.path.append("/Users/vijays/OneDrive - Danmarks Tekniske Universitet/project/tools/scripts")

from useful_functions import get_vibrational_energy
import matplotlib.pyplot as plt
from ase import units
plt.rcParams["figure.figsize"] = (12,16.1)


def energy_avg_wf(is_stat, fs_stat):
    # Returns the list of average wf and rel_energies
    # Takes in the db for IS and FS
    # Needs a list with all database entries

    # Getting relative energy
    is_energy_list = np.array([stat.energy for stat in is_stat ])
    fs_energy_list = np.array([stat.energy for stat in fs_stat ])

    rel_energy = fs_energy_list - is_energy_list

    # Getting avg WF
    is_wf = np.array([stat.wf for stat in is_stat])
    fs_wf = np.array([stat.wf for stat in fs_stat])

    avg_wf = ( is_wf + fs_wf ) / 2 

    return [ avg_wf, rel_energy ]


def energy_surface_charge(is_stat, fs_stat):
    # Returns the binding energy for different charges
    # Takes in the db entry for IS and FS
    # Getting relative energy
    is_energy_list = np.array([stat.energy for stat in is_stat ])
    fs_energy_list = np.array([stat.energy for stat in fs_stat ])

    rel_energy = fs_energy_list - is_energy_list

    # Getting charge
    is_charge = np.array([stat.tot_charge for stat in is_stat])
    fs_charge = np.array([stat.tot_charge for stat in fs_stat])

    # Getting surface area of is and fs
    # Taking the first atoms object 
    # and getting height and volume from there
    atoms = [stat.toatoms() for stat in is_stat]
    height = atoms[0].get_cell()[-1,-1]
    volume = atoms[0].get_volume()
    sa = volume / height

    # Finding surface charge
    is_sigma = is_charge / sa * (1e6 * 1.6e-19 / 1e-16 )
    fs_sigma = fs_charge / sa * (1e6 * 1.6e-19 / 1e-16 )

    return [is_sigma, rel_energy]



def get_order(is_data_list, fs_data_list):
    # Takes the is and fs database lists
    # makes sure that a list is returned with the same 
    # corresponding charges

    # Getting charge
    is_charge = np.array([stat.tot_charge for stat in is_data_list])
    fs_charge = np.array([stat.tot_charge for stat in fs_data_list])
    
    #Sorting based on charges
    is_arg_sorted = np.argsort(is_charge)
    fs_arg_sorted = np.argsort(fs_charge)
    # rearanging charges are db list
    is_charge_sorted = is_charge[is_arg_sorted]
    fs_charge_sorted = fs_charge[fs_arg_sorted]
    is_sorted = []
    fs_sorted = []
    for i in range(len(is_arg_sorted)):
        is_sorted.append(is_data_list[is_arg_sorted[i]])
    for i in range(len(fs_arg_sorted)):
        fs_sorted.append(fs_data_list[fs_arg_sorted[i]])

#    if np.all(is_charge_sorted, fs_charge_sorted):
#        return [is_sorted, fs_sorted]
#    else:

    is_new_sorted = []
    fs_new_sorted = []
    if len(is_charge_sorted) > len(fs_charge_sorted):
        for i in range(len(fs_charge_sorted)):
            charge_select = fs_charge_sorted[i]
            for j in range(len(is_charge_sorted)):
                if is_charge_sorted[j] == charge_select:
                    is_new_sorted.append(is_sorted[j])
        return [ is_new_sorted, fs_sorted ]
    elif len(is_charge_sorted) == len(fs_charge_sorted):
        return [is_sorted, fs_sorted]
    else:
        for i in range(len(is_charge_sorted)):
            charge_select = is_charge_sorted[i]
            for j in range(len(fs_charge_sorted)):
                if fs_sorted == charge_select:
                    fs_new_sorted.append(fs_sorted)[j]
        return [ is_sorted, fs_new_sorted ] 



def fit_to_curve(x, y, s):
    # Curve fitting in a function 
    fit = np.polyfit(x, y, s)
    p = np.poly1d(fit)
    return [p, fit]

def get_sigma_wf(stat, capacitance_order):
    # Takes in a set of atoms objects
    # and the order of capacitance to interpolate if there is no 
    # wf at zero charge available

    # Returns list of sigma, wf and wf at zero charge

    # Getting charge
    charge = np.array([stat.tot_charge for stat in stat])

    # Getting WF
    wf = np.array([stat.wf for stat in stat])

    atoms = [stat.toatoms() for stat in stat]
    height = atoms[0].get_cell()[-1,-1]
    volume = atoms[0].get_volume()
    sa = volume / height
    
    sigma = charge / sa * (1e6 * 1.6e-19 / 1e-16 ) 
    pos_zero = np.where(charge==0)[0]
    if pos_zero.size > 0:
        return [sigma, wf, wf[pos_zero]]
    else:
        # If something does not adsorb at zero charge
        # the wf at zero charge will not be known
        # Extrapolate based on a fixed capacitance

        p_charge_wf_0, fit_charge_wf_0 = fit_to_curve(sigma, wf, 1)

        return [ sigma, wf, p_charge_wf_0[0] ]

def get_taylor_energy(stat):
    # Get the Energy for a given state 
    # Based on a taylor series expansion
    # Give the range of q needed
    # Output is Energy as a function of qrange

    # Get needed variables
    sigma, wf, wf0 = get_sigma_wf(stat, 1)
    p_sigma_wf, fit_sigma_wf = fit_to_curve(wf-wf0, sigma, 1)
    capacNorm = fit_sigma_wf[0]
    # Getting the surface area
    atoms = [stat.toatoms() for stat in stat]
    height = atoms[0].get_cell()[-1,-1]
    volume = atoms[0].get_volume()
    sa = volume / height
    # Multiplying capacitance by area to get capacitance
    capacitance = capacNorm * sa
    # Get charge
    charge = np.array([stat.tot_charge for stat in stat])
    pos_zero = np.where(charge==0)[0]
    # Get energy
    energy = [st.energy for st in stat]
    # Get energy at zero charge
    Eq0 = energy[pos_zero[0]]
    print(Eq0)
    # Plot in the range of q calculated
    qrange = np.arange(min(charge), max(charge), 0.1)
    Eq = np.zeros(len(qrange))
    for i in range(len(qrange)):
        q = qrange[i]
        Eq[i] = Eq0 + wf0 * q + ( q ** 2 ) / ( 2 * capacitance ** 2) 
    return [[charge, energy], [qrange, Eq]]

def list_split(alist, split):
    list1 = alist[:split]
    list2 = alist[split:]
    return [list1, list2]
    
def get_different_capacitances(stat, breakpt):
    # Allow for a break in capcitances
    # Breakpt will take where the capacitances are different
    # Get needed variables
    sigma, wf, wf0 = get_sigma_wf(stat, 1)
    # Getting the surface area
    atoms = [stat.toatoms() for stat in stat]
    height = atoms[0].get_cell()[-1,-1]
    volume = atoms[0].get_volume()
    sa = volume / height
    # Getting charge
    charge = np.array([stat.tot_charge for stat in stat])
    # Get energy
    energy = [st.energy for st in stat]
    # Finding where zero charge is
    pos_zero = np.where(charge==0)[0]
    # Get energy at zero charge
    Eq0 = energy[pos_zero[0]]
    # Finding where to split the lists
    pos_break = np.where(charge==breakpt)[0]
    # Splitting the lists
    charge1, charge2 = list_split(charge, pos_break[0])
    wf1, wf2 = list_split(wf, pos_break[0])
    # Getting two different capacitances
    p1, fit1 = fit_to_curve(wf1-wf0, charge1, 1)
    p2, fit2 = fit_to_curve(wf2-wf0, charge2, 1)
    px_values = [p1, p2]
    fit_values = np.array([fit1, fit2]) *  (1e6 * 1.6e-19 / 1e-16 / sa )
    capac = [fit1[0], fit2[0]]
    qrange = {0:charge1, 1:charge2}
    wfrange = {0:wf1, 1:wf2}
    Eq = {0:[], 1:[]}
    for j in range(len(qrange)):
        C = capac[j]
        q = qrange[j]
        Eq_i = Eq0 + wf0 * q + ( q ** 2 ) / ( 2 * C ** 2)
        Eq[j].append(Eq_i)
    return {'charge_range':qrange, 'Eq':Eq, 'wf_range':wfrange, \
            'px_values':px_values, \
            'fit_values':fit_values, 
            }

def get_stark_shifts(stat):
    # Returns a list of frequencies and a list of charges
    vibrations = {}
    sigma, wf, wf0 = get_sigma_wf(stat, 1)
    vibrations_list = [ st.data.vibrations for st in stat]
    for i in range(len(sigma)):
        vibrations[sigma[i]] = np.array(vibrations_list[i]).real

    return {'sigma':sigma, 'vibrations':vibrations}


    
##############################################################################

if __name__ == '__main__':
    
    # reading in what states to plot
    CLI=argparse.ArgumentParser()
    CLI.add_argument(
            "--states", # What shows up in CLI
            nargs="*", # More than 1 arg expected
            type=str, 
            default=['slab']
            )
    CLI.add_argument(
            "--db",
            nargs="*",
            type=str,
            )
    CLI.add_argument(
            "--symmetric",
            nargs="*",
            type=bool,
            )

    # Parsing what I get from the CLI
    allargs = CLI.parse_args()
    
    states = allargs.states 
    database = allargs.db
    symmetric = allargs.symmetric

    all_objects = {}

    # Reading in the databases
    electrodb = connect(database[0])
    referdb = connect("/Users/vijays/OneDrive - Danmarks Tekniske Universitet/project/tools/references/reference_QE.db")


    # Getting the gas phase energies

    CO2gstat = referdb.get(implicit=False, formula='CO2', functional='RPBE', \
                            pw_cutoff=500.0, paw='SSSP_efficiency_pseudos')
    COgstat = referdb.get(implicit=False, formula='CO', functional='RPBE', \
                            pw_cutoff=500.0, paw='SSSP_efficiency_pseudos')
    H2gstat = referdb.get(implicit=False, formula='H2', functional='RPBE', \
                            pw_cutoff=500.0, paw='SSSP_efficiency_pseudos')
    H2Ogstat = referdb.get(implicit=False, formula='H2O', functional='RPBE', \
                            pw_cutoff=500.0, paw='SSSP_efficiency_pseudos')

    CO2g = get_vibrational_energy(CO2gstat, method='ideal_gas', geometry='linear', \
            symmetrynumber=2)
    COg =  get_vibrational_energy(COgstat, method='ideal_gas', geometry='linear', \
            symmetrynumber=1)
    H2Og =  get_vibrational_energy(H2Ogstat, method='ideal_gas', geometry='nonlinear', \
            symmetrynumber=2)
    H2g =  get_vibrational_energy(H2gstat, method='ideal_gas', geometry='linear', \
            symmetrynumber=2)

    # References dict
    refer_dict = {'COOH':CO2g['E'] + 0.5 * H2g['E'], \
                  'CO':CO2g['E'] + H2g['E'] - H2Og['E'], \
                  'H': 0.5 * H2g['E'], \
                  'CO2': CO2g['E'],\
                  'H2O': H2Og['E'], \
                  'CO_top':2 * COg['E'], \
                  'CO_bridge':2 * COg['E'], \
                  }


    for i in range(len(states)):
        temp_list_storage = []
        for row in electrodb.select(states = states[i]):
            temp_list_storage.append(row)
        all_objects[states[i]] = temp_list_storage
    for i in range(len(states)):
        # Plotting curves for each state
        # 1. Charge vs WF
        
        plt.figure()
        stat = all_objects[states[i]]
        sigma, wf, wf0 = get_sigma_wf(stat, 1)
        p_sigma_wf, fit_sigma_wf = fit_to_curve(wf-wf0, sigma, 1)
        
        plt.plot(wf-wf0, sigma, 'ro')
        plt.plot(wf-wf0, p_sigma_wf(wf-wf0), 'k--')
        plt.xlabel(r'$\Phi - \Phi_{0} [eV] $')
        plt.ylabel(r'$\sigma [ \mu C / cm^{2} ] $')
        plt.annotate(r'$%1.1d \mu F / cm ^{2}$'%fit_sigma_wf[0], \
                xy = (wf[0]-wf0, sigma[0]))
        plt.title(states[i])
        plt.savefig(states[i] + '_sigma_wf.png')
        plt.close()

        plt.figure()
        # 2. Taylor series for of ind state energy
        calculated, taylor = get_taylor_energy(stat)
        calc_charge, calc_energy = calculated
        taylor_charge, taylor_energy = taylor
        plt.plot(calc_charge, calc_energy, 'ro', label='Calculated')
        plt.plot(taylor_charge, taylor_energy, 'k--', label='Taylor series expansion')
        plt.legend(loc='best')
        plt.title(states[i])
        plt.ylabel('E [eV]')
        plt.xlabel('q [e]')
        plt.savefig(states[i] + '_taylor_energy.png')
        plt.close()



        if states[i] != 'slab':
            is_db_data = all_objects['slab']
            fs_db_data = all_objects[states[i]]
        
            is_data, fs_data = get_order(is_db_data, fs_db_data)
            sigma, rel_energy = energy_surface_charge(is_data, fs_data)
            avgwf, rel_energy = energy_avg_wf(is_data, fs_data)
            
            if symmetric == True:
                binding_energy = ( rel_energy - refer_dict[states[i]] ) / 2
            else:
                binding_energy = rel_energy - refer_dict[states[i]]
            # Getting fits
            p_sigma_deltaE, fit_sigma_deltaE = fit_to_curve(sigma, binding_energy, 2)
            p_avgwf_deltaE, fit_avgwf_deltaE = fit_to_curve(avgwf, binding_energy, 2)
            # Plotting the Energy vs sigma curve

            plt.figure()

            plt.plot(sigma, binding_energy, 'ro')
            range_sigma = np.arange(min(sigma), max(sigma))
            plt.plot(range_sigma, p_sigma_deltaE(range_sigma), '--k')
            plt.ylabel(r'$\Delta E [eV]$')
            plt.title(states[i])
            plt.xlabel(r'$\sigma [ \mu C / cm^{2} ]$')
            plt.savefig(states[i] + '_sigma_deltaE.png')
            plt.close()
                
            plt.figure()
            plt.plot(avgwf, binding_energy, 'ro')
            range_avgwf = np.arange(min(avgwf), max(avgwf))
            #plt.plot(range_avgwf, p_avgwf_deltaE(range_avgwf), '--k') 
            plt.ylabel(r'$\Delta E [eV]$')
            plt.title(states[i])
            plt.xlabel(r'$\frac{\Phi_{1} + \Phi_{2}}{2} [eV]$')
            plt.savefig(states[i] + '_avgwf_deltaE.png')
            plt.close() 

            
            # Plotting all frequencies as a function of surface charge
            plt.figure()
            vibrations_charge_dict = get_stark_shifts(stat)
            sigma = vibrations_charge_dict['sigma']
            frequencies_all = vibrations_charge_dict['vibrations']
            # storing all vibrations in a list and transpose
            allvib = np.zeros((len(sigma), len(frequencies_all[sigma[0]])))
            for l in range(len(sigma)):
                allvib[l,:] = np.array(frequencies_all[sigma[l]])
            # Plotting only the last frequency 
            # The largest one should be the stretching one
            # Hard coded need to check 
            plt.plot(sigma, allvib[:,-1], 'ko')
            # Fitting a line though the frequencies
            p_stark_shift, fit_stark_shift = fit_to_curve(sigma, allvib[:,-1],1)
            plt.plot(sigma, p_stark_shift(sigma), 'r--')
            plt.title(states[i])
            plt.ylabel(r'$\nu [cm^{-1}]$')
            plt.xlabel(r'$\sigma [ \mu C / cm^{2} ]$')
            plt.savefig(states[i] + '_stark_shift.png') 



        

