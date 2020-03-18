#!/usr/bin/python

""" Script to compare TPD for different Gold facets """

from useful_classes import experimentalTPD
import numpy as np
from glob import glob
from useful_functions import AutoVivification, get_vibrational_energy
from pprint import pprint
import matplotlib.pyplot as plt
import os, sys
import mpmath as mp
from ase.thermochemistry import HarmonicThermo, IdealGasThermo
from ase.io import read
from ase.db import connect
from matplotlib.ticker import FormatStrFormatter

import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (16.1, 10)

def gaussian(x, a, x0, sigma):
    # called by gaussian_tpd for use with curve fit
    values = a * np.exp( - (x - x0)**2 / ( 2* sigma**2))
    return values

def main(tpd_filename, temprange, tempmin, beta):
    # Running the class experimentalTPD
    expTPD = experimentalTPD(tpd_filename, temprange, tempmin, beta)
    expTPD.collect_tpd_data()
    expTPD.get_normalized_data()
    expTPD.get_gaussian_tpd()
    expTPD.get_desorption_energy()

    # Fit the temperature Ed plot
    fit_Ed = np.polyfit(expTPD.temperature[0:-1], expTPD.Ed, 3)
    p_Ed = np.poly1d(fit_Ed)
    temp_range = np.linspace(min(expTPD.temperature), max(expTPD.temperature), 100)
    fit_gauss = gaussian(temp_range,*expTPD.popt)

    return {'gaussian_tpd':expTPD.gaussian_rates,
            'temperature':expTPD.temperature,
            'norm_rate':expTPD.normalized_rate,
            'Ed':expTPD.Ed,
            'theta':expTPD.theta,
            'p_Ed':p_Ed,
            'fit_gauss':fit_gauss,
            'temp_range':temp_range}


def get_lowest_absolute_energies(lst_stat_CO, lst_stat_slab, largest_cell, COg):
    for stat in lst_stat_CO:
        cell_size = stat.cell_size
        if cell_size == largest_cell:
            energy_CO = stat.energy
    for stat in lst_stat_slab:
        cell_size = stat.cell_size
        if cell_size == largest_cell:
            energy_slab = stat.energy

    return energy_CO - energy_slab - COg

def get_states(stat, functional, cell):
    # for a given database pick out all states
    allstates = []
    for row in stat.select(functional=functional, cell_size=cell):
        allstates.append(row.states)
    unique_states = np.unique(allstates)
    return unique_states

def get_number_CO(atoms):
    nC = [ atom.index for atom in atoms if atom.symbol == 'C']
    return len(nC)

def monotonic(x):
    dx = np.diff(x)
    return np.all(dx <= 0) or np.all(dx >= 0)

def diff_energies(lst_stat, mult_factor, COg, lowE_CO_abs):
    # take a list of database rows and convert it to differential energies
    # and coverages
    cell_lst = [ stat.cell_size for stat in lst_stat ]
    # get mulp factor based on the cell size
    mult_lst = []
    for cell in cell_lst:
        mult_lst.append(mult_factor[cell])

    sorted_lst = np.array(lst_stat)[np.argsort(mult_lst)]
    sorted_mult = np.sort(mult_lst)

    # except TypeError:
        # sorted_lst = lst_stat
        # sorted_mult = [1, 1, 1, 1]

    natoms_lst = []
    abs_ener_lst = []
    cell_sizes_sorted = []

    for i in range(len(sorted_lst)):
        atoms = sorted_lst[i].toatoms()
        cell_sizes_sorted.append(sorted_lst[i].cell_size)
        energies = sorted_lst[i].energy
        natoms = atoms.repeat([sorted_mult[i], 1, 1])
        natoms_lst.append(natoms)
        abs_ener_lst.append(energies)

    # get the number of CO in
    abs_ener_lst = np.array(abs_ener_lst)
    nCO_bigcell = []
    for atom in natoms_lst:
        nCO = get_number_CO(atom)
        nCO_bigcell.append(nCO)

    plot_nCO = nCO_bigcell
    plot_diff_energies = []
    plot_diff_energies.append(lowE_CO_abs)
    nCO_bigcell = np.array(nCO_bigcell)
    # Going from least coverage to most coverage
    for i in range(len(nCO_bigcell)-1):
        nCOdiff = np.array(nCO_bigcell[i+1]) - np.array(nCO_bigcell[i])
        diff_energies = ( sorted_mult[i+1] * abs_ener_lst[i+1] - \
                sorted_mult[i] * abs_ener_lst[i] - \
                nCOdiff * COg ) / nCOdiff
        plot_diff_energies.append(diff_energies)
    return [cell_sizes_sorted, plot_diff_energies]

def atoms_from_db(db, **kwargs):
    # query the database to give EXACTLY one entry
    # return the atoms object
    for row in db.select(**kwargs):
        atoms = row.toatoms()
    return atoms

def get_axis_limits(ax, scale=.9):
    return ax.get_xlim()[1]*scale, ax.get_ylim()[1]*scale

if __name__ == '__main__':

    ## CONSTANTS
    T_switch = 170 #K converting between 111 and 211 step
    T_max = 270 #K Where the TPD spectra ends
    T_min = [250, 300] #K Where the rate becomes zero - baseline
    beta = 3 #k/s Heating rate
    output = 'output/'
    os.system('mkdir -p ' + output)
    kB = 8.617e-05 # eV/K
    pco = 101325. #pressure of CO
    # Vibrational frequency
    # vibration_energies_ads = 0.00012 * np.array([2072.149, 282.507, 182.434, 173.679, 13.996])
    vibration_energies_ads = 0.00012 * np.array([1814.001, 367.591, 340.559, 230.796, 167.919, 69.744])

    thermo_ads = HarmonicThermo(vibration_energies_ads)
    atoms = read('co.traj')
    vibration_energies_gas = 0.00012 * np.array([2121.52, 39.91, 39.45])

    thermo_gas = IdealGasThermo(vibration_energies_gas, atoms = atoms, \
            geometry='linear', symmetrynumber=1, spin=0)

    coverages_cell = {'1x3': 1, '3CO_4x3':0.75, '2x3':0.5, '2CO_4x3':0.5, '3x3':0.33, '1CO_4x3':0.25, '4CO_4x3':1}
    #cell_sizes = ['1x3', '1CO_4x3', '2x3',  '3x3','2CO_4x3', '3CO_4x3']
    # cell_sizes = ['1CO_4x3', '2CO_4x3', '3CO_4x3', '4CO_4x3', ]
    cell_sizes = ['1CO_4x3', '3x3', '2CO_4x3', '3CO_4x3', '1x3',]
    min_cov_cell = '1CO_4x3'
    mult_factor = {'1x3':12, '3CO_4x3':3, '2x3':6, '2CO_4x3':3, '3x3':4, '1CO_4x3':3, '4CO_4x3':3}
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', \
        'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan',\
        'r', 'b', 'g']
    accept_states = ['CO_site_8', 'CO_site_13',] # CO states that are actually plotted
    colors_state = {'CO_site_8':'tab:brown', 'CO_site_13':'tab:olive', \
                    'CO_site_10':'tab:red', 'CO_site_7':'tab:blue'}

    # Stylistic
    colors_facet = {'211':'tab:red', '111':'tab:green'}
    ls_facet = {'211':'-', '111':'--'}

    ############################################################################
    # Take the TPD data from experiment
    input_csv = glob('input_TPD/*.csv')
    results_211 = AutoVivification()
    results_111 = AutoVivification()

    for f in input_csv:
        # For each exposure run the experimental TPD for 211
        exposure = float(f.split('/')[-1].split('.')[0].split('_')[1].replace('p', '.'))
        results_211[exposure] = main(f, [T_switch, T_max], T_min, beta)
        results_111[exposure] = main(f, [0, T_switch], T_min, beta)
    results = {'211':results_211, '111':results_111}
    # First figure - normalized TPD and the Gaussian fit

    # Plot the figure which shows the experimental and theoretical TPD curves
    fig, ax = plt.subplots(2, 2)

    ############################################################################
    # Figure 1: The gaussian fit of the TPD curve
    for facet, results_facet in results.items():
        for index, exposure in enumerate(sorted(results_facet)):
            if facet == '211':
                ax[0,0].plot(results_facet[exposure]['temperature'], results_facet[exposure]['norm_rate'],
                        '.', color=colors[index], alpha=0.40, label=str(exposure) + 'L')
            else:
                ax[0,0].plot(results_facet[exposure]['temperature'], results_facet[exposure]['norm_rate'],
                        '.', color=colors[index], alpha=0.40)
            # plt.plot(results_facet[exposure]['temperature'], results_facet[exposure]['gaussian_tpd'],
            #         '-', alpha=0.5, color=colors[index])
            ax[0,0].plot(results_facet[exposure]['temp_range'], results_facet[exposure]['fit_gauss'],
                    '-',  color=colors[index])

    ax[0,0].set_ylabel(r'$rate_{norm}$ \ $ML / s$')
    ax[0,0].set_xlabel(r'Temperature \ K')
    ax[0,0].annotate('a)', xy=(0., 1.1),xycoords="axes fraction", fontsize=24)


    ############################################################################

    # Plot the desorption energy as a function of temperature
#    fig, ax1 = plt.subplots()
    ax[0,1].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax[0,1].set_xlabel(r'Relative $\theta_{CO}^{vacuum}$ ', fontsize=28)
    ax[0,1].set_ylabel(r'Exp. $\Delta E_{CO^{*}}^{TS}$ \ eV', fontsize=28)


    for index, exposure in enumerate(sorted(results_211)):
        ax[0,1].plot(results_211[exposure]['theta'],   -1 *results_211[exposure]['Ed'], \
            '.', alpha=1, color=colors[index], label=str(exposure) + 'L')#, color='tab:blue')



    ax[0,1].legend(bbox_to_anchor=(1.04,0.5), loc="center left", borderaxespad=0)
    ax[0,1].annotate('b)', xy=(0., 1.1),xycoords="axes fraction", fontsize=24)

    ############################################################################

    # Plot temperature dependent rate and saturation coverages
    temp_range = np.linspace(100, 600, 500)
    ax[1,0].axvline(x=298.15, color='k', ls='--', lw=3, alpha=0.5)
    for facet, results_facet in results.items():
        ax[1,0].plot([], [], label='Au('+facet+')',  color='k', lw=3, ls=ls_facet[facet])
        for ind, exposure in enumerate(results_facet):
            theta_sat = []
            for index, T in enumerate(temp_range):
                entropy_ads = thermo_ads.get_entropy(temperature=T, verbose=False)
                entropy_gas = thermo_gas.get_entropy(temperature=T, \
                                                              pressure=pco, verbose=False)
                entropy_difference =  (entropy_gas - entropy_ads)
                free_correction = -1 * entropy_difference * T
                deltaE = results_facet[exposure]['p_Ed'](T)#results_211[exposure]['Ed'][index]
                deltaG = mp.mpf(deltaE + free_correction)
                # Get the rate constant based on the mpmath value
                K = mp.exp( -1 * deltaG / mp.mpf(kB) / mp.mpf(T) )
                partial_co = pco / 101325
                theta_eq = mp.mpf(1) / ( mp.mpf(1) + K / mp.mpf(partial_co) )
                theta_sat.append(theta_eq)
            ax[1,0].plot(temp_range, theta_sat, color=colors[ind], ls=ls_facet[facet], lw=3)
    ax[1,0].set_ylabel(r'Relative $\theta_{CO}^{eq}$ ', fontsize=28)
    ax[1,0].set_xlabel('Temperature \ K', fontsize=28)
    # ax[1,0].legend(bbox_to_anchor=(1.04,0.5), loc="center left", borderaxespad=0)
    ax[1,0].legend(loc='best', fontsize=18)
    ax[1,0].annotate('c)', xy=(0., 1.1),xycoords="axes fraction", fontsize=24)

    ############################################################################

    ## Get DFT Energies for CO on Gold as a function of the coverage
    # Read in databases
    referencedb = connect('../databases/references_BEEF_VASP_500.db')
    # Get the gas phase energy of CO
    COgstat = referencedb.get(formula='CO', pw_cutoff=500.0)
    COg_E =  get_vibrational_energy(COgstat, [], method='novib', geometry='linear', \
            symmetrynumber=1)['E']
    # Get the energies of all thermodynamics
    thermodb = connect('../databases/Au_CO_coverage.db')
    # Get all enteries for BEEF
    # for the following cell sizes
    # and store it in the following dict
    dict_stat = AutoVivification()
    states = get_states(thermodb, 'BF', '1x3')
    for state in states:
        for cell in cell_sizes:
            print(cell, state)
            stat = thermodb.get(states=state,\
                    functional='BF', \
                    cell_size=cell, \
                    facets='facet_211',
                    )
            dict_stat[state][cell] = stat

    # Now parse and plot the minimum energies
    for state in states:
        lst_stat_slab = []
        for cell in cell_sizes:
            if state == 'state_slab':
                lst_stat_slab.append(dict_stat[state][cell])
    all_diff_energies = []
    for state in states:
        if state != 'state_slab':
            lst_stat_CO = []
            for cell in cell_sizes:
                lst_stat_CO.append(dict_stat[state][cell])

        lowest_CO = get_lowest_absolute_energies(lst_stat_CO, lst_stat_slab, min_cov_cell, \
                COg_E)

        cell_sizes_sort, diff_energy  = diff_energies(lst_stat_CO, mult_factor, COg_E, lowest_CO)
        all_diff_energies.append(np.array(diff_energy))
        coverages = []
        for cell in cell_sizes:
            coverages.append(coverages_cell[cell])



        if diff_energy[0] < -0.3:
            print('Sites which have the CO binding energy in the range of int:')
            print(state)
            if state in accept_states:
                for cell in cell_sizes:
                    atoms = atoms_from_db(thermodb, **{'states':state, 'cell_size':cell, 'functional':'BF'})
                    atoms.write(output + state + '_' + cell + '_plotted.traj')
                # Only taking the negative ones to make the plot look nicer
                # Other sites are rather positive in energy and
                # do not matter
                # add in free energy
                correction_ads = thermo_ads.get_helmholtz_energy(temperature=300, verbose=False)
                correction_gas = thermo_gas.get_gibbs_energy(temperature=300, pressure=pco, verbose=False)
                correction = correction_ads - correction_gas
                ax[1,1].plot(coverages, diff_energy + correction, 'o-',
                     color=colors_state[state])

    ax[1,1].set_xlabel(r'$\theta_{CO^{*}}$ \ ML', fontsize=28)
    ax[1,1].set_ylabel(r'$\Delta G_{CO^{*}}$ \ eV', fontsize=28)
    ax[1,1].annotate('d)', xy=(0., 1.1),xycoords="axes fraction", fontsize=24)


    # plt.figure(3)
    # for exposure in results_111:
    #     plt.plot(results_111[exposure]['theta'], results_111[exposure]['Ed'])
    # plt.ylabel(r'$\Delta E_{d}$ / eV')
    # plt.xlabel(r'$\theta$ / ML')
    # plt.savefig(output + 'desorption_theta_111.pdf')
    # plt.close()

    ############################################################################



    plt.tight_layout()
    plt.savefig(output + 'TPD_compare.svg')
    plt.show()
