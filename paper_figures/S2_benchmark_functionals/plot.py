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
plt.rcParams["figure.figsize"] = (10,6.2)

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



def atoms_from_db(db, **kwargs):
    # query the database to give EXACTLY one entry
    # return the atoms object
    for row in db.select(**kwargs):
        atoms = row.toatoms()
    return atoms


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
    cell_sizes = ['1x3', '2x3', '3x3', '1CO_4x3']
    # cell_sizes = ['1CO_4x3', '3x3', '2CO_4x3', '3CO_4x3', '1x3',]
    coverage_labels = {'1x3':'1', '2x3':r'$\frac{2}{3}$', '3x3':r'$\frac{1}{3}$', '1CO_4x3':r'$\frac{1}{4}$'}
    min_cov_cell = '1CO_4x3'
    mult_factor = {'1x3':12, '3CO_4x3':3, '2x3':6, '2CO_4x3':3, '3x3':4, '1CO_4x3':3, '4CO_4x3':3}
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', \
        'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan',\
        ]
    reg_colors = ['r', 'b', 'g', 'y']
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
    fig, ax = plt.subplots()



    ############################################################################

    # Plot the desorption energy as a function of temperature
#    fig, ax1 = plt.subplots()
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.set_xlabel(r'Relative $\theta_{CO}^{vacuum}$ ', fontsize=28)
    ax.set_ylabel(r'$\Delta E_{CO^{*}}^{TS}$ \ eV', fontsize=28)


    for index, exposure in enumerate(sorted(results_211)):
        ax.plot(results_211[exposure]['theta'],   -1 *results_211[exposure]['Ed'], \
            '.', alpha=1, color=colors[index], label=str(exposure) + 'L')#, color='tab:blue')



    ax.legend(bbox_to_anchor=(1.04,0.5), loc="center left", borderaxespad=0)
    # ax.annotate('b)', xy=(0., 1.1),xycoords="axes fraction", fontsize=24)

    ############################################################################


    ## Get DFT Energies for CO on Gold as a function of the coverage
    # Read in databases
    referencedb = connect('input_DFT/references_BEEF_VASP_500.db')
    # Get the gas phase energy of CO
    COgstat = referencedb.get(formula='CO', pw_cutoff=500.0)
    COg_E =  get_vibrational_energy(COgstat, [], method='novib', geometry='linear', \
            symmetrynumber=1)['E']
    # Get the energies of all thermodynamics
    thermodb = connect('input_DFT/Au_CO_coverage.db')
    # Get all enteries for BEEF
    # for the following cell sizes
    # and store it in the following dict
    dict_stat = AutoVivification()
    states = get_states(thermodb, 'BF', '1x3')
    for state in states:
        for cell in cell_sizes:
            stat = thermodb.get(states=state,\
                    functional='BF', \
                    cell_size=cell, \
                    facets='facet_211',
                    )
            dict_stat[cell][state] = stat



        # ax.axhline(y=diff_energy , ls='-', color=colors_state[state])
    for index, cell in enumerate(cell_sizes):
        energies_all = []
        for state in states:
            if state != 'slab':
                ads_energy = dict_stat[cell][state].energy \
                          -  dict_stat[cell]['slab'].energy \
                          - COg_E

                energies_all.append(ads_energy)
        # do not matter
        # add in free energy
        print(min(energies_all))
        correction_ads = thermo_ads.get_ZPE_correction()
        correction_gas = thermo_gas.get_ZPE_correction()
        correction = correction_ads - correction_gas
        ax.axhline(y=min(energies_all) + correction, color=reg_colors[index], lw=4)
        ax.annotate(coverage_labels[cell] + 'ML', xy=(0.8, min(energies_all)+ correction), color=reg_colors[index]).draggable()


    ############################################################################



    plt.tight_layout()
    # plt.savefig(output + 'TPD_compare.svg')
    plt.show()
