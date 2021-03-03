#!/usr/bin/python

""" Script to compare TPD for different Gold facets """

import numpy as np
from glob import glob
from useful_functions import AutoVivification, get_vibrational_energy
from pprint import pprint
import os, sys
from matplotlib import cm
sys.path.append('../classes/')
from parser_function import get_stable_site_vibrations, get_gas_vibrations, \
                            get_coverage_details, diff_energies, \
                            get_lowest_absolute_energies,\
                            get_differential_energy
from parser_class import ParseInfo
import mpmath as mp
from ase.thermochemistry import HarmonicThermo, IdealGasThermo
from ase.io import read
from ase.db import connect
from matplotlib.ticker import FormatStrFormatter
# matplotlib.rc('text', usetex=True)
# matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
import matplotlib.pyplot as plt
from plot_params import get_plot_params


# get the atoms object
def atoms_from_db(db, **kwargs):
    # query the database to give EXACTLY one entry
    # return the atoms object
    for row in db.select(**kwargs):
        atoms = row.toatoms()
    return atoms

def main_DFT(thermodb, referdb, list_cells, facet, functional):
    parse = ParseInfo(thermodb, referdb, list_cells, facet, functional,ref_type='CO')
    parse.get_pourbaix()
    return parse.absolute_energy, parse.atoms


output = 'output/'
os.system('mkdir -p ' + output)

if __name__ == '__main__':

    get_plot_params()



    """ Frequencies for stable sites """
    # Here we choose the frequencies corresponding to the lowest energy
    # site for adsorption of CO from DFT
    vibration_energies = get_stable_site_vibrations()

    # Store just the ase vibrations object
    thermo_ads = {}
    for facet in vibration_energies:
        thermo_ads[facet] = HarmonicThermo(vibration_energies[facet])

    # Gas phase CO energies
    thermo_gas = get_gas_vibrations()

    """ Specific about coverage """
    # DFT specifics about coverages
    cell_sizes, coverages_cell, coverage_labels = get_coverage_details()

    # states that need to be plotted for each surface facet
    accept_states = {}
    accept_states['211'] = ['CO_site_8', 'CO_site_13']
    accept_states['100'] = ['CO_site_1', 'CO_site_0']
    accept_states['110'] = ['CO_site_4', 'CO_site_1', 'CO_site_5']
    accept_states['111'] = ['CO_site_0', 'CO_site_1', 'CO_site_2']
    accept_states['recon_110'] = ['CO_site_1', 'CO_site_4']

    """ DFT databases """
    # which facets to consider
    facets = ['211', '110', '100', '111']
    ls = ['-', '--', '-.']
    # Get reference for CO in vacuum
    referencedb = connect('../databases/references_BEEF_VASP_500.db')
    # Get the gas phase energy of CO
    COgstat = referencedb.get(formula='CO', pw_cutoff=500.0)
    COg_E =  get_vibrational_energy(COgstat, [], method='novib', geometry='linear', \
            symmetrynumber=1)['E']
    # Get the energies of all thermodynamics
    thermodb = connect('../databases/Au_CO_coverage.db')
    functional = 'BF'


    """ Stylistic """
    colors_facet = {'211':'tab:blue', '111':'tab:green', '100':'tab:red',\
                    '110':'tab:brown', 'recon_110':'tab:cyan'}
    ls_facet = {'211':'-', '111':'--', '110':'-', '100':'--'}
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', \
        'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan',\
        'r', 'b', 'g'] # List of colours
    colors_state = {'CO_site_8':'tab:brown', 'CO_site_13':'tab:olive', \
                    'CO_site_10':'tab:red', 'CO_site_7':'tab:blue'}


    fig, ax = plt.subplots(len(facets), figsize=(8, 20), dpi=600)
    inferno = cm.get_cmap('inferno', 4)
    viridis = cm.get_cmap('viridis', 5)

    #########################################################

    # Plot the IR spectra
    absE_DFT = AutoVivification()
    atoms_DFT = AutoVivification()

    for index, facet in enumerate(facets):
        absE_DFT[facet], atoms_DFT[facet] = main_DFT(thermodb, referencedb, cell_sizes[facet], facet, functional)
        avg_energy = get_differential_energy(absE_DFT[facet], atoms_DFT[facet], facet, COg_E)

        cells = cell_sizes[facet]
        coverages = list(reversed([coverages_cell[facet][cell] for cell in cells]))
        for ind_cell, cell in enumerate(cells):
            ind_state = 0
            for state in avg_energy:
                if state in accept_states[facet]:
                    try:
                        homedir = 'data_IR/facet_' + facet + '/beef_vdw/' + cell + '/' + state + '/dipole/'
                        data_ir = np.loadtxt(homedir + '/ir-spectra.dat')
                        wavenumber, intensity, rel_int = data_ir.transpose()
                        # if facet in ['211', '111']:
                        # ax[index].plot(coverages, avg_energy[state] , 'o-', label='Au('+facet+')', color=colors_facet[facet])
                        ax[index].plot(wavenumber, intensity, color=viridis(ind_cell), lw=3, ls=ls[ind_state])
                        ind_state += 1
                        ax[index].set_xlim([1850, 2150])
                    except OSError:
                        print(homedir)
                        continue
                if 'recon' in facet:
                    ax[index].annotate('Au('+facet.replace('_', '-')+')', xy=(1900, 4), color=colors_facet[facet])
                else:
                    ax[index].annotate('Au('+facet+')', xy=(2075, 3.5), color=colors_facet[facet])
            # if index == 0:
            ax[index].plot([], [], color=viridis(ind_cell), label=coverage_labels[facet][cell] + ' ML', lw=10)
            ax[index].legend(bbox_to_anchor=(0,1.02,1,0.2), loc="lower left",
                                mode="expand", borderaxespad=0, ncol=2)
            if index == int(len(facet)/2):
                ax[index].set_ylabel(r'$ Intensity $ / $ ( D / \AA )^{2} amu^{-1}$')
            if index == 3:
                ax[index].set_xlabel(r'Wavenumber / $cm^{-1}$')

    plt.tight_layout()
    plt.savefig(output + 'IR.svg')
    plt.savefig(output + 'IR.pdf')

    # plt.show()
