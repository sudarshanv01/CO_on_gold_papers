#!/usr/bin/python

""" Script to compare TPD for different Gold facets """

from useful_classes import experimentalTPD
import numpy as np
from glob import glob
from useful_functions import AutoVivification, get_vibrational_energy
from pprint import pprint
import matplotlib.pyplot as plt
import os, sys
from parser_class import Pourbaix
import mpmath as mp
from ase.thermochemistry import HarmonicThermo, IdealGasThermo
from ase.io import read
from ase.db import connect
from matplotlib.ticker import FormatStrFormatter

import matplotlib.pyplot as plt
# plt.rcParams["figure.figsize"] = (20,10)



def get_states(stat, functional, cell, facet):
    # for a given database pick out all states
    allstates = []
    for row in stat.select(functional=functional, cell_size=cell, facets='facet_'+facet):
        allstates.append(row.states)
    unique_states = np.unique(allstates)
    return unique_states


def atoms_from_db(db, **kwargs):
    # query the database to give EXACTLY one entry
    # return the atoms object
    for row in db.select(**kwargs):
        atoms = row.toatoms()
    return atoms


if __name__ == '__main__':


    output = 'output/'
    os.system('mkdir -p ' + output)

    cell_sizes = {
                  '100': ['1x1', '2x2', '3x3'],
                  '211': ['1x3', '2x3', '3x3'],
                  '111': ['1x1', '2x2', '3x3'],
                  '110': ['1x1', '2x2', '3x3'],
                  }

    coverages_cell = {
                      '100': {'1x1':1, '2x2':0.25, '3x3':0.11},
                      '211': {'1x3':1, '2x3':0.66, '3x3':0.33},
                      '111': {'1x1':1, '2x2':0.25, '3x3':0.11},
                      '110': {'1x1':1, '2x2':0.25, '3x3':0.11},
                      }

    coverage_labels = {
                  '100': {'1x1':'1', '2x2':r'$\frac{1}{4}$', '3x3':r'$\frac{1}{9}$'},
                  '211': {'1x3':'1', '2x3':r'$\frac{2}{3}$', '3x3':r'$\frac{1}{3}$'},
                  '111': {'1x1':'1', '2x2':r'$\frac{1}{4}$', '3x3':r'$\frac{1}{9}$'},
                  '110': {'1x1':'1', '2x2':r'$\frac{1}{4}$', '3x3':r'$\frac{1}{9}$'},
                  }

    mult_factor = {
                    '1x3':12, '3CO_4x3':3, '2x3':6,
                    '2CO_4x3':3, '3x3':4, '1CO_4x3':3,
                    '4CO_4x3':3
                    }

    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', \
        'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan',\
        ]

    reg_colors = ['r', 'b', 'g', 'y']


    ## Get DFT Energies for CO on Gold as a function of the coverage
    # Read in databases
    referencedb = connect('../databases/references_BEEF_VASP_500.db')
    # Get the gas phase energy of CO
    COgstat = referencedb.get(formula='CO', pw_cutoff=500.0)
    COg_E =  get_vibrational_energy(COgstat, [], method='novib', geometry='linear', \
            symmetrynumber=1)['E']

    # Get the energies of all thermodynamics
    # Get all enteries for BEEF
    # for the following cell sizes
    # and store it in the following dict
    thermodb = connect('../databases/Au_CO_coverage.db')
    for facet in cell_sizes:
        dict_stat = AutoVivification()

        states = get_states(thermodb, 'BF', cell_sizes[facet][0], facet)
        # print(states)
        for state in states:
            for cell in cell_sizes[facet]:
                print(state, cell, facet)
                stat = thermodb.get(states=state,\
                        functional='BF', \
                        cell_size=cell, \
                        facets='facet_'+facet,
                        )
                dict_stat[cell][state] = stat



        # if facet == '211':
        #     total_rows = int(len(states)/4) + 1
        else:
            total_rows = int(len(states)/4)
        total_columns = min(4, len(states)-1)
        fig, ax = plt.subplots(total_rows,total_columns, figsize=(16 * total_columns / 4,4*total_rows))

        for index, state in enumerate(states):
            if state != 'state_slab':
                energies_all = []
                coverages_all = []
                for cell in cell_sizes[facet]:
                    ads_energy = dict_stat[cell][state].energy \
                              -  dict_stat[cell]['state_slab'].energy \
                              - COg_E

                    energies_all.append(ads_energy)
                    coverages_all.append(coverages_cell[facet][cell])
                # do not matter
                # add in free energy

                row = int(index) / int(4)
                column = index % 4
                if int(total_rows) > 1:
                    ax[int(row), column].plot(coverages_all, energies_all, '-', color='tab:blue', lw=3)
                    ax[int(row), column].plot(coverages_all, energies_all, 'o', color='tab:green', mew=3, )
                    ax[int(row), column].set_xlabel(r'$\theta$ \ ML', )
                    ax[int(row), column].set_ylabel(r'$\Delta E_{CO}^{*}$ \ eV',)
                else:
                    ax[column].plot(coverages_all, energies_all, '-', color='tab:blue', lw=3)
                    ax[column].plot(coverages_all, energies_all, 'o', color='tab:green', mew=3, )
                    ax[column].set_xlabel(r'$\theta$ \ ML', )
                    ax[column].set_ylabel(r'$\Delta E_{CO^{*}}$ \ eV',)


        ############################################################################


        # fig.text(0.3, 0.04, 'common X', ha='center')
        # fig.text(0.04, 0.5, 'common Y', va='center', rotation='vertical')
        # fig.tight_layout()
        # if facet == '211':
            # Delete extra set of points
            # fig.delaxes(ax[-1,-1])
            # fig.delaxes(ax[-1,-2])
        if facet == '110':
            fig.delaxes(ax[-1,-1])
        plt.tight_layout()
        # plt.title('Au'+facet, weight='bold')
        plt.savefig(output + facet + '.pdf')
        plt.show()
