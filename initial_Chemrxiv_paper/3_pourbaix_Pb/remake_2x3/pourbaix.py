#!/usr/bin/python


""" Pourbaix plot """
import numpy as np
from useful_functions import AutoVivification
from pprint import pprint
from ase.db import connect
import matplotlib.pyplot as plt
import os
import csv
plt.rcParams["figure.figsize"] = (22, 9)

class Pourbaix:
    def __init__(self, thermodb, referdb, list_cells,functional):
        # Takes in the databases and the list of cells
        # for that facet
        self.thermodb = thermodb
        self.referdb = referdb
        self.list_cells = list_cells
        self.results = AutoVivification()
        self.rel_energy = AutoVivification()
        self.functional = functional
        self.states = AutoVivification()
        self.min_energy_cell = {}


    def _get_states(self, cell):
        # for a given database pick out all states
        allstates = []
        for row in self.thermodb.select(functional=self.functional, cell_size=cell):
            allstates.append(row.states)
        unique_states = np.unique(allstates)
        return unique_states

    def _parse_energies(self):
        # Gets all energies from the database
        # This is a func specific to this
        for cell in self.list_cells:
            self.states[cell] = self._get_states(cell)
            for state in self.states[cell]:
                try:
                    stat = self.thermodb.get(states=state,
                                        functional=self.functional,
                                        cell_size=cell,
                                        )
                except AssertionError:
                    stat = self.thermodb.get(states=state,
                                        functional=self.functional,
                                        cell_size=cell,
                                        opt=True,
                                        )
                atoms = stat.toatoms()
                output = cell + '/' + state + '/dipole/'
                os.system('mkdir -p ' + output)
                atoms.write(output+'CONTCAR')
                self.results[cell][state] = stat.energy

    def _get_reference(self):
        # TODO: Check if the reference is "correct"
        Pbstat = self.referdb.get(formula='Pb4',
                              functional=self.functional,
                              pw_cutoff=500.0)
        # Four atoms in Pb unit cell
        Pb_energies = Pbstat.energy / 4
        self.energy_reference = Pb_energies


    def get_pourbaix(self):
        # Get the DFT energies
        self._parse_energies()
        # Get the reference energies
        self._get_reference()
        for index, cell in enumerate(self.list_cells):
            energy_cell = []
            for state in self.states[cell]:
                if state != 'slab':

                    self.rel_energy[cell][state] = self.results[cell][state] - \
                                        self.results[cell]['slab'] - \
                                        self.energy_reference
                    energy_cell.append(self.rel_energy[cell][state])
            energy_cell = np.array(energy_cell)
            self.min_energy_cell[cell] = min(energy_cell)


def main(thermodb, referdb, list_cells, functional):
    pour = Pourbaix(thermodb, referdb, list_cells, functional)
    pour.get_pourbaix()
    return pour.min_energy_cell


def get_axis_limits(ax, scale=.9):
    return ax.get_xlim()[1]*scale, ax.get_ylim()[1]*scale



if __name__ == '__main__':
    ## Details
    output = 'output/'
    os.system('mkdir -p ' + output)

    facets = ['111', '100', '110', '211']
    functional = 'BF'
    no_electrons_Pb = 2 # Two electrons are transferred
    U_PB_2_SHE = 0.13 # V To convert to SHE
    referdb = connect('reference_lead.db')
    colors_facet = {'211': 'tab:blue', '100': 'tab:red', '111': 'tab:green'}
    colors_coverage = ['tab:blue', 'tab:green', 'tab:red']
    cell_sizes = {'100': ['1x1', '2x2', '3x3'],
                  '211': ['1x3', '2x3', '3x3'],
                  '111': ['1x1', '2x2', '3x3'],
                  '110': ['1x1', '2x2', '3x3']}

    j_ylim = {'100':[-50, 50], '111':[-120, 120], '211':[-70, 70], '110':[-70, 70]}

    cell_mult = {
                  '100': {'1x1':1, '2x2':1/4, '3x3':1/9},
                  '211': {'1x3':1, '2x3':1/2, '3x3':1/3},
                  '111': {'1x1':1, '2x2':1/4, '3x3':1/9},
                  '100':{'1x1':1, '2x2':1/4, '3x3':1/9},
                }

    coverage_labels = {
                  '100': {'1x1':'1', '2x2':r'$\frac{1}{4}$', '3x3':r'$\frac{1}{9}$'},
                  '211': {'1x3':'1', '2x3':r'$\frac{2}{3}$', '3x3':r'$\frac{1}{3}$'},
                  '111': {'1x1':'1', '2x2':r'$\frac{1}{4}$', '3x3':r'$\frac{1}{9}$'},
                  '110': {'1x1':'1', '2x2':r'$\frac{1}{4}$', '3x3':r'$\frac{1}{9}$'},
                       }

    coverages_cell = {
                      '100': [1, 0.25, 0.11],
                      '211': [1, 0.66, 0.33],
                      '111': [1, 0.25, 0.11],
                      '110': [1, 0.25, 0.11],
                      }
    alphabets = ['a', 'b', 'c', 'd']

    facet_markers = {'100':'o', '111':'o', '211':'o', '110':'o'}
    potential_range = np.linspace(1.4, -1.4)

    results = AutoVivification()
    experiments = AutoVivification()
    thermodb = connect('../databases/Au_Pb_coverage.db')

    for facet in facets:

        results[facet] = main(thermodb, referdb, cell_sizes[facet], functional)
        data_facet = []
        with open('previous/Au' + facet + '.csv') as f:
            #experiments[facet] = np.loadtxt('previous/Au' + facet + '.csv', delimiter=',')
            read_f = csv.reader(f,delimiter=';', dialect=csv.excel,)
            for row in read_f:
                try:
                    data_facet.append([float(a) for a in row])
                except ValueError:
                    print('Error')

        experiments[facet] = np.array(data_facet).transpose()

    # Plot the pourbaix
    fig, ax1 = plt.subplots(2, 4, sharex=True)

    fig.subplots_adjust(hspace=0)
    for ind, facet in enumerate(facets):
        #plt.figure()
        p_all = []
        for index, cell in enumerate(results[facet]):
            nPb = cell_mult[facet][cell]
            # Straight line slope for Pourbaix diagram
            p = np.poly1d([2*nPb, nPb * results[facet][cell]  - nPb * U_PB_2_SHE])
            p_all.append(p)

            # Plot the experimental plot
            ax1[1,ind].plot(experiments[facet][0], experiments[facet][1], color='tab:gray', alpha=0.5,lw=4)
            ax1[0,ind].plot(potential_range, p(potential_range)  ,
                    color=colors_coverage[index], lw=4,  label=r'$\theta = $ ' + coverage_labels[facet][cell] + ' ML')
            # ax1[0,ind].annotate(r'$\theta = $ ' + coverage_labels[facet][cell] + ' ML', \
                # xy=(0.3+0.1*index, 0.4), fontsize=26, color=colors_coverage[index]).draggable()
            # ax1[0,ind].text(-0.8, 0.75, r'Au(' +facet + ')', weight='bold',
            #     color='k',
            #     fontsize=30
            #     )

            ax1[0,ind].tick_params(axis='both', which='major', labelsize=22)
            if ind == 0:
                ax1[0,ind].set_ylabel(r'$\Delta E_{Pb}$ \ eV', fontsize=28)
                ax1[1,ind].set_ylabel(r'$j$ \ $\mu A cm^{-2}$', fontsize=28)
            # if ind == 1:
            ax1[1,ind].set_xlabel(r'Potential vs SHE \ V', fontsize=28)
            ax1[0,ind].set_ylim([-0.7, 0.7])
            ax1[0,ind].set_xlim([-0.7, 0.7])
            ax1[0,ind].legend(bbox_to_anchor=(0,1.02,1,0.2), loc="lower left",
                mode="expand", borderaxespad=0, ncol=3, fontsize=18)
            ax1[0,ind].annotate(r'Au(' +facet + ')', xy=(0.03, 0.9), \
                    xycoords="axes fraction", fontsize=24, color='tab:brown', weight='bold')

            ax1[0,ind].annotate(alphabets[ind] + ')', xy=(-0.1, 1.1),xycoords="axes fraction", fontsize=24)


            ax1[1,ind].tick_params(axis='both', which='major', labelsize=22)
            ax1[1,ind].set_ylim(j_ylim[facet])

        ax1[0,ind].axhline(y=0, color='k', ls='-', lw=4, )

        for i in range(len(p_all)-1):
            points_inter = (p_all[i+1] - p_all[i]).r
            ax1[0,ind].axvline(x=points_inter, ls='--', color='k')
            ax1[1,ind].axvline(x=points_inter, ls='--', color='k')

        # ax1[0,ind].legend(loc='best', fontsize=20)
        # stable_theta = []
        # for poten in potential_range:
        #     energy_values = []
        #     energy_values.append(0.0)
        #     for cell in p_all:
        #         energy_values.append(p_all[cell](poten))
        #     stable_theta.append(min(energy_values))

    plt.tight_layout()
    plt.savefig(output + 'lead_UPD.pdf')
    plt.show()
