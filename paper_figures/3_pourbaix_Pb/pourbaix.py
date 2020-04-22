#!/usr/bin/python


""" Pourbaix plot """


import numpy as np
from useful_functions import AutoVivification
from pprint import pprint
from ase.db import connect
import os, csv, sys
sys.path.append('../classes/')
from parser_class import ParseInfo
import matplotlib
matplotlib.rc('text', usetex=True)
matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"
plt.rc('axes', labelsize=32)    # fontsize of the x and y labels
plt.rcParams['xtick.labelsize'] = 26
plt.rcParams['ytick.labelsize'] = 26

def main(thermodb, referdb, list_cells, facet, functional):
    pour = ParseInfo(thermodb, referdb, list_cells, facet, functional,ref_type='Pb')
    pour.get_pourbaix()
    return pour.min_energy_cell

if __name__ == '__main__':
    ## Details
    output = 'output/'
    os.system('mkdir -p ' + output)

    # facets
    facets = ['111', '100', '110',  '211', ] # 'recon_110'
    functional = 'BF'
    no_electrons_Pb = 2 # Two electrons are transferred
    U_PB_2_SHE = -0.13 # V To convert to SHE

    # databases
    thermodb = connect('../databases/Au_Pb_coverage.db')
    referdb = connect('../databases/reference_lead.db')

    # Stylistic
    colors_coverage = ['tab:blue', 'tab:green', 'tab:red']
    cell_sizes = {
                  '100': ['1x1', '2x2', '3x3'],
                  '211': ['1x3', '2x3', '3x3'],
                  '111': ['1x1', '2x2', '3x3'],
                  '110': ['1x1', '2x2', '3x3'],
                  'recon_110': ['1x1', '2x1', '3x1'],
                  }

    j_ylim = {'100':[-70, 70], '111':[-120, 120], '211':[-70, 70], \
              '110':[-70, 70], 'recon_110':[-70, 70],}

    cell_mult = {
                  '100': {'1x1':1, '2x2':1/4, '3x3':1/9},
                  '211': {'1x3':1, '2x3':1/2, '3x3':1/3},
                  '111': {'1x1':1, '2x2':1/4, '3x3':1/9},
                  '110': {'1x1':1, '2x2':1/4, '3x3':1/9},
                  'recon_110':{'1x1':1, '2x1':1/2, '3x1':1/3},
                }

    coverage_labels = {
                  '100': {'1x1':'1', '2x2':r'$\frac{1}{4}$', '3x3':r'$\frac{1}{9}$'},
                  '211': {'1x3':'1', '2x3':r'$\frac{2}{3}$', '3x3':r'$\frac{1}{3}$'},
                  '111': {'1x1':'1', '2x2':r'$\frac{1}{4}$', '3x3':r'$\frac{1}{9}$'},
                  '110': {'1x1':'1', '2x2':r'$\frac{1}{4}$', '3x3':r'$\frac{1}{9}$'},
                  'recon_110':{'1x1':'1', '2x1':r'$\frac{1}{2}$', '3x1':r'$\frac{1}{3}$'},
                  }

    coverages_cell = {
                      '100': [1, 0.25, 0.11],
                      '211': [1, 0.66, 0.33],
                      '111': [1, 0.25, 0.11],
                      '110': [1, 0.25, 0.11],
                      'recon_110':[1, 0.25, 0.33]
                      }

    alphabets = ['a', 'b', 'c', 'd', 'e']

    facet_markers = {'100':'o', '111':'o', '211':'o', '110':'o'}
    potential_range = np.linspace(1.4, -1.4)

    # Plot related
    fig, ax1 = plt.subplots(2, len(facets), sharex=True, figsize=(35, 9))

    # Store data
    results = AutoVivification()
    experiments = AutoVivification()

    ################################################
    # STORE DATA

    for facet in facets:
        results[facet] = main(thermodb, referdb, cell_sizes[facet], facet, functional)
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

    ################################################
    # PLOT DATA

    fig.subplots_adjust(hspace=0)
    for ind, facet in enumerate(facets):
        #plt.figure()
        p_all = []
        for index, cell in enumerate(results[facet]):
            nPb = cell_mult[facet][cell]
            # Straight line slope for Pourbaix diagram
            p = np.poly1d([2*nPb, nPb * results[facet][cell]  - 2 * nPb * U_PB_2_SHE])
            if index == len(results[facet])-1:
                p_all.append(np.poly1d([0]))
            p_all.append(p)

            # Plot the experimental plot
            ax1[1,ind].plot(experiments[facet][0], experiments[facet][1], color='tab:gray', alpha=0.5,lw=4)
            ax1[0,ind].plot(potential_range, p(potential_range)  ,
                    color=colors_coverage[index], lw=4,  label=r'$\theta = $ ' + coverage_labels[facet][cell] + ' ML')


            # ax1[0,ind].tick_params(axis='both', which='major', labelsize=22)
            if ind == 0:
                ax1[0,ind].set_ylabel(r'$\Delta E_{Pb}$ / eV', fontsize=32)
                ax1[1,ind].set_ylabel(r'$j$ / $\mu A cm^{-2}$', fontsize=32)
            # if ind == 1:
            ax1[1,ind].set_xlabel(r'Potential vs SHE / V', fontsize=32)
            ax1[0,ind].set_ylim([-0.7, 0.7])
            ax1[0,ind].set_xlim([-0.7, 0.7])
            ax1[0,ind].legend(bbox_to_anchor=(0,1.02,1,0.2), loc="lower left",
                mode="expand", borderaxespad=0, ncol=3, fontsize=22)
            # ax1[1,ind].xaxis.set_ticks_position('bottom')
            # ax1[1,ind].spines['top'].set_visible(False)
            # ax1[0,ind].spines['bottom'].set_visible(False)

            ax1[0,ind].annotate(r'Au(' +facet.replace('recon_', 'recons-') + ')', xy=(0.03, 0.9), \
                    xycoords="axes fraction", fontsize=28, color='tab:brown', weight='bold')

            ax1[0,ind].annotate(alphabets[ind] + ')', xy=(-0.1, 1.1),xycoords="axes fraction", fontsize=32)


            # ax1[1,ind].tick_params(axis='both', which='major', labelsize=22)
            ax1[1,ind].set_ylim(j_ylim[facet])

        ax1[0,ind].axhline(y=0, color='k', ls='-', lw=4, )

        for i in range(len(p_all)-1):
            points_inter = (p_all[i+1] - p_all[i]).r
            ax1[0,ind].axvline(x=points_inter, ls='--', color='k')
            ax1[1,ind].axvline(x=points_inter, ls='--', color='k')

    plt.subplots_adjust(hspace=.0)

    plt.tight_layout()
    plt.savefig(output + 'lead_UPD.pdf')
    plt.savefig(output + 'lead_UPD.png')
    plt.show()
