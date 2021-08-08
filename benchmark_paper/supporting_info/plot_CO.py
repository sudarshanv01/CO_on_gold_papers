#!/usr/bin/python

""" Script to compare TPD for different Gold facets """

import numpy as np
from glob import glob
from pprint import pprint
import matplotlib.pyplot as plt
import os, sys
import mpmath as mp
from ase.thermochemistry import HarmonicThermo, IdealGasThermo
from ase.io import read
from ase.db import connect
from matplotlib.ticker import FormatStrFormatter
import matplotlib.pyplot as plt
from plot_params import get_plot_params




def parsedb(database):
    results = {}
    for row in database.select():
        state = row.states.replace('state_','')
        if '110' in row.facets:
            continue
        results.setdefault(row.facets,{}).setdefault(row.cell_size,{})[state] = row.energy
    return results

if __name__ == '__main__':
    get_plot_params()


    output = 'output/'
    os.system('mkdir -p ' + output)


    referencedb = connect('../1_TPD_analysis/input_data/gas_phase.db')
    COgstat = referencedb.get(states='state_CO', functional='BF', pw=500.0)
    COg_E =  COgstat.energy

    thermodb = connect('../1_TPD_analysis/input_data/Au_CO_coverage.db')
    results = parsedb(thermodb)

    fig, ax = plt.subplots(1, len(results), figsize=(14,4), squeeze=False)

    for index_facet, facet in enumerate(results):
        for cell in results[facet]:
            for state in results[facet][cell]:
                if 'slab' in state:
                    continue
                dE = results[facet][cell][state] \
                    - results[facet][cell]['slab'] \
                    - COg_E
                ax[0,index_facet].plot(cell, dE, 'o', color='tab:blue')
        ax[0,index_facet].set_title('Au(%s)'%facet.replace('facet_',''))


    for a in ax.flatten():
        a.set_xlabel(r'$\theta$ \ ML', )
        a.set_ylabel(r'$\Delta E_{CO^{*}}$ \ eV',)

    fig.tight_layout()
    fig.savefig('output/convergence.pdf')