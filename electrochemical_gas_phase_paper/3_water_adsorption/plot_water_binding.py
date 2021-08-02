
"""
Plots the convergence and AIMD data 
for the (310) surface facet
in order to compare with TPD results
"""

import json
import matplotlib.pyplot as plt
from ase.db import connect
import numpy as np 
from ase.io import read
import click 
from ase.data import covalent_radii as radii
from matplotlib.patches import Circle
from plot_params import get_plot_params
import os
from ase.data.colors import jmol_colors
from useful_functions import create_output_directory
from plot_params import get_plot_params_Times
import string

def get_reference(dbame):
    references = {}
    database = connect(dbame)
    for row in database.select():
        species = row.sampling.replace('sampling_','')
        references[species] = row.energy
    return references


@click.command()
@click.option('--filename', default='data/results_310.json')
@click.option('--dbname', default='data/reference_data.db')
def main(filename, dbname):
    ## get the data from the json file
    data = json.load(open(filename, 'r'))
    basedir = 'images/'
    references = get_reference(dbname)

    # fig, ax = plt.subplots(len(results)/2, 2, figsize=(8,6))
    fig = plt.figure(figsize=(8,9), constrained_layout=True)
    gs = fig.add_gridspec(3, int(len(data)/2))
    axd = fig.add_subplot(gs[2,:])
    axe = fig.add_subplot(gs[0,:])
    ax = []
    for i in range(int(len(data)/2)):
        ax.append(fig.add_subplot(gs[1,i]))

    results = {}
    for i, f in enumerate(data):
        homedir = os.path.join(basedir, f)
        atoms = read(os.path.join(homedir, 'POSCAR'))
        n_water = len([atom.index for atom in atoms if atom.symbol == 'O'])
        E_water = data[f]
        dE = ( E_water - references['slab'] - n_water * references['H2O'] ) / n_water
        theta = n_water / 8
        try:
            results[theta].append(dE)
        except KeyError:
            results[theta] = []
            results[theta].append(dE)
        # ax.plot(theta, dE, 'o', color='tab:blue')
        fixed = atoms.constraints[0].index
        if i <= len(data)/2-1:
            for atom in atoms:
                if atom.index in fixed:
                    continue
                color = jmol_colors[atom.number]
                radius = radii[atom.number]
                circle = Circle((atom.y, atom.z), radius, facecolor=color,
                                        edgecolor='k', linewidth=1)
                j = int(len(data)/2 - 1 - i)
                ax[j].add_patch(circle)
                ax[j].axis('equal')
                ax[j].set_xticks([])
                ax[j].set_yticks([])
                ax[j].axis('off')
                ax[j].annotate(r'$\theta = %1.2f$ ML'%theta, fontsize=14, \
                            xy=(0.2, 0.1), xycoords='axes fraction')

    all_theta = []
    all_dE = []
    for theta in results:
        all_theta.append(theta)
        all_dE.append(min(results[theta]))

    axd.plot(all_theta, all_dE, '-o', markersize=16, color='tab:blue')
    axd.set_ylabel(r'$\Delta \left < E \right > $ / eV')
    axd.set_xlabel(r'$\theta_{\mathrm{H}_2\mathrm{O}}$ / ML')
    axd.axhline(y=-0.49,label=r'Juurlink et al. (First Peak)', color='tab:green', ls='--' )
    axd.axhline(y=-0.57,label=r'Juurlink et al. (Second Peak)', color='tab:red', ls='--' )
    # fig.tight_layout()

    low_exposure = np.loadtxt('inputs/low_exposure_peaks.csv', delimiter=',')
    high_exposure = np.loadtxt('inputs/high_exposure_peaks.csv', delimiter=',')

    T_low, R_low = low_exposure.T
    T_high, R_high = high_exposure.T

    axe.plot(T_low, R_low, '.', color='tab:green', label='Low Exposure')
    axe.plot(T_high, R_high, '.', color='tab:red', label='High Exposure')
    axe.set_ylabel(r'TPD Rate / a.u.')
    axe.set_xlabel(r'Temperature / K')
    axe.legend(loc='best', frameon=False, fontsize=12)
    axe.set_yticks([])
    alphabet = list(string.ascii_lowercase)
    for i, a in enumerate([axe] + ax + [axd]):
        a.annotate(alphabet[i]+')', xy=(0.05, 0.75), xycoords='axes fraction', fontsize=20)

    axd.legend(loc='best', frameon=False, fontsize=12)
    fig.savefig('output/coverage_dE_310.pdf')



if __name__ == '__main__':
    get_plot_params_Times()
    create_output_directory()
    main()