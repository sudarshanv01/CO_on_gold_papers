

from ase.io import read
import matplotlib.pyplot as plt
from useful_functions import create_output_directory
from plot_params import get_plot_params
import numpy as np


if __name__ == '__main__':
    get_plot_params()
    create_output_directory()

    neb = read('neb.traj', ':')
    energies = [atoms.get_potential_energy() for atoms in neb]
    energies = energies - np.min(energies)
    fig, ax = plt.subplots(1, 1, figsize=(8,6))

    ax.plot(energies, 'o-', color='tab:blue')
    ax.set_ylabel(r'$\Delta E$ / eV')
    # ax.set_xlabel(r'')
    ax.set_xticks([])
    fig.savefig('output/si_neb.pdf')