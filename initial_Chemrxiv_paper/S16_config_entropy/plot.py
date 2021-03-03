
"""Plot the configurational entropy
"""

import numpy as np
import matplotlib.pyplot as plt
from ase.units import kB
from plot_params import get_plot_params
from useful_functions import create_output_directory


def differential_S():
    """
    Plots the behaviour of the differential 
    configurational entropy
    """
    fig, ax = plt.subplots(1, 1, figsize=(8,6))
    T = 300 # K
    theta = np.linspace(0, 1, 300)
    TdS = -1 * kB * T *  np.log(theta / (1 - theta))

    ax.plot(theta, TdS, '-', color='tab:blue', lw=3)
    ax.set_ylabel(r'$\mathregular{E}_{\mathregular{config}} = -\mathregular{TdS}$ / eV')
    ax.set_xlabel(r'$\theta$ / ML')
    # ax.axvline(0.005, color='tab:red')
    fig.tight_layout()
    fig.savefig('output/diff_config.pdf')

if __name__ == "__main__":
    get_plot_params()
    create_output_directory()
    differential_S()
