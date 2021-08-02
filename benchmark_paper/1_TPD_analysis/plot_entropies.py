
from ase import units
import numpy as np
import matplotlib.pyplot as plt
from plot_params import get_plot_params_Times

def entropy(theta, T=300):
    kB = units.kB
    S_diff = kB * T * np.log(theta / (1 - theta))
    S_int = -kB * T * np.log(theta / (1 - theta)) - kB * T / theta * np.log(1 - theta)
    return S_diff, S_int

if __name__ == '__main__':
    get_plot_params_Times()
    eps=1e-4
    theta = np.linspace(0+eps, 1, 500)
    S_diff, S_int = entropy(theta)

    fig, ax =plt.subplots(1, 1, figsize=(8,6), constrained_layout=True)
    ax.set_ylabel(r'$ST$ / eV')
    ax.set_xlabel(r'$\theta$')
    ax.plot(theta, S_diff, '-', color='tab:red', label=r'$S_{diff}T$')
    ax.plot(theta, S_int, '-', color='tab:blue', label=r'$S_{int}T$')
    ax.legend(loc='best', frameon=False)

    fig.savefig('output/entropy_plot.pdf')