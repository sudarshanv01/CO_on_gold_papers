

import numpy as np 
import matplotlib.pyplot as plt
from useful_functions import create_output_directory
from plot_params import get_plot_params_Times


if __name__ == '__main__':
    create_output_directory()
    get_plot_params_Times()

    low_exposure = np.loadtxt('inputs/low_exposure_peaks.csv', delimiter=',')
    high_exposure = np.loadtxt('inputs/high_exposure_peaks.csv', delimiter=',')

    T_low, R_low = low_exposure.T
    T_high, R_high = high_exposure.T

    fig, ax = plt.subplots(1, 1, figsize=(8,6), constrained_layout=True)

    ax.plot(T_low, R_low, '.', color='tab:red', label='Low Exposure')
    ax.plot(T_high, R_high, '.', color='tab:blue', label='High Exposure')
    ax.set_ylabel(r'TPD Rate / arb. units')
    ax.set_xlabel(r'Temperature / K')
    ax.legend(loc='best', frameon=False)
    ax.set_yticks([])

    fig.savefig('output/TPD_raw.pdf')
