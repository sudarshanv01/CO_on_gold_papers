
import numpy as np
from ase.db import connect
import matplotlib.pyplot as plt
from useful_functions import create_output_directory
from plot_params import get_plot_params


def parsedb(database, results):

    for row in database.select():
        state = row.states
        sampling = int(row.sampling.replace('sampling_',''))
        results.setdefault(sampling,{})[state] = row.energy

    return results


if __name__ == "__main__":
    create_output_directory()
    get_plot_params()

    database = connect('databases/Au_310_convergence.db')
    results = {}
    parsedb(database, results)

    COg = -12.067
    fig, ax = plt.subplots(1, 1, figsize=(8,6))
    for k in results:
        try:
            dE = results[k]['state_CO'] - results[k]['state_slab'] - COg
        except KeyError:
            continue
        ax.plot(k, dE, '-o', color='tab:blue')
    ax.set_ylabel(r'$\Delta$ E / eV')
    ax.set_xlabel(r'k-points')
    fig.tight_layout()
    fig.savefig('output/energy_kpoints_convergence.pdf')
