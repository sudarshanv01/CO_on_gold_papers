#!/usr/bin/python

""" Script to plot the charging curve for different facets """

# This script relies on the fact that there is a database file with all the
# energies here
import numpy as np
from useful_functions import get_vasp_nelect0, AutoVivification, coversions, fit_to_curve
from ase import db
from ase import Atoms
from pprint import pprint
import matplotlib.pyplot as plt
import os
plt.rcParams["figure.figsize"] = (12, 6.25)
from ase.thermochemistry import IdealGasThermo
from ase.thermochemistry import HarmonicThermo
from ase.io import read
class ReturnEnergies:
    def __init__(self, facet, electrodb, referdb):
        # Need to give the script only the facet
        self.facet = facet
        self.electrodb = electrodb
        self.referdb = referdb
        self.energies = AutoVivification()
        self.relenergies = AutoVivification()
        self.gas_energies = AutoVivification()

    def query_electrodb(self, kwargs):
         # Function to query an ase database with the needed kwargs
         self.selected_rows = []
         #print(kwargs)
         for row in self.electrodb.select(**kwargs):
             self.selected_rows.append(row)
    def get_gas_phase(self):
        # Parse the reference database to get the energies and frequenices
        for row in self.referdb.select(functional='BF'):
            formula = str(row.formula)
            electronic_energy = row.energy
            #vibrations =
            self.gas_energies[formula]['E'] = electronic_energy


    def get_unique(self, rows_list, property):
        # For a list of rows returns the unique values
        property_list = []
        for row in rows_list:
            property_list.append(row[property])
        property_list = np.array(property_list)
        self.unique_property = np.unique(property_list)

    def field_to_charge(self, field, surface_area):
        # Return the charge from the field based on Gauss' law
        epsilon = 8.854e-12 / 1e10 # F / Angstrom
        # Surface area based on volume / height
        #surface_area = self.get_surface_area(ase_atoms.get_volume(), ase_atoms.get_cell()[-1, -1]) # Angst^2
        # Sigma based on Gauss law
        sigma_angs2 = field  * epsilon # C / Angs^2
        # Sigma converted to electron units
        sigma = sigma_angs2 * 1e19 / 1.602 # e/ Angst ^ 2
        # Charge from sigma
        # The minus is to match the surface charge sign we normally use
        charge = -1 * surface_area * sigma # e
        return charge

    def parse_energies(self):
         # Get the unique types of states for a facet
         # First get all rows with the facet that we desire
         #rows_facet = self.query_electrodb(dict(facet=self.facet, opt=True))
         self.query_electrodb(dict(facet='facet_' + self.facet,
                                   functional='BF'))
         # Get the unique states
         #unique_states = self.get_unique(rows_facet, 'states')
         self.get_unique(self.selected_rows, 'states')
         for states in self.unique_property:
             # For each states get the energy, charge and the
             # vasp nelect0 assuming the default PAW potentials
             self.query_electrodb(dict(facet='facet_' + self.facet,
                                       states=states,
                                       functional='BF'))
             for row in self.selected_rows:
                 # Get the structure of this row
                 symbols = row.symbols
                 positions = row.positions
                 cell = row.cell
                 field = row.field
                 atoms = Atoms(symbols, positions)
                 atoms.set_cell(cell)
                 atoms.write(structure_dir + row.facet + '_' + states +  '.traj')
                 #atoms = row.get_atoms()
                 # Determine the surface area from the atoms object
                 surface_area = atoms.get_volume() / atoms.get_cell()[-1, -1]
                 charge = round(self.field_to_charge(field, surface_area),2)
                 # nelect0 = get_vasp_nelect0(atoms)
                 energy = row.energy
                 # tot_charge = row.tot_charge
                 # # Negative charge means more electrons
                 # charge = tot_charge - nelect0
                 # Store only the energies and surface charges
                 self.energies[row.states][charge]['energy'] = energy
                 self.energies[row.states][charge]['surface_charge'] = -1 * charge / surface_area

    def get_relative_energies(self):
        # Once the energies are parsed calculate the relative energies
        self.get_gas_phase()
        # Unit to convert from electron to mu C / cm 2
        for states in self.energies:
            if states != 'slab':
                for charge in self.energies[states]:
                    self.relenergies[states][charge]['rel_energy'] = \
                                    self.energies[states][charge]['energy'] -\
                                    self.energies['slab'][charge]['energy'] -\
                                    self.gas_energies['CO']['E']
                    self.relenergies[states][charge]['surface_charge'] = \
                                    self.energies[states][charge]['surface_charge'] * \
                                    coversions('e/m^2TOmuC/cm^2')

def main(facet, electrodb, referdb):
    data = ReturnEnergies(facet=facet, electrodb=electrodb, referdb=referdb)
    data.parse_energies()
    data.get_relative_energies()
    rel_energies = data.relenergies
    return rel_energies

def potential_plot_function(charge):
    potential = charge / 20 + 0.47
    potential_string = [str(round(pot, 2)) for pot in potential]
    return potential_string

def get_lowest_energy_state(dict_energies):
    # Determine the lowest energy to plot at every surface charge
    energy_zero = []
    all_states = []
    for states in dict_energies:
        energy_zero.append(dict_energies[states][4])
        all_states.append(states)
    min_energy_zero = np.argmin(energy_zero)
    print(min_energy_zero)

    return all_states[min_energy_zero]

def get_harmonic(vib_list, temperature, pressure):
    cmtoeV = 0.00012
    vib_energies = vib_list * cmtoeV
    thermo = HarmonicThermo(vib_energies = vib_energies, \
                            #potentialenergy = potential_energy, \
                            )
    gibbs = thermo.get_helmholtz_energy(temperature, pressure)
    #entropy = thermo.get_entropy(temperature, pressure)
    #enthalpy = thermo.get_internal_energy(temperature, pressure)
    return gibbs

def gas_phase():
    COg = {}
    vib_list = [2121.52, 39.91, 39.45]
    cmtoeV = 0.00012
    COg_atoms = read('COg/CO.traj')
    vib_ener = cmtoeV * np.array(vib_list)
    thermo = IdealGasThermo( vib_ener,\
                            atoms = COg_atoms, \
                            geometry='linear', \
                            symmetrynumber = 1, \
                            spin = 0 \
                            )
    COg['G'] = thermo.get_gibbs_energy(temperature=298.15, pressure=101325)
    COg['S'] = thermo.get_entropy(temperature=298.15, pressure=101325)
    return COg

if __name__ == '__main__':
    output = 'output/'
    os.system('mkdir -p ' + output)
    structure_dir = 'output_structure/'
    os.system('mkdir -p ' + structure_dir)
    # Facets to parse for and plot on one graph
    facets = ['100','111', '211']
    facet_markers = {'100':'o', '111':'o', '211':'o'}
    facet_color = {'100':'tab:blue', '111':'tab:green', '211':'tab:red'}
    # Get vibrational contibutions for each facet
    vib = {'211':np.array([1814.001, 367.591, 340.559, 230.796, 167.919, 69.744]),
           '100':np.array([1815.706, 371.374, 358.245, 248.225, 172.559, 78.188]),
           '111':np.array([2028.235, 290.764, 290.207, 221.973, 68.15, 65.48]),
           }
    gibbs_energy = {}
    for facet in vib:
        gibbs_energy[facet] = get_harmonic(vib[facet], 298.15, 101325)
    # get gas energies
    COg = gas_phase()
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twiny()
    electrodb = db.connect('Au_field.db')
    referdb = db.connect('BEEF_vdw_VASP_500.db')
    range_sigma = np.linspace(-5, 5, 6)
    for facet in facets:
        rel_energies = main(facet, electrodb, referdb)
        # Now plot energies for different states
        state_energy = {}
        for states in rel_energies:
            surface_charges = []
            energies = []
            for charges in rel_energies[states]:
                energies.append(rel_energies[states][charges]['rel_energy'])
                surface_charges.append(rel_energies[states][charges]['surface_charge'])
            state_energy[states] = energies
        lowest_energy_state = get_lowest_energy_state(state_energy)
        #for state in state_energy:
        # Lowest energy state at negative surface charges
        gibbs_correction = gibbs_energy[facet] - COg['G']
        p, fit = fit_to_curve(surface_charges, state_energy[lowest_energy_state] + gibbs_correction, 2)
        ax1.plot(range_sigma, p(range_sigma),  lw=3,
                    color=facet_color[facet], alpha=0.75,
                    label='Au(' + facet + ')')
        ax1.plot(surface_charges, state_energy[lowest_energy_state] + gibbs_correction,
                marker=facet_markers[facet], color=facet_color[facet],
                fillstyle='none', mew=2,  ls='none')
    ax2.set_xlim(ax1.get_xlim())
    ax2.set_xticks(range_sigma)
    ax2.set_xticklabels(potential_plot_function(np.array(range_sigma)))

    ax1.tick_params(axis='x',  labelsize=24)
    ax2.tick_params(axis='x', labelsize=24)
    ax1.tick_params(axis='y', labelsize=24)


    ax1.set_ylabel(r'$\Delta G_{CO^{*}}$ \ eV', fontsize=28)
    ax1.set_xlabel(r'$\sigma$ \ $ \mu C / cm^{2} $', fontsize=28)
    ax2.set_xlabel(r'Potential vs SHE \ V', fontsize=28)
    ax1.legend(loc='best', fontsize=20)
    ax1.legend(bbox_to_anchor=(1.04,1), loc="upper left")
    fig.tight_layout()
    plt.savefig(output + 'binding_surface_charge.pdf')
    plt.savefig(output + 'binding_surface_charge.svg')
