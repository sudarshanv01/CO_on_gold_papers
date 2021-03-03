import numpy as np
from glob import glob
from useful_functions import AutoVivification, get_vibrational_energy
from pprint import pprint
import os, sys
from scipy.optimize import curve_fit
from matplotlib import cm
import mpmath as mp
from ase.thermochemistry import HarmonicThermo, IdealGasThermo
from ase.io import read
from ase.db import connect
import matplotlib

def get_gas_vibrations():
    atoms = read('input_data/co.traj')
    vibration_energies_gas = 0.00012 * np.array([2100.539983, 24.030662, 24.018143])
    thermo_gas = IdealGasThermo(vibration_energies_gas, atoms = atoms, \
            geometry='linear', symmetrynumber=1, spin=0)

    return thermo_gas

def get_stable_site_vibrations():
    vibration_energies = {}
    vibration_energies['211'] = 0.00012 * np.array([2044.1, 282.2, 201.5, 188.5, 38.3, 11.5])
    vibration_energies['111'] = 0.00012 * np.array([2084.8, 201.7, 110.3, 110.2, 70.7, 70.8])
    vibration_energies['100'] = 0.00012 * np.array([1886.5, 315.2, 273.4, 222.2, 152.7, 49.8])
    vibration_energies['110'] = 0.00012 * np.array([2054.5, 262.9, 183.4, 147.3, 30.9, 30.])
    vibration_energies['310'] = 0.00012 * np.array([2054.5, 262.9, 183.4, 147.3, 30.9, 30.])

    return vibration_energies

def get_coverage_details():
    cell_sizes = {
                  '100': ['1x1', '2x2', '3x3'],
                  '211': ['1x3', '2x3', '3x3', '4x3',],
                  '111': ['1x1', '2x2', '3x3'],
                  '110': ['1x1', '2x2', '3x3'],
                  'recon_110': ['1x1', '2x1', '3x1'],
                  }
    coverages_cell = {
                  '100': {'1x1':1, '2x2':1/4, '3x3':1/9},
                  '211': {'1x3':1, '2x3':1/2, '3x3':1/3, '4x3':1/4, '6x3':1/6 },
                  '111': {'1x1':1, '2x2':1/4, '3x3':1/9},
                  '110': {'1x1':1, '2x2':1/4, '3x3':1/9},
                  'recon_110':{'1x1':1, '2x1':1/2, '3x1':1/3},
                  '310':{'2x4':1, '4x4':1/2, '6x4':1/3, '8x4':1/4, \
                            '4x8':1/4, '6x8':1/6, '8x8':1/8, '6x12':1/9, },
                }
    coverage_labels = {
                  '100': {'1x1':'1', '2x2':r'$\frac{1}{4}$', '3x3':r'$\frac{1}{9}$'},
                  '211': {'1x3':'1', '2x3':r'$\frac{2}{3}$', '3x3':r'$\frac{1}{3}$', '1CO_4x3':r'$\frac{1}{4}$'},
                  '111': {'1x1':'1', '2x2':r'$\frac{1}{4}$', '3x3':r'$\frac{1}{9}$'},
                  '110': {'1x1':'1', '2x2':r'$\frac{1}{4}$', '3x3':r'$\frac{1}{9}$'},
                  'recon_110':{'1x1':'1', '2x1':r'$\frac{1}{2}$', '3x1':r'$\frac{1}{3}$'},
                  }
    return cell_sizes, coverages_cell, coverage_labels

class PlotDFT():
    def __init__(self, database, reference_database, facet, functional):
        self.vibration_energies = {}
        self.thermo_ads = {}
        self.thermo_gas = {}
        self.coverages_cell = []
        self.coverage_labels = []
        self.database = database
        self.reference_database = reference_database
        self.reference = 0.0
        self.facet = facet
        self.functional = functional
        self.absE = {}
        self.atoms = {}
        self.dEdiff = [] # differential energy for the chosen coverage
        self.dEint = [] # integral energy
        self.coverage = [] # chosen coverage to plot against 
        self.free_diff = 0.0
        self.results = {}


        self.beef_error = {}
        self.beef_error['211'] = [0.1225,0.14629, 0.11908, 0.13701]
        self.beef_error['100'] = [0.16693, 0.14192, 0.18838, ]
        self.beef_error['111'] = [0.14959, 0.13724, 0.17666, ]
        self.beef_error['110'] = [0.2806, 0.15287, 0.17385, ]
        self.beef_error['310'] = [0.1520, 0.15515, 0.13882, 0.16298]

        # self.beef_error_diff = {}
        # self.beef_error_diff['211'] = [0.1225, 0.24416, 0.23704, 1.0295]
        # self.beef_error_diff['100'] = [0.16693,  ]
        # self.beef_error_diff['111'] = [0.14959, 0.65484,  ]
        # self.beef_error_diff['110'] = [0.2806, ]
        # self.beef_error_diff['310'] = [0.1520, 0.24815, 0.33387, 0.9753 ]

        self.chosen_sites = {}
        self.chosen_sites['211'] = 'CO_site_8'
        self.chosen_sites['310'] = 'CO_site_11'
        self.chosen_sites['111'] = 'CO_site_0'
        self.chosen_sites['100'] = 'CO_site_1'

        # Outline 
        self.get_data()

    def parse_data(self):
        for row in self.database.select(facets='facet_%s'%self.facet, \
                                        functional='BF'):
            self.results.setdefault(row.cell_size,{})[row.states.replace('state_','')] = row.energy

    def get_data(self):
        """
        Gets data about vibrations and energies from different places
        """

        """ 1.  Frequencies for stable sites """
        # Here we choose the frequencies corresponding to the lowest energy
        # site for adsorption of CO from DFT
        self.vibration_energies = get_stable_site_vibrations()
        self.thermo_ads = HarmonicThermo(self.vibration_energies[self.facet])
        # Gas phase CO energies
        self.thermo_gas = get_gas_vibrations()
        """2.  Specific about coverage """
        # DFT specifics about coverages
        self.cell_sizes, self.coverages_cell, self.coverage_labels = get_coverage_details()
        self.parse_data()

        """ 3. Adsorption energies """
        # get the reference energy for COg
        COg = self.reference_database.get(formula='CO',
                                    functional=self.functional,
                                    pw=500.0).energy
        ## get the multiplication factors
        facet_factors = []
        for cell in self.results:
            factor = cell.split('x')[0]
            facet_factors.append(int(factor))

        # Find the lowest common multiple of the cell size factors
        lcm = np.lcm.reduce(facet_factors)
        
        # get the data to plot
        absolute_energies = {}
        slab_energies = {}
        n_CO = {}
        for cell in self.results:
            theta = self.coverages_cell[self.facet][cell]
            factor = cell.split('x')[0] 
            n = lcm / int(factor)  #factor/theta
            ## find the minimum energy
            all_CO_energies = []
            all_sites = []
            if self.facet in ['211', '310']:
                mult_factor = n
            else:
                mult_factor = n**2
            n_CO[theta] = mult_factor
            print(self.facet, cell, mult_factor)
            for state in self.results[cell]:
                # if state != 'slab':
                if state in self.chosen_sites[self.facet]:
                    all_CO_energies.append(mult_factor * self.results[cell][state])
                    all_sites.append(state)
            absolute_energies[theta] = np.min(all_CO_energies)
            # print(all_sites[np.argmin(all_CO_energies)], cell)
            slab_energies[theta] = mult_factor * self.results[cell]['slab']

        all_Ediff = []
        all_Eint = []
        all_E = []
        all_theta = [] 
        all_n = []
        delta_zpe =  self.thermo_ads.get_ZPE_correction() - self.thermo_gas.get_ZPE_correction()
        free_ads = self.thermo_ads.get_helmholtz_energy(temperature=298, verbose=False)
        free_gas = self.thermo_gas.get_gibbs_energy(temperature=298, \
                                                    pressure=101325, verbose=False)
        for i, theta in enumerate(sorted(absolute_energies)):
            if i == 0:
                Ediff = ( absolute_energies[theta] - slab_energies[theta] - n_CO[theta] * COg ) / n_CO[theta]  \
                         + free_ads - free_gas #+ delta_zpe 
            else:
                delta_n = n_CO[theta] - all_n[i-1] 
                Ediff = (absolute_energies[theta] - all_E[i-1] - delta_n * COg ) / delta_n \
                            + free_ads - free_gas #+ delta_zpe 
            Eint = ( absolute_energies[theta] - slab_energies[theta] - n_CO[theta] *  COg )  / n_CO[theta]  \
                         + delta_zpe
            all_Eint.append(Eint)
            all_Ediff.append(Ediff)
            all_E.append(absolute_energies[theta])
            all_n.append(n_CO[theta])
            all_theta.append(theta)

        
        self.dEdiff = all_Ediff
        self.coverage = all_theta
        self.dEint = all_Eint