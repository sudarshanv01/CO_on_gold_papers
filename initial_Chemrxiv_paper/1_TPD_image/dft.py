#!/usr/bin/python

import numpy as np
from glob import glob
from useful_functions import AutoVivification, get_vibrational_energy
from pprint import pprint
import os, sys
from scipy.optimize import curve_fit
from matplotlib import cm
sys.path.append('../classes/')

from parser_function import get_stable_site_vibrations, get_gas_vibrations, \
                            get_coverage_details, diff_energies, \
                            get_lowest_absolute_energies,\
                            get_differential_energy, \
                            get_constants, \
                            accept_states, stylistic_exp, \
                            get_adsorbate_vibrations
from parser_class import ParseInfo, experimentalTPD
import mpmath as mp
from ase.thermochemistry import HarmonicThermo, IdealGasThermo
from ase.io import read
from ase.db import connect
import matplotlib


class PlotDFT():
    # Plots the dft data

    def __init__(self, database, reference_database, facet, functional):
        self.vibration_energies = {}
        self.thermo_ads = {}
        self.thermo_gas = {}
        self.cell_sizes = []
        self.coverages_cell = []
        self.coverage_labels = []
        self.accept_states = []
        self.database = database
        self.reference_database = reference_database
        self.facet = facet
        self.functional = functional
        self.absE = {}
        self.atoms = {}
        self.dE = [] # differential energy for the chosen coverage
        self.coverage = [] # chosen coverage to plot against 
        self.free_diff = 0.0


        self.beef_error = AutoVivification()
        self.beef_error['211'] = [0.1225,0.14629, 0.11908, 0.13701, ]
        self.beef_error['100'] = [0.16693, 0.14192, 0.18838, ]
        self.beef_error['111'] = [0.14959, 0.13724, 0.17666, ]
        self.beef_error['110'] = [0.2806, 0.15287, 0.17385, ]

        # Outline 
        self.get_data()

    def _parse_DFT_data(self, facet):
        parse = ParseInfo(
                          thermodb=self.database, 
                          referdb=self.reference_database, 
                          list_cells=self.cell_sizes[facet], 
                          facet=facet, 
                          functional=self.functional,
                          ref_type='CO'
                          )
        parse.get_pourbaix()
        return parse.absolute_energy, parse.atoms

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
        self.accept_states = accept_states() # CO states that will be plotted

        """ 3. Adsorption energies """
        # get the reference energy for COg
        COstat = self.reference_database.get(formula='CO',
                                    functional=self.functional,
                                    pw=500.0)
        self.absE, self.atoms = self._parse_DFT_data(self.facet)
        avg_energy = get_differential_energy(absolute_energy=self.absE, 
                                                atoms_dict=self.atoms,
                                                facet=self.facet, 
                                                COg=COstat.energy
                                            )
        
        for state in avg_energy:
            free_ads = self.thermo_ads.get_helmholtz_energy(temperature=298, verbose=False)
            free_gas = self.thermo_gas.get_gibbs_energy(temperature=298, \
                                                        pressure=101325, verbose=False)
            entropy_ads = self.thermo_ads.get_entropy(temperature=298, verbose=False)
            entropy_gas = self.thermo_gas.get_entropy(temperature=298, \
                                                        pressure=101325, verbose=False)
            self.free_diff = -1 * 298 * (entropy_ads - entropy_gas)
            delta_zpe =  self.thermo_ads.get_ZPE_correction() - self.thermo_gas.get_ZPE_correction()
            self.coverage = list(reversed([self.coverages_cell[self.facet][cell] for cell in self.cell_sizes[self.facet]]))
            if state in self.accept_states[self.facet]:
                self.dE = avg_energy[state] + delta_zpe

