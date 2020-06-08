  #!/usr/bin/python

import numpy as np
from useful_functions import AutoVivification
from pprint import pprint
from ase.db import connect
import matplotlib.pyplot as plt
import os
import csv
from scipy.optimize import curve_fit


class ParseInfo:
    def __init__(self, thermodb, referdb, list_cells, facet, functional, \
                ref_type):
        # Takes in the databases and the list of cells
        # for that facet
        self.thermodb = thermodb
        self.referdb = referdb
        self.list_cells = list_cells
        self.results = AutoVivification()
        self.rel_energy = AutoVivification()
        self.absolute_energy = AutoVivification()
        self.functional = functional
        self.states = AutoVivification()
        self.min_energy_cell = {}
        self.facet = facet
        self.ref_type = ref_type
        self.atoms = AutoVivification()



    def _get_states(self, cell):
        # for a given database pick out all states
        allstates = []

        for row in self.thermodb.select(functional=self.functional, cell_size=cell,\
                                        facets='facet_'+self.facet):
            allstates.append(row.states)
        unique_states = np.unique(allstates)
        return unique_states

    def _parse_energies(self):
        # Gets all energies from the database
        # This is a func specific to this
        for cell in self.list_cells:
            self.states[cell] = self._get_states(cell)
            for state in self.states[cell]:
                stat = self.thermodb.get(states=state,
                                    functional=self.functional,
                                    cell_size=cell,
                                    facets='facet_'+self.facet,
                                    )
                self.results[cell][state] = stat.energy
                self.atoms[cell][state] = stat.toatoms()

    def _get_reference(self):
        if self.ref_type == 'Pb':
            Pbstat = self.referdb.get(formula='Pb4',
                              functional=self.functional,
                              pw_cutoff=500.0)
        # Four atoms in Pb unit cell
            Pb_energies = Pbstat.energy / 4
            self.energy_reference = Pb_energies
        elif self.ref_type == 'CO':
            COstat = self.referdb.get(formula='CO',
                                      functional=self.functional,
                                      pw_cutoff=500.0)
            self.energy_reference = COstat.energy


    def get_pourbaix(self):
        # Get the DFT energies
        self._parse_energies()
        # Get the reference energies
        self._get_reference()
        for index, cell in enumerate(self.list_cells):
            energy_cell = []
            for state in self.states[cell]:
                self.absolute_energy[cell][state] = self.results[cell][state]
                if self.ref_type == 'CO':
                    if state != 'state_slab':
                        self.rel_energy[cell][state] = self.results[cell][state] - \
                                            self.results[cell]['state_slab'] - \
                                            self.energy_reference
                        energy_cell.append(self.rel_energy[cell][state])
                elif self.ref_type == 'Pb':
                    if state != 'slab':
                        self.rel_energy[cell][state] = self.results[cell][state] - \
                                            self.results[cell]['slab'] - \
                                            self.energy_reference
                        energy_cell.append(self.rel_energy[cell][state])
            energy_cell = np.array(energy_cell)
            self.min_energy_cell[cell] = min(energy_cell)


#  Experimental TPD parsing

class experimentalTPD:
    def __init__(self, tpd_filename, temprange, tempmin, beta):
        self.tpd_filename = tpd_filename
        # Experimental temperatures and rates
        self.beta = beta
        self.temprange = temprange,
        self.tempmin = tempmin,
        self.exp_temperature = []
        self.exp_rates = []
        self.normalized_rate = []
        self.temperature = []
        self.gaussian_rates = []
        self.Ed = []
        self.theta = []
        self.popt = [] # For fitting the Gaussian later

    # Collects the data which has been taken from a publication
    def collect_tpd_data(self):
        # First get the tpd data frm the file created by WebPlotDigitizer
        text_max_exposure = np.genfromtxt(self.tpd_filename,  delimiter=',')
        # Get the temperatures and rates from the experiment
        self.exp_temperature = text_max_exposure[:,0]
        self.exp_rate = text_max_exposure[:,1]

    def normalize_TPD_baseline(self, kwargs):
        # Called by get_normalized_data
        import numpy as np
        low, high = kwargs['min_range'][0]
        exp_temperature = self.exp_temperature
        rate_to_av = []
        rates_range = []
        temperature_range = []
        for i in range(len(exp_temperature)):
            if exp_temperature[i] > low and exp_temperature[i] < high:
                # will be part of range now
                rate_to_av.append(self.exp_rate[i])
        rate_to_av = np.array(rate_to_av)
        average_low = rate_to_av.mean()

        rlow, rhigh = kwargs['val_range'][0]
        for i in range(len(exp_temperature)):
            if exp_temperature[i] > rlow and exp_temperature[i] < rhigh:
                temperature_range.append(exp_temperature[i])
                rates_range.append(self.exp_rate[i] - average_low)

        data_norm = {'normalized_rate':rates_range, 'temperature':temperature_range}

        return data_norm

    def gaussian(self, x, a, x0, sigma):
        # called by gaussian_tpd for use with curve fit
        values = a * np.exp( - (x - x0)**2 / ( 2* sigma**2))
        return values

    # Normalize the TPD data by subtracting the ends
    def get_normalized_data(self):
        # Normalized the TPD spectra and write out the
        # right range of temperature and rates
        kwargs = {'min_range':self.tempmin, 'val_range':self.temprange}
        normalized_data = self.normalize_TPD_baseline(kwargs)
        self.normalized_rate = normalized_data['normalized_rate']
        self.temperature = normalized_data['temperature']

    def get_gaussian_tpd(self):
        # Fit a Gaussian to the TPD curve to get the the low-coverage tail
        temperature = np.array(self.temperature)
        rate_exp = np.array(self.normalized_rate)
        mean_rate = np.sum(temperature*rate_exp) / np.sum(rate_exp)
        sigma_rate = np.sqrt(np.sum(rate_exp *(temperature - mean_rate)**2) / np.sum(rate_exp) )
        rate_max = np.max(rate_exp)
        popt, pcov = curve_fit(self.gaussian, temperature, rate_exp,\
                                p0=[rate_max, mean_rate, sigma_rate])
        self.gaussian_rates = self.gaussian(self.temperature, *popt)
        self.popt = popt

    def Ed_temp_dependent(self, temperature, rate, beta):
        kB = 8.617e-05 # eV/K
        h = 4.135e-15 # eV.s
        theta = []
        for i in range(len(temperature)):
            cov = np.trapz(rate[i:], temperature[i:])
            theta.append(cov)
        theta = np.array(theta)
        dtheta_dT = np.diff(theta) / np.diff(temperature)
        dtheta_dt = beta * dtheta_dT
        temperature = np.array(temperature)
        nu = kB * temperature[0:-1] / h

        Ed = -kB * temperature[0:-1] * np.log( -1 * dtheta_dt / (nu * theta[0:-1]))

        return {'Ed':Ed, 'theta':theta}

    def get_desorption_energy(self):
        # Get desorption energy profiles with coverage for the chosen TPD spectra

        desorption_data = self.Ed_temp_dependent(self.temperature,
                                                self.gaussian_rates,
                                                self.beta
                                                )
        self.Ed = desorption_data['Ed']
        theta = desorption_data['theta'] / max(desorption_data['theta'])
        self.theta = theta[0:-1]
