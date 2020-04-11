#!/usr/bin/python

""" Plots the TPD curves with temperature dependent pre-factors and compared enery """

import numpy as np
from useful_functions import get_vasp_nelect0, AutoVivification, coversions, fit_to_curve
from ase import db
from ase import Atoms
from pprint import pprint
import matplotlib.pyplot as plt
import os
from scipy.optimize import curve_fit

def gaussian(x, a, x0, sigma):
    # called by gaussian_tpd for use with curve fit
    values = a * np.exp( - (x - x0)**2 / ( 2* sigma**2))
    return values

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

    def collect_tpd_data(self):
        # First get the tpd data frm the file created by WebPlotDigitizer
        text_max_exposure = np.genfromtxt(tpd_filename,  delimiter=',')
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
        popt, pcov = curve_fit(gaussian, temperature, rate_exp,\
                                p0=[rate_max, mean_rate, sigma_rate])
        self.gaussian_rates = gaussian(self.temperature, *popt)

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

def main(tpd_filename, temprange, tempmin, beta):
    # Running the class experimentalTPD
    expTPD = experimentalTPD(tpd_filename, temprange, tempmin, beta)
    expTPD.collect_tpd_data()
    expTPD.get_normalized_data()
    expTPD.get_gaussian_tpd()
    expTPD.get_desorption_energy()
    Ed = expTPD.Ed
    theta = expTPD.theta
    return {'Ed':Ed, 'theta':theta}

output = 'output/'
os.system('mkdir -p ' + output)

if __name__ == '__main__':
    ## Inputs to be supplied
    # Where the exposure experimental file is stored
    tpd_filename = 'TPD_spectra/gold_211_exposure.csv'
    # Temperature range where the analysis will be done
    temprange = [170, 270] # min and max values
    # Temperature where a minimum is located so that the curves can be normalized
    tempmin = [250, 300]
    # Heating rate
    beta = 3 # K / s
    # First plot the binding energy vs coverage graph obtained from experiments
    data_Au = main(tpd_filename, temprange, tempmin, beta)
    # Not repeat the same for Pt
    tpd_filename = 'TPD_spectra/Pt_211.csv'
    temprange = [500, 650]
    tempmin = [650, 700]
    beta = 4
    data_Pt = main(tpd_filename, temprange, tempmin, beta)
    # Now plotting for poly-crystalline copper
    tpd_filename = 'TPD_spectra/Cu_PC.csv'
    temprange = [200, 300]
    tempmin = [300, 350]
    beta = 2
    data_Cu = main(tpd_filename, temprange, tempmin, beta)
    # For PC Pd
    tpd_filename = 'TPD_spectra/Pd_PC.csv'
    temprange = [350, 480]
    tempmin = [450, 500]
    beta = 5
    data_Pd = main(tpd_filename, temprange, tempmin, beta)
    # Now plot the experimental graphs
    plt.figure()
    plt.plot(data_Au['theta'], -1 * data_Au['Ed'], color='gold', label='Au(211)')
    plt.plot(data_Pt['theta'], -1 * data_Pt['Ed'], color='tab:blue', label='Pt(211)')
    plt.plot(data_Cu['theta'], -1 * data_Cu['Ed'], color='tab:red', label='Cu(PC)')
    plt.plot(data_Pd['theta'], -1 * data_Pd['Ed'], color='mediumturquoise', label='Pd(PC)')
    plt.ylabel(r'$\Delta E_{CO^{*}}$')
    plt.xlabel(r'$\theta$')
    plt.xticks([])
    # Plot the zero coverage slopes
    plt.axhline(y=-0.5, color='gold', ls='--')
    plt.axhline(y=-1.915, color='tab:blue', ls='--')
    plt.axhline(y=-0.71, color='tab:red', ls='--')
    plt.axhline(y=-1.75, color='mediumturquoise', ls='--')
    #plt.legend(loc='best')
    plt.legend(bbox_to_anchor=(1.04,0), loc="lower left", borderaxespad=0)
    plt.savefig(output + 'TPD_comparison.pdf', bbox_inches="tight")
