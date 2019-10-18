#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt
import os
from useful_functions import get_spline_TPD, get_normalized_TPD, \
        get_desorption_energy

from glob import glob
from scipy.interpolate import interp1d
from scipy.interpolate import UnivariateSpline
#####################
# Constants in experiment
# dT / dt heating rate
beta = 3 # K/s
kB = 8.617e-05 # eV/K
#####################

output = 'output/'
os.system('mkdir -p ' + output)

# Plotting a fit to the largest TPD and a few more exposures

## First get data for all initial coverages (exposures)
data = glob('/Users/vijays/Documents/project/2_gold/TPD/Cu/data_points/*')
homedir = 'data_points/'
exposures = []

for i in range(len(data)):
    exposure = float(data[i].split('.')[0].split('_')[-1].replace('p', '.'))    
    exposures.append(exposure)

exposures = np.array(exposures)
tpd_data = {}
for exposure in exposures:
    text_max_exposure = np.genfromtxt(homedir + '/exposure_' \
            + str(exposure).replace('.', 'p') + '.csv', delimiter=',')
    temperature = text_max_exposure[:,0]
    rate = text_max_exposure[:,1]
    tpd_data[exposure] = [temperature, rate]

max_exposure = max(exposures)
"""
Plot all TPD and perform spline fits on the largest exposure one
"""
plt.figure()
for key in tpd_data:
    temperature, rate = tpd_data[key]
    plt.plot(temperature, rate, '.', label='exposure = ' + str(key) )
    if key == max_exposure:
        spline_TPD = get_spline_TPD(temperature, rate)
        plt.plot(temperature, spline_TPD(temperature), 'k--')
plt.ylabel('TPD rates')
plt.xlabel('Temperature \ K')
plt.savefig(output + 'TPD_from_experiment.pdf')
print('Maximum Exposure used is %1.1f'%max_exposure)
"""
Plot coverage as a function of temperature for the largest TPD
Find the maximum coverage
ASSUMPTION: Maximum coverage is attained for highest exposure
"""
data_TPD_largest = get_normalized_TPD(tpd_data[max_exposure][0], tpd_data[max_exposure][1])
coverages_largest = data_TPD_largest['coverages']
spline_norm_rate_largest = data_TPD_largest['spline_normalized_rate']
spline_coverages_largest = data_TPD_largest['spline_coverages']
maximum_sites_from_spline = data_TPD_largest['maximum_sites_from_spline']

# plotting coverages of largest TPD
plt.figure()
plt.plot(tpd_data[max_exposure][0], spline_coverages_largest(tpd_data[max_exposure][0]))
plt.ylabel(r'$\theta$')
plt.xlabel('Temperature \ K')
plt.savefig(output + 'coverages_largest.pdf')
plt.close()
quant_TPD = {}
for exposure in exposures:
    quant_TPD[exposure] = {}
    if exposure == max_exposure:
        print('Considered max exposure as coverage of 1')
        quant_TPD[exposure]['coverages'] = coverages_largest
        quant_TPD[exposure]['spline_normalized_rate'] = spline_norm_rate_largest
        quant_TPD[exposure]['spline_coverages'] = spline_coverages_largest
    else:
        temperature = tpd_data[exposure][0]
        rate = tpd_data[exposure][1]
        data_TPD = get_normalized_TPD(temperature, rate,\
                maximum_sites_from_spline=maximum_sites_from_spline)
        quant_TPD[exposure]['coverages'] = data_TPD['coverages']
        quant_TPD[exposure]['spline_normalized_rate'] = data_TPD['spline_normalized_rate']
        quant_TPD[exposure]['spline_coverages'] = data_TPD['spline_coverages']

"""
Plot desorption energies for a given TPD curve
"""
random_nu = np.array([ 10e9, 10e10, 10e11, 10e12, 10e13])

for exposure in exposures:
    temperature = tpd_data[exposure][0]
    temp_to_coverage = quant_TPD[exposure]['spline_coverages']
    plt.figure()
    for nu in random_nu:
        Ed_list = []
        theta_list = []
        
        for i in range(len(temperature)):
            data_desorption_energy = get_desorption_energy(temperature[i], \
                    beta, nu, temp_to_coverage)
            theta = data_desorption_energy['theta']
            Ed = data_desorption_energy['Ed']
            theta_list.append(theta)
            Ed_list.append(Ed)
        Ed_list = np.array(Ed_list)
        theta_list = np.array(theta_list)
        plt.plot(theta_list, Ed_list, label=r'$log\nu$ = ' + str(np.log10(nu)))
    plt.ylabel('Desorption Energy / eV')
    plt.xlabel(r'$\theta $ / ML')
    plt.legend(loc='best')
    plt.savefig(output + 'desorp_energy_coverage_' + str(exposure).replace('.', 'p') + '.pdf')
    plt.close()

"""
Plot the TPD for all chosen spectra with the desoprtion energy profile of 
maximum exposure
"""
fit_Ed_maxEx_nu = {}
plt.figure()
temperature = tpd_data[max_exposure][0]
temp_to_coverage = quant_TPD[max_exposure]['spline_coverages']
for nu in random_nu:
    Ed_maxEx_list = []
    theta_maxEx_list = []
    
    for i in range(len(temperature)):
        data_desorption_energy = get_desorption_energy(temperature[i], \
                beta, nu, temp_to_coverage)
        theta = data_desorption_energy['theta']
        Ed = data_desorption_energy['Ed']
        theta_maxEx_list.append(theta)
        Ed_maxEx_list.append(Ed)
    Ed_maxEx_list = np.array(Ed_maxEx_list)
    theta_maxEx_list = np.array(theta_maxEx_list)
    Ed_maxEx_clean = [ a for a in Ed_maxEx_list if np.isfinite(a) ]
    theta_maxEx_clean = [ a for a in theta_maxEx_list if np.isfinite(a) ]
    # Get a master fit for seeing how much prefactor matters with other exposures
    theta_maxEx_sorted, Ed_maxEx_sorted = zip(*sorted(zip(theta_maxEx_clean, Ed_maxEx_clean)))
    fit_Ed_max = UnivariateSpline(theta_maxEx_sorted,  Ed_maxEx_sorted, check_finite=True)
    fit_Ed_max.set_smoothing_factor(0.0) # fit all points
    fit_Ed_maxEx_nu[nu] = fit_Ed_max
    rate_maxEx_plot = nu * np.exp(-1 * Ed_maxEx_list / kB / temperature ) \
           * theta_maxEx_list
    plt.plot(temperature, rate_maxEx_plot, label = r'$log\nu$ = ' + str(np.log10(nu)))

plt.plot(temperature, beta * quant_TPD[max_exposure]['spline_normalized_rate'] , 'k.', label='TPD data')
plt.ylabel('TPD rate')
plt.xlabel('Temperature / K')
plt.legend(loc='best')
plt.savefig(output + 'simulated_TPD_largest_TPD.pdf')
plt.close()
"""
Plot the adsorption energy fits from the maximum coverage spectra
"""
plt.figure()
for key in fit_Ed_maxEx_nu:
    # plot over different values of nu
    coverage_range =  quant_TPD[max_exposure]['coverages']
    plt.plot(coverage_range, fit_Ed_maxEx_nu[key](coverage_range), label=r'$log\nu$ = ' + str(np.log10(nu)))

plt.xlabel(r'$\theta \ ML$')
plt.ylabel('Desorption Energy \ eV')
plt.ylim([0, 1])
plt.savefig(output + 'desoprtion_energy_fit.png')
plt.close()

"""
Plot the above diagram for all exposures on one graph
"""
plt.figure()
for nu in random_nu:
    for exposure in exposures:
        coverages =  quant_TPD[exposure]['coverages']
        temperature = tpd_data[exposure][0]
        Ed = fit_Ed_maxEx_nu[nu](coverages)
        rate = nu * np.exp( - Ed / kB / temperature ) * coverages
        plt.plot(temperature, rate / beta, '--', label=r'$log\nu$ = ' + str(np.log10(nu)))
        
for exposure in exposures:
    temperature = tpd_data[exposure][0]
    plt.plot(temperature,  quant_TPD[exposure]['spline_normalized_rate'] , 'k.')
plt.ylabel('TPD rate')
plt.xlabel('Temperature / K')
plt.legend(loc='best')
plt.savefig(output + 'all_TPD_Edmax.pdf')
plt.close()
