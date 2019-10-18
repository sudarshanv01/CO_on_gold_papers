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
saturation_cov = 0.17 #ML - taken from paper
eVtokjoulepmol = 96
#####################

output = 'output/'
os.system('mkdir -p ' + output)

# Plotting a fit to the largest TPD and a few more exposures

## First get data for all initial coverages (exposures) that is to be plotted
data = glob('data_points/*')
homedir = 'data_points'
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
    #if key == max_exposure:
    spline_TPD = get_spline_TPD(temperature, rate, scaling_smoothing=0.0)
    plt.plot(temperature, spline_TPD(temperature), 'k--')

plt.ylabel('TPD rates \ ML/s')
plt.xlabel('Temperature \ K')
plt.savefig(output + 'TPD_from_experiment.pdf')
print('Maximum Exposure used is %1.1f'%max_exposure)
"""
Plot coverage as a function of temperature for the largest TPD
Find the maximum coverage
ASSUMPTION: Maximum coverage is attained for highest exposure
"""
data_TPD_largest = get_normalized_TPD(tpd_data[max_exposure][0], \
        tpd_data[max_exposure][1], beta, saturation_cov, scaling_smoothing=0.0)
coverages_largest = data_TPD_largest['coverages']
spline_norm_rate_largest = data_TPD_largest['spline_normalized_rate']
spline_coverages_largest = data_TPD_largest['spline_coverages']
maximum_sites_from_spline = data_TPD_largest['maximum_sites_from_spline']

# plotting coverages of largest TPD
plt.figure()
plt.plot(tpd_data[max_exposure][0], spline_coverages_largest(tpd_data[max_exposure][0]))
plt.ylabel(r'$\theta$')
plt.ylim([0, 1])
plt.xlabel('Temperature \ K')
plt.savefig(output + 'coverages_largest.pdf')
plt.close()

quant_TPD = {}
for exposure in exposures:
    quant_TPD[exposure] = {}
    if exposure == max_exposure:
        temperature = tpd_data[exposure][0]
        print('Considered max exposure as coverage of 1')
        quant_TPD[exposure]['coverages'] = coverages_largest
        quant_TPD[exposure]['spline_normalized_rate'] = spline_norm_rate_largest
        quant_TPD[exposure]['spline_coverages'] = spline_coverages_largest
        quant_TPD[exposure]['temperature'] = temperature
    else:
        temperature = tpd_data[exposure][0]
        rate = tpd_data[exposure][1]
        data_TPD = get_normalized_TPD(temperature, rate,\
                maximum_sites_from_spline=maximum_sites_from_spline, beta=beta,\
                scaling_smoothing=0., 
                saturation_coverage=saturation_cov, 
                )
        quant_TPD[exposure]['coverages'] = data_TPD['coverages']
        quant_TPD[exposure]['spline_normalized_rate'] = data_TPD['spline_normalized_rate']
        quant_TPD[exposure]['spline_coverages'] = data_TPD['spline_coverages']
        assert (temperature==data_TPD['temperature']).all()
        quant_TPD[exposure]['temperature'] = data_TPD['temperature']

# plotting all coverages
plt.figure()
for exposure in exposures:
    plt.plot(quant_TPD[exposure]['temperature'], \
            quant_TPD[exposure]['coverages'], label='exposure: ' + str(exposure) + 'L')
plt.ylabel(r'$\theta$ \ ML')
#plt.ylim([0, 1])
plt.xlabel('Temperature \ K')
plt.legend(loc='best')
plt.savefig(output + 'coverages_all.pdf')
plt.close()

""" 
Sanity check to see if dtheta / dt matches the rate 
"""
plt.figure()
for exposure in exposures:
    spl_coverage = quant_TPD[exposure]['spline_coverages']
    temperature = quant_TPD[exposure]['temperature']
    dtheta_dT = spl_coverage.derivative()
    dtheta_dt = dtheta_dT(temperature) * beta
    # plot dtheta/dt for each exposure
    plt.plot(temperature, -1 * dtheta_dt, '-')
    plt.plot(temperature, quant_TPD[exposure]['spline_normalized_rate'], '.')
plt.ylabel('rate')
plt.xlabel('Temperature \ K')
plt.legend(loc='best')
plt.savefig(output + 'sanity_rates_dtheta_dt.pdf')
plt.close()

"""
Plot desorption energies for a given TPD curve
"""
#DFT_binding_energies = [[0.33, 0.36], [0.33, 0.35], [0.33, 0.7], [0.1, 0.282]]
#annotations_DFT = ['RPBE', 'BEEF-vdW', 'RPBE-D3', 'RPBE']
random_nu = np.array([ 10e0, 10e1, 10e2, 10e13, 10e14])
colors = ['sienna', 'olive', 'navy', 'orchid', 'r']
colors_nu = dict(zip(random_nu, colors))


for exposure in exposures:
    # for a given exposure get the temperature and spline which convert the 
    # temperature to a coverage
    temperature = quant_TPD[exposure]['temperature']
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
        plt.plot(theta_list, Ed_list, label=r'$log\nu$ = ' + str(np.log10(nu)), color=colors_nu[nu])
    #for i in range(len(DFT_binding_energies)):
    #    coverage, binding_energy = DFT_binding_energies[i]
    #    plt.plot(coverage, binding_energy, 'o')
    #    plt.annotate(annotations_DFT[i], xy = (coverage, binding_energy))
    plt.ylabel('Desorption Energy / eV')
    plt.xlabel(r'$\theta $ / ML')
    plt.xlim([0, 1])
    plt.legend(loc='best')
    #plt.ylim([0.4,0.9])
    plt.savefig(output + 'desorp_energy_coverage_' + str(exposure).replace('.', 'p') + '.pdf')
    plt.close()

"""
Plot the TPD for all chosen spectra with the desoprtion energy profile of 
maximum exposure
"""
fit_Ed_maxEx_nu = {}
Ed_maxEx_nu = {}
plt.figure()
temp_to_coverage = quant_TPD[max_exposure]['spline_coverages']
for nu in random_nu:
    temperature = quant_TPD[exposure]['temperature']
    Ed_maxEx_list = []
    theta_maxEx_list = []
    # get Ed and theta for random choice of nu 
    for i in range(len(temperature)):
        data_desorption_energy = get_desorption_energy(temperature[i], \
                beta, nu, temp_to_coverage)
        theta = data_desorption_energy['theta']
        Ed = data_desorption_energy['Ed']
        theta_maxEx_list.append(theta)
        Ed_maxEx_list.append(Ed)

    Ed_maxEx_list = np.array(Ed_maxEx_list)
    Ed_maxEx_nu[nu] = Ed_maxEx_list
    theta_maxEx_list = np.array(theta_maxEx_list)
    # clean up by removing inf and nan
    Ed_maxEx_args = [ i for i in range(len(Ed_maxEx_list)) if np.isfinite(Ed_maxEx_list[i]) ]
    Ed_maxEx_clean = Ed_maxEx_list[Ed_maxEx_args] #[ a for a in Ed_maxEx_list if np.isfinite(a) ]
    theta_maxEx_clean = theta_maxEx_list[Ed_maxEx_args] #[ a for a in theta_maxEx_list if np.isfinite(a) ]
    temperature_clean = temperature[Ed_maxEx_args]
    # Get a master fit for seeing how much prefactor matters with other exposures
    theta_maxEx_sorted, Ed_maxEx_sorted = zip(*sorted(zip(theta_maxEx_clean, Ed_maxEx_clean)))
    fit_Ed_max = UnivariateSpline(theta_maxEx_sorted,  Ed_maxEx_sorted, check_finite=True)
    fit_Ed_max.set_smoothing_factor(0.02) # fit all points
    fit_Ed_maxEx_nu[nu] = fit_Ed_max ## This is the fit for Ed with theta for the largest exposure
    rate_maxEx_plot = nu * np.exp(-1 * np.array(Ed_maxEx_clean) / kB / temperature_clean ) \
           * np.array(theta_maxEx_clean)
    plt.plot(temperature_clean, rate_maxEx_plot, label = r'$log\nu$ = ' + str(np.log10(nu)), color=colors_nu[nu])

plt.plot(quant_TPD[max_exposure]['temperature'], quant_TPD[max_exposure]['spline_normalized_rate'] , 'k.', label='TPD data')
        
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
    plt.plot(coverage_range,  fit_Ed_maxEx_nu[key](coverage_range), label=r'$log\nu$ = ' + str(np.log10(key)), color=colors_nu[key])
    # Now plot the exact data points
    #plt.plot(coverage_range, Ed_maxEx_nu[key], 'o')

plt.xlabel(r'$\theta \ ML$')
plt.ylabel('Desorption Energy \ eV')
plt.legend(loc='best')
#plt.xlim([0, 1.2])
plt.ylim([0, 1])
plt.savefig(output + 'desoprtion_energy_fit.png')
plt.close()

"""
Plot the above diagram for all exposures on one graph
"""
plt.figure()
for nu in random_nu:
    for exposure in exposures:
        rate = []
        coverages =  quant_TPD[exposure]['coverages']
        temperature = quant_TPD[exposure]['temperature']
        Ed = fit_Ed_maxEx_nu[nu](coverages)
        for i in range(len(Ed)):
            rate_ind = nu * np.exp( -1 * Ed[i] / ( kB * temperature[i] )) * coverages[i] 
            rate.append(rate_ind)
        #rate = nu * np.exp( - Ed / kB / temperature ) * coverages
        plt.plot(temperature, rate , '--', color=colors_nu[nu])
    plt.plot([], [],  label=r'$log\nu$ = ' + str(np.log10(nu)), color=colors_nu[nu]) 
for exposure in exposures:
#    temperature = tpd_data[exposure][0]
    temperature = quant_TPD[exposure]['temperature']
    plt.plot(temperature, quant_TPD[exposure]['spline_normalized_rate'], 'k.')
#    plt.plot(temperature,  beta * \
#            (quant_TPD[exposure]['spline_normalized_rate']) , 'k.')
plt.ylabel('TPD rate')
plt.xlabel('Temperature / K')
plt.legend(loc='best')
plt.xlim([160, 280])
plt.savefig(output + 'all_TPD_Edmax.pdf')
plt.close()

"""
Plot a range of binding energies to see which one would give the best fit to 
the TPD for a pre-factor of 1e13
TODO check this
"""
colors = ['r', 'b', 'g', 'y', 'm', 'p']
nu_fixed = 1e13
fict_binding = np.arange(0.6, 0.9, 0.1)
plt.figure()
for i in range(len(fict_binding)):
    binding_energy = fict_binding[i]
    for exposure in exposures:
        temp_to_theta = quant_TPD[exposure]['spline_coverages']
        temperature = tpd_data[exposure][0]
        theta = temp_to_theta(temperature)
        rate = nu_fixed * np.exp ( - binding_energy / kB / temperature ) * theta
        plt.plot(temperature, rate)

for exposure in exposures:
    temperature = tpd_data[exposure][0]
#    plt.plot(temperature,  beta * \
#            (quant_TPD[exposure]['spline_normalized_rate']) , 'k.')
plt.ylabel('TPD rate')
plt.xlabel('Temperature / K')
#plt.xlim([100, 300])
plt.savefig(output + 'guessed_TPD.pdf')
