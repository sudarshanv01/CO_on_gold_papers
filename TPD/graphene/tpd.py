#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt 
import os
from glob import glob
from useful_functions import (
                                normalize_TPD_baseline, 
                                normalize_TPD_coverage, 
                                get_area, 
                             )
from scipy.integrate import odeint
import csv

""" Get TPD data points """

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
    tpd_data[exposure] = {}
    tpd_data[exposure]['temperature'] = temperature
    tpd_data[exposure]['rate'] = rate

max_exposure = max(exposures)

""" Step 0: Plot experimental TPD as it is """
plt.figure()
for exposure in exposures:
    plt.plot(tpd_data[exposure]['temperature'], tpd_data[exposure]['rate'], 'k')
plt.ylabel('TPD rate \ Arbitrary Units')
plt.xlabel('Temperature \ K')
plt.savefig(output + 'TPD_directly_experiment.pdf')
plt.close()

""" Step 1: Normalize TPD spectrum """
## Normalize data based on a zero somewhere in the TPD 
## start with averaging data points between 250 and 300K
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Define the needed constants
temp_range_min = [600, 700] # Temperature range to find minimum rate
temp_range = [200, 700] # Temperature range to plot the TPD in
sat_coverage = 0.1 # Maximum coverage possible in ML
sat_area = 500 # based on the area under curve vs exposure graph
beta = 20/60 # heating rate in K/s
random_nu = np.array([ 10e12, 10e13, 10e14])
colors = ['sienna', 'olive', 'navy', 'orchid', 'r']
colors_nu = dict(zip(random_nu, colors))
markers = ['v', 'o', 'x', 'd', '^', '1', '2']
markers_exp = dict(zip(exposures, markers))
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# subtracting baseline
plt.figure()
for exposure in exposures:
    # Plot normalized data to a baseline
    normalized_data = normalize_TPD_baseline(tpd_data[exposure], \
            min_range=temp_range_min, val_range=temp_range)
    rates_norm = normalized_data['normalized_rate']
    temperature = normalized_data['temperature']
    plt.plot(temperature, rates_norm, 'k', label='exposure: ' + str(exposure))
plt.ylabel('TPD rates \ arbitrary units')
plt.xlabel('Temperature \ K')
plt.savefig(output + 'norm_baseline.pdf')
plt.close()

# plotting area with exposure
plt.figure()
for exposure in exposures:
    normalized_data = normalize_TPD_baseline(tpd_data[exposure], \
            min_range=temp_range_min, val_range=temp_range)
    area = get_area(normalized_data)
    plt.plot(exposure, area, 'ro')

plt.xlabel('Exposure \ L')
plt.ylabel('Area under curve \ Arbitrary units')
plt.xscale('log')
plt.axhline(500, color='k', ls='--', lw=2)
plt.savefig(output + 'area_exposure.pdf')
plt.close()

# Plotting normalized TPD with coverage
plt.figure()
for exposure in exposures:
    normalized_data = normalize_TPD_baseline(tpd_data[exposure], \
            min_range=temp_range_min, val_range=temp_range)
    norm_data_theta = normalize_TPD_coverage(normalized_data, sat_area, sat_coverage) 
    rate_theta = norm_data_theta['rate']
    temperature = norm_data_theta['temperature']

    plt.plot(temperature, rate_theta, 'k')

plt.xlabel('Temperature \ K')
plt.ylabel('TPD rate \ ML / s')
plt.savefig(output + 'normalized_TPD.pdf')
ndata = {}
# writing out csv file to use with old script
for exposure in exposures:
    f = open(output + '/exposure_'+str(exposure).replace('.', 'p') +'.csv', 'w')
    normalized_data = normalize_TPD_baseline(tpd_data[exposure], \
            min_range=temp_range_min, val_range=temp_range)
    norm_data_theta = normalize_TPD_coverage(normalized_data, sat_area, sat_coverage) 
    ndata[exposure] = norm_data_theta
    rate_theta = norm_data_theta['rate']
    temperature = norm_data_theta['temperature']
    writer = csv.writer(f)
    for i in range(len(temperature)):
        writer.writerow([temperature[i], rate_theta[i]])

""" Step 2: Simulate the TPD """

# Find the coverage as a function of T
ncoverage = {}
for exposure in exposures:
    rates = ndata[exposure]['rate']
    temperature = ndata[exposure]['temperature']
    ncoverage[exposure] = []
    for i in range(len(temperature)):
        temp_range = temperature[i:]
        rates_range = rates[i:]
        cov_temp = np.trapz(rates_range, temp_range)
        ncoverage[exposure].append(cov_temp)
    plt.plot(temperature, ncoverage[exposure])
plt.ylabel(r'$\theta$ \ ML')
plt.xlabel('Temperature \ K')
plt.savefig(output + 'coverage.pdf')
plt.close()

# Find the desorption energy for the maximum exposure and random 
# prefactor values
def get_Ed(theta, nu, beta, T):
    # For a given set of coverages and corresponding temperatures \
    # find the desorption energy
    theta = np.array(theta)
    T = np.array(T)
    kB = 8.617e-05 # eV/K
    dtheta_dT = np.diff(theta) / np.diff(T)
    Ed = - kB * T[0:-1] * np.log(beta * -1 *dtheta_dT / nu / theta[0:-1])

    data = {'Ed':Ed, 'theta':theta[0:-1], 'dtheta_dT':dtheta_dT, 'T':T[0:-1]}

    return data

plt.figure()
for nu in random_nu:
    rates = ndata[max(exposures)]['rate']
    temperature = ndata[max(exposures)]['temperature']
    theta = ncoverage[max(exposures)]
    data_Ed = get_Ed(theta, nu, beta, temperature)
    Ed = data_Ed['Ed']
    theta = data_Ed['theta']
    plt.plot(theta, Ed, color=colors_nu[nu], label=r'$log \nu = $'+str(np.log10(nu)))
plt.ylabel(r'$E_{d}$ \ eV')
plt.xlabel(r'$\theta$ \ ML')
plt.legend(loc='best')
plt.savefig(output + 'desorption_energy.pdf')
plt.close()

# Use the desorption energy profiles and find the rates
# Compare this with TPD values obtained from the experiment

def simulated_rate(theta, nu, Ed, T):
    import numpy as np
    kB = 8.617e-05 # eV/K
    rate = nu * np.exp( - Ed / kB / T ) * theta 

    return rate

def interpolate_des(theta, Ed):
    from scipy import interpolate
    f = interpolate.interp1d(theta, Ed, fill_value="extrapolate")
    return f
plt.figure()
for nu in random_nu:
    # get the desorption energy vs theta curve needed 
    rates = ndata[max(exposures)]['rate']
    temperature = ndata[max(exposures)]['temperature']
    theta = ncoverage[max(exposures)]
    data_Ed = get_Ed(theta, nu, beta, temperature)
    Ed = data_Ed['Ed']
    theta = data_Ed['theta']
    interp = interpolate_des(theta, Ed)
    
    # simulate the TPD with this desorption profile
    for exposure in exposures[::2]:
        rates_ex = ndata[exposure]['rate']
        theta_ex = ncoverage[exposure]
        temperature_ex = ndata[exposure]['temperature']
        # plot the data points first
        plt.plot(temperature_ex, rates_ex, 'kv')
        # plot similated TPD
        Ed_simulated = interp(theta_ex)
        rates_simulated = simulated_rate(theta_ex, nu, Ed_simulated, temperature_ex)
        plt.plot(temperature_ex, rates_simulated / beta, color=colors_nu[nu], ls='--')
    plt.plot([], [], color=colors_nu[nu], label=r'$log \nu = $'+str(np.log10(nu)))
plt.ylabel('TPD rates \ ML / s')
plt.xlabel('Temperature \ K')
plt.legend(loc='best')
plt.savefig(output + 'simulated_TPD.pdf')


""" Step 3: Use TST to get a best case estimate for desorption energy """

# Assume a fixed nu 
fixed_nu = 10e13
kB = 8.617e-05 # eV/K
# start by assuming a constant Ed that does not change with coverage
# question - what is the Ed that we would get for each TPD spectrum for a 
# fixed nu
plt.figure()
for nu in random_nu:
    plt.plot([], [], color=colors_nu[nu], label=r'$log \nu = $'+str(np.log10(nu)))
    for exposure in exposures:
        rates = ndata[exposure]['rate']
        temperature = np.array(ndata[exposure]['temperature'])
        theta = np.array(ncoverage[exposure])
        dtheta_dt = beta * np.diff(theta) / np.diff(temperature)
        Ed = -kB * temperature[0:-1] * np.log(-1 *dtheta_dt / nu / theta[0:-1])
        plt.plot(theta[0:-1], Ed, color=colors_nu[nu], marker=markers_exp[exposure])
        
for exposure in exposures:  
    plt.plot([], [], 'k', marker=markers_exp[exposure], label='exposure: ' + str(exposure))

plt.ylabel(r'$\Delta E_{des}$')
plt.xlabel(r'$\theta$ \ ML')
plt.legend(loc='best')
plt.savefig(output + 'fixed_nu_desorption.pdf')


