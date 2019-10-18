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
from scipy.integrate import odeint, quad
import csv
from ase.thermochemistry import HarmonicThermo, IdealGasThermo
from ase.io import read
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d

""" Get TPD data points """

output = 'output/'
os.system('mkdir -p ' + output)
newtpd = 'newtpd'
os.system('mkdir -p ' + newtpd)

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
temp_range_min = [100, 120] # Temperature range to find minimum rate
temp_range = [120, 300] # Temperature range to plot the TPD in
sat_coverage = 0.17 # Maximum coverage possible in ML
sat_area = 0.2 # based on the area under curve vs exposure graph
beta = 3 # heating rate in K/s
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
plt.axhline(sat_area, color='k', ls='--', lw=2)
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

def gaussian(x, a, x0, sigma):
    values = a * np.exp( - (x - x0)**2 / ( 2* sigma**2))
    return values

plt.figure()
""" trial: try to simulate the step TPD to full coverage """
for exposure in exposures:
    f = open(newtpd + '/exposure_'+str(exposure).replace('.', 'p') +'.csv', 'w')
    rate_exp = ndata[exposure]['rate']
    temperature = ndata[exposure]['temperature']
    plt.plot(temperature, rate_exp, 'kv')
    # fit a parabola to this and see what comes out
    #fit = np.polyfit(temperature, rate_exp, 10)
    #p = np.poly1d(fit)
    # fit to gaussian distribution 
    mean_rate = np.sum(temperature*rate_exp) / np.sum(rate_exp) #np.mean(rate_exp)
    sigma_rate = np.sqrt(np.sum(rate_exp *(temperature - mean_rate)**2) / np.sum(rate_exp) )#np.std(rate_exp)
    rate_max = rate_exp.max()
    popt, pcov = curve_fit(gaussian, temperature, rate_exp,\
            p0=[rate_max, mean_rate, sigma_rate])
    temper_simu = np.linspace(90, 300)
    writer = csv.writer(f)
    for i in range(len(temperature)):
        writer.writerow([temper_simu[i], gaussian(temper_simu, *popt)[i]])
    plt.plot(temper_simu, gaussian(temper_simu, *popt), '--', lw=2)

plt.ylabel('TPD rate \ ML / s')
plt.xlabel('Temperature \ K ')
plt.savefig(output + '/parabola_fit.pdf')
plt.close()


""" Step 2: Simulate the TPD """

# Find the coverage as a function of T
ncoverage = {}
plt.figure()
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
Ed_values = {}
for nu in random_nu:
    Ed_values[nu] = {}
    plt.plot([], [], color=colors_nu[nu], label=r'$log \nu = $'+str(np.log10(nu)))
    for exposure in exposures:
        rates = ndata[exposure]['rate']
        temperature = np.array(ndata[exposure]['temperature'])
        theta = np.array(ncoverage[exposure])
        dtheta_dt = beta * np.diff(theta) / np.diff(temperature)
        Ed = -kB * temperature[0:-1] * np.log(-1 *dtheta_dt / nu / theta[0:-1])
        Ed_values[nu][exposure] = Ed
        plt.plot(theta[0:-1], Ed, color=colors_nu[nu], marker=markers_exp[exposure])
        
for exposure in exposures:  
    plt.plot([], [], 'k', marker=markers_exp[exposure], label='exposure: ' + str(exposure))

plt.ylabel(r'$\Delta E_{des}$')
plt.xlabel(r'$\theta$ \ ML')
plt.legend(loc='best')
plt.savefig(output + 'fixed_nu_desorption.pdf')

# Plot the best fit desorption energy profile for a given pre-factor 
for nu in random_nu:
    plt.figure()
    Ed_exposure = Ed_values[nu]
    for key in Ed_exposure:
        Ed = Ed_exposure[key]
        rates = ndata[key]['rate']
        temperature = ndata[key]['temperature']
        theta = ncoverage[key]
        # plot the rates directly from experiment
        plt.plot(temperature, rates * beta, 'kv')
        # plot the rates calulated from coverages derived
        rate_simulated = simulated_rate(theta[0:-1], nu, Ed, temperature[0:-1])
        plt.plot(temperature[0:-1], rate_simulated)
    plt.ylabel('TPD rates \ ML / s')
    plt.xlabel('Temperature \ K')
    plt.legend(loc='best')
    plt.savefig(output + str(int(np.log10(nu))) + 'simulated_TPD.pdf')
    plt.close()

def sum_squared_error(right, exp):
    #assert len(right) == len(exp)
    sse = []
    if len(right) > len(exp):
        for i in range(len(exp)):
            sse.append( (right[i] - exp[i] ) ** 2)
    else:
        for i in range(len(right)):
            sse.append( (right[i] - exp[i] ) ** 2)
        
    sse = np.array(sse)
    return sse

# find the best fit simulated rate and plot that for each prefactor
# the idea is to get one desorption rate per prefactor for each exposure
Ed_plot = {}
arg_exposure_plot = {}
for nu in random_nu:
    plt.figure()
    Ed_exposure = Ed_values[nu]
    rmse_all = []
    exposure_count = []
    # pick a desorption energy profile
    for exposure in Ed_exposure:
        exposure_count.append(exposure)
        sse = []
        Ed = Ed_exposure[exposure]
        temperature = ndata[exposure]['temperature']
        theta = ncoverage[exposure]
        rate_simulated = simulated_rate(theta[0:-1], nu, Ed, temperature[0:-1])
        # now comapare the fit with all experimental rates
        for exposure in Ed_exposure:
            rates_exp = ndata[exposure]['rate']
            sse.append(sum_squared_error(rates_exp, rate_simulated))
        sse_flat = np.array([item for sublist in sse for item in sublist])
        rmse = np.sqrt(sse_flat.mean() / len(sse_flat))
        rmse_all.append(rmse)
    # For the one with the minimum rmse plot the comparison with experiment
    arg_minrmse = np.argmin(rmse_all)
    exposure_minrmse = exposure_count[arg_minrmse]
    arg_exposure_plot[nu] = arg_minrmse
    Ed_min = Ed_exposure[exposure_minrmse]
    Ed_plot[nu] = Ed_min
    for exposure in Ed_exposure:
        temperature = ndata[exposure]['temperature']
        rates = ndata[exposure]['rate']
        theta = ncoverage[exposure]
        rate_simulated = simulated_rate(theta[0:len(Ed_min)], nu, Ed_min,\
                temperature[0:len(Ed_min)])
        plt.plot(temperature[0:len(Ed_min)], rate_simulated, color='indianred', lw=3)
        plt.plot(temperature, rates * beta, 'k-')

plt.ylabel('TPD rates \ ML / s')
plt.xlabel('Temperature \ K')
plt.legend(loc='best')
plt.plot([], [], color='indianred', label='Simulated')
plt.plot([], [], color='k', label='Experimental fit')
#plt.savefig(output + str(int(np.log10(nu))) + 'best_fit_TPD.pdf')
plt.axvline(190, color='k', lw=3, ls='--')
plt.legend(loc='best')
plt.savefig(output + 'best_fit_TPD.pdf')
plt.close()
    

plt.figure()
for nu in random_nu:
    exposure = exposures[arg_exposure_plot[nu]]
    theta = ncoverage[exposure]
    plt.plot(theta[0:len(Ed_plot[nu])], Ed_plot[nu], color=colors_nu[nu], lw=3, label=r'$log \nu = $'+str(np.log10(nu)))
plt.ylabel(r'$\Delta E_{des} / eV$')
plt.xlabel(r'$\theta$')
plt.savefig(output + 'best_fit_desorption.pdf')
plt.close()


""" Plotting equilibirum coverage vs temperature """
T_plot_fit = np.linspace(120, 600, 200)
plt.figure()
for nu in random_nu:
    Ed = Ed_values[nu][max(exposures)]
    T_plot = ndata[max(exposures)]['temperature']
    p = interp1d(T_plot[0:-1], Ed, fill_value=(Ed[0], Ed[-1]), bounds_error=False)
    #p = interp1d(T_plot[0:-1], Ed, fill_value="extrapolate")
    plt.plot(T_plot[0:-1], Ed,  'kv')
    plt.plot(T_plot_fit, p(T_plot_fit), '--', label=r'$log \nu = $'+str(np.log10(nu)))
plt.ylabel(r'$E_{d}$ \ eV')
plt.xlabel('Temperature \ K')
plt.legend(loc='best')
plt.savefig(output + 'interp_temp_Ed.pdf')


vibration_energies_ads = 0.00012 * np.array([129.65, 155.55, 189.8, 227.5, 2073.25])
thermo_ads = HarmonicThermo(vibration_energies_ads)
# gas CO thermochemistry
atoms = read('co.traj')
vibration_energies_gas = 0.00012 * np.array([89.8, 127.2, 2145.5])
thermo_gas = IdealGasThermo(vibration_energies_gas, atoms = atoms, \
        geometry='linear', symmetrynumber=1, spin=0)
#T_range = np.linspace(160, 300, num=200) # K
kB = 8.617e-05 # eV/K
pco = 101325. #pressure of CO
## Doing this for the maximum exposure
plt.figure()
for nu in random_nu:
    Ed = Ed_values[nu][max(exposures)]
    T_plot = ndata[max(exposures)]['temperature']
    #p = interp1d(T_plot[0:-1], Ed, fill_value=(Ed[0], Ed[-1]), bounds_error=False)
    p = interp1d(T_plot[0:-1], Ed, fill_value="extrapolate")
    # convering all these values to equlibrium delta G
    theta_co_eq = []
    for i in range(len(T_plot_fit)):
        T = T_plot_fit[i]
        deltaE = p(T)

        entropy_ads = thermo_ads.get_entropy(temperature=T) 
        entropy_gas = thermo_gas.get_entropy(temperature=T, \
                                                      pressure=pco)
        #print('Temperature: %d'%T)
        #print('Entropy of gas: %1.7f'%entropy_gas)
        #print('Entropy of adsorbate: %1.7f'%entropy_ads)
        entropy_difference =  (entropy_gas - entropy_ads)
        #print('Entropic difference: %1.5f'%entropy_difference)
        free_correction = -1 * entropy_difference * T #free_correction_ads - free_correction_gas
        #print('-T Delta S %1.2f'%(free_correction))
        deltaG = deltaE + free_correction
        #print('Delta G')
        print(deltaE, deltaG)
        K = np.exp( -1 * deltaG / kB / T )
        partial_co = pco / 101325
        theta_eq = 1 / ( 1 + K / partial_co )
        theta_co_eq.append(theta_eq)

    theta_co_eq = np.array(theta_co_eq)
    plt.plot(T_plot_fit, theta_co_eq, lw=3)

plt.ylabel(r'$\theta_{CO}^{eq}$ \ ML', fontsize=28)
plt.xlabel('Temperature \ K', fontsize=28)
plt.savefig(output + 'equilibrium_co.pdf')



""" Now with temperature dependent pre factor """
def Ed_temp_dependent(temperature, rate, beta):
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

rates_maxExposure = ndata[max(exposures)]['rate']
T_maxExposure = ndata[max(exposures)]['temperature']
data_tempDep = Ed_temp_dependent(T_maxExposure, rates_maxExposure, beta)
Ed_tempDep = data_tempDep['Ed']
theta_tempDep = data_tempDep['theta']
#p = interp1d(theta_tempDep[0:-1], Ed_tempDep, fill_value=(Ed_tempDep[0], Ed_tempDep[-1]), bounds_error=False)
p = interp1d(theta_tempDep[0:-1], Ed_tempDep, fill_value="extrapolate")
#p = interp1d(T_plot[0:-1], Ed_tempDep, fill_value="extrapolate")
plt.figure()
## Desorption energy as a function of coverage
plt.plot(theta_tempDep[0:-1], Ed_tempDep, 'kv', fillstyle='none', mew=2)
plt.plot(theta_tempDep, p(theta_tempDep),'--')
plt.ylabel(r'$\Delta E_{des}$ \ eV', fontsize=28)
plt.xlabel(r'$\theta $ \ ML', fontsize=28)
plt.savefig(output + 'temp_dependent_Ed.pdf')
plt.close()
import pickle
data_pickle = {'Ed':Ed_tempDep, 'theta':theta_tempDep[0:-1]}
f = open('desorption_energy.pickle', 'wb')
pickle.dump(data_pickle, f)
f.close()
## Coverage as a function of temperature
p_theta_T = interp1d(T_maxExposure, theta_tempDep, fill_value="extrapolate")
plt.figure()
plt.plot(T_maxExposure, theta_tempDep, 'kv', fillstyle='none', mew=2)
plt.plot(T_plot_fit, p_theta_T(T_plot_fit), '--', lw=3)
plt.ylabel(r'$\theta $ \ ML', fontsize=28)
plt.xlabel('Temperature \ K')
plt.savefig(output + '/coverage_temperature_dep.pdf')
plt.close()

# convering all these values to equlibrium delta G
theta_co_eq = []
plt.figure()
# plotting coverage dependent binding energies

for i in range(len(T_plot_fit)):
    T = T_plot_fit[i]
    theta_co_tpd = p_theta_T(T)
    deltaE = p(theta_co_tpd)

    entropy_ads = thermo_ads.get_entropy(temperature=T) 
    entropy_gas = thermo_gas.get_entropy(temperature=T, \
                                                  pressure=pco)
    #print('Temperature: %d'%T)
    #print('Entropy of gas: %1.7f'%entropy_gas)
    #print('Entropy of adsorbate: %1.7f'%entropy_ads)
    entropy_difference =  (entropy_gas - entropy_ads)
    #print('Entropic difference: %1.5f'%entropy_difference)
    free_correction = -1 * entropy_difference * T #free_correction_ads - free_correction_gas
    #print('-T Delta S %1.2f'%(free_correction))
    deltaG = deltaE + free_correction
    #print('Delta G')
    print(deltaE, deltaG)
    K = np.exp( -1 * deltaG / kB / T )
    partial_co = pco / 101325
    theta_eq = sat_coverage * 1 / ( 1 + K / partial_co )
    theta_co_eq.append(theta_eq)

theta_co_eq = np.array(theta_co_eq)
plt.plot(T_plot_fit, theta_co_eq, 'k', lw=3, label=r'Coverage dependent $\Delta E_{des}$')

# plotting curve for a fixed binding energy
deltaE_CO_plot = np.array([0.3, 0.4, 0.5, 0.6, 0.7])
theta_co_vary = {}
for deltaE in deltaE_CO_plot:
    theta_co_eq = []
    for i in range(len(T_plot_fit)):
        T = T_plot_fit[i]

        entropy_ads = thermo_ads.get_entropy(temperature=T) 
        entropy_gas = thermo_gas.get_entropy(temperature=T, \
                                                      pressure=pco)
        #print('Temperature: %d'%T)
        #print('Entropy of gas: %1.7f'%entropy_gas)
        #print('Entropy of adsorbate: %1.7f'%entropy_ads)
        entropy_difference =  (entropy_gas - entropy_ads)
        #print('Entropic difference: %1.5f'%entropy_difference)
        free_correction = -1 * entropy_difference * T #free_correction_ads - free_correction_gas
        #print('-T Delta S %1.2f'%(free_correction))
        deltaG = deltaE + free_correction
        #print('Delta G')
        print(deltaE, deltaG)
        K = np.exp( -1 * deltaG / kB / T )
        partial_co = pco / 101325
        theta_eq = sat_coverage / ( 1 + K / partial_co )
        theta_co_eq.append(theta_eq)
    theta_co_eq = np.array(theta_co_eq)
    theta_co_vary[deltaE] = theta_co_eq

for deltaE in theta_co_vary:
    plt.plot(T_plot_fit, theta_co_vary[deltaE],'--', label=r'$\Delta E_{des} = ' + str(deltaE) + '$ eV')

plt.ylabel(r'$\theta_{CO}^{eq}$ ', fontsize=28)
plt.xlabel('Temperature \ K', fontsize=28)
#plt.legend(loc='best')
plt.legend(bbox_to_anchor=(0,1.02,1,0.2), loc="lower left", mode="expand", borderaxespad=0, ncol=2)
#plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")
plt.yticks([])

#plt.tight_layout()
plt.savefig(output + 'temp_dep_Ed_equilibrium_co.pdf',  bbox_inches="tight")

