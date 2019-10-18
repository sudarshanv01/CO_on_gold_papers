#!/usr/bin/python

# script to analyse TPD results from Koel 2006 JPCB

import numpy as np
from glob import glob
from scipy.integrate import quad, simps
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline

#####################
# Constants in experiment
# dT / dt heating rate
beta = 3 # K/s
kB = 8.617e-05 # eV/K
#####################

def integrand(temperature, rate):
    return rate

# First get data for all initial coverages (exposures)
data = glob('/Users/vijays/Documents/project/2_gold/TPD/data_points/*')
homedir = 'data_points/'
exposures = []

for i in range(len(data)):
    exposure = float(data[i].split('.')[0].split('_')[-1].replace('p', '.'))    
    exposures.append(exposure)

exposures = np.array(exposures)
# Get the coverage relationship for the maximum exposure
# !! ASSUMPTION : maximum exposure corresponds to a coverage of 1
max_exposure = max(exposures)
text_max_exposure = np.genfromtxt(homedir + '/exposure_' \
        + str(int(max_exposure)) + '.csv', delimiter=',')
temperature = text_max_exposure[:,0]
rate = text_max_exposure[:,1]
# Apply a spline fit to the rates with respect to temperature and use that 
sorted_temperature, sorted_rate = zip(*sorted(zip(temperature, rate))) 
spline_temperature_rate = UnivariateSpline(sorted_temperature, sorted_rate)
spline_temperature_rate.set_smoothing_factor(0.01)
"""
Plot TPD and perform a spline fit over it
"""
plt.figure()
plt.plot(temperature, rate, label='TPD experiment')
plt.plot(temperature, spline_temperature_rate(temperature), 'k--',  label='Spline fit')
plt.ylabel('TPD rate \ au ')
plt.xlabel('Temperature \ K')
plt.legend(loc='best')
plt.savefig('TPD_largest_TPD.pdf')
plt.close()
# Getting desorption energies as a function of theta
# !! ASSUMPTION : Assume that the prefactor is independent of coverages
## taking direct TPD measurements data
min_rate_measurements = min(rate)
rate_normalized_measurements = rate - min_rate_measurements
min_rate_from_spline = spline_temperature_rate(temperature[np.argmin(rate)]) #assuming max temp lowest rate
spline_temperature_rate_normalized = UnivariateSpline(sorted_temperature, rate - min_rate_from_spline)
spline_temperature_rate_normalized.set_smoothing_factor(0.01)
maximum_sites_measurements = simps(integrand(temperature, rate_normalized_measurements))
maximum_sites_from_spline = spline_temperature_rate_normalized.integral(min(temperature), max(temperature))
number_sites_measurements = []
number_sites_from_spline = []
for i in range(len(rate)):
    # taking from TPD measurements
    rate_range = rate_normalized_measurements[i:]
    temperature_range = temperature[i:]
    sites_measurements = simps(integrand(temperature_range, rate_range))
    sites_from_spline = spline_temperature_rate_normalized.integral(temperature[i], temperature[-1])
    number_sites_measurements.append(sites_measurements)
    number_sites_from_spline.append(sites_from_spline)

number_sites_measurements = np.array(number_sites_measurements)
number_sites_from_spline = np.array(number_sites_from_spline)
# Taking spline fit to see difference
coverage_ftemperature = number_sites_from_spline / maximum_sites_from_spline
#coverage_ftemperature = number_sites_measurements / maximum_sites_measurements
#print(coverage_ftemperature)
#norm_rate = rate / maximum_sites_measurements
norm_rate = spline_temperature_rate_normalized(temperature) / maximum_sites_from_spline
print(norm_rate - norm_rate[-1])

"""
Plotting coverage as a function of temperature
Testing different ways to fit a line to that theta vs T curve
"""
plt.figure()
plt.plot(temperature, coverage_ftemperature, '.')
# plot coverage as a function of temperature
fit_T_theta = np.polyfit(temperature, coverage_ftemperature, 6)
fit_theta_T = np.polyfit(coverage_ftemperature, temperature, 6)
p_theta_T = np.poly1d(fit_theta_T)
p_T_theta = np.poly1d(fit_T_theta)
plt.plot(temperature, p_T_theta(temperature), 'm--', label='polynomial fit')
# numerically differetiating list
diff_theta = np.diff(coverage_ftemperature)
diff_temperature = np.diff(temperature)
dtheta_dT_numerical = diff_theta / diff_temperature
# spline fit - this would be the best way
sorted_temperature, sorted_coverage = zip(*sorted(zip(temperature, coverage_ftemperature)))
spline_temperature_theta = UnivariateSpline(sorted_temperature, sorted_coverage)
spline_temperature_theta.set_smoothing_factor(0.0)
sorted_coverage, sorted_temperature = zip(*sorted(zip(coverage_ftemperature, temperature )))
spline_theta_temperature = UnivariateSpline(sorted_coverage, sorted_temperature)
plt.plot(temperature, spline_temperature_theta(temperature), 'k-.', label='spline fit')
# plot spline fit
plt.xlabel('Temperature \ K')
plt.ylabel(r'$\theta$')
plt.legend(loc='best')
plt.savefig('temperature_coverage_largest_TPD.pdf')
plt.close()

"""
Plot the rate of change of coverage with time
Using different fits based on theta vs T curve
"""
plt.figure()
# First plot the fitted polynomial function
## take differential based on polynomial
dtheta_dT_polynomial = np.polyder(p_T_theta)
## mind the sign multiplied by -1
plt.plot(temperature, -1 *beta * dtheta_dT_polynomial(temperature), label='fitted function')
# Next plot the spline function
dtheta_dT_spline = spline_temperature_theta.derivative()
plt.plot(temperature, -1 * beta * dtheta_dT_spline(temperature), label='spine function')
plt.ylabel(r'$ - \frac{d\theta}{dt}$')
plt.xlabel('Temperature \ K')
plt.legend(loc='best')
#plt.ylim([0, None])
plt.savefig('temperature_dtheta_dt_largest_TPD.pdf')

"""
Plot the desorption energies as a function of theta
"""
# Ed = -kBT ln( -dtheta / dt / nu / theta)
random_nu = np.array([10e13, 10e14, 10e15, 10e16, 10e17])
## Fit a large order polynomial to the theta vs T curve
plt.figure()
for nu in random_nu:
    Ed_theta_list = []
    theta_list = []
    for i in range(len(temperature)):
        theta = spline_temperature_theta(temperature[i])
        theta_list.append(theta)
        Ed_theta = -1 * kB * temperature[i] \
                * np.log( -1 * dtheta_dT_spline(temperature[i]) * beta / nu / theta )
        Ed_theta_list.append(Ed_theta)
    Ed_theta_list = np.array(Ed_theta_list)
    theta_list = np.array(theta_list)
    plt.plot(theta_list, Ed_theta_list, label = r'$log\nu$ = ' + str(np.log10(nu)))

plt.ylabel('Desorption Energy / eV')
plt.xlabel(r'$\theta $ / ML')
plt.legend(loc='best')
plt.savefig('coverage_desorption_energy_largest_TPD.pdf')
plt.close()

""" 
Sanity check with back substitution to find rate
based on Polyani Wigner equation
"""
plt.figure()
for nu in random_nu:
    Ed_theta_list = []
    theta_list = []
    for i in range(len(temperature)):
        theta = spline_temperature_theta(temperature[i])
        theta_list.append(theta)
        Ed_theta = -1 * kB * temperature[i] \
                * np.log( -1 * dtheta_dT_spline(temperature[i]) * beta / nu / theta )
        Ed_theta_list.append(Ed_theta)
    Ed_theta_list = np.array(Ed_theta_list)
    theta_list = np.array(theta_list)
    rate_sanity_plot = nu * np.exp(-1 * Ed_theta_list / kB / temperature ) * theta_list
    plt.plot(temperature, rate_sanity_plot, label = r'$log\nu$ = ' + str(np.log10(nu)))

# plot the data points from the actual TPD 
plt.plot(temperature, (norm_rate - norm_rate[-1]) * beta , 'k.', label='TPD data')
plt.ylabel('TPD rate')
plt.xlabel('Temperature / K')
plt.legend(loc='best')
plt.savefig('simulated_TPD_largest_TPD.pdf')



