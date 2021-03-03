#!/usr/bin/python

import numpy as np
from glob import glob
from useful_functions import AutoVivification, get_vibrational_energy
from pprint import pprint
import os, sys
from scipy.optimize import curve_fit, least_squares, minimize
from matplotlib import cm
sys.path.append('../classes/')
from parser_function import get_stable_site_vibrations, get_gas_vibrations, \
                            get_coverage_details, diff_energies, \
                            get_lowest_absolute_energies,\
                            get_differential_energy,\
                            accept_states, \
                            get_constants, \
                            stylistic_comp, stylistic_exp

from parser_class import ParseInfo, experimentalTPD
import mpmath as mp
from ase.thermochemistry import HarmonicThermo, IdealGasThermo
from ase.io import read
from ase.db import connect
import matplotlib
import csv
from ase.units import kB
from mpmath import mpf

### PLOT related preferences
# matplotlib.rc('text', usetex=True)
# matplotlib.rcParams['text.latex.preamble']=[r'\usepackage{amsmath}']
# import matplotlib.pyplot as plt
# plt.rcParams["font.family"] = "Times New Roman"
# plt.rc('axes', labelsize=32)    # fontsize of the x and y labels
# plt.rcParams['xtick.labelsize'] = 24
# plt.rcParams['ytick.labelsize'] = 24
from scipy import optimize
from matplotlib.ticker import FormatStrFormatter
from matplotlib import cm
from scipy.optimize import newton
import matplotlib.pyplot as plt
plt.style.use('science')

class PlotTPD():

    """Get all needed parameters for the TPD analysis 
    """

    def __init__(self, exp_data, order, constants, color_map,
                    thermo_ads, correct_background, thermo_gas, bounds=[], plot_temperature=np.linspace(100,400),
                    pco=101325):
        """Perform the temperature programmed desorption analysis for a surface 
        based on configurational entropy an interaction parameters and a zero coverage
        energy term

        Args:
            exp_data (list): globbed files with csv
            order (int): Order of the reaction
            constants (list): Parameters to parse TPD
            color_map (obj): Color map for different exposures
            thermo_ads (obj): HarmonicThermo for the adsorbate
            thermo_gas (obj): IdealGasThermo for the gas
            bounds (list, optional): Bounds within to fit the coverage of the TPD. Defaults to [].
            plot_temperature (array, optional): Temperature range to plot the equilbirum coverage. Defaults to np.linspace(100,400).
            pco (float, optional): Pressure of CO(g). Defaults to 101325.
        """

        # Define all the __init__ variables
        self.constants = constants # Constants for TPD parsing
        self.exp_data = exp_data # globbed files with csv
        self.order = order
        self.color_map = color_map # color map to distinguish different exposures
        self.thermo_ads = thermo_ads # adsorbate HarmonicThermo
        self.thermo_gas = thermo_gas # gas phase IdealGasThermo
        self.plot_temperature = plot_temperature # temperature for plotting equilibrium coverages
        self.pco = pco # partial pressure of CO(g)
        self.bounds = bounds
        self.correct_background = correct_background

        self.font_number = []
        
        # Results
        self.norm_results = AutoVivification()
        self.results = AutoVivification()
        self.Ed = AutoVivification()
        self.theta_rel = AutoVivification()
        self.theta_eq = AutoVivification()
        self.theta_eq_p = AutoVivification()
        self.theta_eq_n = AutoVivification()
        self.E0 = AutoVivification()
        self.b = AutoVivification()
        self.theta_sat = AutoVivification()
        self.dG = AutoVivification()
        self.error = AutoVivification() # Error in Ed fit
        self.temperature_range = {} # for plotting in main figure

        self._define_other_plots()
        self._get_styles()
        self.get_results()



    def _define_other_plots(self):
        """Defines the plots to be made while parsing the TPD 
        (SI plots)
        """
        plot_number = len(self.constants[0]) + 1
        # Figure of the TPD data with correction for CO pumping
        self.figP, self.axP = plt.subplots(1, 2, figsize=(10, 6))
        # Plot desorption energy as a function of the coverage
        self.figc, self.axc = plt.subplots(plot_number, 1,figsize=(8,8))
        # Matching Ed with fitted coverage
        self.figC, self.axC = plt.subplots(1, plot_number,figsize=(16,6))
        # Plot of the free energy of adsorption
        self.figG, self.axG = plt.subplots(1, plot_number, figsize=(16,6))
        # Plot the equilibirum coverage as a function of the temperature
        self.figT, self.axT = plt.subplots(1, plot_number, figsize=(15, 5))
        # Plot the saturation coverage as a function of exposure 
        self.figsat, self.axsat = plt.subplots(1, plot_number, figsize=(15, 5), sharey=True)
        # Plot the interaction parameter as a function of exposure 
        self.figb, self.axb = plt.subplots(1, plot_number, figsize=(15, 5))
        # Plot the contributions to the desorption energy 
        self.figEc, self.axEc = plt.subplots(1, 1, figsize=(8,6))

    def _get_styles(self):
        """get the color maps
        """
        # Picks up the styles for making the plots
        self.inferno, self.viridis, self.font_number = stylistic_exp()

    def _exponential_fit(self, temperature, a, k):
        """Exponential fit to the tail of the TPD to remove pumping 
        related rates

        Args:
            temperature (list): temperature list based on TPD
            a (float): amplitude of exponent 
            k (float): argument of exponent 

            Returns:
            list: rates for each temperature
        """
        rate = a * np.exp(-k * temperature)
        return rate

    def get_results(self):
        """Perform the TPD analysis
        """
        T_switch, T_max, T_rate_min, beta = self.constants

        # Create the temperature range based on the switch data
        assert len(T_switch) < 3 , 'Only implemented switch lower than 2 '
        if len(T_switch) == 1:
            temperature_ranges = [
                                    [0, T_switch[0]], ## Terrace
                                    [T_switch[0], T_max] ## Step
                                ]
        elif len(T_switch) == 2:
            temperature_ranges = [
                [0, T_switch[0]], # Terrace 1
                [T_switch[0], T_switch[1]], # Terrace 2
                [T_switch[1], T_max] # step
            ]

        # range of temperatures for different TPD values
        self.temperature_range = temperature_ranges

        # 1. Get the TPD results with includes background subtraction
        # for each exposure
        for index, f in enumerate(sorted(self.exp_data)):
            exposure = float(f.split('/')[-1].split('.')[0].split('_')[1].replace('p', '.'))
            self.norm_results[exposure] = experimentalTPD(tpd_filename=f,
                                temprange=temperature_ranges,
                                tempmin=T_rate_min,
                                beta=beta,
                                order=self.order, 
                                correct_background=self.correct_background,
                                )

            # Iterate over the different facets in temperature ranges
            for surface_index in range(len(temperature_ranges)):

                T_range = temperature_ranges[surface_index]
                indices = [ a for a in range(len(self.norm_results[exposure].temperature)) \
                        if T_range[0] < self.norm_results[exposure].temperature[a] < T_range[1] ] 

                self.results[surface_index][exposure]['temperature'] = self.norm_results[exposure].temperature[indices]
                self.results[surface_index][exposure]['normalized_rate'] = self.norm_results[exposure].normalized_rate[indices]

                # create variable within look for easy calling 
                temperatures = self.results[surface_index][exposure]['temperature']
                rates = self.results[surface_index][exposure]['normalized_rate']

                #2. For each point get the energy of desorption as a function of the coverage
                data = self._Ed_temp_dependent(
                            temperature=temperatures, 
                            rate=rates,
                            beta=beta,
                            )

                ## if there are nans
                args_accept = [i for i in range(len(data[0])) \
                                    if np.isfinite(data[0][i]) and \
                                    data[1][i] > 0]

                self.Ed[surface_index][exposure], self.theta_rel[surface_index][exposure] = data
                self.Ed[surface_index][exposure] = self.Ed[surface_index][exposure][args_accept]
                self.theta_rel[surface_index][exposure] = self.theta_rel[surface_index][exposure][args_accept]
                temperature_fit = self.norm_results[exposure].temperature[indices][args_accept]

                ## TODO: Is bounds needed in code?
                # # check if relative theta has bounds
                # if self.bounds:
                #     index_bounds = [i for i in range(len(self.theta_rel[surface_index][exposure])) \
                #              if min(self.bounds) <=  self.theta_rel[surface_index][exposure][i] <= max(self.bounds)]
                #     self.Ed[surface_index][exposure] = self.Ed[surface_index][exposure][index_bounds]
                #     self.theta_rel[surface_index][exposure] = self.theta_rel[surface_index][exposure][index_bounds]
                #     temperature_fit = temperature_fit[index_bounds]
                    
                # 3. Fit the Desorption energy curve to the desorption energy equation
                ## First get good initial guesses for parameters
                guess_E0 = self.Ed[surface_index][exposure][0]
                linear_region = [i for i in range(len(self.Ed[surface_index][exposure])) if 0.25 < self.Ed[surface_index][exposure][i] < 0.75  ]
                guess_b = -1 * np.polyfit(self.theta_rel[surface_index][exposure][linear_region], self.Ed[surface_index][exposure][linear_region], 1)[0]

                popt, pcov = curve_fit(\
                lambda temp, E0, b, theta_sat: self._fit_Ed_theta(temp, E0, b, theta_sat, \
                                            self.theta_rel[surface_index][exposure]), \
                                            xdata=temperature_fit,
                                            ydata=self.Ed[surface_index][exposure],
                                            p0=[guess_E0, guess_b, 0.9], 
                                                )

                # # Least squares routine
                # TODO better minimisation routine?
                # res = minimize( 
                #     self._least_sq_Ed_theta, 
                #     x0=[guess_E0, guess_b, 0.8], 
                #     args=(temperature_fit, self.theta_rel[surface_index][exposure], self.Ed[surface_index][exposure]),
                #     bounds=bounds,
                #     # options={'ftol': 1e-2 }
                #     tol=1e-6,
                # ) 
                # popt = res.x
                # print(popt, guess_b)
                # print(res.message)
                # print(self._least_sq_Ed_theta(res.x, temperature_fit, self.theta_rel[surface_index][exposure], self.Ed[surface_index][exposure] ))

                residual = self._least_sq_Ed_theta(popt, temperature=temperature_fit, theta_rel=self.theta_rel[surface_index][exposure], \
                    Ed_real=self.Ed[surface_index][exposure])

                self.E0[surface_index][exposure], self.b[surface_index][exposure], self.theta_sat[surface_index][exposure], \
                                = popt
                self.error[surface_index][exposure] = residual#np.sqrt(np.diag(pcov)) #self._least_sq_Ed_theta(res.x, temperature_fit, self.theta_rel[surface_index][exposure], self.Ed[surface_index][exposure] ) #np.sqrt(np.diag(pcov))
                # 4. Calculate the coverage at equilbirum
                self.theta_eq[surface_index][exposure], self.dG[surface_index][exposure]\
                 = self._get_equilibirum_coverage(
                            E0=self.E0[surface_index][exposure], 
                            b=self.b[surface_index][exposure],
                            )
                self.theta_eq_p[surface_index][exposure], _ \
                 = self._get_equilibirum_coverage(
                            E0=self.E0[surface_index][exposure]+self.error[surface_index][exposure], 
                            b=self.b[surface_index][exposure],
                            )
                self.theta_eq_n[surface_index][exposure], _ \
                 = self._get_equilibirum_coverage(
                            E0=self.E0[surface_index][exposure]-self.error[surface_index][exposure],
                            b=self.b[surface_index][exposure],
                            )
                
                ######## PLOTS ##########

                ## Corrected TPD plots
                if surface_index == len(temperature_ranges)-1: # only plot the corrected TPD once
                    # Plot the original TPD and the corrected portion
                    self.axP[0].plot(self.norm_results[exposure].exp_temperature, \
                                     self.norm_results[exposure].exp_rate,# self.results[exposure].exp_rate, \
                                     '.', \
                                     color=self.color_map(index),)
                    self.axP[0].plot(
                                     self.norm_results[exposure].exp_temperature,
                                     self.norm_results[exposure].background_correction,
                                            color=self.color_map(index),  lw=3,
                                                    )

                    # Then plot the normalized TPD
                    self.axP[1].plot(temperatures, rates, '.', 
                                     color=self.color_map(index), alpha=0.40
                                     )

                ## Desorption energy as a function of the coverage
                # Plot the desorption energy as a function of the coverage
                self.axc[surface_index].plot(self.theta_rel[surface_index][exposure], self.Ed[surface_index][exposure],
                                '.', \
                                color=self.color_map(index), alpha=0.40)

                ## Fitted desoprtion energy as a function of the coveage
                self.axC[surface_index].plot(self.theta_rel[surface_index][exposure],\
                                    self.Ed[surface_index][exposure],
                                '.', \
                                color=self.color_map(index), alpha=0.40)
                self.axC[surface_index].plot(self.theta_rel[surface_index][exposure],\
                                    self._fit_Ed_theta(temperature_fit, \
                                                        *popt, self.theta_rel[surface_index][exposure]),
                                '--', \
                                color=self.color_map(index), alpha=0.40)

                ## Plot the Free energy variation as a function of the temperature
                self.axG[surface_index].plot(self.plot_temperature, self.dG[surface_index][exposure], \
                                color=self.color_map(index), alpha=0.40)

                ## Plot the equilibirum coverage
                self.axT[surface_index].plot(self.plot_temperature, self.theta_eq[surface_index][exposure], \
                                color=self.color_map(index), alpha=0.40)
                
                ## plot the saturation coverage as a function of exposure 
                self.axsat[surface_index].plot(exposure, self.theta_sat[surface_index][exposure], 'o', 
                        color=self.color_map(index), alpha=0.4)
                self.axsat[surface_index].errorbar(exposure, self.theta_sat[surface_index][exposure], \
                                    self.error[surface_index][exposure], color=self.color_map(index), alpha=0.4)
                
                ## plot the total coverage 
                total_coverage = self.theta_sat[surface_index][exposure] * self.theta_rel[surface_index][exposure]
                interaction_term = - self.b[surface_index][exposure] * self.theta_sat[surface_index][exposure] * self.theta_rel[surface_index][exposure]
                config_term = - kB * temperature_fit * np.log(self.theta_sat[surface_index][exposure]*self.theta_rel[surface_index][exposure] / ( 1 - self.theta_sat[surface_index][exposure]*self.theta_rel[surface_index][exposure]))
                if surface_index == len(temperature_ranges)-1:
                    self.axEc.plot(total_coverage, interaction_term, color=self.color_map(index), ls='dashdot' )
                    self.axEc.plot(total_coverage, config_term, color=self.color_map(index), ls='--')
                    self.axEc.axhline(self.E0[surface_index][exposure], color=self.color_map(index), ls=':')
                    self.axEc.plot(total_coverage, self._fit_Ed_theta(temperature_fit, \
                                                        *popt, self.theta_rel[surface_index][exposure]), color=self.color_map(index), ls='-' )


        ######## Legends and Plot details
        self.axP[0].set_xlabel(r'Temperature / K ')
        self.axP[1].set_xlabel(r'Temperature / K ')
        self.axP[0].set_ylabel(r'TPD rate / arb units ')
        self.axP[1].set_ylabel(r'Norm. TPD rate / arb units')
        for i in range(len(self.axc)):
            self.axc[i].set_xlabel(r'$\theta_{rel}$')
            self.axc[i].set_ylabel(r'$G_{d}$ / eV')
            self.axc[i].set_title(r'Fitted temperature range {f}'.format(f=temperature_ranges[i]))
        for i in range(len(self.axC)):
            self.axC[i].set_xlabel(r'$\theta_{rel}$')
            self.axC[i].set_ylabel(r'$G_{d}$ / eV')
            self.axC[i].set_title(r'Fitted temperature range {f}'.format(f=temperature_ranges[i]))
        # self.axC[0].set_title('Terrace')
        # self.axC[1].set_title('Step')
        for i in range(len(self.axG)):
            self.axG[i].set_xlabel(r'Temperature')
            self.axG[i].set_ylabel(r'$\Delta G$ / eV')
            self.axG[i].set_title(r'Fitted temperature range {f}'.format(f=temperature_ranges[i]))
        for i in range(len(self.axT)):
            self.axT[i].set_xlabel(r'Temperature')
            self.axT[i].set_ylabel(r'$\theta_{eq}$ / ML')
            self.axT[i].set_title(r'Fitted temperature range {f}'.format(f=temperature_ranges[i]))
        for i in range(len(self.axsat)):
            self.axsat[i].set_xlabel(r'Exposure / L ')
            self.axsat[i].set_ylabel(r'$ \theta_{sat} $ / ML')
            self.axsat[i].set_title(r'Fitted temperature range {f}'.format(f=temperature_ranges[i]))
        # for i in range(len(self.axEc)):
        self.axEc.set_xlabel(r'$\theta_{rel} \theta_{sat} = \theta$')
        self.axEc.set_ylabel(r'Contribution to $G_{d}$ ')
        self.axEc.plot([], [], label=r'$-b\theta_{sat}\theta_{rel}$', color='k', ls='dashdot')
        self.axEc.plot([], [], label=r'$-k_{b}T log \left ( \frac{\theta_{rel} \theta_{sat}}{1 - \theta_{rel} \theta_{sat}}  \right )$', color='k', ls='--')
        self.axEc.plot([], [], label=r'$E_{\theta \to 0}$', color='k', ls=':')
        self.axEc.plot([], [], label=r'$G_{d}$', color='k', ls='-')
        self.axEc.set_title(r'$G_{d} = E_{\theta \to 0} - b\theta_{rel}\theta_{sat} -k_{b}T log \left ( \frac{\theta_{rel} \theta_{sat}}{1 - \theta_{rel} \theta_{sat}}  \right )$')
        self.axEc.legend(loc='best')



    def _Ed_temp_dependent(self, temperature, rate, beta):
        """Gets the desorption energy as a function of the temperature
        1. Do trapezoidal integration to get the coverage by integrating over the 
           rate and temperature 
        2. Get the desorption energy by fitting to the form 
                Ed = -kBT log(-dtheta/dt / mu / theta)
        3. Normalise theta by dividing my maximum coverage

        Args:
            temperature (list): temperatures corresponding to the TPD
            rate (list): rate from the TPD
            beta (float): Rate of heating

        Returns:
            list: Desorption energy and coverage
        """
        h = 4.135e-15 # eV.s

        theta = []
        for i in range(len(temperature)):
            cov = np.trapz(rate[i:], temperature[i:])
            theta.append(cov)
        
        theta = np.array(theta)
        rate = np.array(rate)
        dtheta_dT = np.diff(theta) / np.diff(temperature)
        dtheta_dt = beta * dtheta_dT #rate[0:-1]
        temperature = np.array(temperature)
        nu = kB * temperature[0:-1] / h
        Ed = -kB * temperature[0:-1] * np.log( -1 * dtheta_dt / (nu * theta[0:-1]))

        return [Ed, theta[0:-1]/max(theta[0:-1])]

    def _least_sq_Ed_theta(self, x, temperature, theta_rel, Ed_real):
        E_0, b, theta_sat = x
        Ed_fit = []
        for i in range(len(temperature)):
            Ed = E_0 \
                - kB * temperature[i] * np.log(theta_sat * theta_rel[i] / ( 1 - theta_sat * theta_rel[i] ) ) \
                - b * theta_rel[i] * theta_sat
            Ed_fit.append(Ed)
        residual = Ed_real - Ed_fit
        mea = np.mean([np.abs(a) for a in residual])
        # square_error = np.square([np.abs(a) for a in residual])
        # mean_sq_error = np.mean(square_error)
        # rms = np.sqrt(mean_sq_error) 
        return mea #square_error #mea#Ed_real - Ed_fit

    def _fit_Ed_theta(self, temperature, E_0, b, theta_sat, theta_rel):
        """Fit the desorption energy to the relative coverage
        Fed into scipy curve fit

        Args:
            temperature (list): temperature range
            E_0 (float): energy at zero coverage
            b (float): interaction parameter
            theta_sat (float): saturation coverage of TPD 
            theta_rel (float): relative coverage 

        Returns:
            list: Desorption energy based on fit
        """
        Ed_all = []
        for i in range(len(temperature)):
            Ed = E_0 \
            -  kB * temperature[i] * np.log(theta_sat*theta_rel[i] / ( 1 - theta_sat*theta_rel[i]))
            -  b * theta_rel[i] * theta_sat
            Ed_all.append(Ed)
        return Ed_all

    def _eq_coverage_function(self, theta, T, G0, b, p):
        """Function to implicitly solve the equilibrium coverage

        Args:
            theta (float): Guessed coverage 
            T (float) : temperature
            G0 (float): Free energy at the half a mono-layer coverage
            b (float): Interaction parameter
            p (float): partial pressure of CO
        """
        kBT = kB * T
        ## start by calculating the equilibirum constant 
        K = np.exp( -1 * ( G0 + b * ( theta - 1./2. ) ) / kBT ) 
        return theta - ( K / ( 1 + K ) )

    def _jacobian(self, theta, T, G0, b, p):
        """Jacobian function for finding the root 

        Args:
            theta (list): Guessed coverage
            T ([type]): [description]
            G0 (float): Free energy at the half a mono-layer coverage
            b (float): Interaction parameter
            p (float): partial pressure of CO
        """
        kBT = kB * T
        ## start by calculating the equilibirum constant 
        K = np.exp( -1 * ( G0 + b * ( theta - 1./2. ) ) / kBT ) 

        return 1 + K / (1+K)**2 * b / kBT 

    def _get_equilibirum_coverage(self, E0, b):
        """Equilibirum coverage based on equilibrium constant that is coverage dependent

        Args:
            E0 (float): Desorption energy at zero coverage
            b (float): Interaction parameter

        Returns:
            list: equilibrium coverage and free energy of CO adsorption
        """

        theta_eq = []
        dG_eq = []

        for index, T in enumerate(self.plot_temperature):
            entropy_ads = self.thermo_ads.get_entropy(temperature=T, verbose=False)
            entropy_gas = self.thermo_gas.get_entropy(temperature=T, \
                                                          pressure=self.pco, verbose=False)

            # converting from energies to free energies
            entropy_difference =  entropy_ads - entropy_gas
            partial_co = self.pco / 101325.
            # convert desorption energy into adsorption energy
            dG0 = -1 * E0 -1 * T * entropy_difference
            # K = np.exp( - dG / kB / T)
            # theta =  K * partial_co / (1 + K * partial_co)
            K_guess = np.exp( -1 * dG0 / kB / T )
            theta_guess = K_guess / ( 1 + K_guess ) 

            try:
                theta = newton(
                    func = lambda x: self._eq_coverage_function(x, T, dG0, b, partial_co ), 
                    fprime = lambda x: self._jacobian(x, T, dG0, b, partial_co ),
                    x0=theta_guess, 
                )
            except RuntimeError:
                theta = 0


            dG = ( -dG0 + b * ( theta - 1./2. )  )
            theta_eq.append(theta)
            dG_eq.append(dG)

        return theta_eq, dG_eq
