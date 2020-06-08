#!/usr/bin/python

""" Script to compare TPD for different Gold facets """

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
                            get_differential_energy

from parser_class import ParseInfo, experimentalTPD
import mpmath as mp
from ase.thermochemistry import HarmonicThermo, IdealGasThermo
from ase.io import read
from ase.db import connect
import matplotlib

### PLOT related preferences
matplotlib.rc('text', usetex=True)
matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"
plt.rc('axes', labelsize=32)    # fontsize of the x and y labels
plt.rcParams['xtick.labelsize'] = 24
plt.rcParams['ytick.labelsize'] = 24
from scipy import optimize
from matplotlib.ticker import FormatStrFormatter
import csv


# Gaussian for fitting purposes
def gaussian(x, a, x0, sigma):
    # called by gaussian_tpd for use with curve fit
    values = a * np.exp( - (x - x0)**2 / ( 2* sigma**2))
    return values

def log_scale(x, a, b):
    return a * np.log(x * b)

def linear_scale(x, a):
    return a * x


# Main function for the TPD plot
def main(tpd_filename, temprange, tempmin, beta):
    # Running the class experimentalTPD
    expTPD = experimentalTPD(tpd_filename, temprange, tempmin, beta)
    expTPD.collect_tpd_data()
    expTPD.get_normalized_data()
    expTPD.get_gaussian_tpd()
    expTPD.get_desorption_energy()

    # Fit the temperature Ed plot
    try:
        popt, pcov = curve_fit(log_scale,expTPD.temperature[0:-1], expTPD.Ed, p0=[0.1, 1])#np.polyfit(expTPD.temperature[0:-1], expTPD.Ed, 2)
        p_Ed = lambda x : log_scale(x, *popt)#np.poly1d(fit_Ed)
    except (RuntimeError, ValueError):
        popt, pcov = curve_fit(linear_scale,expTPD.temperature[0:-1], expTPD.Ed)#np.polyfit(expTPD.temperature[0:-1], expTPD.Ed, 2)
        p_Ed = lambda x : linear_scale(x, *popt)#np.poly1d(fit_Ed)

    temp_range = np.linspace(min(expTPD.temperature), max(expTPD.temperature), 100)
    fit_gauss = gaussian(temp_range,*expTPD.popt)

    return {'gaussian_tpd':expTPD.gaussian_rates,
            'temperature':expTPD.temperature,
            'norm_rate':expTPD.normalized_rate,
            'Ed':expTPD.Ed,
            'theta':expTPD.theta,
            'p_Ed':p_Ed,
            'fit_gauss':fit_gauss,
            'temp_range':temp_range,
            'popt':expTPD.popt}

# get the states for a chosen config
def get_states(stat, functional, cell, facet):
    # for a given database pick out all states
    allstates = []
    for row in stat.select(functional=functional, cell_size=cell, facets=facet):
        allstates.append(row.states)
    unique_states = np.unique(allstates)
    return unique_states

# get the atoms object
def atoms_from_db(db, **kwargs):
    # query the database to give EXACTLY one entry
    # return the atoms object
    for row in db.select(**kwargs):
        atoms = row.toatoms()
    return atoms

# Main function for DFT parsing
def main_DFT(thermodb, referdb, list_cells, facet, functional):
    parse = ParseInfo(thermodb, referdb, list_cells, facet, functional,ref_type='CO')
    parse.get_pourbaix()
    return parse.absolute_energy, parse.atoms

# Fit the energy of desorption to the configurational entropy equation
def fit_config_entropy(temperature, a, b, theta_sat, popt, popt_max):
    # print(temperature)
    total_temperature_range = np.linspace(175, 300, 500)
    temp_range = temperature

    rates = gaussian(temp_range, *popt)
    rates_all = gaussian(total_temperature_range, *popt)
    rates_max = gaussian(total_temperature_range, *popt_max)

    # integrate the rate to get the relative coverage
    # print(rates, temp_range)
    theta_all = np.trapz(rates_max, total_temperature_range)
    theta_rel = []
    for i in range(len(temp_range)):
        temp_i = min(range(len(total_temperature_range)), key=lambda j: abs(total_temperature_range[j]-temp_range[i]))
        # print(temp_range[i], total_temperature_range[temp_i])
        theta_val = np.trapz(rates_all[temp_i:], total_temperature_range[temp_i:])
        theta_rel.append(theta_val / theta_all)
    theta_rel = np.array(theta_rel)
    # b = np.piecewise(theta_rel, [theta_rel < 0.2, theta_rel >= 0.2], [0, b])
    # theta_sat = 0.3 # Fixed based on DFT calculation
    Ed = a \
    -  8.617e-5 * temperature * np.log(theta_sat*theta_rel / ( 1 - theta_sat*theta_rel)) \
    - b * theta_rel * theta_sat
    # print(Ed, a, theta)
    return Ed

# gets the differential configurational entropy
def get_config_entropy(temperature, theta):
    temperature = np.array(temperature)
    theta = np.array(theta)
    config = 8.617e-5 * temperature * np.log(theta / (1-theta))
    return config

output = 'output/'
os.system('mkdir -p ' + output)

if __name__ == '__main__':

    """ Constants """
    kB = 8.617e-05 # eV/K
    pco = 101325. #pressure of CO

    """ TPD data """
    # For 211 TPD
    T_switch_211 = [110, 165, 165, 175] #K converting between 111 and 211 step
    T_max_211 = 250 #K Where the TPD spectra ends
    T_min_211 = [250, 300] #K Where the rate becomes zero - baseline
    beta_211 = 3 #k/s Heating rate

    # For 310 TPD
    T_switch_310 = [150] # K converting between 100 and 110
    T_max_310 = 250 # K Where the TPD spectra ends
    T_min_310 = [225, 250] #K Where the rate becomes zero - baseline
    beta_310 = 5 #k/s Heating rate

    """ Frequencies for stable sites """
    # Here we choose the frequencies corresponding to the lowest energy
    # site for adsorption of CO from DFT
    vibration_energies = get_stable_site_vibrations()

    # Store just the ase vibrations object
    thermo_ads = {}
    for facet in vibration_energies:
        thermo_ads[facet] = HarmonicThermo(vibration_energies[facet])

    # Gas phase CO energies
    thermo_gas = get_gas_vibrations()

    """ Specific about coverage """
    # DFT specifics about coverages
    cell_sizes, coverages_cell, coverage_labels = get_coverage_details()


    # states that need to be plotted for each surface facet
    accept_states = {}
    accept_states['211'] = ['CO_site_8']
    accept_states['100'] = ['CO_site_1']
    accept_states['110'] = ['CO_site_4']
    accept_states['111'] = ['CO_site_0']
    accept_states['recon_110'] = ['CO_site_1']

    """ DFT databases """
    # which facets to consider
    facets = ['211', '111-0', '111-1', '100', '110',]
    # Get reference for CO in vacuum
    referencedb = connect('../databases/references_BEEF_VASP_500.db')
    # Get the gas phase energy of CO
    COgstat = referencedb.get(formula='CO', pw_cutoff=500.0)
    COg_E =  get_vibrational_energy(COgstat, [], method='novib', geometry='linear', \
            symmetrynumber=1)['E']
    # Get the energies of all thermodynamics
    thermodb = connect('../databases/Au_CO_coverage.db')
    functional = 'BF'


    """ Stylistic """
    colors_facet = {'211':'tab:blue', '111-0':'tab:green','111-1':'tab:green', '100':'tab:red',\
                    '110':'tab:brown', 'recon_110':'tab:cyan', '111':'tab:green'}
    ls_facet = {'211':'-', '111-0':'--', '111-1':'-.', '110':'-', '100':'--', '111':'--'}
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', \
        'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan',\
        'r', 'b', 'g'] # List of colours
    colors_state = {'CO_site_8':'tab:brown', 'CO_site_13':'tab:olive', \
                    'CO_site_10':'tab:red', 'CO_site_7':'tab:blue'}

    ## LIST OF FIGURES
    fig, ax = plt.subplots(4 , 2, figsize=(22, 22)) # figure for the paper
    figc, axc = plt.subplots(5, 1,figsize=(6,16)) # Energy as a function of temp
    figs, axs = plt.subplots(1, 2,figsize=(16,6)) # saturation coverage figure
    figG, axG = plt.subplots(1, 1, figsize=(8,6)) # Plot of the free energy of adsorption
    figO, axO = plt.subplots(1, 2, figsize=(15, 5))
    inferno = cm.get_cmap('inferno', 12)
    viridis = cm.get_cmap('viridis', 5)
    font_number = 32 # size of a b c ...
    ############################################################################
    # Take the TPD data from experiment
    input_csv_211 = glob('input_TPD/Au_211/*.csv')
    input_csv_310 = glob('input_TPD/Au_310/*.csv')

    images_211 = plt.imread('surface_sites/gold_211.png') #Image('surface_sites/gold_211.svg')
    images_310 = plt.imread('surface_sites/gold_310.png') #Image('surface_sites/gold_310.svg')
    height = images_211.size
    results = AutoVivification()

    for f in input_csv_211:
        # For each exposure run the experimental TPD for 211
        exposure = float(f.split('/')[-1].split('.')[0].split('_')[1].replace('p', '.'))
        results['211'][exposure] = main(f, [T_switch_211[-2], T_max_211], T_min_211, beta_211)
        # results['intermediate'][exposure] = main(f, [T_switch_211[1], T_switch_211[2]], T_min_211, beta_211)
        results['111-0'][exposure] = main(f, [0, T_switch_211[0]], T_min_211, beta_211)
        results['111-1'][exposure] = main(f, [T_switch_211[0], T_switch_211[1]], T_min_211, beta_211)

    for f in input_csv_310:
        # For each exposure run the experimental TPD for 310
        exposure = round(float(f.split('/')[-1].split('.')[0].split('_')[1].replace('p', '.')) * 12,2)
        results['110'][exposure] = main(f, [T_switch_310, T_max_310], T_min_310, beta_310)
        results['100'][exposure] = main(f, [0, T_switch_310], T_min_310, beta_310)

    # First figure - normalized TPD and the Gaussian fit
    # Plot the figure which shows the experimental and gaussian fit TPD curves
    ############################################################################
    # Figure 1 a b : The gaussian fit of the TPD curve
    for facet, results_facet in results.items():
        for index, exposure in enumerate(sorted(results_facet)):
            if facet in ['211', '111-0', '111-1']:
                ax[0,0].plot(results_facet[exposure]['temperature'], results_facet[exposure]['norm_rate'],
                        '.', color=inferno(index), alpha=0.40)
                # ax[0,0].plot(results_facet[exposure]['temp_range'], results_facet[exposure]['fit_gauss'],
                        # '-',  color=inferno(index))
            elif facet in ['100', '110']:
                ax[0,1].plot(results_facet[exposure]['temperature'], results_facet[exposure]['norm_rate'],
                        '.', color=viridis(index), alpha=0.40)
                # ax[0,1].plot(results_facet[exposure]['temp_range'], results_facet[exposure]['fit_gauss'],
                        # '-',  color=viridis(index))

    # Annotations
    ax[0,0].annotate(r'$\mathrm{Au(211)_{terrace}}$', xy=(125, 30), xycoords='data',xytext=(225,30), \
    arrowprops=dict(shrink=0.05, color=colors_facet['111-0']),
    horizontalalignment='right', verticalalignment='center', color=colors_facet['111-0'], fontsize=22)
    ax[0,1].annotate(r'$\mathrm{Au(310)_{terrace}}$', xy=(125, 55), xycoords='data',xytext=(200,55), \
    arrowprops=dict(shrink=0.05, color=colors_facet['100']),
    horizontalalignment='right', verticalalignment='center', color=colors_facet['100'], fontsize=22)
    ax[0,0].annotate(r'$\mathrm{Au(211)_{step}}$', xy=(190, 12), xycoords='data',xytext=(190,20), \
    arrowprops=dict(shrink=0.05, color=colors_facet['211']),
    horizontalalignment='center', verticalalignment='center', color=colors_facet['211'], fontsize=22)
    ax[0,1].annotate(r'$\mathrm{Au(310)_{step}}$', xy=(190, 20), xycoords='data',xytext=(250,20), \
    arrowprops=dict(shrink=0.05, color=colors_facet['110']),
    horizontalalignment='right', verticalalignment='center', color=colors_facet['110'], fontsize=22)
    ax[0,0].set_ylabel(r'rate / arb. unit')
    ax[0,0].set_xlabel(r'Temperature / K')
    ax[0,0].annotate('a)', xy=(-0.2, 1.1),xycoords="axes fraction", fontsize=font_number)
    ax[0,1].set_ylabel(r'rate / arb. unit')
    ax[0,1].set_xlabel(r'Temperature / K')
    ax[0,1].annotate('b)', xy=(-0.2, 1.1),xycoords="axes fraction", fontsize=font_number)
    newax = fig.add_axes([0.37, 0.8, 0.12, 0.12], anchor='NE', zorder=-1)
    newax.imshow(images_211)
    newax.axis('off')
    newax2 = fig.add_axes([0.87, 0.8, 0.12, 0.12], anchor='NE', zorder=-1)
    newax2.imshow(images_310)
    newax2.axis('off')
    ############################################################################
    # Figure 1 c d: Plot the desorption energy as a function of relative coverage
    ax[1,0].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax[1,0].set_xlabel(r'$\theta_{CO}^{rel}$ ')
    ax[1,0].set_ylabel(r'$E_{d}$ / eV')
    ax[1,1].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax[1,1].set_xlabel(r'$\theta_{CO}^{rel}$ ')
    ax[1,1].set_ylabel(r'$E_{d}$ / eV')

    for index, exposure in enumerate(sorted(results['211'])):
        ax[1,0].plot(results['211'][exposure]['theta'], results['211'][exposure]['Ed'] , \
            '.', alpha=1, color=inferno(index), label=str(exposure) + 'L')#, color='tab:blue')

    ax[1,0].legend(bbox_to_anchor=(1.04,0.5), loc="center left", borderaxespad=0)
    ax[1,0].annotate('c)', xy=(-0.2, 1.1),xycoords="axes fraction", fontsize=font_number)
    ax[1,0].annotate(r'$\mathrm{Au(211)_{step}}$', xy=(0.5, 0.55), color=colors_facet['211'], fontsize=24, weight='bold')

    for index, exposure in enumerate(sorted(results['110'])):
        ax[1,1].plot(results['110'][exposure]['theta'],  results['110'][exposure]['Ed'], \
            '.', alpha=1, color=viridis(index), label=str(exposure) + 'L')#, color='tab:blue')
    ax[1,1].annotate(r'$\mathrm{Au(310)_{step}}$', xy=(0.5, 0.55), color=colors_facet['110'], fontsize=24, weight='bold')

    ax[1,1].legend(bbox_to_anchor=(1.04,0.5), loc="center left", borderaxespad=0)
    ax[1,1].annotate('d)', xy=(-0.2, 1.1),xycoords="axes fraction", fontsize=font_number)

    ax[1,0].set_ylim([0.40, 0.67])
    ax[1,1].set_ylim([0.40, 0.67])

    # Plot the 111 and the 110 surfaces
    for ind_facet, facet in enumerate(['111-0', '111-1']):
        for index, exposure in enumerate(sorted(results[facet])):
            if ind_facet == 0:
                axO[0].plot(results[facet][exposure]['theta'], results[facet][exposure]['Ed'] , \
                '.', alpha=1, color=inferno(index), label=str(exposure) + 'L')
            else:
                axO[0].plot(results[facet][exposure]['theta'], results[facet][exposure]['Ed'] , \
                'v', alpha=1, color=inferno(index))
    axO[0].set_title('(111)')
    for facet in ['100']:
        for index, exposure in enumerate(sorted(results[facet])):
            axO[1].plot(results[facet][exposure]['theta'], results[facet][exposure]['Ed'] , \
                '.', alpha=1, color=viridis(index), label=str(exposure) + 'L')
    axO[1].set_title('(100)')
    axO[0].legend(bbox_to_anchor=(1.04,0.5), loc="center left", borderaxespad=0)
    axO[0].annotate('a)', xy=(-0.2, 1.1),xycoords="axes fraction", fontsize=font_number)

    axO[1].legend(bbox_to_anchor=(1.04,0.5), loc="center left", borderaxespad=0)
    axO[1].annotate('b)', xy=(-0.2, 1.1),xycoords="axes fraction", fontsize=font_number)

    axO[0].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    axO[0].set_xlabel(r'$\theta_{CO}^{rel}$ ')
    axO[0].set_ylabel(r'$E_{d}$ / eV')
    axO[1].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    axO[1].set_xlabel(r'$\theta_{CO}^{rel}$ ')
    axO[1].set_ylabel(r'$E_{d}$ / eV')

    ############################################################################
    # Figure 1 e: fit energy at zero coverage that is the dilute limit
    ax[2,0].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    fit_theta_lim = [1] # the choice of where the fit end
    f1 = open('211.csv','w')
    # write out the csv such that the data is stored
    writer=csv.writer(f1, delimiter='\t',lineterminator='\n',)

    # FOR 211
    # If more than one fit limit is required
    for fit_theta in fit_theta_lim:
        for index, exposure in enumerate(sorted(results['211'])):
            fit_Ed = [results['211'][exposure]['Ed'][i] \
                            for i in range(len(results['211'][exposure]['Ed'])) if \
                             0.001 < results['211'][exposure]['theta'][i] < fit_theta ]
            fit_T = [results['211'][exposure]['temperature'][i] \
                            for i in range(len(results['211'][exposure]['Ed'])) if \
                            0.001 < results['211'][exposure]['theta'][i] < fit_theta ]
            fit_theta_range = [results['211'][exposure]['theta'][i] \
                            for i in range(len(results['211'][exposure]['Ed'])) if \
                            0.001 < results['211'][exposure]['theta'][i] < fit_theta ]
            # Find the coverage through a fit
            rates_popt = results['211'][exposure]['popt']
            rates_popt_max = results['211'][50.0]['popt']
            popt, pcov = curve_fit(\
                                    lambda temp, a, b, theta_sat: fit_config_entropy(temp, a, b, theta_sat, rates_popt, rates_popt), \
                                    fit_T, fit_Ed, [0.2, 0.01, 0.1])
            # Plot the Desorption energy as a fit to clarify fitting

            axs[0].plot(fit_theta_range, fit_Ed, 'o', color=inferno(index), label=str(exposure)+'L')
            axs[0].plot(fit_theta_range,\
            fit_config_entropy(np.array(fit_T) , *popt, rates_popt, rates_popt), '-', color=inferno(index))
            axs[0].set_ylabel(r'$E_{d}$ / eV')
            axs[0].set_xlabel(r'$\theta_{rel} $')

            # Write out data
            e0, cov_dep, theta_max = popt
            error = np.sqrt(np.diag(pcov))
            row = [exposure, *popt, *error]
            writer.writerow(row)

            # Store the results
            results['211'][exposure]['e0'] = e0
            results['211'][exposure]['e0_error'] = error[0]
            results['211'][exposure]['cov_dep'] = cov_dep
            results['211'][exposure]['cov_dep_error'] = error[1]
            results['211'][exposure]['theta_max'] = theta_max

            ax[2,0].plot(float(exposure), e0, 'o', color='tab:blue')
            ax[2,0].errorbar(float(exposure),e0, error[0],  color='tab:blue' )

    f1.close()
    axs[0].legend(bbox_to_anchor=(1.04,0.5), loc="center left", borderaxespad=0)
    ax[2,0].set_ylabel(r'$\Delta E_{0}$ / eV')
    ax[2,0].set_ylim([0.45, 0.55])
    ax[2,0].annotate('e)', xy=(-0.2, 1.1),xycoords="axes fraction", fontsize=font_number)
    ax[2,0].set_xlabel(r'Initial exposure / L')

    # For 110
    fit_theta_lim = [0.8]
    # write out data
    f2 = open('110.csv','w')
    writer=csv.writer(f2, delimiter='\t',lineterminator='\n',)
    for fit_theta in fit_theta_lim:
        for index, exposure in enumerate(sorted(results['110'])):

            fit_Ed = [results['110'][exposure]['Ed'][i] \
                            for i in range(len(results['110'][exposure]['Ed'])) if \
                             0.001 < results['110'][exposure]['theta'][i] < fit_theta ]
            fit_T = [results['110'][exposure]['temperature'][i] \
                            for i in range(len(results['110'][exposure]['Ed'])) if \
                            0.001 < results['110'][exposure]['theta'][i] < fit_theta ]
            fit_theta_range = [results['110'][exposure]['theta'][i] \
                            for i in range(len(results['110'][exposure]['Ed'])) if \
                            0.001 < results['110'][exposure]['theta'][i] < fit_theta ]
            # Find the coverage through a fit
            rates_popt = results['110'][exposure]['popt']
            popt, pcov = curve_fit(\
                                    lambda temp, a, b, theta_sat: fit_config_entropy(temp, a, b, theta_sat, rates_popt, rates_popt), \
                                    fit_T, fit_Ed, [0.2, 0.1, 0.1])

            axs[1].plot(fit_theta_range, fit_Ed, 'o', color=viridis(index), label=str(exposure)+'L')
            axs[1].plot(fit_theta_range,\
            fit_config_entropy(np.array(fit_T) , *popt, rates_popt, rates_popt), '-', color=inferno(index))
            axs[1].set_ylabel(r'$E_{d}$ / eV')
            axs[1].set_xlabel(r'$\theta_{rel} $')

            # write out data
            e0, cov_dep, theta_max= popt
            error = np.sqrt(np.diag(pcov))
            row = [exposure, *popt, *error]
            writer.writerow(row)

            # store data
            results['110'][exposure]['e0'] = e0
            results['110'][exposure]['cov_dep'] = cov_dep
            results['110'][exposure]['theta_max'] = theta_max
            results['110'][exposure]['e0_error'] = error[0]
            results['110'][exposure]['cov_dep_error'] = error[1]


            ax[2,0].plot(float(exposure), e0, 'o', color='tab:red')
            ax[2,0].errorbar(float(exposure),e0, error[0], color='tab:red')

    f2.close()
    axs[1].legend(bbox_to_anchor=(1.04,0.5), loc="center left", borderaxespad=0)
    ax[2,0].plot([], [], 'o', color='tab:blue', label=r'$\mathrm{Au(211)_{step}}$')
    ax[2,0].plot([], [], 'o',  color='tab:red', label=r'$\mathrm{Au(310)_{step}}$')
    ax[2,0].legend(loc='best')

    ############################################################################
    # Figure 1f: Plot equilbirum coverage as a function of the temperature

    temp_range = np.linspace(175, 600)
    # count = 0
    # for facet, results_facet in results.items():
    #     for ind, exposure in enumerate(results_facet):
    #         theta_sat = []
    #         temperature_Exp = np.array(results_facet[exposure]['temperature'][0:-1])
    #         deltaE = results_facet[exposure]['p_Ed'](temperature_Exp)
    #         axc[count].plot(temperature_Exp, deltaE, color=inferno(ind), ls=ls_facet[facet], lw=3)
    #         axc[count].plot(temperature_Exp, results_facet[exposure]['Ed'], '.', \
    #                         mew=3, fillstyle='none',color=inferno(ind))
    #         axc[count].set_xlabel(r'Temperature \ K')
    #         axc[count].set_ylabel(r'$E_{d}$')
    #         axc[count].set_title(facet)
    #     count += 1
    # figc.tight_layout()
    # figc.savefig(output + 'desorption_energy_temp.pdf')

    def equilibrium_coverage(theta_eq, pCO, T, entropy_diff, cov_dep, e0):
        kb = 8.617e-5
        # entropy diff is gas minus ads
        # do everything in the form given in the paper
        # so all adsorption energies
        # Free energy at half a monolayer coverage will contain only the
        # energy at dilute coverage the interaction term
        # and the entropic difference from ads to gas
        deltaG = -1 * ( e0 - cov_dep * theta_eq ) -T * ( -entropy_diff )
        # Equilibrium constant for adsorption of CO
        K = np.exp( - deltaG / kb / T)
        # residual
        res = theta_eq - K * pCO / (1 + K * pCO)
        return res

    def deltaG_plot(theta_eq, pCO, T, entropy_diff):
        deltaG = -1 * ( e0 - cov_dep * theta_eq ) -T * ( -entropy_diff )
        return deltaG

    # Plot temperature dependent rate and saturation coverages
    ax[2,1].axvline(x=298.15, color='k', ls='--', lw=3, alpha=0.5)
    ax[2,1].axvline(x=298.15, color='k', ls='--', lw=3, alpha=0.5)

    for facet, results_facet in results.items():
        if facet in [ '211', '110' ]:
            for ind, exposure in enumerate(results_facet):
                theta_sat = []
                theta_err_plus = []
                theta_err_min = []
                for index, T in enumerate(temp_range):
                    entropy_ads = thermo_ads[facet.split('-')[0]].get_entropy(temperature=T, verbose=False)
                    entropy_gas = thermo_gas.get_entropy(temperature=T, \
                                                                  pressure=pco, verbose=False)
                    # converting from energies to free energies
                    entropy_difference =  (entropy_gas - entropy_ads)
                    partial_co = pco / 101325
                    cov_dep = results_facet[exposure]['cov_dep']
                    e0 = results_facet[exposure]['e0']
                    # Solve for the equilibrium coverage and then find error bars
                    theta_eq = optimize.root(\
                        lambda theta : equilibrium_coverage(theta, partial_co, T, entropy_difference, cov_dep, e0 ),
                        x0=[0.5])
                    theta_sat.append(theta_eq.x[0])
                    theta_eq_pos = optimize.root(\
                        lambda theta : equilibrium_coverage(theta, partial_co, T, \
                            entropy_difference, cov_dep+results_facet[exposure]['cov_dep_error'],\
                            e0+results_facet[exposure]['e0_error'] ),
                                x0=[0.5])
                    theta_err_plus.append(theta_eq_pos.x[0])
                    theta_eq_neg = optimize.root(\
                        lambda theta : equilibrium_coverage(theta, partial_co, T, \
                            entropy_difference, cov_dep-results_facet[exposure]['cov_dep_error'],\
                            e0-results_facet[exposure]['e0_error'] ),
                                x0=[0.5])
                    theta_err_min.append(theta_eq_pos.x[0])

                    # Plot free energy with temperature
                    axG.plot(T, deltaG_plot(theta_eq.x[0], partial_co, T, entropy_difference),  'o', color=inferno(ind), ls=ls_facet[facet], lw=3)

                # Plot the equilibrium coverage curves
                if facet in ['111-0', '111-1', '211']:
                    ax[2,1].plot(temp_range, theta_sat, color=inferno(ind), ls=ls_facet[facet], lw=3)
                    ax[2,1].fill_between(temp_range, theta_err_min, theta_err_plus, color=inferno(ind), alpha=0.5)
                elif facet in ['100', '110']:
                    ax[2,1].plot(temp_range, theta_sat, color=viridis(ind), ls=ls_facet[facet], lw=3)
                    ax[2,1].fill_between(temp_range, theta_err_min, theta_err_plus,color=viridis(ind), alpha=0.5 )

    ax[2,1].set_ylabel(r'Equilibirum $\theta_{CO}$ / ML')
    ax[2,1].set_xlabel('Temperature / K')
    ax[2,1].annotate('f)', xy=(-0.2, 1.1),xycoords="axes fraction", fontsize=font_number)
    ax[2,1].set_ylabel(r'Equilibirum $\theta_{CO}$ / ML')
    ax[2,1].set_xlabel('Temperature / K')

    # annotate the equilibirum coverage plot
    ax[2,1].annotate(r'$\mathrm{Au(211)_{step}}$', xy=(350, 0.5), xycoords='data',xytext=(500,0.5), \
    arrowprops=dict(shrink=0.05, color=colors_facet['211']),
    horizontalalignment='right', verticalalignment='center', color=colors_facet['211'], fontsize=22)
    ax[2,1].annotate(r'$\mathrm{Au(310)_{step}}$', xy=(350, 0.25), xycoords='data',xytext=(275,0.25), \
    arrowprops=dict(shrink=0.05, color=colors_facet['110']),
    horizontalalignment='right', verticalalignment='center', color=colors_facet['110'], fontsize=22)
    ############################################################################
    # Plot differential energies with DFT
    absE_DFT = AutoVivification()
    atoms_DFT = AutoVivification()
    relE_DFT = AutoVivification()

    # BEEF Error
    beef_error = AutoVivification()
    beef_error['211'] = [0.1225,0.14629, 0.11908, 0.13701, ]
    beef_error['100'] = [0.16693, 0.14192, 0.18838, ]
    beef_error['111'] = [0.14959, 0.13724, 0.17666, ]
    beef_error['110'] = [0.2806, 0.15287, 0.17385, ]

    i = 0
    for facet in facets:
        if '-' in facet:
            facet = facet.split('-')[0]
            i+= 1
        absE_DFT[facet], atoms_DFT[facet] = main_DFT(thermodb, referencedb, cell_sizes[facet], facet, functional)
        avg_energy = get_differential_energy(absE_DFT[facet], atoms_DFT[facet], facet, COg_E)

        # Useful for checking stable sites comment out when needed
        # selected_states = [] # states which contain the lowest energy state
        # i=0
        # selected_states = []
        # while i < len(avg_energy['CO_site_0'])-1:
        #     lowest_energy = 6.
        #     for state in avg_energy:
        #         if avg_energy[state][i] < lowest_energy:
        #             lowest_energy = avg_energy[state][i]
        #             state_current_lowest = state
        #     selected_states.append(state_current_lowest)
        #     i+=1
        # print(selected_states)

        cells = cell_sizes[facet]
        coverages = list(reversed([coverages_cell[facet][cell] for cell in cells]))
        # config = get_config_entropy(298, coverages)
        # print(config)
        #
        for state in avg_energy:
            free_ads = thermo_ads[facet].get_helmholtz_energy(temperature=298, verbose=False)
            free_gas = thermo_gas.get_gibbs_energy(temperature=298, \
                                                          pressure=101325, verbose=False)
            entropy_ads = thermo_ads[facet].get_entropy(temperature=298, verbose=False)
            entropy_gas = thermo_gas.get_entropy(temperature=298, \
                                                          pressure=101325, verbose=False)
            #
            free_diff = -1 * 298 * (entropy_ads - entropy_gas)
            delta_zpe =  thermo_ads[facet].get_ZPE_correction() - thermo_gas.get_ZPE_correction()
            if state in accept_states[facet]:
                if facet in ['111']:
                    ax[3,0].plot(coverages, avg_energy[state] + delta_zpe , 'o-', color=colors_facet[facet], label='Au('+facet+')' if i == 1 else '')
                    ax[3,0].errorbar(coverages, avg_energy[state] + delta_zpe, beef_error[facet],  color=colors_facet[facet])
                elif facet in '211':
                    ax[3,0].plot(coverages, avg_energy[state] + delta_zpe, 'o-', label='Au('+facet+')', color=colors_facet[facet])
                    ax[3,0].errorbar(coverages, avg_energy[state] + delta_zpe, beef_error[facet], color=colors_facet[facet])
                    ax[3,0].axhline(y=-1*free_diff, color='k', alpha=0.5, ls='--')
                    # to compare with TPD result first remove the zpe
                    # delta_zpe = thermo_gas.get_ZPE_correction() - thermo_ads[facet].get_ZPE_correction()
                    tpd_e0 = -1 * ( results['211'][50.]['e0'])
                    # tpd_g0 = -1 * ( results['211'][50.]['e0'] - 298 * (entropy_gas - entropy_ads))
                    # tpd_g0 = -1 * ( results['211'][50.]['e0'] + entropy_gas - entropy_ads )
                    ax[3,0].axhline(y=tpd_e0, color=colors_facet[facet], ls='-', label=r'$\Delta E_{0} (211)_{step}$')
                elif facet in ['110', 'recon_110', '100']:
                    ax[3,1].plot(coverages, avg_energy[state] + delta_zpe, 'o-', \
                        label='Au('+facet.replace('_', '-')+')', \
                        color=colors_facet[facet])
                    ax[3,1].errorbar(coverages, avg_energy[state] + delta_zpe, beef_error[facet],  color=colors_facet[facet])
                    if facet == '110':
                        # delta_zpe = thermo_gas.get_ZPE_correction() - thermo_ads[facet].get_ZPE_correction()
                        tpd_e0 = -1 * ( results['110'][12]['e0'] )
                        # tpd_g0 = -1 * ( results['110'][12]['e0'] - 298 * (entropy_gas - entropy_ads))
                        ax[3,1].axhline(y=tpd_e0, color=colors_facet[facet], ls='-', label=r'$\Delta E_{0} (310)_{step}$')
                        ax[3,1].axhline(y=-free_diff, color='k', alpha=0.5, ls='--')


    # ax[3,0].axhline(y=0, color='k', alpha=0.5, ls='--')
    # ax[3,1].axhline(y=0, color='k', alpha=0.5, ls='--')
    #
    ax[3,0].set_xlabel(r'$\theta_{CO}$ / ML')
    ax[3,0].set_ylabel(r'$\Delta E_{CO}^{diff}$ / eV')
    ax[3,0].annotate('g)', xy=(-0.2, 1.1),xycoords="axes fraction", fontsize=font_number)
    ax[3,0].legend(bbox_to_anchor=(1.04,0.5), loc="center left", borderaxespad=0)
    ax[3,1].set_xlabel(r'$\theta_{CO}$ / ML')
    ax[3,1].set_ylabel(r'$\Delta E_{CO}^{diff}$ / eV')
    ax[3,1].annotate('h)', xy=(-0.2, 1.1),xycoords="axes fraction", fontsize=font_number)
    ax[3,1].legend(bbox_to_anchor=(1.04,0.5), loc="center left", borderaxespad=0)


    ############################################################################

    fig.tight_layout()
    figs.tight_layout()
    figG.tight_layout()
    figO.tight_layout()
    fig.savefig(output + 'TPD_compare.svg')
    fig.savefig(output + 'TPD_compare.pdf')
    figs.savefig(output + 'saturation.pdf')
    figG.savefig(output + 'free_energy.pdf')
    figO.savefig(output + 'other_facets.pdf')

    # plt.show()
