#!/usr/bin/python

import numpy as np
import os
import sys
### PLOT related preferences
import matplotlib
matplotlib.rc('text', usetex=True)
matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"
plt.rc('axes', labelsize=32)    # fontsize of the x and y labels
plt.rcParams['xtick.labelsize'] = 24
plt.rcParams['ytick.labelsize'] = 24
plt.rcParams['xtick.major.size'] = 8
plt.rcParams['ytick.major.size'] = 8
from scipy import optimize, integrate
from matplotlib.ticker import FormatStrFormatter
import csv
from pprint import pprint

def gaussian(x, a, x0, sigma):
    # called by gaussian_tpd for use with curve fit
    values = a * np.exp( - (x - x0)**2 / ( 2* sigma**2))
    return values

def integrated_peak(wavenumber_all, spectra_all):
    global counter_spectra
    wavenumber_all = np.array(wavenumber_all)
    spectra_all = np.array(spectra_all)
    args = [i for i in range(len(wavenumber_all)) if 2000 < wavenumber_all[i] < 2200 ]
    spectra = spectra_all[args]
    wavenumber = wavenumber_all[args]
    y_bg_a = np.average(spectra[0:50])
    y_bg_b = np.average(spectra[-50::])
    x_bg_a = np.average(wavenumber[0:50])
    x_bg_b = np.average(wavenumber[-50::])
    # print(len(x_bg_b))
    y_bg = y_bg_a + (y_bg_b - y_bg_a)/(x_bg_b - x_bg_a) * (wavenumber - wavenumber[0])
    true_spectra = spectra - y_bg


    # mean = np.average(true_spectra)
    std_dev = np.std(true_spectra)
    try:
        popt, pcov = optimize.curve_fit(gaussian, wavenumber[50:-50], true_spectra[50:-50], p0=[0.002, 2100, 20])
    except RuntimeError:
        popt, pcov = optimize.curve_fit(gaussian, wavenumber[50:-50], true_spectra[50:-50], p0=[0.002, 2100, std_dev])
        popt = [0, 0, 1]
    fit_spectra = gaussian(wavenumber, *popt)
    fig_spectra, ax_spectra = plt.subplots(1,1)

    ax_spectra.plot(wavenumber[50:-50], true_spectra[50:-50], 'o')
    ax_spectra.plot(wavenumber, fit_spectra)
    fig_spectra.savefig(output + 'spectra'+str(counter_spectra)+'.pdf')
    plt.close(fig=fig_spectra)
    # area = np.trapz(gaussian(wavenumber[50:-50], *popt), wavenumber[50:-50])
    # print(gaussian(wavenumber[50:-50], *popt))
    # print(area)
    area = integrate.quad(lambda x : gaussian(x, *popt), wavenumber[50], wavenumber[-50])
    # print(popt)
    # print(area)
    # sys.exit()
    return -1 * area[0]

output = 'output/'
os.system('mkdir -p ' + output)

if __name__ == '__main__':

    # Plots out the experimental graph in the format of
    fig, ax = plt.subplots(2,3, figsize=(16.1,10))
    counter_spectra = 0

    gs = ax[1, 1].get_gridspec()
    # fig_
    scan_rate = 2e-3 #V/s
    for axs in ax[:,0]:
        axs.remove()
    for axs in ax[:,1]:
        axs.remove()
    ax[0,0] = fig.add_subplot(gs[:,0])
    ax[0,1] = fig.add_subplot(gs[:,1])
    #
    # gs = ax[1].get_gridspec()
    # axi = fig.add_subplot(gs[1])

    schemes = ['clean', 'with_Pb']
    colors_schemes = {'clean':'tab:blue', 'with_Pb':'tab:green'}
    # DEBUG:
    debug_number = 8
    data_ranges = {'clean':np.arange(270, 470, 8),
                   'with_Pb':np.arange(163, 380, 8)}
    integration_ranges = {'clean':np.arange(270, 470, debug_number),
                   'with_Pb':np.arange(163, 380, debug_number)}
    # data_ranges = {'clean':np.arange(270, 272),
    #                'with_Pb':np.arange(163, 165)}
    factor_mult = {'clean':0.00725, 'with_Pb':0.0065}
    potential_limits = {'clean':[0.634, -0.166],
                        'with_Pb':[0.634, -0.226]}

    basedir = {'with_Pb':'input_exp_data/with_Pb/20190628_Au_eless_on_Si_HS_L005_0p1_M_HClO4_intro_of_Pb_and_upd_cont_1_14_15_14.',
               'clean':'input_exp_data/clean/20190628_Au_eless_on_Si_HS_L001_0p1_M_HClO4_intro_of_CO_13_10_25.'}
    # cvbasedir = 'input_exp_data/'
    cvbasedir = {'clean':{'forward':'input_exp_data/20180628CV1Cat.csv', \
                          'reverse':'input_exp_data/20180628CV1Cat.csv'},
                 'with_Pb':{'forward':'input_exp_data/20180628CV2Cat.csv',
                            'reverse':'input_exp_data/20180628CV2An.csv'}}
    cvfile = 'input_exp_data/20180628EC.csv'
    font_number = 32
    ########################
    # Plots a, b
    for index, scheme in enumerate(schemes):
        arrow_points = []

        for index_data, dat in enumerate(data_ranges[scheme]):
            data = np.loadtxt(basedir[scheme]+str(dat), delimiter=',')
            data = data.transpose()

            shift_up = index_data * factor_mult[scheme]
            data[1] = data[1] + shift_up
            ax[0,index].plot(data[0], data[1], color='k', alpha=0.4)
            if index_data == 13:
                y_val = data[1][np.abs(data[0] - 2000).argmin()]
                t = ax[0,index].text(2050, y_val, potential_limits[scheme][1])
                t.set_bbox(dict(facecolor='white', alpha=1, edgecolor='white'))
                arrow_points.append(y_val)

            elif index_data in [0, len(data_ranges[scheme])-1]:
                y_val = data[1][np.abs(data[0] - 2000).argmin()]
                t = ax[0,index].text(2050, y_val, potential_limits[scheme][0])
                t.set_bbox(dict(facecolor='white', alpha=1, edgecolor='white'))
                arrow_points.append(y_val)



        # ax[0,index].arrow(2050, arrow_points[0], 0, arrow_points[1] - arrow_points[0],\
        #         color='k', lw=1)
        # ax[0,index].arrow(2050, arrow_points[1], 0, arrow_points[2] - arrow_points[1],\
        #         color='k',ls=':', lw=1)
        ax[0,index].annotate('', xy=(2050, arrow_points[1]), \
                    xycoords="data", xytext=(2050, arrow_points[0]), \
        arrowprops=dict(shrink=0.05, color='k'))
        ax[0,index].annotate('', xy=(2050, arrow_points[2]), \
                    xycoords="data", xytext=(2050, arrow_points[1]), \
        arrowprops=dict(color='k',  shrink=0.05))

        ax[0,index].arrow(2180, arrow_points[0], 0, 0.01,\
                color='k', lw=1)
        ax[0,index].text(2170, arrow_points[0]-0.005, '0.01a.u.')
        # ax[0,2].plot(potential_forward, area_forward, color=colors_schemes[scheme])
        # ax[0,2].plot(potential_reverse, area_reverse, '--', color=colors_schemes[scheme])
    for index, scheme in enumerate(schemes):
        ax[0,index].set_xlim([2200, 2000])
        ax[0,index].set_ylim([-0.01, 0.2])
        if index == 0:
            ax[0,index].set_ylabel('Absorbance / arb. unit.')
        ax[0,index].yaxis.set_ticks([])
        ax[0,index].set_xlabel(r'Wavenumber / $\mathrm{cm^{-1}}$')

    ax[0,0].set_title(r'$\mathbf{CO \ w/o \ Pb}$', color=colors_schemes['clean'])
    ax[0,1].set_title(r'$\mathbf{CO \ w/ \ Pb}$', color=colors_schemes['with_Pb'])
    ax[0,0].annotate('a)', xy=(-0.1, 1.05),xycoords="axes fraction", fontsize=font_number)
    ax[0,1].annotate('b)', xy=(-0.1, 1.05),xycoords="axes fraction", fontsize=font_number)
    # ax[0,0].annotate(r'$\mathbf{+Pb \ UPD}$', xy=(1.1, 0.5), \
    #             xycoords="axes fraction", fontsize=24, color='tab:purple',)
    # ax[0,0].annotate('', xy=(1.1, 0.5), \
    #             xycoords="axes fraction", xytext=(190,20), \
    # arrowprops=dict(shrink=0.05, color='tab:purple'))
    ########################
    # Plot c
    for index, scheme in enumerate(schemes):
        for direction in cvbasedir[scheme]:
            data = np.loadtxt(cvbasedir[scheme][direction], delimiter=',', skiprows=1)
            data = data.transpose()
            potential, j = data
            if direction == 'reverse':
                ax[0,2].plot(potential, j, color=colors_schemes[scheme], ls='--')
            else:
                ax[0,2].plot(potential, j, color=colors_schemes[scheme])
    ax[0,2].annotate('CO (w/ Pb)', color=colors_schemes['with_Pb'], xy=(0.20, 0.05))
    ax[0,2].annotate('CO (w/o Pb)', color=colors_schemes['clean'], xy=(-0.23, 0.125))
    ax[0,2].set_xlabel(r'Potential / V vs. SHE')
    ax[0,2].set_ylabel(r'Peak Area / $\mathrm{a.u. cm^{-1}}$')
    ax[0,2].annotate('c)', xy=(-0.2, 1.1),xycoords="axes fraction", fontsize=font_number)
    ax[0,2].annotate('', xy=(0.25, 0.075), \
                xycoords="data", xytext=(0.2, 0.15), \
    arrowprops=dict(color=colors_schemes['with_Pb'],  shrink=0.05))
    ax[0,2].annotate('', xy=(0.1, 0.075), \
                xycoords="data", xytext=(0.15, 0.15), \
    arrowprops=dict(color=colors_schemes['with_Pb'],  shrink=0.05))

    # ax[1,2].set_ylim([-0.01, 0.01])

    ########################
    # Plot d
    data = np.loadtxt(cvfile, delimiter=',', skiprows=1)
    data = data.transpose()
    time, v, j = data
    args_clean = [ i for i in range(len(time)) if 3050 < time[i] < 3850]
    args_Pb = [ i for i in range(len(time)) if 6220 < time[i] < 7380]

    ax[1,2].plot(v[args_clean], j[args_clean], color=colors_schemes['clean'])
    ax[1,2].plot(v[args_Pb], j[args_Pb], color=colors_schemes['with_Pb'])
    ax[1,2].set_ylim([-0.01, 0.01])
    ax[1,2].annotate('d)', xy=(-0.2, 1.1),xycoords="axes fraction", fontsize=font_number)
    ax[1,2].set_ylabel(r'j / $\mathrm{mAcm^{-2}}$')
    ax[1,2].set_xlabel(r'Potential / V vs SHE')

    ########################
    fig.tight_layout()
    plt.savefig('figure2.pdf')
    plt.show()


    # for index, scheme in enumerate(schemes):
    #     area_forward = []
    #     area_reverse = []
    #     potential_forward = []
    #     potential_reverse = []
    #     for index_data, dat in enumerate(integration_ranges[scheme]):
    #         data = np.loadtxt(basedir[scheme]+str(dat), delimiter=',')
    #         data = data.transpose()
    #         area = integrated_peak(data[0], data[1])
    #         if index_data < len(integration_ranges[scheme]) / 2:
    #             area_forward.append(area)
    #             potential_forward.append(potential_limits[scheme][0] - index_data * scan_rate * 4 * debug_number)
    #         elif index_data >= len(integration_ranges[scheme]) / 2:
    #             area_reverse.append(area)
    #             new_index = index_data - len(integration_ranges[scheme]) / 2
    #             potential_reverse.append(potential_limits[scheme][1] + new_index * scan_rate * 4 * debug_number)
    #         counter_spectra += 1
    #
    #     ax[0,2].plot(potential_forward, area_forward, color=colors_schemes[scheme])
    #     ax[0,2].plot(potential_reverse, area_reverse, '--', color=colors_schemes[scheme])
    #     ax[0,2].annotate('CO (w/ Pb)', color=colors_schemes['with_Pb'], xy=(0.20, 0.05))
    #     ax[0,2].annotate('CO (w/o Pb)', color=colors_schemes['clean'], xy=(-0.23, 0.125))
    # ax[0,2].set_xlabel(r'Potential / V vs. SHE')
    # ax[0,2].set_ylabel(r'Peak Area / $\mathrm{a.u. cm^{-1}}$')
    # ## Plot CV figure
