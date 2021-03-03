
""" Main Script to plot Figure 1 of the Gold paper """

import numpy as np
from glob import glob
import os, sys
from ase.thermochemistry import HarmonicThermo, IdealGasThermo
import mpmath as mp
from ase.io import read
from ase.db import connect
import matplotlib
from ase.units import kB
import csv
from scipy.optimize import curve_fit
from matplotlib import cm
import matplotlib.pyplot as plt
from tpd_analyse.tpd import PlotTPD
from plot_params import get_plot_params
from useful_functions import create_output_directory
import matplotlib
from dft import PlotDFT
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
import string

def get_TPD_constants():

    """ TPD data """
    # For 211 TPD
    T_switch_211 = [170] #[110, 170] #[110, 165, 175] #K converting between 111 and 211 step
    T_max_211 = 250 #K Where the TPD spectra ends
    T_rate_min_211 = [250, 300] #K Where the rate becomes zero - baseline
    beta_211 = 3 #k/s Heating rate
    data_211 = [T_switch_211, T_max_211, T_rate_min_211, beta_211]


    # For 310 TPD
    T_switch_310 = [150] # K converting between 100 and 110
    T_max_310 = 240 # K Where the TPD spectra ends
    T_rate_min_310 = [230 , 250] #K Where the rate becomes zero - baseline
    beta_310 = 5 #k/s Heating rate
    data_310 = [T_switch_310, T_max_310, T_rate_min_310, beta_310]

    return {
            '211':data_211,
            '310':data_310
            }



if __name__ == '__main__':

    get_plot_params()
    create_output_directory()
    

    fig1, ax1 = plt.subplots(2, 3, figsize=(14,8), squeeze=False)
    fig2, ax2 = plt.subplots(1, 2, figsize=(10,4.5), squeeze=False)
    fig3, ax3 = plt.subplots(1, 1, figsize=(7,5), squeeze=False)
    fig4, ax4 = plt.subplots(1, 2, figsize=(10,5), squeeze=False)

    ## Supplementary information plots
    figS1, axS1 = plt.subplots(1, 2, figsize=(8,5), squeeze=False )
    figS2, axS2 = plt.subplots(2, 2, figsize=(10,8), squeeze=False)
    figS3, axS3 = plt.subplots(2, 2, figsize=(10,8), squeeze=False)

    """ Plotting TPD graphs """

    ## We look at only two facets
    facets = ['211', '310']
    cmap_facet = {'211':'Blues', '310':'Oranges'}
    correct_background = {'211':True, '310':False}
    ## Optional: Image to put next to the TPD plot
    image_data = {'211':plt.imread('surface_sites/gold_211.png'),
                  '310':plt.imread('surface_sites/gold_310.png')}
    ## Order of the reaction
    order = 1
    ## initial guess for theta
    initial_guess_theta = 0.9
    ## gas atoms
    atoms = read('input_data/co.traj')
    thermo_gas = 0.00012 * np.array([2100.539983, 24.030662, 24.018143 ])
    vibration_energies_gas = IdealGasThermo(thermo_gas, atoms = atoms, \
            geometry='linear', symmetrynumber=1, spin=0)
    vibration_energies_ads = HarmonicThermo(0.00012 * np.array([2044.1, 282.2, 201.5, 188.5, 38.3, 11.5]))

    for index_facet, facet in enumerate(facets):
        ## Files with the different exposure
        files = glob('input_data/Au_%s/*.csv'%facet)
        TPDClass = PlotTPD(exp_data=files,
                            order=order,
                            thermo_ads=vibration_energies_ads,
                            thermo_gas=vibration_energies_gas,
                            plot_temperature=np.linspace(100, 500, 50), 
                            constants=get_TPD_constants()[facet],
                            initial_guess_theta=initial_guess_theta,
                            correct_background=correct_background[facet],
                            )
        TPDClass.get_results()
        ## Nested dict of facets and 
        # cmap = cm.get_cmap(cmap_facet[facet], len(files))
        if facet == '211':
            cmap = cm.Blues(np.linspace(0, 1, len(files)))
        elif facet == '310':
            cmap = cm.Oranges(np.linspace(0, 1, len(files)))
        cmap = matplotlib.colors.ListedColormap(cmap[1:,:-1])
        all_E0 = []

        for surface_index, facet_no in enumerate(TPDClass.results):
            # Plot the TPD in normalised form
            for index_exposure, exposure in enumerate(sorted(TPDClass.results[surface_index])):
                ax1[index_facet,0].plot(TPDClass.results[surface_index][exposure]['temperature'], \
                                TPDClass.results[surface_index][exposure]['normalized_rate'], 
                                '.', color=cmap(index_exposure),
                                markersize=6
                                )
                if surface_index == len(TPDClass.results)-1:
                    theta = TPDClass.theta_rel[surface_index][exposure] * TPDClass.theta_sat[surface_index][exposure] 
                    theta_rel = TPDClass.theta_rel[surface_index][exposure]
                    interaction_term = - TPDClass.b[surface_index][exposure] * TPDClass.theta_sat[surface_index][exposure] * TPDClass.theta_rel[surface_index][exposure]
                    config_term = - kB * TPDClass.results[surface_index][exposure]['temperature_fit']* np.log(TPDClass.theta_sat[surface_index][exposure]*TPDClass.theta_rel[surface_index][exposure] / ( 1 - TPDClass.theta_sat[surface_index][exposure]*TPDClass.theta_rel[surface_index][exposure]))
                    # total_E = 1 * interaction_term + config_term + TPDClass.E0[surface_index][exposure] 
                    total_E = TPDClass.Ed_fitted[surface_index][exposure]

                    ax1[index_facet,1].plot(theta_rel,
                                    TPDClass.Ed[surface_index][exposure],
                                    'o', \
                                    color=cmap(index_exposure), \
                                    label=str(exposure)+'L', 
                                    markersize=6
                                    )
                    ax1[index_facet,1].plot(theta_rel,
                                total_E, '--', 
                                color=cmap(index_exposure), 
                                # alpha=0.25,
                                )
                    ax2[0,0].plot(exposure, TPDClass.E0[surface_index][exposure], 'o',\
                                color=cmap(index_exposure))
                    ax2[0,1].plot(TPDClass.theta_rel[surface_index][exposure], 
                                interaction_term, 
                                '-.', color=cmap(index_exposure),
                    )
                    ax2[0,1].plot(TPDClass.theta_rel[surface_index][exposure], 
                                config_term, 
                                '-', color=cmap(index_exposure),
                    )
                    ax2[0,0].errorbar(exposure, TPDClass.E0[surface_index][exposure], TPDClass.error[surface_index][exposure], color=cmap(index_exposure))
                    ax3[0,0].plot(TPDClass.plot_temperature, \
                                TPDClass.theta_eq[surface_index][exposure],
                                color=cmap(index_exposure),
                                ls='-', lw=4)
                    ax3[0,0].fill_between(TPDClass.plot_temperature,
                                TPDClass.theta_eq_p[surface_index][exposure],
                                TPDClass.theta_eq_n[surface_index][exposure],
                                color=cmap(index_exposure),
                                alpha=0.5)
                    all_E0.append(TPDClass.E0[surface_index][exposure])
                
                axS2[index_facet,surface_index].plot(TPDClass.theta_rel[surface_index][exposure],
                                TPDClass.Ed[surface_index][exposure],
                                'o', \
                                color=cmap(index_exposure), \
                                markersize=6
                                )
                axS2[index_facet,surface_index].plot(TPDClass.theta_rel[surface_index][exposure],
                            TPDClass.Ed_fitted[surface_index][exposure], '--', 
                            color=cmap(index_exposure), 
                            )
                axS3[index_facet,surface_index].plot(exposure,
                            TPDClass.theta_sat[surface_index][exposure], 'o', 
                            color=cmap(index_exposure), 
                            )

        ax2[0,0].plot([], [], 'o-',  label='Au(%s)'%facet, color=cmap(index_exposure))
        ax3[0,0].plot([], [], '-',  label='Au(%s)'%facet, color=cmap(index_exposure))
        for i in range(len(ax1)):
            ax1[i,1].set_xlabel(r'$\theta_{\mathregular{rel}}$')
            ax1[i,1].set_ylabel(r'$G_{\mathregular{d}}$ / eV')
            ax1[i,0].set_ylabel(r'Norm. rate')
            ax1[i,0].set_xlabel(r'Temperature / K')
        ax2[0,0].set_ylabel(r'$\Delta \mathregular{E}_{\theta \to 0}$ / eV')
        ax2[0,0].set_xlabel(r'Exposure / L')
        ax2[0,1].set_ylabel(r'G$_{\mathregular{d}}$ contributions')
        ax2[0,1].set_xlabel(r'$\theta_{\mathregular{rel}}$')
        ax3[0,0].axvline(298.15, color='k', ls='--')
        ax3[0,0].set_ylabel(r'Equilibrium $\theta$ / ML')
        ax3[0,0].set_xlabel(r'Temperature / K')
        ax4[0,0].axhspan(-1 * min(all_E0), -1 * max(all_E0), color=cmap(index_exposure), alpha=0.25, \
                    label=r'$\Delta \mathregular{E}_{ \theta \to 0} \ \mathregular{Au}(%s)$'%facet)
            
        for index_exposure, exposure in enumerate(sorted(TPDClass.results[surface_index])):
                ## Plot experimental SI plots
                axS1[0,index_facet].plot(TPDClass.norm_results[exposure].temperature, \
                                        TPDClass.norm_results[exposure].background_correction, '-', color=cmap(index_exposure))
                axS1[0,index_facet].plot(TPDClass.norm_results[exposure].temperature, \
                                        TPDClass.norm_results[exposure].exp_rate, '.', color=cmap(index_exposure))
                if index_exposure == 0:
                    axS1[0,index_facet].set_title('Au(%s)'%facet)

    ax1[0,0].annotate(r'$\mathrm{Au(111)_{terrace}}$', xy=(125, 30), xycoords='data',xytext=(225,30), \
    arrowprops=dict(shrink=0.05, color='k'),
    horizontalalignment='right', verticalalignment='center', fontsize=12)
    ax1[1,0].annotate(r'$\mathrm{Au(100)_{terrace}}$', xy=(125, 55), xycoords='data',xytext=(225,55), \
    arrowprops=dict(shrink=0.05, color='k'),
    horizontalalignment='right', verticalalignment='center', fontsize=12)
    ax1[0,0].annotate(r'$\mathrm{Au(100)_{step}}$', xy=(190, 12), xycoords='data',xytext=(190,20), \
    arrowprops=dict(shrink=0.05, color='k'),
    horizontalalignment='center', verticalalignment='center', fontsize=12)
    ax1[1,0].annotate(r'$\mathrm{Au(110)_{step}}$', xy=(190, 30), xycoords='data',xytext=(190,45), \
    arrowprops=dict(shrink=0.05, color='k'),
    horizontalalignment='center', verticalalignment='center',  fontsize=12)

    ax1[0,1].annotate('Configurational \nEntropy', xy=(0.12, 0.7), xycoords='axes fraction', \
                color='k', fontsize=12)
    ax1[0,1].annotate('CO*-CO* \ninteractions', xy=(0.6, 0.3), xycoords='axes fraction', \
                color='k', fontsize=12)
    ax1[1,1].annotate('Configurational \nEntropy', xy=(0.12, 0.7), xycoords='axes fraction', \
                color='k', fontsize=12)
    ax1[1,1].annotate('CO*-CO* \ninteractions', xy=(0.6, 0.35), xycoords='axes fraction', \
                color='k', fontsize=12)
    ax1[0,1].legend(bbox_to_anchor=(1.04,0.5), loc="center left", borderaxespad=0, frameon=False, fontsize=12)
    ax1[1,1].legend(bbox_to_anchor=(1.04,0.5), loc="center left", borderaxespad=0, frameon=False, fontsize=12)

    for ax in ax1.flatten():
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.tick_params(which='minor', length=4)
    ax1[0,0].set_yticks([])
    ax1[1,0].set_yticks([])

    ax2[0,1].plot([], [], ls='-', color='k', label='Configurational entropy ')
    ax2[0,1].plot([], [], ls='-.',color='k', label='CO*-CO* Interactions ')
    ax2[0,1].legend(loc='best', frameon=False, fontsize=12)
    ax2[0,0].legend(loc='lower right', frameon=False, fontsize=12)
    ax3[0,0].legend(loc='best', frameon=False, fontsize=12)
    ax3[0,0].annotate(r'$p_{\mathregular{CO}(\mathregular{g})} = 1 \mathregular{bar}$', \
                           xy=(0.1, 0.3), xycoords='axes fraction' )


    for i, facet in enumerate(image_data):
        ax1[i,2].imshow(image_data[facet])
        ax1[i,2].axis('off')

    ## Label the diagram
    alphabet = list(string.ascii_lowercase)
    for i, a in enumerate(ax1.flatten()):
        a.annotate(alphabet[i]+')', xy=(-0.1, 1.1), xycoords='axes fraction', fontsize=20)
    for i, a in enumerate(ax2.flatten()):
        a.annotate(alphabet[i]+')', xy=(-0.1, 1.1), xycoords='axes fraction', fontsize=20)

    for ax in axS1.flatten():
        ax.set_ylabel(r'TPD Rate')
        ax.set_xlabel(r'Temperature / K')
        ax.set_yticks([])

    for ax in axS2.flatten():
        ax.set_ylabel(r'$\mathregular{G}_{d}$')
        ax.set_xlabel(r'$\theta_{\mathregular{rel}}$ ')
        
    for ax in axS3.flatten():
        ax.set_ylabel(r'$\theta_{sat}$ / ML')
        ax.set_xlabel(r'Exposure / L ')

    for i, a in enumerate(axS1.flatten()):
        a.annotate(alphabet[i]+')', xy=(-0.1, 1.1), xycoords='axes fraction', fontsize=20)
    for i, a in enumerate(axS2.flatten()):
        a.annotate(alphabet[i]+')', xy=(-0.1, 1.1), xycoords='axes fraction', fontsize=20)
    for i, a in enumerate(axS3.flatten()):
        a.annotate(alphabet[i]+')', xy=(-0.1, 1.1), xycoords='axes fraction', fontsize=20)

    fig1.tight_layout()
    fig1.savefig(os.path.join('output/figure_1.pdf'))

    fig2.tight_layout()
    fig2.savefig(os.path.join('output/figure_2.pdf'))

    fig3.tight_layout()
    fig3.savefig(os.path.join('output/figure_3.pdf'))

    figS1.tight_layout()
    figS1.savefig(os.path.join('output/figure_S1.pdf'))

    figS2.tight_layout()
    figS2.savefig(os.path.join('output/figure_S2.pdf'))
    
    figS3.tight_layout()
    figS3.savefig(os.path.join('output/figure_S3.pdf'))


    """ Plotting DFT graphs """
    dft_facets = ['211', '310', '111', '100', ]
    colors_facet = {'211':'tab:blue',  '100':'tab:red',\
                    '110':'tab:brown', 'recon_110':'tab:cyan', '111':'tab:green', 
                    '310':'tab:orange'}

    thermodb = connect('input_data/Au_CO_coverage.db')
    referencedb = connect('input_data/gas_phase.db')

    for index_facet, facet in enumerate(dft_facets):
        dft_data = PlotDFT(database=thermodb,
                reference_database=referencedb,
                facet=facet,
                functional='BF')
        ax4[0,1].plot(dft_data.coverage, dft_data.dEdiff, \
                marker='o', ls='--', color=colors_facet[facet], \
                    label='Au('+facet+')')
        ax4[0,0].plot(dft_data.coverage, dft_data.dEint, \
                marker='o', ls='--', color=colors_facet[facet], \
                    # label='Au('+facet+')', )
        )
        ax4[0,0].errorbar(dft_data.coverage, dft_data.dEint, dft_data.beef_error[facet], alpha=0.25,\
             color=colors_facet[facet])
        ax4[0,1].errorbar(dft_data.coverage, dft_data.dEdiff, dft_data.beef_error[facet],alpha=0.25,\
              color=colors_facet[facet])
    # handles, labels = ax4[0,0].get_legend_handles_labels()
    ax4[0,1].legend(loc="best", frameon=False, fontsize=12)
    ax4[0,0].legend(loc='upper left', frameon=False, fontsize=12)

    # fig4.legend(handles, labels, bbox_to_anchor=(0,1.02,1,0.2), loc="lower left",
                # mode="expand", borderaxespad=0, ncol=2)
    ax4[0,1].axhline(y=0, color='k', ls='-')
    ax4[0,0].set_xlabel(r'$\theta$ / ML')
    ax4[0,0].set_ylabel(r'$\Delta \mathregular{E} + \Delta \mathregular{ZPE}$ / eV')
    ax4[0,1].set_xlabel(r'$\theta$ / ML')
    ax4[0,1].set_ylabel(r'$\Delta \mathregular{G}_{\mathregular{diff}}$ / eV')


    for i, a in enumerate(ax4.flatten()):
        a.annotate(alphabet[i]+')', xy=(-0.1, 1.1), xycoords='axes fraction', fontsize=20)

    fig4.tight_layout()
    fig4.savefig('output/figure_4.pdf')