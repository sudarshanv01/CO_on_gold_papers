
""" Main Script to plot Figure 1 of the Gold paper """

import numpy as np
from glob import glob
import os, sys
import json
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
from pathlib import Path
import matplotlib
from dft import PlotDFT
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
import string
from ase import build
try:
    from plot_params import get_plot_params
    get_plot_params()
except Exception:
    pass
Path('output').mkdir(exist_ok=True)

if __name__ == '__main__':

    fig1, ax1 = plt.subplots(2, 3, figsize=(14,8), squeeze=False)
    fig2, ax2 = plt.subplots(1, 2, figsize=(12.5,4.5), squeeze=False)
    fig3, ax3 = plt.subplots(1, 1, figsize=(7,5), squeeze=False)
    fig4, ax4 = plt.subplots(1, 3, figsize=(14,5), squeeze=False)

    ## Supplementary information plots
    figS1, axS1 = plt.subplots(1, 2, figsize=(8,5), squeeze=False )
    figS2, axS2 = plt.subplots(2, 2, figsize=(10,8), squeeze=False)
    figS3, axS3 = plt.subplots(2, 2, figsize=(10,8), squeeze=False)

    """ Plotting TPD graphs """

    ## We look at only two facets
    facets = ['211', '310']
    cmap_facet = {'211':'Blues', '310':'Oranges'}

    ## Optional: Image to put next to the TPD plot
    image_data = {'211':plt.imread('surface_sites/gold_211.png'),
                  '310':plt.imread('surface_sites/gold_310.png')}
    ## gas atoms
    atoms = build.molecule('CO')
    thermo_gas = 0.00012 * np.array([2100.539983, 24.030662, 24.018143 ])
    vibration_energies_gas = IdealGasThermo(thermo_gas, atoms = atoms, \
            geometry='linear', symmetrynumber=1, spin=0)
    vibration_energies_ads = HarmonicThermo(0.00012 * np.array([2044.1, 282.2, 201.5, 188.5, 38.3, 11.5]))
    functional_color = {'RPBE':'tab:cyan', 'PBE':'tab:olive', 'RPBE-D3':'tab:pink', 'PBE-D3':'tab:gray', 'BEEF-vdW':'tab:brown'}

    for index_facet, facet in enumerate(facets):
        ## Files with the different exposure
        files = glob('input_data/Au_%s/*.csv'%facet)
        with open(os.path.join('input_data', f'Au_{facet}', 'inputs.json')) as handle:
            inputs = json.load(handle)

        TPDClass = PlotTPD( exp_data=files,
                            thermo_ads=vibration_energies_ads,
                            thermo_gas=vibration_energies_gas,
                            plot_temperature=np.linspace(100, 500, 50), 
                            **inputs
                            )

        TPDClass.get_results()
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
                theta_rel = TPDClass.results[surface_index][exposure]['theta_rel']
                if surface_index == len(TPDClass.results)-1:
                    theta = TPDClass.results[surface_index][exposure]['theta_rel'] * TPDClass.results[surface_index][exposure]['theta_sat'] 
                    interaction_term = TPDClass.results[surface_index][exposure]['ads_ads_interaction'] 
                    config_term = TPDClass.results[surface_index][exposure]['configurational_entropy'] 
                    total_E = TPDClass.results[surface_index][exposure]['Ed_fitted']

                    ax1[index_facet,1].plot(theta_rel,
                                    TPDClass.results[surface_index][exposure]['Ed'],
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
                    ax2[0,0].plot(exposure, TPDClass.results[surface_index][exposure]['E0'], 'o',\
                                color=cmap(index_exposure))
                    ax2[0,1].plot(theta_rel, interaction_term, '-.', color=cmap(index_exposure))
                    ax2[0,1].plot(theta_rel, config_term, '-', color=cmap(index_exposure),
                                label='Au(%s):%1.2fL'%(facet,exposure))
                    ax2[0,0].errorbar(exposure, TPDClass.results[surface_index][exposure]['E0'],\
                         TPDClass.results[surface_index][exposure]['error'], color=cmap(index_exposure))
                    ax3[0,0].plot(TPDClass.plot_temperature, \
                                TPDClass.theta_eq[surface_index][exposure],
                                color=cmap(index_exposure),
                                label='Au(%s):%1.2fL'%(facet,exposure),
                                ls='-', lw=4)
                    ax3[0,0].fill_between(TPDClass.plot_temperature,
                                TPDClass.theta_eq_p[surface_index][exposure],
                                TPDClass.theta_eq_n[surface_index][exposure],
                                color=cmap(index_exposure),
                                alpha=0.5)
                    all_E0.append(TPDClass.results[surface_index][exposure]['E0'])
                
                axS2[index_facet,surface_index].plot(theta_rel,
                                TPDClass.results[surface_index][exposure]['Ed'],
                                'o', \
                                color=cmap(index_exposure), \
                                markersize=6
                                )
                axS2[index_facet,surface_index].plot(theta_rel,
                            TPDClass.results[surface_index][exposure]['Ed_fitted'],
                            color=cmap(index_exposure), 
                            )
                axS3[index_facet,surface_index].plot(exposure,
                            TPDClass.results[surface_index][exposure]['theta_sat'], 'o',
                            color=cmap(index_exposure), 
                            )

        for i in range(len(ax1)):
            ax1[i,1].set_xlabel(r'$\theta_{\mathregular{rel}}$')
            ax1[i,1].set_ylabel(r'$G_{\mathregular{d}}$ / eV')
            ax1[i,0].set_ylabel(r'Rate')
            ax1[i,0].set_xlabel(r'Temperature / K')
        ax2[0,0].set_ylabel(r'$\Delta \mathregular{E}_{\theta \to 0}$ / eV')
        ax2[0,0].set_xlabel(r'Exposure / L')
        ax2[0,1].set_ylabel(r'G$_{\mathregular{d}}$ contributions')
        ax2[0,1].set_xlabel(r'$\theta_{\mathregular{rel}}$')
        ax3[0,0].axvline(298.15, color='k', ls='--')
        ax3[0,0].set_ylabel(r'Equilibrium $\theta$ / ML')
        ax3[0,0].set_xlabel(r'Temperature / K')
        if facet == '211':
            ax4[0,0].axhspan(-1 * min(all_E0), -1 * max(all_E0), color=cmap(index_exposure), alpha=0.25,
                    label=r'$\Delta \mathregular{E}_{ \theta \to 0} \ \mathregular{Au}(%s)$'%facet)
        elif facet == '310':
            ax4[0,1].axhspan(-1 * min(all_E0), -1 * max(all_E0), color=cmap(index_exposure), alpha=0.25,
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
    ax2[0,0].set_xscale('log')
    ax2[0,1].legend(bbox_to_anchor=(1.04,1), borderaxespad=0, frameon=False, fontsize=10)
    # ax2[0,0].legend(loc='lower right', frameon=False, fontsize=12)
    ax3[0,0].legend(loc='lower left', frameon=False, fontsize=10)
    ax3[0,0].annotate(r'$p_{\mathregular{CO}(\mathregular{g})} = 1 \mathregular{bar}$', \
                           xy=(0.6, 0.7), xycoords='axes fraction' )


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
    fig1.savefig(os.path.join('output/figure_1.png'), dpi=450)

    fig2.tight_layout()
    fig2.savefig(os.path.join('output/figure_2.png'), dpi=450)

    fig3.tight_layout()
    fig3.savefig(os.path.join('output/figure_3.png'), dpi=450)

    figS1.tight_layout()
    figS1.savefig(os.path.join('output/figure_S1.png'), dpi=450)

    figS2.tight_layout()
    figS2.savefig(os.path.join('output/figure_S2.png'), dpi=450)
    
    figS3.tight_layout()
    figS3.savefig(os.path.join('output/figure_S3.png'), dpi=450)


    """ Plotting DFT graphs """
    data_functional = json.load(open('input_data/energies.json', 'r'))
    gas_vibrations = json.load(open('input_data/gas_phase_vib.json', 'r'))
    ads_vibrations = json.load(open('input_data/vibrations.json', 'r'))

    # Now plot data for the different functionals
    for functional in data_functional:
        for facet in ['211', '310']:
            data_energies = data_functional[functional][facet]
            data_energies = np.array(data_energies).T
            data_energies = np.sort(data_energies)
            # get the gas molecule vibrations for the right functional
            gas_vib = np.array(gas_vibrations[functional])
            # get the adsorbate vibrations for the right coverage
            ads_vib = np.array(ads_vibrations[functional][facet]["1.0"])
            ads_vib[ads_vib < 0] = 30 # replace negative values with 30

            cm_to_eV = 0.00012
            # Free energies 
            thermo_gas = IdealGasThermo(cm_to_eV * gas_vib[0:3], atoms = build.molecule('CO'), \
                    geometry='linear', symmetrynumber=1, spin=0)
            thermo_ads = HarmonicThermo(cm_to_eV * ads_vib)

            delta_zpe = thermo_ads.get_ZPE_correction() - thermo_gas.get_ZPE_correction()
            dG = thermo_ads.get_helmholtz_energy(298.15, verbose=False) - thermo_gas.get_gibbs_energy(298.15, 101325, verbose=False) 

            diff_energies = []
            for i in range(len(data_energies[0])-1):
                if i == 0:
                    diff_energies.append(0)
                else:
                    diff_energies.append(data_energies[1][i+1]-data_energies[1][i])
            diff_energies = np.array(diff_energies)
            diff_energies += data_energies[1][0]
            diff_energies += dG 

            if facet == '211': 
                index = 0
                line = '-'
                marker = 'o'
                ax4[0,index].plot(data_energies[0], data_energies[1]+delta_zpe, marker+line,  color=functional_color[functional])
                ax4[0,2].plot(data_energies[0][:-1], diff_energies, marker+line, label=f'{functional} ({facet})', color=functional_color[functional])
            elif facet == '310': 
                index = 1
                line = '--'
                marker = 'v'
                ax4[0,index].plot(data_energies[0], data_energies[1]+delta_zpe, marker+line,  color=functional_color[functional])
                ax4[0,2].plot(data_energies[0][:-1], diff_energies, marker+line, label=f'{functional} ({facet})', color=functional_color[functional])
            else: raise ValueError 


    ax4[0,0].set_ylabel(r'$\Delta \mathregular{E} + \Delta \mathregular{ZPE}$ / eV')
    ax4[0,0].set_xlabel(r'$\theta$ / ML')
    ax4[0,1].set_ylabel(r'$\Delta \mathregular{E} + \Delta \mathregular{ZPE}$ / eV')
    ax4[0,1].set_xlabel(r'$\theta$ / ML')
    ax4[0,2].set_ylabel(r'$\Delta G_\mathregular{diff}$ / eV') 
    ax4[0,2].set_xlabel(r'$\theta$ / ML')

    ax4[0,0].axhline(-0.58, label='Redhead analysis', color='tab:purple', ls='-.')
    ax4[0,1].axhline(-0.58, label='Redhead analysis', color='tab:purple', ls='-.')
    ax4[0,2].axhline(-0.0, label='$\Delta G_{\mathregular{diff}}=0$', color='k', ls='-.')

    ax4[0,0].set_ylim([-0.8, 0.1])
    ax4[0,1].set_ylim([-0.8, 0.1])

    ax4[0,0].legend(loc='best', fontsize=10)
    ax4[0,1].legend(loc='best', fontsize=10)
    ax4[0,2].legend(bbox_to_anchor=(1.04,1), loc="upper left", fontsize=10)
    ax4[0,0].set_title('Au(211)',) 
    ax4[0,1].set_title('Au(310)',) 



    for i, a in enumerate(ax4.flatten()):
        a.annotate(alphabet[i]+')', xy=(-0.12, 1.1), xycoords='axes fraction', fontsize=20)

    fig4.tight_layout()
    fig4.savefig('output/figure_4.png', dpi=450)