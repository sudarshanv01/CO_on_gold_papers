#!/usr/bin/python

""" Main Script to plot Figure 1 of the Gold paper """

import numpy as np
from glob import glob
from useful_functions import AutoVivification, get_vibrational_energy
from pprint import pprint
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
sys.path.append('../classes/')
from parser_class import ParseInfo, experimentalTPD
from parser_function import get_stable_site_vibrations, get_gas_vibrations, \
                            get_coverage_details, diff_energies, \
                            get_lowest_absolute_energies,\
                            get_differential_energy, \
                            get_constants, \
                            accept_states, stylistic_exp, \
                            get_adsorbate_vibrations, \
                            stylistic_comp
from plot_params import get_plot_params

from tpd import PlotTPD
from dft import PlotDFT

from scipy import optimize
from docx import Document 
from docx.shared import Cm, Pt
import matplotlib.pyplot as plt

if __name__ == '__main__':

    ## This is where all the output is stored
    output = 'output_figures/'
    os.system('mkdir -p ' + output)

    ## Create the needed plot parameters
    get_plot_params()

    # Main figure for the paper
    fig, ax = plt.subplots(4, 2, figsize=(20, 22)) # figure for the paper

    # facets to consider in this plot
    facets = ['211', '310'] ## These are the two TPD spectra we have
    correct_background = {'211':True, '310':False} ## Choose to correct the background
    bounds_facets = {'211':[], '310':[]} ## Specify bounds for the coverage

    # the DFT facets to consider 
    dft_facets = ['211', '111', '100', '110']
    ## Must have a correspoding csv file with TPD data
    exp_data = {'211': glob('input_TPD/Au_211/*.csv'),
                '310': glob('input_TPD/Au_310/*.csv')}

    ## Optional: Image to put next to the TPD plot
    image_data = {'211':plt.imread('surface_sites/gold_211.png'),
                  '310':plt.imread('surface_sites/gold_310.png')}

    # Reaction order
    order = 1
    # Get defaults constants for CO on Au
    ## If something else change here
    constants = get_constants()
    colors = ['tab:blue', 'tab:red']
    ls = ['-', '--']

    # Styles to be used in the plot
    inferno, viridis, font_number = stylistic_exp()
    color_maps = [viridis, inferno]
    colors_facet, ls_facet, colors, colors_state = stylistic_comp()

    # Get vibrational energies for adsorbates and gas phase species
    vibrations_ads_data = get_adsorbate_vibrations()
    vibrations_gas_data = get_gas_vibrations()

    # databases 
    thermodb = connect('input_DFT/Au_CO_coverage.db')
    referencedb = connect('input_DFT/gas_phase.db')
    allE0 = {}
    for facet in facets:
        allE0[facet] = []
    
    # Word document to store the tables
    word_document = Document()

    for index, facet in enumerate(facets):
        # T_switch, T_max, T_rate_min, beta = constants[facet]
        TPDClass = PlotTPD(exp_data=exp_data[facet],
                           order=order,
                           constants=constants[facet],
                           color_map=color_maps[index],
                           thermo_ads=vibrations_ads_data[facet],
                           thermo_gas=vibrations_gas_data,
                           plot_temperature=np.linspace(100, 500, 50), 
                           bounds=bounds_facets[facet],
                           correct_background=correct_background[facet],
                           )
        # Plot the main figure
        ## Iterate over every surface facet ( 211 , 111, etc)
        for surface_index, facet_no in enumerate(TPDClass.results):

            ## Prepare word document for the TPD
            table = word_document.add_table(0,0)
            table.style = 'TableGrid'
            table.add_column(Cm(10)) ## Exposure
            table.add_column(Cm(10)) ## Temperature range
            table.add_column(Cm(5)) ## E0
            table.add_column(Cm(5)) ## error
            table.add_column(Cm(5)) ## b  
            table.add_row()
            row = table.rows[0]
            row.cells[0].text = 'Exposure (L)'
            row.cells[1].text = 'Temperature Range (K)'
            row.cells[2].text = r'\Delta E_{\theta \to 0} \ (eV)'
            row.cells[3].text = 'Residual (eV)'
            row.cells[4].text = 'b (eV)'

            # iterate over the exposures
            # Plot the TPD in normalised form
            for index_exposure, exposure in enumerate(sorted(TPDClass.results[surface_index])):
                ax[0,index].plot(TPDClass.results[surface_index][exposure]['temperature'],
                                TPDClass.results[surface_index][exposure]['normalized_rate'],
                                '.', \
                                color=color_maps[index](index_exposure), \
                                )
                # Also plot where one TPD end and the other starts 
                for i in range(len(TPDClass.temperature_range)-1):
                    if index_exposure == 1:
                        ax[0,index].axvline(TPDClass.temperature_range[i][-1], color='k', ls='--', alpha=0.2)
                # Plot the desorption energy as a function of the relative coverage
                # only for the 100 and 110 step sites
                if surface_index == len(TPDClass.results)-1:
                    ax[1,index].plot(TPDClass.theta_rel[surface_index][exposure],
                                 TPDClass.Ed[surface_index][exposure],
                                 '.', \
                                 color=color_maps[index](index_exposure), \
                                 label=str(exposure)+'L'
                                 )
                    # Plot the E0 for the 100 and 110 step sites
                    ax[2,0].plot(exposure,
                                    TPDClass.E0[surface_index][exposure],
                                    'o',
                                    color=colors[index],
                                    )
                    # Plot the associated error bar
                    ax[2,0].errorbar(exposure, TPDClass.E0[surface_index][exposure], \
                                        TPDClass.error[surface_index][exposure], color=colors[index],)
                    # Plot the equilibirum coverage
                    ax[2,1].plot(TPDClass.plot_temperature, \
                                TPDClass.theta_eq[surface_index][exposure],
                                color=color_maps[index](index_exposure),
                                ls=ls[index], lw=4)
                    # # if facet == '211':
                    ax[2,1].fill_between(TPDClass.plot_temperature, 
                                TPDClass.theta_eq_p[surface_index][exposure], 
                                TPDClass.theta_eq_n[surface_index][exposure], 
                                color=color_maps[index](index_exposure), 
                                alpha=0.1)
                    # Store the min and max E0 value for use in the last plot 
                    allE0[facet].append(TPDClass.E0[surface_index][exposure])

                    # write out E0 and b value in the form of a word table 
                    ## This paper was written with word 
                table.add_row()
                row = table.rows[index_exposure+1]
                row.cells[0].text = str(exposure)
                row.cells[1].text = str(TPDClass.temperature_range[surface_index])
                row.cells[2].text = str(round(TPDClass.E0[surface_index][exposure],2))
                row.cells[3].text = str(round(TPDClass.error[surface_index][exposure],3))
                row.cells[4].text = str(round(TPDClass.b[surface_index][exposure],2))
            word_document.add_page_break()

        ## Managing labels for main plot
        # for i in range(len(ax[0,:])):
        ax[0,index].set_xlabel(r'Temperature / K')
        ax[0,index].set_ylabel(r'$rate_{norm}$ / arb. units')
        ax[1,index].set_xlabel(r'$\theta_{rel}$')
        ax[1,index].set_ylabel(r'$G_{d}$ / eV')
        ax[1,1].legend(bbox_to_anchor=(1.04,0.5), loc="center left", borderaxespad=0)
        ax[1,0].legend(bbox_to_anchor=(1.04,0.5), loc="center left", borderaxespad=0)
        ax[2,0].set_xlabel(r'Exposure / L')
        ax[2,0].set_ylabel(r'$\Delta E_{\theta \to 0}$ / eV')
        ax[2,1].set_xlabel(r'Temperature / K')
        ax[2,1].set_ylabel(r'Equilibrium $\theta$ / ML')

        # Managing extra plots
        ## Background removing from TPD plots
        TPDClass.figP.tight_layout()
        TPDClass.figP.savefig(os.path.join(output,facet+'_background.pdf'))
        TPDClass.figP.savefig(os.path.join(output,facet+'_background.svg'))
        ## Desorption energy vs theta plots
        TPDClass.figc.tight_layout()
        TPDClass.figc.savefig(os.path.join(output,facet+'_Ed_theta.pdf'))
        TPDClass.figc.savefig(os.path.join(output,facet+'_Ed_theta.svg'))
        ## Fitted Desorption energy vs theta plots
        TPDClass.figC.tight_layout()
        TPDClass.figC.savefig(os.path.join(output,facet+'_Ed_temperature_fit.pdf'))
        TPDClass.figC.savefig(os.path.join(output,facet+'_Ed_temperature_fit.svg'))
        ## Free energy as a function of temperature
        TPDClass.figG.tight_layout()
        TPDClass.figG.savefig(os.path.join(output,facet+'_dG_T.pdf'))
        ## Coverage as a function of temperature
        TPDClass.figT.tight_layout()
        TPDClass.figT.savefig(os.path.join(output,facet+'_theta_T.pdf'))
        ## Saturation coverage as a function of exposure
        TPDClass.figsat.tight_layout()
        TPDClass.figsat.savefig(os.path.join(output,facet+'_exposure_sattheta.pdf'))
        ## Terms configuration and interaction term plotted against coverage
        TPDClass.figEc.tight_layout()
        TPDClass.figEc.savefig(os.path.join(output,facet+'_coverage_terms.pdf'))

    ## save the word document 
    word_document.save(os.path.join(output, 'tables.docx'))

    ax[0,0].annotate(r'$\mathrm{Au(111)_{terrace}}$', xy=(125, 30), xycoords='data',xytext=(225,30), \
    arrowprops=dict(shrink=0.05, color=colors_facet['111-0']),
    horizontalalignment='right', verticalalignment='center', color=colors_facet['111-0'], fontsize=22)
    ax[0,1].annotate(r'$\mathrm{Au(100)_{terrace}}$', xy=(125, 48), xycoords='data',xytext=(200,48), \
    arrowprops=dict(shrink=0.05, color=colors_facet['100']),
    horizontalalignment='right', verticalalignment='center', color=colors_facet['100'], fontsize=22)
    ax[0,0].annotate(r'$\mathrm{Au(100)_{step}}$', xy=(190, 12), xycoords='data',xytext=(190,20), \
    arrowprops=dict(shrink=0.05, color=colors_facet['211']),
    horizontalalignment='center', verticalalignment='center', color=colors_facet['211'], fontsize=22)
    ax[0,1].annotate(r'$\mathrm{Au(110)_{step}}$', xy=(185, 25), xycoords='data',xytext=(185,35), \
    arrowprops=dict(shrink=0.05, color=colors_facet['110']),
    horizontalalignment='center', verticalalignment='center', color=colors_facet['110'], fontsize=22)

    ax[0,0].annotate('a)', xy=(-0.2, 1.1),xycoords="axes fraction", fontsize=font_number)
    ax[0,1].annotate('b)', xy=(-0.2, 1.1),xycoords="axes fraction", fontsize=font_number)
    ax[1,0].annotate('c)', xy=(-0.2, 1.1),xycoords="axes fraction", fontsize=font_number)
    ax[1,1].annotate('d)', xy=(-0.2, 1.1),xycoords="axes fraction", fontsize=font_number)
    ax[2,0].annotate('e)', xy=(-0.2, 1.1),xycoords="axes fraction", fontsize=font_number)
    ax[2,1].annotate('f)', xy=(-0.2, 1.1),xycoords="axes fraction", fontsize=font_number)
    for index, facet in enumerate(facets):
        actual_facet = facet.replace('211', '100').replace('310', '110')
        ax[2,0].plot([], [], 'o', color=colors[index], label=r'Au('+actual_facet+')$_{step}$')
        ax[2,1].plot([], [], color='k', ls=ls[index], label=r'Au('+actual_facet+')$_{step}$')
    ax[2,0].legend(loc='best')
    ax[2,1]. axvline(x=298.15, color='k', lw=3, ls='--')
    ax[2,1].legend(loc='best')

    ## DFT data 
    for facet in dft_facets:
        dft_data = PlotDFT(database=thermodb, 
                reference_database=referencedb, 
                facet=facet, 
                functional='BF')
        # print(dft_data.dE, dft_data.coverage)
        if facet in ['111', '211',]:
            ax[3,0].plot(dft_data.coverage, dft_data.dE, \
                 'o-', color=colors_facet[facet], \
                     label='Au('+facet+')')
            ax[3,0].errorbar(dft_data.coverage, dft_data.dE, dft_data.beef_error[facet],  color=colors_facet[facet])
            if facet == '211':
                ax[3,0].axhline(y=-1*dft_data.free_diff, color='k', alpha=0.5, ls='--', label=r'$\Delta G_{diff}=0$')
        if facet in ['100', '110']:
            ax[3,1].plot(dft_data.coverage, dft_data.dE, \
                 'o-', color=colors_facet[facet], \
                     label='Au('+facet+')')
            ax[3,1].errorbar(dft_data.coverage, dft_data.dE, dft_data.beef_error[facet],  color=colors_facet[facet])
            if facet == '110':
                ax[3,1].axhline(y=-1*dft_data.free_diff, color='k', alpha=0.5, ls='--', label=r'$\Delta G_{diff}=0$')
        
    # Plot the E0 term in the DFT plot as well as the Delta E corresponding to Delta G
    for facet, aE0 in allE0.items():
        minmax = [min(aE0), max(aE0)]

        if facet == '211':
            xbound = ax[3,0].get_xbound()
            minbound = -1 * minmax[0] * np.ones(len(xbound))
            maxbound = -1 * minmax[1] * np.ones(len(xbound))
            ax[3,0].fill_between(xbound, minbound, maxbound, color='tab:blue', alpha=0.25, label=r'$\Delta E_{\theta \to 0}$')

        if facet == '310':
            xbound = ax[3,1].get_xbound()
            minbound = -1 * minmax[0] * np.ones(len(xbound))
            maxbound = -1 * minmax[1] * np.ones(len(xbound))
            ax[3,1].fill_between(xbound, minbound, maxbound, color='tab:brown', alpha=0.25, label=r'$\Delta E_{\theta \to 0}$')

    ax[3,0].set_xlabel(r'$\theta$ / ML')
    ax[3,0].set_ylabel(r'$\Delta E_{diff}$ / eV')
    ax[3,0].annotate('g)', xy=(-0.2, 1.1),xycoords="axes fraction", fontsize=font_number)
    ax[3,0].legend(bbox_to_anchor=(1.04,0.5), loc="center left", borderaxespad=0)
    ax[3,1].set_xlabel(r'$\theta$ / ML')
    ax[3,1].set_ylabel(r'$\Delta E_{diff}$ / eV')
    ax[3,1].annotate('h)', xy=(-0.2, 1.1),xycoords="axes fraction", fontsize=font_number)
    ax[3,1].legend(bbox_to_anchor=(1.04,0.5), loc="center left", borderaxespad=0)

    # Add schematics of 211 and 310 
    for facet in image_data:
        if facet == '211':    
            newax = fig.add_axes([0.38, 0.8, 0.12, 0.12], anchor='NE', zorder=-1)
        elif facet == '310':
            newax = fig.add_axes([0.86, 0.8, 0.12, 0.12], anchor='NE', zorder=-1)
        newax.imshow(image_data[facet])
        newax.axis('off')

    fig.tight_layout()
    fig.savefig(os.path.join(output,'main_figure.pdf'))
