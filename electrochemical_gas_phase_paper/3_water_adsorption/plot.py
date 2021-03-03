


import matplotlib.pyplot as plt
from useful_functions import create_output_directory
from plot_params import get_plot_params_Times

if __name__ == "__main__":
    create_output_directory()
    get_plot_params_Times()

    fig, ax = plt.subplots(1, 1, figsize=(6,5), squeeze=False)    
    # image1 = plt.imread('schematic/schematic_co.png')
    image2 = plt.imread('schematic/schematic_water.png')

    ax[0,0].imshow(image2)
    # ax[0,1].imshow(image2)
    ax[0,0].axis('off')
    # ax[0,1].axis('off')
    ax[0,0].set_title(r'$\Delta \left < \mathrm{E}_{\mathrm{H}_2\mathrm{O}} \right >  = -0.49 \pm 0.1 \ \mathrm{eV}$ ', )
    # ax[0,0].set_title(r'$\Delta \mathrm{E}_{\textrm{CO}^*} = -0.47 \ \mathrm{eV} $ ', )
    fig.savefig('output/water_schematic.pdf')