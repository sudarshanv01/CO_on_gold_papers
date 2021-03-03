

def get_plot_params():
    """Create the plot parameters used in the plotting 
    all the figures in the paper
    """
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    mpl.rcParams['font.family'] = 'Serif'
    mpl.rcParams['font.serif'] = 'Times New Roman'
    plt.rcParams['font.size'] = 18
    plt.rcParams['axes.linewidth'] = 2

    plt.rcParams['xtick.major.size'] = 10
    plt.rcParams['xtick.major.width'] = 2
    plt.rcParams['xtick.minor.size'] = 5
    plt.rcParams['xtick.minor.width'] = 2
    plt.rcParams['ytick.major.size'] = 10
    plt.rcParams['ytick.major.width'] = 2
    plt.rcParams['ytick.minor.size'] = 5
    plt.rcParams['ytick.minor.width'] = 2

    plt.rcParams['axes.labelsize'] = 28
