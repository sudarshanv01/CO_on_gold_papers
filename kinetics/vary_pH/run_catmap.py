from catmap import ReactionModel
import matplotlib

mkm_file = 'catmap.mkm'
model = ReactionModel(setup_file=mkm_file)
model.output_variables += ['rate', 'production_rate']
model.run()

from catmap import analyze

vm = analyze.VectorMap(model)
vm.plot_variable = 'production_rate' #tell the model which output to plot
vm.log_scale = True #rates should be plotted on a log-scale
vm.min = 1e-20 #minimum rate to plot
vm.max = 1e4 #maximum rate to plot
vm.threshold = 1e-25 #anything below this is considered to be 0
#vm.subplots_adjust_kwargs = {'left':0.2,'right':0.8,'bottom':0.15}
vm.plot(save='production_rate.pdf')

vm.plot_variable = 'rate' #tell the model which output to plot
vm.log_scale = True #rates should be plotted on a log-scale
vm.min = 1e-2 #minimum rate to plot
vm.max = 1e9 #maximum rate to plot
vm.plot(save='rate.pdf')

ma = analyze.MechanismAnalysis(model)
ma.energy_type = 'free_energy' #can also be free_energy/potential_energy
ma.include_labels = False #way too messy with labels
ma.pressure_correction = True #assume all pressures are 1 bar (so that energies are the same as from DFT)
ma.include_labels = True
fig = ma.plot(plot_variants=[0.0], save='FED.png')
print(ma.data_dict)  # contains [energies, barriers] for each rxn_mechanism defined

vm.plot_variable = 'coverage'
vm.log_scale = False
vm.min = 0
vm.max = 1
vm.plot(save='coverage.pdf')
