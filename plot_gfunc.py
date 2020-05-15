from samd_functions import plot_age_metallicity_distribution
import pickle

# Choose a G function to plot (by index)
Gindex = 41

with open('gfuncs_MWlike_turnoff.p', 'rb') as f:
    gfunc_data = pickle.load(f)

gfuncs = gfunc_data['gfuncs']
tau_grid = gfunc_data['tau_grid']
feh_grid = gfunc_data['feh_grid']

plot_age_metallicity_distribution(gfuncs[Gindex], tau_grid, feh_grid)
