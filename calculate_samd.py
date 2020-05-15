import numpy as np
import pickle
import samd_functions as sf

# Set beta range (start, step, max). A step of 0.001 is usually low enough.
# A maximum value of 0.1 is sufficient in this case, but not always.
betas = (0.001, 0.001, 0.1)
# Set the regularization parameter, alpha.
# In this case the optimal value is in the range 10.000 to 100.000.
alpha = 50000

# Load the G functions and grid.
with open('gfuncs_MWlike_turnoff.p', 'rb') as f:
    data = pickle.load(f)
gfuncs = data['gfuncs']
tau_grid = data['tau_grid']
feh_grid = data['feh_grid']

# Estimate the SAMD.
print('Calculating SAMD...')
samd = sf.estimate_samd(gfuncs, tau_grid, feh_grid, case='2D',
                        betas=betas, alpha=alpha)[0]
# Plot the SAMD. samd is a list containing the SAMD for each value of beta.
# We are only interested in the solution for beta_max, so we take samd[-1].
print('Plotting SAMD...\n')
sf.plot_age_metallicity_distribution(samd[-1], tau_grid, feh_grid)



# We can also sum over the metallicity axis and apply the method on the
# 1D age distributions to estimate the SAD.
gfuncs = np.sum(gfuncs, axis=2)

# We use the same value for alpha which works in this case,
# but this is not always the case.
print('Calculating SAD...')
samd = sf.estimate_samd(gfuncs, tau_grid, case='1D',
                        betas=betas, alpha=alpha)[0]
# Plot the SAD
print('Plotting SAD...')
sf.plot_age_distribution(samd[-1], tau_grid)



