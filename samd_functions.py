import numpy as np
from scipy.linalg import block_diag

import matplotlib.pyplot as plt
import matplotlib.style as mpls
mpls.use('classic')

def estimate_samd(gfuncs, tau_grid, feh_grid=None, case='1D',
                  betas=None, alpha=0,
                  grid_slice=None, grid_thin=None,
                  max_iter=1, min_tol=1.e-20):
    '''
    Function for estimating the sample age-metallicity distribution (SAMD) OR
    simply the sample age distribution (SAD).

    This uses a Newton-Raphson minimisation to find the function phi which
    maximises the likelihood L(phi) = sum(L_i(phi)), where
        L_i(phi) = integral(G_i(theta)*phi(theta)) ,
    G_i are the G functions, and theta is either the age (in the 1D SAD case)
    or both the age and metallicity (in the 2D SAMD case).

    Parameters
    ----------
    gfuncs : array
        Array containing the 2D or 1D G functions.

    tau_grid : array
        Age grid on which the G functions are defined.

    feh_grid : array, optional
        Metallicity grid on which the 2D G function is defined.
        Not used in the 1D case.

    case : str, optional
        Determines whether the 2D (SAMD) or 1D (sad) is calculated.
        '2D' for SAMD and '1D' for SAD. Default is '1D'.

    betas : tuple, optional
        Beta is a regularization parameter which regulates how strongly the
        solution favors a flat (constant) function (0 is most strict, higher
        numbers are less strict).
        betas should be a tuple containing the three floats beta, dbeta, and
        beta_max. beta is the initial value, dbeta is the step, and beta_max is
        the maximum value which, if hit, stops the computation.
        beta (the initial value) should be close to 0 and dbeta not too large
        to allow a gentle convergence towards a sensible solution.
        Default is None in which case the values (0.001, 0.001, 0.5) are used.

    alpha : int, optional
        Value of the smoothing parameter. Higher values will favor solutions
        with smaller point-to-point variations (second derivatives).
        Default value is 0.

    grid_slice : tuple of ints, optional
        Grid slice indices. If specified, it must be a tuple of four integers,
        and only the age grid points from grid_slice[0] to grid_slice[1] and the
        metallicity grid points from grid_slice[2] to grid_slice[3] are
        considered.
        This increases performance by decresaing the size of the problem.
        Default value is None in which case all grid points are considered.
        Note that if G functions are saved as 1D, only the first two integers
        are used.

    grid_thin : tuple of ints, optional
        Thinning factor. If specified, it must be a tuple of two integers, and
        only every `grid_thin[0]`th age and every `grid_thin[1]`th metallicity
        grid point is considered. This increases performance by decreasing the
        size of the problem.
        This thinning is performed after slicing with grid_slice.
        Default is None in which case all grid points (in the grid_slice
        selection) are considered.

    max_iter : int, optional
        Maximum number of Newton-Raphson iterations per beta.
        Default value is 1 (if dbeta is small enough one iteration should
        suffice.)

    min_tol : float, optional
        Minimum value that the SAMD/SAD is allowed to reach.
        Default value is 1e-20.

    Returns
    -------
    samd : list
        List of SAMD/SAD with one entry for each value of beta.

    Q : list
        List of same length as `samd`. Each entry is a list giving the values of
        beta, the negative log-likelihood of the solution, its entropy, and the
        regularization term.

    tau_grid : array
        Age grid on which the output SAMD/SAD is defined.

    feh_grid : array
        Metallicity grid on which the output SAMD is defined.
        Not returned in the 1D case.
    '''

    # Check case
    if case == '1D':
        sad = True
    elif case == '2D':
        sad = False
    else:
        raise ValueError('"case" must be either "1D" or "2D".')

    # Ensure that the data and grid are compatible with the case
    if sad:
        assert gfuncs.ndim == 2 and gfuncs.shape[1] == len(tau_grid)
    else:
        assert gfuncs.ndim == 3 and gfuncs.shape[1:] == (len(tau_grid),
                                                         len(feh_grid))

    # Make grid smaller and/or more coarse (optionally, increases performance)
    if grid_slice is not None:
        tau_grid = tau_grid[grid_slice[0]:grid_slice[1]]
        if sad:
            gfuncs = gfuncs[:, grid_slice[0]:grid_slice[1]]
        else:
            gfuncs = gfuncs[:, grid_slice[0]:grid_slice[1],
                               grid_slice[2]:grid_slice[3]]
            feh_grid = feh_grid[grid_slice[2]:grid_slice[3]]
    if grid_thin is not None:
        tau_grid = tau_grid[::grid_thin[0]]
        if sad:
            gfuncs = gfuncs[:, ::grid_thin[0]]
        else:
            gfuncs = gfuncs[:, ::grid_thin[0], ::grid_thin[1]]
            feh_grid = feh_grid[::grid_thin[1]]

    # Number of tau/feh-values and number of stars
    n = gfuncs.shape[0]
    m = len(tau_grid)
    if not sad:
        l = len(feh_grid)

    # Define the G matrix (in 2D case each G-function is flattened)
    if sad:
        G = gfuncs
        k = m
    else:
        k = m*l
        G = gfuncs.reshape(n, k, order='F')
    # Add small number to avoid log(0)
    G += 1e-10

    # Setting up for iterations
    # Weights for integrals over theta
    w = np.ones(k) / k

    # Constant prior, normalized
    Phi = np.ones(k)
    Phi /= np.dot(w, Phi)

    # Initial guess for phi and lambda (solutions for beta=0)
    phi = Phi
    lamda = -1

    # Initial beta and step
    if betas is None:
        beta, dbeta, beta_max = 0.001, 0.001, 0.5
    else:
        beta, dbeta, beta_max = betas

    # Gw = G matrix, with each column j multiplied by w(j)
    Gw = G * w

    # Set up derivative matrix T
    if sad:
        T = np.diag([-1]+(m-2)*[-2]+[-1])
        T += np.diag((m-1)*[1], k=1)
        T += np.diag((m-1)*[1], k=-1)
    else:
        T1 = np.diag(np.ones(m)*(-3))
        T1[0][0] = T1[-1][-1] = -2
        T1 += np.diag(np.ones(m-1), k=1)
        T1 += np.diag(np.ones(m-1), k=-1)
        T2 = np.diag(np.ones(m)*(-4))
        T2[0][0] = T2[-1][-1] = -3
        T2 += np.diag(np.ones(m-1), k=1)
        T2 += np.diag(np.ones(m-1), k=-1)
        T_repeat = [T1] + [T2 for i in range(l-2)] + [T1]
        T = block_diag(*T_repeat)
        T += np.diag(np.ones(k-m), k=m)
        T += np.diag(np.ones(k-m), k=-m)

    # Tw = T matrix, with each column j multiplied by w(j)
    Tw = T * w

    # List to hold beta, L, E, R for each iteration
    Q = []
    # List to hold phi (the SAMD/SAD)
    samd = []

    # Perform Newton-Raphson minimisation
    finished = False
    while not finished:
        for iterr in range(max_iter):
            # u = dot product of Gw matrix with phi
            u = np.dot(Gw, phi)
            # v = dot product of Tw matrix with phi
            v = np.dot(Tw, phi)

            u[u == 0] = 1 # Avoid division by zero
            # Gwu = Gw matrix with each row divided by u(i)
            Gwu = Gw / u[:, np.newaxis]
            # Twu = Tw matrix with each row multiplied by v(k)
            Twv = Tw * v[:, np.newaxis]

            # Residuals
            r = w * (1 + np.log(phi/Phi)) - beta * np.sum(Gwu, 0) \
                + 2*alpha*beta * np.sum(Twv, 0) + lamda * w
            R = np.dot(w, phi) - 1

            # Hessian
            H = np.diag(w / phi) + beta * np.dot(Gwu.T, Gwu) \
                + 2*alpha*beta * np.dot(Tw.T, Tw)

            # Full matrix
            M = np.zeros((k+1, k+1))
            M[:k, :k] = H
            M[-1, :k] = M[:k, -1] = w
            h = np.append(-r, -R)

            s = np.append(1/np.sqrt(np.diag(H)), 1.)
            S = np.diag(s)
            M1 = np.dot(np.dot(S, M), S)
            h1 = np.dot(S, h)

            con = np.linalg.cond(M1)
            if con is np.inf:
                print('WARNING: Singular matrix - aborting at beta =', beta)
                finished = True
                break

            Delta1 = np.linalg.solve(M1, h1)
            Delta = np.dot(S, Delta1)

            Delta_phi = Delta[:k]
            Delta_lambda = Delta[-1]

            f = 1.
            phi_test = phi + f * Delta_phi
            while min(phi_test) < 0:
                f *= 0.5
                phi_test = phi + f * Delta_phi
            phi = phi_test
            lamda += f * Delta_lambda

            phi[phi < min_tol] = min_tol
            if beta >= beta_max:
                finished = True
                break

        # Renormalise to avoid exponential growth of rounding errors
        phi /= np.dot(w, phi)

        # Save SAD/SAMD
        if sad:
            samd.append(phi)
        else:
            # For SAMD, the solution is reshaped back into 2D
            samd.append(phi.reshape(m, l, order='F'))

        # Relative entropy of phi with respect to Phi
        E = np.sum(w * phi * np.log(phi / Phi))

        # Total negative log-likelihood of solution phi
        L = -np.sum(np.log(np.dot(Gw, phi)))

        # Total regularization term
        R = np.sum(np.dot(Tw, phi)**2)

        # Add these terms to Q
        Q.append([beta, L, E, R])

        # Increment beta until beta_max
        beta += dbeta

    if sad:
        return samd, Q, tau_grid
    else:
        return samd, Q, tau_grid, feh_grid


def plot_age_metallicity_distribution(phi, tau_grid, feh_grid):
    fig = plt.figure(figsize=(7,6))

    ax0 = plt.subplot2grid((4, 18), (1, 0), colspan=17, rowspan=3)
    ax1 = plt.subplot2grid((4, 18), (0, 0), colspan=17, sharex=ax0)
    ax2 = plt.subplot2grid((4, 18), (1, 17), rowspan=3)

    phi = phi.T
    dtau = tau_grid[1]-tau_grid[0]
    dfeh = feh_grid[1]-feh_grid[0]
    plot_lims = (tau_grid[0]-dtau/2, tau_grid[-1]+dtau/2,
                 feh_grid[0]-dfeh/2, feh_grid[-1]+dfeh/2)
    cim = ax0.imshow(phi, cmap='viridis', origin='lower', extent=plot_lims,
                     aspect='auto', interpolation='nearest')
    cbar = plt.gcf().colorbar(cim, cax=ax2)
    cbar.set_label('Relative probability')
    #ax0.contour(*np.meshgrid(tau_grid, feh_grid), phi, levels=[0.5*np.amax(phi)], colors='k')

    ax0.set_xlabel('Age [Gyr]')
    ax0.set_ylabel('[Fe/H]')
    ax0.grid()

    ax0.set_xlim(plot_lims[:2])
    ax0.set_ylim(plot_lims[2:])

    phi_age = np.sum(phi, axis=0)
    phi_age /= np.sum(phi_age*dtau)
    ax1.plot(tau_grid, phi_age, c='k', linewidth=2)
    ax1.set_ylabel('Relative probability')

    fig.tight_layout()
    plt.show()


def plot_age_distribution(phi, tau_grid):
    fig, ax = plt.subplots(figsize=(7,3))

    dtau = tau_grid[1]-tau_grid[0]
    plot_lims = (tau_grid[0]-dtau/2, tau_grid[-1]+dtau/2)

    ax.plot(tau_grid, phi/np.sum(phi*dtau), c='k', linewidth=2)
    ax.set_ylabel('Relative probability')
    ax.set_xlabel('Age [Gyr]')
    ax.grid()

    ax.set_xlim(plot_lims[:2])

    fig.tight_layout()
    plt.show()
