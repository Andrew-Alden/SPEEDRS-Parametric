# Code to simulate the Multi-GBM was adapted from http://dx.doi.org/10.2139/ssrn.3829701

import numpy as np
from torch.quasirandom import SobolEngine
import scipy.stats as si
import matplotlib.pyplot as plt


def generate_sobol_normal_rvs_uncorrelated(num_time_steps, num_sim, scramble=True, seed=None):

    """
    Generate normal random variables using Sobol sequences.
    :param num_time_steps: Number of time steps in the discretisation.
    :param num_sim: Number of simulations.
    :param scramble: Boolean indicating whether to scramble the Sobol sequences. Default is True.
    :param seed: Scrambling seed. Default is None. In this case a random seed is selected. Default is None.
    :return: Array containing the random variables of shape [Number of Simulations, Number of time steps].
    """

    engine = SobolEngine(num_time_steps, scramble=scramble, seed=seed)
    totalW = si.norm.ppf(np.array(engine.draw(num_sim)) * (1 - 2e-7) + 1e-7)
    return totalW


def ConstantCorrelationCov(sigma, rho):
    """
    Generate covariance matrix.
    :param sigma: List of volatilities.
    :param rho: Correlation.
    :return: Covariance matrix.
    """

    dimS = len(sigma)
    corr = np.ones([dimS, dimS])
    corr *= rho
    np.fill_diagonal(corr, 1.0)
    cov = np.multiply(corr, np.outer(sigma, sigma))
    return cov


def BlackScholesExactSimulationSobolNDim(S0, r, sigma, cov, T, num_sim, num_time_steps, decomp_needed=True,
                                         scramble=False, seed=None):

    """
    Generate sample paths using exact solution of SDE.
    :param S0: Initial spot values.
    :param r: Risk-free interest rate.
    :param sigma: List of volatilities.
    :param cov: Covariance matrix.
    :param T: Time horizon.
    :param num_sim: Number of simulations.
    :param num_time_steps: Number of time steps
    :param decomp_needed: Boolean indicating whether Cholesky decomposition is needed. Default is True.
    :param scramble: Boolean indicating whether to scramble the Sobol sequences. Default is True.
    :param seed: Scrambling seed. Default is None. In this case a random seed is selected. Default is None.
    :return: List of sample paths. Output is of shape [Num time steps, Num Sim].
    """

    h = np.divide(T, num_time_steps)

    S = []

    dimS = len(S0)

    if decomp_needed:
        L = np.linalg.cholesky(cov)
        sqrtCov = L
    else:
        sqrtCov = cov

    totalW = generate_sobol_normal_rvs_uncorrelated(dimS * num_time_steps, num_sim, scramble, seed)
    totalW = np.array(np.hsplit(totalW, num_time_steps))

    currentPath = np.ones([num_sim, dimS]) * S0
    S.append(np.hstack((np.array(currentPath), np.ones(num_sim)[:, None])))

    for i in range(num_time_steps):
        totalW[i] = ((totalW[i] - np.average(totalW[i], 0)) / np.sqrt(np.var(totalW[i], 0)))
        currentPath *= np.exp((r - (0.5 * np.array(sigma) ** 2)) * h + (np.sqrt(h) * sqrtCov @ totalW[i].T).T)
        S.append(np.hstack((np.array(currentPath), (np.ones(num_sim) * (i + 2))[:, None])))
    return np.asarray(S)


def BlackScholesExactSimulationSobolNDim_autocallable(S0, r, sigma, cov, T_end, num_sim, dt, scramble=True, seed=None):
    """
    Generate sample paths using exact solution of SDE.
    :param S0: Initial spot values.
    :param r: Risk-free interest rate.
    :param sigma: List of volatilities.
    :param cov: Covariance matrix.
    :param T_end: Final fixing date.
    :param num_sim: Number of simulations.
    :param dt: Time increment.
    :param scramble: Boolean indicating whether to scramble the Sobol sequences. Default is True.
    :param seed: Scrambling seed. Default is None. In this case a random seed is selected. Default is None.
    :return: List of sample paths. Output is of shape [Num time steps, Num Sim].
    """

    out = []

    dimS = len(S0)

    L = np.linalg.cholesky(cov)
    sqrtCov = L

    num_time_steps = 0
    t = 0
    while t < T_end:
        t += dt
        num_time_steps += 1

    totalW = generate_sobol_normal_rvs_uncorrelated(dimS * num_time_steps, num_sim, scramble, seed)
    totalW = np.array(np.hsplit(totalW, num_time_steps))

    currentPath = np.ones([num_sim, dimS]) * S0
    out.append(np.hstack((np.array(currentPath), np.ones(num_sim)[:, None])))

    for i in range(num_time_steps):
        totalW[i] = ((totalW[i] - np.average(totalW[i], 0)) / np.sqrt(np.var(totalW[i], 0)))
        currentPath *= np.exp((r - (0.5 * np.array(sigma) ** 2)) * dt + (np.sqrt(dt) * sqrtCov @ totalW[i].T).T)
        out.append(np.hstack((np.array(currentPath), (np.ones(num_sim) * (i + 2))[:, None])))
    return np.asarray(out)
