# Code to simulate the Heston process was adapted from https://github.com/HeKrRuTe/OptStopRandNN

import numpy as np


def volatility_drift(speed, prev_vol, mean_volatility):

    """
    Calculate the volatility drift.
    :param speed: Speed of mean-reversion (kappa).
    :param prev_vol: Volatility at the previous time step.
    :param mean_volatility: Long variance (theta).
    :return: Volatility drift.
    """

    return speed * (mean_volatility - prev_vol)


def volatility_diffusion(vol_of_vol, pre_vol):

    """
    Calculate the diffusion of the volatility process.
    :param vol_of_vol: Volatility of the volatility (xi).
    :param pre_vol: Volatility at the previous time step.
    :return: Volatility diffusion.
    """

    prev_vol_positive = np.maximum(pre_vol, 0.0)
    return vol_of_vol * np.sqrt(prev_vol_positive)


def asset_drift(prev_price, asset_drift):

    """
    Calculate the asset drift coefficient.
    :param prev_price: Asset price at the previous time step.
    :param asset_drift: Asset drift (mu).
    :return: Asset drift coefficient.
    """

    return prev_price * asset_drift


def asset_diffusion(prev_price, vol):

    """
    Calculate the asset diffusion.
    :param prev_price: Asset price at the previous time step.
    :param vol: Volatility at current time step.
    :return: Asset diffusion coefficient.
    """

    vol_positive = np.maximum(vol, 0.0)
    return prev_price * np.sqrt(vol_positive)


def heston_sample_paths(S0, v0, r, correlation, mean_volatility, speed, vol_of_vol, T, num_sim, num_time_steps,
                        barrier=None, strike=None):

    """
    Generate the Heston sample paths.
    :param S0: Initial spot rate.
    :param v0: Initial volatility.
    :param r: Risk-free interest rate.
    :param correlation: Brownian Motion correlation (rho).
    :param mean_volatility: Long variance (theta).
    :param speed: Speed of mean-reversion (kappa).
    :param vol_of_vol: Volatility of the volatility (xi).
    :param T: Maturity.
    :param num_sim: Number of simulations.
    :param num_time_steps: Number of time steps.
    :param barrier: Option barrier. Default is None.
    :param strike: Option strike.Default is None.
    :return: 1) Heston sample paths and time-steps.
                Array of dimension [Number of time steps + 1, Number of Simulations, 2].
             2) Volatility sample paths and time-steps.
                Array of dimension [Number of time steps + 1, Number of Simulations, 2].
             3) Heston and volatility sample paths along with time-steps.
                Array of dimension [Number of time steps + 1, Number of Simulations, 3].
             4) Array of final spot prices or strike. Intended use is for down-and-in options.
                If the running minimum is greater than Barrier, set the final spot price to the strike. By doing this,
                the European call option has a payoff of 0 at maturity. If barrier is None, None is returned.
    """


    h = np.divide(T, num_time_steps)
    normal_rvs_1 = np.random.normal(loc=0, scale=1, size=(num_time_steps, num_sim))
    normal_rvs_2 = np.random.normal(loc=0, scale=1, size=(num_time_steps, num_sim))
    dW = normal_rvs_1 * np.sqrt(h)
    dZ = (correlation * normal_rvs_1 + np.sqrt(1 - correlation ** 2) * normal_rvs_2) * np.sqrt(h)

    S = np.ones(num_sim) * S0
    V = np.ones(num_sim) * v0

    sample_paths = []
    sample_volatility_paths = []
    sample_price_vol_paths = []

    sample_paths.append(np.hstack(((S.copy())[:, None], np.zeros(num_sim)[:, None])))
    sample_volatility_paths.append(np.hstack(((V.copy())[:, None], np.zeros(num_sim)[:, None])))

    sample_price_vol_paths.append(np.hstack(((S.copy())[:, None], (V.copy())[:, None], np.zeros(num_sim)[:, None])))

    for i in range(num_time_steps):
        V += volatility_drift(speed, V, mean_volatility) * h + np.multiply(volatility_diffusion(vol_of_vol, V), dZ[i])
        S += asset_drift(S, r) * h + np.multiply(asset_diffusion(S, V), dW[i])

        sample_paths.append(np.hstack(((S.copy())[:, None], (np.ones(num_sim) * ((i + 1) * h))[:, None])))
        sample_volatility_paths.append(np.hstack(((V.copy())[:, None], (np.ones(num_sim) * ((i + 1) * h))[:, None])))
        sample_price_vol_paths.append(
            np.hstack(((S.copy())[:, None], (V.copy())[:, None], (np.ones(num_sim) * ((i + 1) * h))[:, None])))

    heston_down_in = None

    if barrier is not None:
        path_array = np.asarray(sample_paths)[:, :, 0].copy()
        running_min = np.amin(path_array, axis=0)

        heston_down_in = np.where(running_min > barrier, strike, path_array[-1, :])

    return np.asarray(sample_paths), np.asarray(sample_volatility_paths), np.asarray(
        sample_price_vol_paths), heston_down_in


def heston_sample_paths_v2(S0, v0, r, correlation, mean_volatility, speed, vol_of_vol, T, num_sim, num_time_steps,
                               barrier=None, strike=None):
    """
    Generate the Heston sample paths.
    :param S0: Initial spot rate.
    :param v0: Initial volatility.
    :param r: Risk-free interest rate.
    :param correlation: Brownian Motion correlation (rho).
    :param mean_volatility: Long variance (theta).
    :param speed: Speed of mean-reversion (kappa).
    :param vol_of_vol: Volatility of the volatility (xi).
    :param T: Maturity.
    :param num_sim: Number of simulations.
    :param num_time_steps: Number of time steps.
    :param barrier: Option barrier. Default is None.
    :param strike: Option strike.Default is None.
    :return: 1) Heston sample paths and time-steps.
                Array of dimension [Number of time steps + 1, Number of Simulations, 2].
             2) Volatility sample paths and time-steps.
                Array of dimension [Number of time steps + 1, Number of Simulations, 2].
             3) Heston and volatility sample paths along with time-steps.
                Array of dimension [Number of time steps + 1, Number of Simulations, 3].
             4) Array of final spot prices or strike. Intended use is for down-and-in options.
                If the running minimum is greater than Barrier, set the final spot price to the strike. By doing this,
                the European call option has a payoff of 0 at maturity. If barrier is None, None is returned.
    """

    h = np.divide(T, num_time_steps)
    normal_rvs_1 = np.random.normal(loc=0, scale=1, size=(num_time_steps, num_sim))
    normal_rvs_2 = np.random.normal(loc=0, scale=1, size=(num_time_steps, num_sim))
    dW = normal_rvs_1 * np.sqrt(h)
    dZ = (correlation * normal_rvs_1 + np.sqrt(1 - correlation ** 2) * normal_rvs_2) * np.sqrt(h)

    S = np.ones((num_time_steps + 1, num_sim))
    V = np.ones((num_time_steps + 1, num_sim))

    S[0, :] = np.ones(num_sim) * S0
    V[0, :] = np.ones(num_sim) * v0

    time_steps = [0]

    for i in range(1, num_time_steps + 1):
        V[i, :] = V[i - 1, :] + volatility_drift(speed, V[i - 1, :], mean_volatility) * h + \
                  np.multiply(volatility_diffusion(vol_of_vol, V[i - 1, :]), dZ[i - 1])

        S[i, :] = S[i - 1, :] + asset_drift(S[i - 1, :], r) * h + np.multiply(asset_diffusion(S[i - 1, :], V[i, :]),
                                                                              dW[i - 1])

        time_steps.append(i * h)

    sample_paths = np.concatenate((S[:, :, None],
                                   np.repeat(np.asarray(time_steps)[:, None, None], repeats=num_sim,
                                             axis=1)),
                                  axis=2)

    sample_volatility_paths = np.concatenate((V[:, :, None],
                                              np.repeat(np.asarray(time_steps)[:, None, None], repeats=num_sim,
                                                        axis=1)),
                                             axis=2)

    sample_price_vol_paths = np.concatenate((S[:, :, None], V[:, :, None],
                                             np.repeat(np.asarray(time_steps)[:, None, None], repeats=num_sim,
                                                       axis=1)),
                                            axis=2)

    heston_down_in = None

    if barrier is not None:
        path_array = np.asarray(sample_paths)[:, :, 0].copy()
        running_min = np.amin(path_array, axis=0)

        heston_down_in = np.where(running_min > barrier, strike, path_array[-1, :])

    return sample_paths, sample_volatility_paths, sample_price_vol_paths, heston_down_in


def heston_sample_paths_inv(S0, v0, r, correlation, mean_volatility, speed, vol_of_vol, T, num_sim, num_time_steps,
                            barrier=None, strike=None):

    """
    Generate the Heston sample paths.
    :param S0: Initial spot rate.
    :param v0: Initial volatility.
    :param r: Risk-free interest rate.
    :param correlation: Brownian Motion correlation (rho).
    :param mean_volatility: Long variance (theta).
    :param speed: Speed of mean-reversion (kappa).
    :param vol_of_vol: Volatility of the volatility (xi).
    :param T: Maturity.
    :param num_sim: Number of simulations.
    :param num_time_steps: Number of time steps.
    :param barrier: Option barrier. Default is None.
    :param strike: Option strike.Default is None.
    :return: 1) Heston sample paths and time-steps.
                Array of dimension [Number of time steps + 1, Number of Simulations, 2].
             2) Volatility sample paths and time-steps.
                Array of dimension [Number of time steps + 1, Number of Simulations, 2].
             3) Heston and volatility sample paths along with time-steps.
                Array of dimension [Number of time steps + 1, Number of Simulations, 3].
             4) Array of final spot prices or strike. Intended use is for down-and-in options.
                If the running minimum is greater than Barrier, set the final spot price to the strike. By doing this,
                the European call option has a payoff of 0 at maturity. If barrier is None, None is returned.
    """

    h = np.divide(T, num_time_steps)
    normal_rvs_1 = np.random.normal(loc=0, scale=1, size=(num_time_steps, num_sim))
    normal_rvs_2 = np.random.normal(loc=0, scale=1, size=(num_time_steps, num_sim))
    dW = normal_rvs_1 * np.sqrt(h)
    dZ = (correlation * normal_rvs_1 + np.sqrt(1 - correlation ** 2) * normal_rvs_2) * np.sqrt(h)

    S = np.ones(num_sim) * S0
    V = np.ones(num_sim) * v0

    sample_paths = []
    sample_volatility_paths = []
    sample_price_vol_paths = []

    sample_paths.append(np.hstack(((S.copy())[:, None], np.zeros(num_sim)[:, None])))
    sample_volatility_paths.append(np.hstack(((V.copy())[:, None], np.zeros(num_sim)[:, None])))

    sample_price_vol_paths.append(np.hstack(((S.copy())[:, None], (V.copy())[:, None], np.zeros(num_sim)[:, None])))

    for i in range(num_time_steps):
        S += asset_drift(S, r) * h + np.multiply(asset_diffusion(S, V), dW[i])
        V += volatility_drift(speed, V, mean_volatility) * h + np.multiply(volatility_diffusion(vol_of_vol, V), dZ[i])

        sample_paths.append(np.hstack(((S.copy())[:, None], (np.ones(num_sim) * ((i + 1) * h))[:, None])))
        sample_volatility_paths.append(np.hstack(((V.copy())[:, None], (np.ones(num_sim) * ((i + 1) * h))[:, None])))
        sample_price_vol_paths.append(
            np.hstack(((S.copy())[:, None], (V.copy())[:, None], (np.ones(num_sim) * ((i + 1) * h))[:, None])))

    heston_down_in = None

    if barrier is not None:
        path_array = np.asarray(sample_paths)[:, :, 0].copy()
        running_min = np.amin(path_array, axis=0)

        heston_down_in = np.where(running_min > barrier, strike, path_array[-1, :])

    return np.asarray(sample_paths), np.asarray(sample_volatility_paths), np.asarray(
        sample_price_vol_paths), heston_down_in


def heston_sample_paths_inv_v2(S0, v0, r, correlation, mean_volatility, speed, vol_of_vol, T, num_sim, num_time_steps,
                               barrier=None, strike=None):

    """
    Generate the Heston sample paths.
    :param S0: Initial spot rate.
    :param v0: Initial volatility.
    :param r: Risk-free interest rate.
    :param correlation: Brownian Motion correlation (rho).
    :param mean_volatility: Long variance (theta).
    :param speed: Speed of mean-reversion (kappa).
    :param vol_of_vol: Volatility of the volatility (xi).
    :param T: Maturity.
    :param num_sim: Number of simulations.
    :param num_time_steps: Number of time steps.
    :param barrier: Option barrier. Default is None.
    :param strike: Option strike.Default is None.
    :return: 1) Heston sample paths and time-steps.
                Array of dimension [Number of time steps + 1, Number of Simulations, 2].
             2) Volatility sample paths and time-steps.
                Array of dimension [Number of time steps + 1, Number of Simulations, 2].
             3) Heston and volatility sample paths along with time-steps.
                Array of dimension [Number of time steps + 1, Number of Simulations, 3].
             4) Array of final spot prices or strike. Intended use is for down-and-in options.
                If the running minimum is greater than Barrier, set the final spot price to the strike. By doing this,
                the European call option has a payoff of 0 at maturity. If barrier is None, None is returned.
    """

    h = np.divide(T, num_time_steps)
    normal_rvs_1 = np.random.normal(loc=0, scale=1, size=(num_time_steps, num_sim))
    normal_rvs_2 = np.random.normal(loc=0, scale=1, size=(num_time_steps, num_sim))
    dW = normal_rvs_1 * np.sqrt(h)
    dZ = (correlation * normal_rvs_1 + np.sqrt(1 - correlation ** 2) * normal_rvs_2) * np.sqrt(h)

    S = np.ones((num_time_steps+1, num_sim))
    V = np.ones((num_time_steps+1, num_sim))

    S[0, :] = np.ones(num_sim) * S0
    V[0, :] = np.ones(num_sim) * v0

    time_steps = [0]

    for i in range(1, num_time_steps+1):
        S[i, :] = S[i-1, :] + asset_drift(S[i-1, :], r) * h + np.multiply(asset_diffusion(S[i-1, :], V[i-1, :]),
                                                                          dW[i-1])
        V[i, :] = V[i-1, :] + volatility_drift(speed, V[i-1, :], mean_volatility) * h +\
                  np.multiply(volatility_diffusion(vol_of_vol, V[i-1, :]), dZ[i-1])
        time_steps.append(i * h)

    sample_paths = np.concatenate((S[:, :, None],
                                   np.repeat(np.asarray(time_steps)[:, None, None], repeats=num_sim,
                                             axis=1)),
                                  axis=2)

    sample_volatility_paths = np.concatenate((V[:, :, None],
                                   np.repeat(np.asarray(time_steps)[:, None, None], repeats=num_sim,
                                             axis=1)),
                                  axis=2)

    sample_price_vol_paths = np.concatenate((S[:, :, None], V[:, :, None],
                                   np.repeat(np.asarray(time_steps)[:, None, None], repeats=num_sim,
                                             axis=1)),
                                  axis=2)

    heston_down_in = None

    if barrier is not None:
        path_array = np.asarray(sample_paths)[:, :, 0].copy()
        running_min = np.amin(path_array, axis=0)

        heston_down_in = np.where(running_min > barrier, strike, path_array[-1, :])

    return sample_paths, sample_volatility_paths, sample_price_vol_paths, heston_down_in
