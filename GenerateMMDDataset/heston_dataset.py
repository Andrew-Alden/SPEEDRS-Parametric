import numpy as np
import random
import torch
from GenerateMMDDataset.mmd_dataset_base import GenerateMMDDataset
from StochasticModels.heston_model import heston_sample_paths_inv


class GenerateHestonDistanceDataset(GenerateMMDDataset):

    """
    Class used to generate the Heston dataset. Inherits from mmd_dataset_base.py
    """

    def __init__(self, N, device, rbf_kernel_1_sigma, dyadic_order_1, rbf_kernel_2_sigma=None, dyadic_order_2=None,
                 _naive_solver=False, starting_value=0.5, time_series_index=0):

        """
        Constructor.
        :param N: Number of base stochastic processes.
        :param device: PyTorch device.
        :param rbf_kernel_1_sigma: Sigma parameter value for the RBF kernel corresponding to the first order Gram
                                   matrix.
        :param dyadic_order_1: Dyadic order corresponding to the first set of PDEs.
        :param rbf_kernel_2_sigma: Sigma parameter value for the RBF kernel corresponding to the second order Gram
                                   matrix. Default value is None. When this is None, the value will be set to
                                   rbf_kernel_1_sigma.
        :param dyadic_order_2: Dyadic order corresponding to the second set of PDEs. Default value is None. When this is
                               None, the value will be set to dyadic_order_1.
        :param _naive_solver: Boolean indicating whether to use the naive solver when solving the PDEs. Default value is
                              False.
        :param starting_value: To compute the MMD, all realisations will start from this value. The default is 0.5.
        :param time_series_index: Index of returned element from Heston function. Default is 0.
        """

        super(GenerateHestonDistanceDataset, self).__init__(N, device, rbf_kernel_1_sigma, dyadic_order_1,
                                                            rbf_kernel_2_sigma, dyadic_order_2, _naive_solver,
                                                            starting_value)

        self.time_series_index = time_series_index

    def generate_mmd_dataset(self, M, num_time_steps, num_sim, T_range, vol_of_vol_range, speed_range,
                             mean_volatility_range, v0_range, _round=2, order=2, lambda_=1e-5, estimator='b'):

        """
        Generate the MMD Dataset.
        :param M: Number of samples.
        :param num_time_steps: Number of discretisation steps.
        :param num_sim: Number of simulations.
        :param T_range: Tuple (T_min, T_max) specifying the range of maturities.
        :param vol_of_vol_range: Tuple (vol_of_vol_min, vol_of_vol_max) [(xi_min, xi_max)] specifying the
                                 range of the volatility of volatility.
        :param speed_range: Tuple (speed_min, speed_max) [(kappa_min, kappa_max)] specifying the range of the
                            speed of mean-reversion.
        :param mean_volatility_range: Tuple (mean_vol_min, mean_vol_max) [(theta_min, theta_max)] specifying the range
                                      of the long variance range.
        :param v0_range: Tuple (v0_min, v0_max) specifying the range of starting volatilities.
        :param _round: Number of decimal places to round S0, vol, and strike. Default is set to 2.
        :param order: Order of the MMD. Default is set to 2.
        :param lambda_: Parameter for calculating the inner product of the conditional kernel mean embeddings. Default
                        is set to 1e-5.
        :param estimator: 'b' for biased estimator, 'ub' for unbiased. Default is 'b'.
        :return: 1) mmd_dataset -> List of size M. Each element of the list is a list of N MMDs.
                 2) path_dataset -> path_dataset -> List of size M. Each element of the list is a tensor of shape
                                    [num_sim, num_time_steps + 1, 2] containing the path trajectories and time steps.
                 3) sample_path_param_dict -> Dictionary containing parameter values used to generate the dataset.
        """

        path_dataset, sample_path_param_dict = self._generate_heston_paths(M, num_time_steps, num_sim, T_range,
                                                                           vol_of_vol_range, speed_range,
                                                                           mean_volatility_range, v0_range, _round)

        sample_path_param_dict['N'] = self.N
        sample_path_param_dict['Num Time Steps'] = num_time_steps
        sample_path_param_dict['Num Sim'] = num_sim
        sample_path_param_dict['T Range'] = T_range
        sample_path_param_dict['Vol of Vol Range'] = vol_of_vol_range
        sample_path_param_dict['Speed Range'] = speed_range
        sample_path_param_dict['Mean Volatility Range'] = mean_volatility_range
        sample_path_param_dict['V0 Range'] = v0_range
        sample_path_param_dict['round'] = _round

        mmd_dataset = self._compute_distances(M, path_dataset, order, lambda_, estimator)

        return mmd_dataset, path_dataset, sample_path_param_dict

    def _generate_heston_paths(self, M, num_time_steps, num_sim, T_range, vol_of_vol_range, speed_range,
                               mean_volatility_range, v0_range, _round=2):

        """
        Generate the sample paths.
        :param M: Number of samples.
        :param num_time_steps: Number of discretisation steps.
        :param num_sim: Number of simulations.
        :param T_range: Tuple (T_min, T_max) specifying the range of maturities.
        :param vol_of_vol_range: Tuple (vol_of_vol_min, vol_of_vol_max) [(xi_min, xi_max)] specifying the
                                 range of the volatility of volatility.
        :param speed_range: Tuple (speed_min, speed_max) [(kappa_min, kappa_max)] specifying the range of the
                            speed of mean-reversion.
        :param mean_volatility_range: Tuple (mean_vol_min, mean_vol_max) [(theta_min, theta_max)] specifying the range
                                      of the long variance range.
        :param v0_range: Tuple (v0_min, v0_max) specifying the range of starting volatilities.
        :param _round: Number of decimal places to round S0, vol, and strike. Default is set to 2.
        :return: 1) path_dataset -> List of size M. Each element of the list is a tensor of shape
                                    [num_sim, num_time_steps + 1, 2] containing the path trajectories and time steps.
                 2) sample_path_param_dict -> Dictionary containing parameter values used to generate the dataset.
        """

        if T_range[0] == T_range[1]:
            T = [T_range[0] for _ in range(M)]
        else:
            T = np.random.randint(T_range[0], T_range[1], M)
        vol_of_vol = np.round(np.random.uniform(vol_of_vol_range[0], vol_of_vol_range[1], M), _round)
        speed = np.round(np.random.uniform(speed_range[0], speed_range[1], M), _round)
        mean_volatility = np.round(np.random.uniform(mean_volatility_range[0], mean_volatility_range[1], M), _round)
        correlation = np.round(np.random.uniform(-1, 1, M), _round)
        v0 = np.round(np.random.uniform(v0_range[0], v0_range[1], M), _round)

        path_dataset = [torch.transpose(torch.from_numpy(
            heston_sample_paths_inv(self.starting_value, v0[i], 0, correlation[i], mean_volatility[i], speed[i],
                                vol_of_vol[i], T[i], num_sim,
                                num_time_steps)[self.time_series_index]), 0, 1).to(device=self.device)
                        for i in range(M)]

        sample_path_param_dict = {
            'M': M,
            'T': T,
            'Vol of Vol': vol_of_vol,
            'Mean Vol': mean_volatility,
            'Speed': speed,
            'Correlation': correlation,
            'V0' : v0
        }

        return path_dataset, sample_path_param_dict
