import numpy as np
import random
import torch
from GenerateMMDDataset.mmd_dataset_base import GenerateMMDDataset
from StochasticModels.geometric_brownian_motion import BlackScholesExactSimulationSobolNDim, \
    ConstantCorrelationCov


class GenerateMultiDimensionalGBM(GenerateMMDDataset):

    """
    Class used to generate the Multi-Dimensional GBM dataset. Inherits from mmd_dataset_base.py
    """

    def __init__(self, N, device, rbf_kernel_1_sigma, dyadic_order_1, dim, rbf_kernel_2_sigma=None, dyadic_order_2=None,
                 _naive_solver=False, starting_value=0.5):

        """
        Constructor.
        :param N: Number of base stochastic processes.
        :param device: PyTorch device.
        :param rbf_kernel_1_sigma: Sigma parameter value for the RBF kernel corresponding to the first order Gram
                                   matrix.
        :param dyadic_order_1: Dyadic order corresponding to the first set of PDEs.
        :param dim: Number of assets.
        :param rbf_kernel_2_sigma: Sigma parameter value for the RBF kernel corresponding to the second order Gram
                                   matrix. Default value is None. When this is None, the value will be set to
                                   rbf_kernel_1_sigma.
        :param dyadic_order_2: Dyadic order corresponding to the second set of PDEs. Default value is None. When this is
                               None, the value will be set to dyadic_order_1.
        :param _naive_solver: Boolean indicating whether to use the naive solver when solving the PDEs. Default value is
                              False.
        :param starting_value: To compute the MMD, all realisations will start from this value. The default is 0.5.
        """

        super(GenerateMultiDimensionalGBM, self).__init__(N, device, rbf_kernel_1_sigma, dyadic_order_1,
                                                          rbf_kernel_2_sigma, dyadic_order_2, _naive_solver,
                                                          starting_value)

        self.dim = dim
        self.starting_values = [starting_value for _ in range(dim)]

    def generate_mmd_dataset(self, M, num_time_steps, num_sim, T_range, vol_range, corr_range=(0.05, 0.95),
                             _round=2, order=2, lambda_=1e-5, estimator='b'):

        """
        Generate the sample paths.
        :param M: Number of samples.
        :param num_time_steps: Number of discretisation steps
        :param num_sim: Number of simulations.
        :param T_range: Tuple (T_min, T_max) specifying the range of maturities.
        :param vol_range: Tuple (vol_min, vol_max) specifying the volatility range.
        :param corr_range: Tuple (corr_min, corr_max) specifying the correlation range. Default is
                           (0.05, 0.95).
        :param _round: Number of decimal places to round S0, vol, and strike. Default is set to 2.
        :param order: Order of MMD.
        :param lambda_: Parameter for calculating the inner product of the prediction conditional kernel mean
                        embeddings. Default is set to 1e-5.
        :param estimator: 'b' for biased estimator, 'ub' for unbiased. The default is 'b'.
        :return: 1) mmd_dataset -> List of size M. Each element of the list is a list of N MMDs.
                 2) path_dataset -> path_dataset -> List of size M. Each element of the list is a tensor of shape
                                    [num_sim, num_time_steps + 1, Number of Assets + 1] containing the path
                                    trajectories and time steps.
                3) sample_path_param_dict -> Dictionary containing parameter values used to generate the dataset.
        """

        path_dataset, sample_path_param_dict = self._generate_multi_dim_gbm_paths(M, num_time_steps, num_sim,
                                                                                  T_range, vol_range, corr_range,
                                                                                  _round)

        sample_path_param_dict['N'] = self.N
        sample_path_param_dict['Num Time Steps'] = num_time_steps
        sample_path_param_dict['Num Sim'] = num_sim
        sample_path_param_dict['T Range'] = T_range
        sample_path_param_dict['Vol Range'] = vol_range
        sample_path_param_dict['Correlation Range'] = corr_range
        sample_path_param_dict['round'] = _round

        mmd_dataset = self._compute_distances(M, path_dataset, order, lambda_, estimator)

        return mmd_dataset, path_dataset, sample_path_param_dict

    def _generate_multi_dim_gbm_paths(self, M, num_time_steps, num_sim, T_range, vol_range, corr_range=(0.05, 0.95),
                                      _round=2):

        """
        Generate the sample paths.
        :param M: Number of samples.
        :param num_time_steps: Number of discretisation steps
        :param num_sim: Number of simulations.
        :param T_range: Tuple (T_min, T_max) specifying the range of maturities.
        :param vol_range: Tuple (vol_min, vol_max) specifying the volatility range.
        :param corr_range: Tuple (corr_min, corr_max) specifying the correlation range. Default is
                           (0.05, 0.95).
        :param _round: Number of decimal places to round S0, vol, and strike. Default is set to 2.
        :return: 1) path_dataset -> List of size M. Each element of the list is a tensor of shape
                                    [num_sim, num_time_steps + 1, Number of Assets + 1] containing the path trajectories
                                     and time steps.
                 2) sample_path_param_dict -> Dictionary containing parameter values used to generate the dataset.
        """

        if T_range[0] == T_range[1]:
            T = np.asarray([1 for _ in range(M)])
        else:
            T = random.sample(range(T_range[0], T_range[1]), M)

        sigma_list = [np.round(np.random.uniform(vol_range[0], vol_range[1], self.dim), 2) for _ in range(M)]
        correlation = np.round(np.random.uniform(corr_range[0], corr_range[1], M), _round)

        path_dataset = [torch.transpose(torch.from_numpy(
            BlackScholesExactSimulationSobolNDim(self.starting_values, 0.0, sigma_list[i],
                                                 ConstantCorrelationCov(sigma_list[i], correlation[i]),
                                                 T[i], num_sim, num_time_steps, True)), 0, 1).to(
            device=self.device) for i in range(M)]

        sample_path_param_dict = {
            'M': M,
            'Dim': self.dim,
            'T': T,
            'Sigma': sigma_list,
            'Correlation': correlation
        }

        return path_dataset, sample_path_param_dict
