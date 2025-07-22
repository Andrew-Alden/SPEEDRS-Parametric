from MMD.mmd import RBFKernel, SigKernel
import torch
from tqdm.auto import tqdm
import pickle


class GenerateMMDDataset:

    """
    Base Class for the MMD datasets.
    """

    def __init__(self, N, device, rbf_kernel_1_sigma, dyadic_order_1, rbf_kernel_2_sigma=None, dyadic_order_2=None,
                 _naive_solver=False, starting_value=0.5):

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
        """

        self.N = N
        self.device = device
        self.rbf_kernel_1_sigma = rbf_kernel_1_sigma
        self.dyadic_order_1 = dyadic_order_1
        self.dyadic_order_2 = dyadic_order_2
        self._naive_solver = _naive_solver
        self.starting_value = starting_value

        if rbf_kernel_2_sigma is None:
            self.rbf_kernel_2_sigma = rbf_kernel_1_sigma
        else:
            self.rbf_kernel_2_sigma = rbf_kernel_2_sigma

        if dyadic_order_2 is None:
            self.dyadic_order_2 = dyadic_order_1
        else:
            self.dyadic_order_2 = dyadic_order_2

        self.signature_kernel = self.construct_signature_kernel()

    def construct_signature_kernel(self):

        """
        Construct the signature kernel object used to compute the MMD.
        :return: SigKernel object.
        """

        static_kernel_1 = RBFKernel(sigma=self.rbf_kernel_1_sigma)
        static_kernel_2 = RBFKernel(sigma=self.rbf_kernel_2_sigma)
        return SigKernel([static_kernel_1, static_kernel_2], [self.dyadic_order_1, self.dyadic_order_2],
                         self._naive_solver)

    def _compute_distances(self, M, path_dataset, order=2, lambda_=1e-5, estimator='b'):

        """
        Compute the MMD distances.
        :param M: Number of samples.
        :param path_dataset: List of size M. Each element of the list is a tensor of shape
                             [num_sim, num_time_steps + 1, 2] containing the path trajectories and time steps.
        :param order: Order of the MMD. Default is 2.
        :param lambda_: Parameter for calculating the inner product of the prediction conditional kernel mean
                        embeddings. Default is set to 1e-5.
        :param estimator: 'b' for biased estimator, 'ub' for unbiased. The default is 'b'.
        :return: List of size M. Each element of the list is a list of N MMDs.
        """

        mmd_dataset = [[None for _ in range(self.N)] for _ in range(M)]

        print(f'Computing Initial Distances')

        for i in tqdm(range(self.N)):
            mmd_dataset[i][i] = torch.maximum(self.signature_kernel.compute_mmd(path_dataset[i], path_dataset[i],
                                                                                lambda_=lambda_, order=order,
                                                                                estimator=estimator),
                                              torch.tensor(0.0))
            for j in range(i + 1, self.N):
                distance = torch.maximum(self.signature_kernel.compute_mmd(path_dataset[i], path_dataset[j],
                                                                           lambda_=lambda_, order=order,
                                                                           estimator=estimator),
                                         torch.tensor(0.0))
                mmd_dataset[i][j] = distance
                mmd_dataset[j][i] = distance

        print(f'Finished Computing Initial Distances')
        print(f'{"-" * 100}\n\n')
        print(f'Computing Next Batch of Distances')

        for i in tqdm(range(self.N, M)):
            for j in range(self.N):
                mmd_dataset[i][j] = torch.maximum(self.signature_kernel.compute_mmd(path_dataset[i], path_dataset[j],
                                                                                    lambda_=lambda_, order=order,
                                                                                    estimator=estimator),
                                                  torch.tensor(0.0))

        print(f'Finished Computing All Distances')
        print(f'{"-" * 100}\n\n')

        return mmd_dataset

    def _compute_augmented_distances(self, M, old_paths, path_dataset, order=2, lambda_=1e-5, estimator='b'):

        """
        Compute the MMD distances.
        :param M: Number of samples.
        :param old_paths: List of path trajectories. The first N elements of the list contain trajectories from the base
                          stochastic processes. The MMD between the base paths and the new paths will be computed.
        :param path_dataset: List of size M. Each element of the list is a tensor of shape
                             [num_sim, num_time_steps + 1, 2] containing the path trajectories and time steps.
        :param order: Order of the MMD. Default is 2.
        :param lambda_: Parameter for calculating the inner product of the prediction conditional kernel mean
                        embeddings. Default is set to 1e-5.
        :param estimator: 'b' for biased estimator, 'ub' for unbiased. The default is 'b'.
        :return: List of size M. Each element of the list is a list of N MMDs.
        """

        mmd_dataset = [[None for _ in range(self.N)] for _ in range(M)]

        print(f'Computing Distances')

        for i in tqdm(range(M)):
            for j in tqdm(range(self.N), leave=False):
                distance = self.signature_kernel.compute_mmd(path_dataset[i][0], old_paths[j][0], lambda_=lambda_,
                                                             order=order, estimator=estimator)
                mmd_dataset[i][j] = torch.maximum(distance, torch.tensor(0.0))

        print(f'Finished Computing Distances')
        print(f'{"-" * 100}\n\n')

        return mmd_dataset


# ===========================================================================================================


# ===========================================================================================================
# Various utility functions to save and load the datasets.
# ===========================================================================================================

def save_dataset(path_dataset, mmd_dataset, original_paths_name, mmd_dataset_name):

    """
    Function: Save the path and MMD dataset.
    Parameters: * path_dataset -> The path dataset.
                * mmd_dataset -> The MMD dataset.
                * original_paths_name -> Path dataset file name.
                * mmd_dataset_name -> MMD dataset file name.
    """

    torch.save(path_dataset, f'{original_paths_name}.pt')
    torch.save(mmd_dataset, f'{mmd_dataset_name}.pt')


def load_dataset(original_paths_name, mmd_dataset_name):

    """
    Function: Load the dataset.
    Parameters: * original_paths_name -> Path dataset file name.
                * mmd_dataset_name -> MMD dataset file name.
    Returns: 1) MMD dataset.
             2) Path dataset.
    """

    return torch.load(f'{mmd_dataset_name}.pt'), torch.load(f'{original_paths_name}.pt')


def save_path_params(sample_path_param_dict, file_name):

    """
    Function: Save the param dictionary.
    Parameters: * sample_path_param_dict -> The param dictionary.
                * file_name -> Param dictionary file name.
    """

    with open(file_name, 'wb') as f:
        pickle.dump(sample_path_param_dict, f)


def load_path_params(file_name):

    """
    Function: Load the param dictionary.
    Parameters: * file_name -> The param dictionary file name.
    Returns: The param dictionary.
    """

    with open(file_name, 'rb') as f:
        return pickle.load(f)

# ===========================================================================================================
