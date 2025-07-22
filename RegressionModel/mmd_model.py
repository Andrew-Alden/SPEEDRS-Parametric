from RegressionModel.regression_model import RegressionModel
import pickle
import torch


class SecondOrderMMDApprox(RegressionModel):

    """
    Class which handles the Second Order MMD Approximation. Inherits from class RegressionModel.
    """

    def __init__(self, model_param_dict, training_param_dict=None, dataset_loader_params=None, loss_type=None,
                 device="cpu", scaler_file_name='mmd_scaler.pkl'):

        """
        Constructor.
        :param model_param_dict: Dictionary specifying the regression model architecture. The keys are:
                                 -> input_dimension - input dimension of the model
                                 -> intermediate_dimensions - list of hidden layer dimensions
                                 -> activation_functions - list of activation functions
                                 -> add_layer_norm - list of Booleans indicating whether to add layer normalisation
                                                     before a neural network layer
                                 -> output_dimension - output dimension of the model
                                 -> output_activation_fn - output layer activation function
        :param training_param_dict: Dictionary specifying the model training procedure. The keys are:
                                    -> lr - learning rate
                                    -> Epochs - number of epochs
                                    -> l2_weight - L2-regularisation weight
                                    -> l1_weight - L1-regularisation weight
                                    -> Train/Val Split - Percentage of data used for training as opposed to validation
                                    -> exp_sigma - sigma parameter for exponential smoothing of distances
        :param dataset_loader_params: Dictionary specifying the loading of the dataset. The keys are:
                                      -> batch_size - the batch size
                                      -> shuffle - Boolean indicating whether to shuffle the order when loading the data
                                      -> num_workers - specify how many processes are simultaneously loading the data.
                                                       If num_workers=0, the main process loads the data.
        :param loss_type: PyTorch or custom loss function. Default is None.
        :param device: PyTorch device. Default is "cpu".
        :param scaler_file_name: The Scaler file name. Default is 'mmd_scaler.pkl'.
        """

        super(SecondOrderMMDApprox, self).__init__(model_param_dict, training_param_dict, dataset_loader_params,
                                                   loss_type, device)

        self.scaler_file_name = scaler_file_name

    def fit(self, X, y=None, **fit_params):

        """
        Fit the model to the data.
        :param X: Input to the model. Tensor of shape [Number of samples, Input dimension].
        :param y: Labels. Tensor of shape [Number of samples].
        :param fit_params: Dictionary of additional parameters.
        :return: Nothing
        """

        # Standardise the features.
        scaled_features = self.standard_scaler.fit_transform(X)

        pickle.dump(self.standard_scaler, open(self.scaler_file_name, 'wb'))

        self.train_model(scaled_features, y, **fit_params)

    def transform(self, X):

        """
        Transform the data.
        :param X: Input data. Tensor of shape [Number of samples, Input dimension].
        :return: Predicted output. Tensor of shape [Number of samples, 1].
        """

        X = self.standard_scaler.transform(X)
        X = torch.tensor(X).to(device=self.device)
        return self.model(X.float())
