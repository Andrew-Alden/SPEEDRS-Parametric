from RegressionModel.regression_model import RegressionModel
import pickle
import torch


class PricingModel(RegressionModel):

    """
    Class which handles the pricing model. Inherits from class RegressionModel.
    """

    def __init__(self, model_param_dict, training_param_dict, dataset_loader_params, loss_type, device,
                 scaler_file_name='scaler.pkl', num_additional_inputs=3, scheduler_step_size=50, scheduler_gamma=0.1,
                 use_scheduler=False, milestones=None, scheduler_type="multi_step"):

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
        :param loss_type: PyTorch or custom loss function.
        :param device: PyTorch device.
        :param scaler_file_name: The Scaler file name. Default is 'scaler.pkl'.
        :param num_additional_inputs: Number of inputs to scale. If X is the input, scale X[:-num_additional_inputs].
        :param scheduler_step_size: step size for the StepLR scheduler. Default is 50.
        :param scheduler_gamma: scheduler decay factor. Default is 0.1.
        :param use_scheduler: Boolean indicating whether to use a learning rate scheduler. Default is False.
        :param milestones: List of milestones for the MultiStepLR scheduler. Default is None.
        :param scheduler_type: String specifying which scheduler to use during training. Either 'step', 'exponential'
                               or 'multi_step'. Default is 'multi_step'.
        """

        super(PricingModel, self).__init__(model_param_dict, training_param_dict, dataset_loader_params, loss_type,
                                           device, scheduler_step_size, scheduler_gamma, use_scheduler, milestones,
                                           scheduler_type)

        self.scaler_file_name = scaler_file_name
        self.num_additional_inputs = num_additional_inputs
        self.get_delta = False

    def set_get_delta(self, get_delta):
        self.get_delta = get_delta

    def fit(self, X, y=None, **fit_params):

        """
        Fit the model to the data.
        :param X: Input to the model. Tensor of shape [Number of samples, Input dimension].
        :param y: Labels. Tensor of shape [Number of samples].
        :param fit_params: Dictionary of additional parameters.
        :return: Nothing
        """

        if 'num_additional_inputs' in fit_params:
            self.num_additional_inputs = fit_params['num_additional_inputs']
            del fit_params['num_additional_inputs']

        strike_spot_r_combs = X[:, -self.num_additional_inputs:]
        standard_strike_spot_r_combs = self.standard_scaler.fit_transform(strike_spot_r_combs)
        X[:, -self.num_additional_inputs:] = torch.from_numpy(standard_strike_spot_r_combs)

        pickle.dump(self.standard_scaler, open(self.scaler_file_name, 'wb'))

        self.train_model(X, y, **fit_params)

    def transform(self, X):

        """
        Transform the data.
        :param X: Input data. Tensor of shape [Number of samples, Input dimension].
        :return: Predicted output. Tensor of shape [Number of samples, 1].
        """

        strike_spot_r_combs = X[:, -self.num_additional_inputs:]
        standard_strike_spot_r_combs = self.standard_scaler.transform(strike_spot_r_combs)
        X[:, -self.num_additional_inputs:] = torch.from_numpy(standard_strike_spot_r_combs)
        X = X.to(device=self.device)
        if self.get_delta:
            X.requires_grad_()
            price = self.model(X)
            return price, torch.autograd.grad(price, X)

        return self.model(X)
