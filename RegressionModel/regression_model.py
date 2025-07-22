import torch
from torch import nn
import torch.cuda
from torch.optim.lr_scheduler import ExponentialLR
from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import MultiStepLR
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin

from tqdm.auto import tqdm
import pickle
import shutil
from collections import defaultdict
import numpy as np


class DatasetDataLoader(torch.utils.data.Dataset):

    """
    Class responsible for loading the dataset with PyTorch compatibility. Inherits from torch.utils.data.Dataset
    Code adapted from https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel
    """

    def __init__(self, IDs, features, labels):

        """
        Constructor.
        :param IDs: List of indices.
        :param features: Feature vectors.
        :param labels: Output labels (derivative price).
        """

        super(DatasetDataLoader, self).__init__()

        self.IDs = IDs
        self.features = features
        self.labels = labels

    def __len__(self):

        """
        Get dataset length.
        :return: Length.
        """

        return len(self.IDs)

    def __getitem__(self, index):

        """
        Get a dataset item.
        :param index: Retrieve feature and label at position IDs[index].
        :return: Feature vector and label corresponding to the ID at position index.
        """

        ID = self.IDs[index]

        X = self.features[ID]
        y = self.labels[ID]

        return X, y


class RegressionNeuralNetwork(nn.Module):

    """
    Class handling the regression neural network. Inherits from the class nn.Module.
    """

    def __init__(self, input_dimension, intermediate_dimensions, activation_functions, add_layer_norm, output_dimension,
                 output_activation_fn=None, use_input_layer_norm=False, use_output_layer_norm=False,
                 use_residual=False):

        """
        Constructor.
        :param input_dimension: Input dimension of the model.
        :param intermediate_dimensions: List of hidden unit dimensions.
        :param activation_functions: List of activation functions equal in length to the number of hidden units + 1.
                                     If no activation function is used between layer i and layer (i+1), set
                                     activation_functions[i] to None.
        :param add_layer_norm: List of Booleans indicating whether to use layer normalisation before each hidden unit.
        :param output_dimension: The output dimension.
        :param output_activation_fn: The output activation function. Default is None.
        :param use_input_layer_norm: Boolean indicating whether to pass the input through a layer normalisation layer.
        :param use_output_layer_norm: Boolean indicating whether to use layer normalisation before the output layer.
        :param use_residual: Boolean indicating whether to apply a residual block from the input to the output of the
                             layer before the final layer. For this to work, the last element of intermediate_dimensions
                             must equal input_dimension.
        """

        super(RegressionNeuralNetwork, self).__init__()

        self.use_residual = use_residual
        if use_residual:
            assert input_dimension == intermediate_dimensions[-1]

        self.use_input_layer_norm = use_input_layer_norm
        self.input_layer_norm = nn.LayerNorm(input_dimension)

        self.input_layer = nn.Linear(input_dimension, intermediate_dimensions[0])

        if activation_functions[0] is not None:
            self.input_activation = activation_functions[0]
        else:
            self.input_activation = None

        module_list = []
        for i in range(len(intermediate_dimensions) - 1):
            if add_layer_norm[i]:
                module_list.append(nn.LayerNorm(intermediate_dimensions[i]))
            module_list.append(nn.Linear(intermediate_dimensions[i], intermediate_dimensions[i + 1]))
            if activation_functions[i + 1] is not None:
                module_list.append(activation_functions[i + 1])
        self.layers_list = nn.ModuleList(module_list)

        self.use_output_layer_norm = use_output_layer_norm
        self.output_layer_norm = nn.LayerNorm(intermediate_dimensions[-1])
        self.output_layer = nn.Linear(intermediate_dimensions[-1], output_dimension)
        if output_activation_fn is not None:
            self.output_activation = output_activation_fn
        else:
            self.output_activation = None

    def forward(self, x):

        """
        Forward pass through the network.
        :param x: Input to the model. Tensor of shape [Batch Size, Input Dimension].
        :return: Output of the model. Tensor of shape [Batch Size, Output Dimension].
        """

        if self.use_input_layer_norm:
            output = self.input_layer_norm(x)
            output = self.input_layer(output)
        else:
            output = self.input_layer(x)

        if self.input_activation is not None:
            output = self.input_activation(output)

        for i, l in enumerate(self.layers_list):
            output = l(output)

        if self.use_residual:
            output = output + x

        if self.use_output_layer_norm:
            output = self.output_layer_norm(output)

        output = self.output_layer(output)
        if self.output_activation is not None:
            output = self.output_activation(output)

        return output


class RegressionModel(BaseEstimator, TransformerMixin):

    """
    Class which handles the regression model. Inherits from the sklearn classes BaseEstimator and TransformerMixin.
    """

    def __init__(self, model_param_dict, training_param_dict, dataset_loader_params, loss_type, device,
                 scheduler_step_size=50, scheduler_gamma=0.1, use_scheduler=False, milestones=None,
                 scheduler_type="multi_step"):

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
        :param scheduler_step_size: step size for the StepLR scheduler. Default is 50.
        :param scheduler_gamma: scheduler decay factor. Default is 0.1.
        :param use_scheduler: Boolean indicating whether to use a learning rate scheduler. Default is False.
        :param milestones: List of milestones for the MultiStepLR scheduler. Default is None.
        :param scheduler_type: String specifying which scheduler to use during training. Either 'step', 'exponential'
                               or 'multi_step'. Default is 'multi_step'.
        """

        self.model_param_dict = model_param_dict
        self.training_param_dict = training_param_dict
        self.dataset_loader_params = dataset_loader_params
        self.standard_scaler = StandardScaler()
        self.model = RegressionNeuralNetwork(**self.model_param_dict)
        self.device = device
        self.model.to(self.device)
        self.criterion = loss_type
        if self.training_param_dict is not None:
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.training_param_dict['lr'])
            self.use_scheduler = use_scheduler
            if self.use_scheduler:
                self.scheduler_type = scheduler_type
                if scheduler_type == "step":
                    self.scheduler = StepLR(self.optimizer, step_size=scheduler_step_size, gamma=scheduler_gamma)
                elif scheduler_type == "exponential":
                    self.scheduler = ExponentialLR(self.optimizer, gamma=scheduler_gamma)
                else:
                    self.scheduler = MultiStepLR(self.optimizer, milestones=milestones, gamma=scheduler_gamma)

    def fit(self, X, y=None, **fit_params):

        """
        Fit the model to the data.
        :param X: Input to the model. Tensor of shape [Number of samples, Input dimension].
        :param y: Labels. Tensor of shape [Number of samples].
        :param fit_params: Dictionary of additional input.
        :return: NotImplementedError
        """

        raise NotImplementedError

    def transform(self, X):

        """
        Transform the data.
        :param X: Input data. Tensor of shape [Number of samples, Input dimension].
        :return: NotImplementedError
        """

        raise NotImplementedError

    # @staticmethod
    def generate_train_valid_partition(self, train_val_split, num_samples):

        """
        Partition the dataset into train and validation. Static method.
        :param train_val_split: Percentage of data in the training set.
        :param num_samples: Number of samples in the dataset.
        :return: Dictionary with keys 'train' and 'validation'. The value corresponding to each key is a list of indices
                 used to partition the dataset.
        """

        partition = defaultdict(list)

        random_nums = np.random.uniform(0, 1, num_samples)

        for i in range(num_samples):

            if random_nums[i] < train_val_split:

                partition['train'].append(i)

            else:

                partition['validation'].append(i)

        return partition

    @staticmethod
    def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', best_model_filename='model_best.pth.tar'):

        """
        Save checkpoint. Static method.
        :param state: Model and training state dictionary.
        :param is_best: Boolean indicating whether the current model parameters give the best validation score.
        :param filename: File name to save checkpoint. Default is checkpoint.pth.tar
        :param best_model_filename: File name to save the best model. Default is model_best.pth.tar
        :return:
        """

        torch.save(state, filename)
        if is_best:
            shutil.copyfile(filename, best_model_filename)

    def load_best_model(self, best_model_filename='model_best.pth.tar', standard_scaler_filename='scaler.pkl'):

        """
        Load the best model and the fitted StandardScaler.
        :param best_model_filename: File name of the best model. Default is model_best.pth.tar
        :param standard_scaler_filename: File name of the StandardScaler. Default is scaler.pkl
        :return: Nothing
        """

        checkpoint = torch.load(best_model_filename, map_location=self.device)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.standard_scaler = pickle.load(open(standard_scaler_filename, 'rb'))

    def train_model(self, X, y, filename='checkpoint.pth.tar', best_model_filename='model_best.pth.tar'):

        """
        Train the model.
        :param X: Input to the model. Tensor of shape [Number of samples, Input dimension].
        :param y: Labels. Tensor of shape [Number of samples].
        :param filename: Checkpoint file name. Default is checkpoint.pth.tar
        :param best_model_filename: Best model file name. Default is model_best.pth.tar
        :return: Nothing
        """

        M = X.shape[0]
        partition = self.generate_train_valid_partition(self.training_param_dict['Train/Val Split'], M)
        print(len(partition['train']))

        training_set = DatasetDataLoader(partition['train'], X, y)
        training_generator = torch.utils.data.DataLoader(training_set, **self.dataset_loader_params)

        validation_set = DatasetDataLoader(partition['validation'], X, y)
        validation_generator = torch.utils.data.DataLoader(validation_set, **self.dataset_loader_params)

        best_val_loss = -1.0

        for epoch in tqdm(range(self.training_param_dict['Epochs'])):

            valid_loss = torch.tensor(0.0).to(device=self.device)
            is_best = False

            train_outputs = None
            train_labels = None
            val_outputs = None
            val_labels = None

            # Switch to training mode
            self.model.train()

            for local_batch, local_labels in training_generator:

                # Transfer to GPU
                local_batch, local_labels = local_batch.to(self.device), local_labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(local_batch.float())

                if train_outputs is None:
                    train_outputs = torch.squeeze(outputs, 1)
                    train_labels = local_labels
                else:
                    train_outputs = torch.cat((train_outputs, torch.squeeze(outputs, 1)), 0)
                    train_labels = torch.cat((train_labels, local_labels), 0)

                loss = self.criterion(torch.squeeze(outputs, 1).float(), local_labels.float())

                l2_penalty, l1_penalty = 0.0, 0.0
                if self.training_param_dict['l2_weight'] != 0:
                    l2_penalty = self.training_param_dict['l2_weight'] * sum(
                        [(p ** 2).sum() for p in self.model.parameters()])

                if self.training_param_dict['l1_weight'] != 0:
                    l1_penalty = self.training_param_dict['l1_weight'] * sum(
                        [p.abs().sum() for p in self.model.parameters()])

                loss = loss + l2_penalty + l1_penalty

                loss.backward()
                self.optimizer.step()

            # Evaluation mode
            self.model.eval()
            with torch.no_grad():
                for local_batch, local_labels in validation_generator:

                    # Transfer to GPU
                    local_batch, local_labels = local_batch.to(self.device), local_labels.to(self.device)
                    outputs_val = self.model(local_batch.float())

                    loss = self.criterion(torch.squeeze(outputs_val, 1), local_labels)
                    valid_loss += loss

                    if val_outputs is None:
                        val_outputs = torch.squeeze(outputs_val, 1)
                        val_labels = local_labels
                    else:
                        val_outputs = torch.cat((val_outputs, torch.squeeze(outputs_val, 1)), 0)
                        val_labels = torch.cat((val_labels, local_labels), 0)

            valid_loss = self.criterion(val_outputs, val_labels)

            if epoch == 0:
                best_val_loss = valid_loss.item()
                is_best = True
            else:
                if valid_loss <= best_val_loss:
                    is_best = True
                    best_val_loss = valid_loss.item()

            # Save checkpoint
            self.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.state_dict(),
                'valid_loss': best_val_loss,
                'optimizer': self.optimizer.state_dict(),
                'train_loss': self.criterion(train_outputs, train_labels),
            }, is_best, filename, best_model_filename)

            if is_best:
                print(
                    f'Epoch: {epoch + 1} \t Training Loss: {self.criterion(train_outputs, train_labels)} \t '
                    f'Validation Loss: {self.criterion(val_outputs, val_labels)} \t Saving Checkpoint')
            else:
                print(
                    f'Epoch: {epoch + 1} \t Training Loss: {self.criterion(train_outputs, train_labels)} \t '
                    f'Validation Loss: {self.criterion(val_outputs, val_labels)}')

            if self.use_scheduler:
                self.scheduler.step()
