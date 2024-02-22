import cProfile
import pstats
from io import StringIO
from typing import Union, Tuple


# import memory_profiler
import numpy as np
from activation import ReLU
from data_loader import RegressionDataLoader
from linear import Linear
from conv2d import Conv2d
from pooling import AvgPool2d
from output_updater import OutputUpdater
from sequential import Sequential
from tqdm import tqdm

import pytagi.metric as metric
from pytagi import Normalizer as normalizer
from pytagi import HierarchicalSoftmax, Utils

import time
import csv
import os

# /home/bd/projects/cuTAGI

import sys
print(sys.path)
sys.path.append('/home/bd/projects/cuTAGI') # always append the path to the root directory

class Regression:
    """Test regression"""
    utils: Utils = Utils()

    def __init__(
        self,
        num_epochs: int,
        data_loader: dict,
        batch_size: int,
        num_inputs: int,
        num_hidden_layers: int,
        num_outputs: int,
        dtype=np.float32,
    ) -> None:
        self.num_epochs = num_epochs
        self.data_loader = data_loader
        self.batch_size = batch_size
        self.num_inputs = num_inputs
        self.num_hidden_layers = num_hidden_layers
        self.num_outputs = num_outputs
        self.dtype = dtype

        # FNN
        self.network = Sequential(
            Linear(self.num_inputs, self.num_hidden_layers), 
            ReLU(), 
            Linear(self.num_hidden_layers, self.num_hidden_layers), 
            ReLU(), 
            Linear(self.num_hidden_layers, 1)
        )

        # CNN
        #self.network = Sequential(
        #    Conv2d(1, 16, 4, padding=1, in_width=28, in_height=28),
        #    ReLU(),
        #    AvgPool2d(3, 2),
        #    Conv2d(16, 32, 5),
        #    ReLU(),
        #    AvgPool2d(3, 2),
        #    Linear(32 * 4 * 4, 100),
        #    ReLU(),
        #    Linear(100, 11),
        #)

        self.network.set_threads(8)
        self.network.to_device("cuda")


    def train(self) -> None:
        """Train the network using TAGI"""
        # Updater for output layer (i.e., equivalent to loss function)
        output_updater = OutputUpdater(self.network.device)

        # Inputs
        batch_size = self.batch_size

        # Outputs
        var_obs, _ = self.init_outputs(batch_size)
        print(var_obs)

        input_data, output_data = self.data_loader["train"]
        num_data = input_data.shape[0]
        num_iter = int(num_data / batch_size)
        pbar = tqdm(range(self.num_epochs))
        for epoch in pbar:
            
            var_obs = var_obs * 0.0 + 0.3**2
            for i in range(num_iter):
                # Get data
                idx = np.random.choice(num_data, size=batch_size)
                x_batch = input_data[idx, :]
                mu_obs_batch = output_data[idx, :]
                

                # Feed forward
                self.network(x_batch.flatten())

                # Update output layer
                output_updater.update(
                    output_states=self.network.output_z_buffer,
                    mu_obs=mu_obs_batch.flatten(),
                    var_obs=var_obs.flatten(),
                    delta_states=self.network.input_delta_z_buffer,
                )

                # Update hidden states
                self.network.backward()
                self.network.step()

                # Loss
                norm_pred, std_pred = self.network.get_outputs()
                print(norm_pred)
                print(std_pred)
                pred = normalizer.unstandardize(
                    norm_data=norm_pred,
                    mu=self.data_loader["y_norm_param_1"],
                    std=self.data_loader["y_norm_param_2"],
                )
                obs = normalizer.unstandardize(
                    norm_data=mu_obs_batch,
                    mu=self.data_loader["y_norm_param_1"],
                    std=self.data_loader["y_norm_param_2"],
                )
                mse = metric.mse(pred, obs)
                pbar.set_description(
                    f"Epoch# {epoch: 0}|{i * batch_size + len(x_batch):>5}|{num_data: 1}\t mse: {mse:>7.2f}"
                )
        pbar.close()

    def predict(self, std_factor: int = 1) -> None:
        """Make prediction using TAGI"""
        # Inputs
        batch_size = self.batch_size
        Sx_batch, Sx_f_batch = self.init_inputs(batch_size)

        mean_predictions = []
        variance_predictions = []
        y_test = []
        x_test = []
        for x_batch, y_batch in self.data_loader["test"]:
            # Predicitons
            self.network.forward(x_batch, Sx_batch)
            ma, Sa = self.network.get_outputs()
            print(f"The expected values are: {ma}")
            print(f"The variance values are: {Sa}")
            mean_predictions.append(ma)
            variance_predictions.append(Sa+0.3**2) #+ 0.3**2
            x_test.append(x_batch)
            y_test.append(y_batch)

        mean_predictions = np.stack(mean_predictions).flatten()
        std_predictions = (np.stack(variance_predictions).flatten()) ** 0.5
        y_test = np.stack(y_test).flatten()
        x_test = np.stack(x_test).flatten()

        # Unnormalization
        mean_predictions = normalizer.unstandardize(
            norm_data=mean_predictions,
            mu=self.data_loader["y_norm_param_1"],
            std=self.data_loader["y_norm_param_2"],
        )
        std_predictions = normalizer.unstandardize_std(
            norm_std=std_predictions, std=self.data_loader["y_norm_param_2"]
        )

        # x_test = normalizer.unstandardize(
        #     norm_data=x_test,
        #     mu=self.data_loader["x_norm_param_1"],
        #     std=self.data_loader["x_norm_param_2"],
        # )
        y_test = normalizer.unstandardize(
            norm_data=y_test,
            mu=self.data_loader["y_norm_param_1"],
            std=self.data_loader["y_norm_param_2"],
        )

        # Compute log-likelihood
        mse = metric.mse(mean_predictions, y_test)
        rmse = mse**0.5
        log_lik = metric.log_likelihood(
            prediction=mean_predictions, observation=y_test, std=std_predictions
        )

        print("#############")
        print(f"MSE           : {mse: 0.2f}")
        print(f"Log-likelihood: {log_lik: 0.2f}")

        return mse, rmse, log_lik


    def init_inputs(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray]:
        """Initnitalize the covariance matrix for inputs"""
        Sx_batch = np.zeros((batch_size, 1), dtype=self.dtype)

        Sx_f_batch = np.array([], dtype=self.dtype)

        return Sx_batch, Sx_f_batch

    def init_outputs(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray]:
        """Initnitalize the covariance matrix for outputs"""
        # Outputs
        V_batch = (
            np.zeros((batch_size, 1), dtype=self.dtype)
            + 0.3**2
        )
        ud_idx_batch = np.zeros((batch_size, 1), dtype=np.int32)

        return V_batch, ud_idx_batch


## Functions 

def create_data_loader(raw_input: np.ndarray, raw_output: np.ndarray, batch_size) -> list:
        """Create dataloader based on batch size"""
        num_input_data = raw_input.shape[0]
        num_output_data = raw_output.shape[0]
        assert num_input_data == num_output_data

        # Even indices
        even_indices = split_evenly(num_input_data, batch_size)

        if np.mod(num_input_data, batch_size) != 0:
            # Remider indices
            rem_indices = split_reminder(num_input_data, batch_size)
            even_indices.append(rem_indices)

        indices = np.stack(even_indices)
        input_data = raw_input[indices]
        output_data = raw_output[indices]
        dataset = []
        for x_batch, y_batch in zip(input_data, output_data):
            dataset.append((x_batch, y_batch))
        return dataset


def split_evenly(num_data, chunk_size: int):
    """split data evenly"""
    indices = np.arange(int(num_data - np.mod(num_data, chunk_size)))

    return np.split(indices, int(np.floor(num_data / chunk_size)))

def split_reminder(num_data: int, chunk_size: int):
        """Pad the reminder"""
        indices = np.arange(num_data)
        reminder_start = int(num_data - np.mod(num_data, chunk_size))
        num_samples = chunk_size - (num_data - reminder_start)
        random_idx = np.random.choice(indices, size=num_samples, replace=False)
        reminder_idx = indices[reminder_start:]

        return np.concatenate((random_idx, reminder_idx))


## Load the data
data_names = ["Boston_housing"]
# data_names = ["Boston_housing","Concrete","Energy", "Yacht", "Wine", \
#               "Kin8nm","Naval",\
#               "Power-plant","Protein"]

for j in range(len(data_names)):
    
    # check if the results folder already exists; create it if does not exist or remove the existing one
    if not os.path.exists("results_small_UCI_TAGI/{}".format(data_names[j])):
        os.makedirs("results_small_UCI_TAGI/{}".format(data_names[j]))
    elif os.path.isfile("results_small_UCI_TAGI/{}/RMSEtest.txt".format(data_names[j])) and \
        os.path.isfile("results_small_UCI_TAGI/{}/LLtest.txt".format(data_names[j])) and \
        os.path.isfile("results_small_UCI_TAGI/{}/runtime_train.txt".format(data_names[j])):
        
        os.remove("results_small_UCI_TAGI/{}/RMSEtest.txt".format(data_names[j]))
        os.remove("results_small_UCI_TAGI/{}/LLtest.txt".format(data_names[j]))
        os.remove("results_small_UCI_TAGI/{}/runtime_train.txt".format(data_names[j]))
    

    # File paths for the results
    RESULTS_RMSEtest = "results_small_UCI_TAGI/"+data_names[j]+"/RMSEtest.txt"
    RESULTS_LLtest = "results_small_UCI_TAGI/"+data_names[j]+"/LLtest.txt"
    RESULTS_RUNTIME = "results_small_UCI_TAGI/"+data_names[j]+"/runtime_train.txt"
    
    # getting data name
    data_name = '/home/bd/projects/cuTAGI/data/UCI/' + data_names[j]
    print(data_name)
    
    # load data
    data = np.loadtxt(data_name + '/data/data.txt')
    
    # We load the indexes for the features and for the target
    index_features = np.loadtxt(data_name +'/data/index_features.txt').astype(int)
    index_target   = np.loadtxt(data_name +'/data/index_target.txt').astype(int)
    
    

    # User-input
    n_splits = 20 # no. of splits
    num_inputs = len(index_features)  #len(index_features) 
    num_outputs = 1
    num_epochs = 40
    batch_size = 10
    num_hidden_layers = 50
    
    # Change batch size for wine and yacht
    if data_names[j] == "Yacht":
        BATCH_SIZE = 5
    
    # Change number of splits for Protein data to 5
    if data_names[j] == "Protein":
        n_splits = 5
        
    # sigma V values for each dataset
    sigma_v_values = {"Boston_housing": 0.3, "Concrete": 0.3, "Energy": 0.1, "Yacht": 0.1, "Wine": 0.7, \
                        "Kin8nm": 0.3, "Naval": 0.6, "Power-plant": 0.2, "Protein": 0.7}

    # Input data and output data
    X = data[ : , index_features.tolist() ]
    Y = data[ : , index_target.tolist() ]
    input_dim = X.shape[1]
    
    mse_list = []
    log_lik_list = []
    rmse_list = []
    runtime_list = []
    for i in range(n_splits):
        index_train = np.loadtxt(data_name +"/data/index_train_{}.txt".format(i)).astype(int)
        index_test = np.loadtxt(data_name +"/data/index_test_{}.txt".format(i)).astype(int)

        # Train and Test data for the current split
        x_train = X[ index_train.tolist(), ]
        y_train = Y[ index_train.tolist() ]
        y_train = np.reshape(y_train,[len(y_train),1]) #BD
        x_test  = X[ index_test.tolist(), ]
        y_test  = Y[ index_test.tolist() ]
        y_test = np.reshape(y_test,[len(y_test),1])    #BD
        
        # Normalizer
        from pytagi import Normalizer, Utils

        normalizer: Normalizer = Normalizer()

        x_mean, x_std = normalizer.compute_mean_std(x_train)
        y_mean, y_std = normalizer.compute_mean_std(y_train)

        # x_mean, x_std = normalizer.compute_mean_std(
        #     np.concatenate((x_train, x_test))
        # )
        # y_mean, y_std = normalizer.compute_mean_std(
        #     np.concatenate((y_train, y_test))
        # )


        x_train = normalizer.standardize(data=x_train, mu=x_mean, std=x_std)
        y_train = normalizer.standardize(data=y_train, mu=y_mean, std=y_std)
        x_test = normalizer.standardize(data=x_test, mu=x_mean, std=x_std)
        y_test = normalizer.standardize(data=y_test, mu=y_mean, std=y_std)

        print(x_train.shape)
        print(y_train.shape)
        
        # Dataloader
        data_loader = {}
        data_loader["train"] = (x_train, y_train)
        data_loader["test"] = create_data_loader(
            raw_input=x_test, raw_output=y_test, batch_size=batch_size
        )
        data_loader["x_norm_param_1"] = x_mean
        data_loader["x_norm_param_2"] = x_std
        data_loader["y_norm_param_1"] = y_mean
        data_loader["y_norm_param_2"] = y_std

        print(data_loader["train"][0].shape)
        
        reg_data_loader = RegressionDataLoader(num_inputs=num_inputs,
                                        num_outputs=num_outputs,
                                        batch_size=batch_size)


        reg_task = Regression(
            num_epochs=num_epochs, data_loader=data_loader, batch_size=batch_size, num_inputs=num_inputs, num_hidden_layers=num_hidden_layers, num_outputs=1
        )
        
        # Train the network
        start_time = time.time()
        reg_task.train()
        # time to run max epochs
        runtime = time.time()-start_time
        # Predict for one split
        mse, log_lik, rmse = reg_task.predict()
        # Store the results
        mse_list.append(mse)
        log_lik_list.append(log_lik)
        rmse_list.append(rmse)
        runtime_list.append(runtime)

    # Print the average results
    print("Average MSE: ", np.mean(mse_list))
    print("Average Log-likelihood: ", np.mean(log_lik_list))
    print("Average RMSE: ", np.mean(rmse_list))
    print("Average Runtime: ", np.mean(runtime_list))
    
    # Save the average results
    with open(RESULTS_RMSEtest, "a") as file:
        file.write(str(np.mean(rmse_list)) + "\n")
    with open(RESULTS_LLtest, "a") as file:
        file.write(str(np.mean(log_lik_list)) + "\n")
    with open(RESULTS_RUNTIME, "a") as file:
        file.write(str(np.mean(runtime_list)) + "\n")
    