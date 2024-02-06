## Libraries
import csv
import os
import pandas as pd
import numpy as np
from typing import Union, Tuple
from tqdm import tqdm
# go one level down to import the data_loader and regression classes
# os.chdir('..')
import sys
print(sys.path)
sys.path.append('/Users/bhargobdeka/Desktop/cuTAGI') # always append the path to the root directory

from python_examples.data_loader import RegressionDataLoader
from python_examples.regression import Regression
from pytagi import NetProp

np.random.seed(1)

text_file_path = '/Users/bhargobdeka/Desktop/cuTAGI/data/UCI/Concrete/data/data.txt'

data_name = 'data/UCI/Concrete'

data = np.loadtxt(data_name + '/data/data.txt')

# We load the indexes for the features and for the target

index_features = np.loadtxt(data_name +'/data/index_features.txt').astype(int)
index_target   = np.loadtxt(data_name +'/data/index_target.txt').astype(int)

# print(index_features)
# print(index_target)

# splits
n_splits  = 20

# User-input
num_inputs = len(index_features)     # 1 explanatory variable
num_outputs = 1     # 1 predicted output
num_epochs = 40     # row for 50 epochs
BATCH_SIZE = 10     # batch size

# Input data and output data
X = data[ : , index_features.tolist() ]
Y = data[ : , index_target.tolist() ]
input_dim = X.shape[1]


## classes
class HeterosUCIMLP(NetProp):
    """Multi-layer preceptron for regression task where the
    output's noise varies overtime"""

    def __init__(self) -> None:
        super().__init__()
        self.layers =       [1, 1, 1]
        self.nodes =        [num_inputs, 50, 1]  # output layer = [mean, std]
        self.activations =  [0, 4, 0]
        self.batch_size =   BATCH_SIZE
        self.sigma_v =      0.3
        self.sigma_v_min =  0
        self.noise_gain =   1.0
        # self.noise_type =   "homosce" # "heteros" or "homosce"
        self.init_method =  "He"
        self.device =       "cpu"

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


mse_list = []
log_lik_list = []
rmse_list = []
for i in range(n_splits):
    index_train = np.loadtxt(data_name +"/data/index_train_{}.txt".format(i)).astype(int)
    index_test = np.loadtxt(data_name +"/data/index_test_{}.txt".format(i)).astype(int)

    # print(index_train)
    # print(index_test)

    #Check for intersection of elements
    ind = np.intersect1d(index_train,index_test)
    if len(ind)!=0:
        print('Train and test indices are not unique')
        break

    # Train and Test data for the current split
    x_train = X[ index_train.tolist(), ]
    y_train = Y[ index_train.tolist() ]
    y_train = np.reshape(y_train,[len(y_train),1]) #BD
    x_test  = X[ index_test.tolist(), ]
    y_test  = Y[ index_test.tolist() ]
    y_test = np.reshape(y_test,[len(y_test),1])    #BD

    # print(x_train.shape)

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
        raw_input=x_test, raw_output=y_test, batch_size=BATCH_SIZE
    )
    data_loader["x_norm_param_1"] = x_mean
    data_loader["x_norm_param_2"] = x_std
    data_loader["y_norm_param_1"] = y_mean
    data_loader["y_norm_param_2"] = y_std

    print(data_loader["train"][0].shape)


    # Model
    net_prop = HeterosUCIMLP()


    reg_data_loader = RegressionDataLoader(num_inputs=num_inputs,
                                       num_outputs=num_outputs,
                                       batch_size=net_prop.batch_size)


    reg_task = Regression(num_epochs=num_epochs,
                      data_loader=data_loader,
                      net_prop=net_prop)

    reg_task.train()
    # Predict for one split
    mse, log_lik, rmse = reg_task.predict()
    # Store the results
    mse_list.append(mse)
    log_lik_list.append(log_lik)
    rmse_list.append(rmse)

# Print the average results
print("Average MSE: ", np.mean(mse_list))
print("Average Log-likelihood: ", np.mean(log_lik_list))
print("Average RMSE: ", np.mean(rmse_list))



    
    