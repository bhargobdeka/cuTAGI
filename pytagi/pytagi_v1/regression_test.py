import cProfile
import pstats
from io import StringIO
from typing import Union, Tuple

import fire
import memory_profiler
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



class Regression:
    """Test regression"""
    utils: Utils = Utils()

    def __init__(
        self,
        num_epochs: int,
        data_loader: dict,
        batch_size: int,
        dtype=np.float32,
    ) -> None:
        self.num_epochs = num_epochs
        self.data_loader = data_loader
        self.batch_size = batch_size
        self.dtype = dtype

        # FNN
        self.network = Sequential(
            Linear(1, 50), ReLU(), Linear(50, 50), ReLU(), Linear(50, 1)
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

        #self.network.set_threads(8)
        #self.network.to_device("cuda")


    def train(self) -> None:
        """Train the network using TAGI"""
        # Updater for output layer (i.e., equivalent to loss function)
        output_updater = OutputUpdater(self.network.device)

        # Inputs
        batch_size = self.batch_size

        # Outputs
        var_obs, _ = self.init_outputs(batch_size)

        input_data, output_data = self.data_loader["train"]
        num_data = input_data.shape[0]
        num_iter = int(num_data / batch_size)
        pbar = tqdm(range(self.num_epochs))
        for epoch in pbar:

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
                norm_pred, _ = self.network.get_outputs()
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

            mean_predictions.append(ma)
            variance_predictions.append(Sa + 0.3**2)
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

        x_test = normalizer.unstandardize(
            norm_data=x_test,
            mu=self.data_loader["x_norm_param_1"],
            std=self.data_loader["x_norm_param_2"],
        )
        y_test = normalizer.unstandardize(
            norm_data=y_test,
            mu=self.data_loader["y_norm_param_1"],
            std=self.data_loader["y_norm_param_2"],
        )

        # Compute log-likelihood
        mse = metric.mse(mean_predictions, y_test)
        log_lik = metric.log_likelihood(
            prediction=mean_predictions, observation=y_test, std=std_predictions
        )

        print("#############")
        print(f"MSE           : {mse: 0.2f}")
        print(f"Log-likelihood: {log_lik: 0.2f}")



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
            + 0.1**2
        )
        ud_idx_batch = np.zeros((batch_size, 0), dtype=np.int32)

        return V_batch, ud_idx_batch



# @memory_profiler.profile
def reg_runner():
    """Run classification training"""
    # User-input
    num_inputs = 1
    num_outputs = 1
    num_epochs = 50
    batch_size = 5
    x_train_file = "../../data/toy_example/x_train_1D_full_cov.csv"
    y_train_file = "../../data/toy_example/y_train_1D_full_cov.csv"
    x_test_file = "../../data/toy_example/x_test_1D_full_cov.csv"
    y_test_file = "../../data/toy_example/y_test_1D_full_cov.csv"

    # Data loader
    reg_data_loader = RegressionDataLoader(
        num_inputs=num_inputs, num_outputs=num_outputs, batch_size=batch_size
    )
    data_loader = reg_data_loader.process_data(
        x_train_file=x_train_file,
        y_train_file=y_train_file,
        x_test_file=x_test_file,
        y_test_file=y_test_file,
    )

    reg_task = Regression(
        num_epochs=num_epochs, data_loader=data_loader, batch_size=batch_size
    )
    reg_task.train()
    reg_task.predict() #std_factor=3



def memory_profiling_main():
    reg_runner()


def profiler():
    """Run profiler"""
    pr = cProfile.Profile()
    pr.enable()

    # Run the main function
    memory_profiling_main()

    pr.disable()
    s = StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats("time")
    ps.print_stats(20)  # Print only the top 20 functions

    # Print cProfile output to console
    print("Top 20 time-consuming functions:")
    print(s.getvalue())


def main(profile: bool = False):
    """Test API"""
    if profile:
        print("Profile training")
        profiler()
    else:
        reg_runner()


if __name__ == "__main__":
    fire.Fire(main)