import numpy as np 
import os 
import matplotlib.pyplot as plt
from config_model import VALIDATION_CONFIG 


class DatasetValidator:

    def __init__(self, dataset_path):

        self.dataset_path = dataset_path

        self.X_train = None
        self.y_train = None

        self.X_val = None
        self.y_val = None

        self.X_test = None
        self.y_test = None


    def load_dataset(self):

        self.X_train = np.load(os.path.join(self.dataset_path, "X_train.npy"))
        self.y_train = np.load(os.path.join(self.dataset_path, "y_train.npy"))

        self.X_val = np.load(os.path.join(self.dataset_path, "X_val.npy"))
        self.y_val = np.load(os.path.join(self.dataset_path, "y_val.npy"))

        self.X_test = np.load(os.path.join(self.dataset_path, "X_test.npy"))
        self.y_test = np.load(os.path.join(self.dataset_path, "y_test.npy"))

        print("Datasets loaded successfully.")


    def check_shapes(self):

        expected_features = VALIDATION_CONFIG["expected_features"]
        expected_seq = VALIDATION_CONFIG["expected_seq_len"]

        for name, X in [
            ("Train", self.X_train),
            ("Validation", self.X_val),
            ("Test", self.X_test)
        ]:

            if X.shape[1] != expected_seq:
                raise ValueError(f"{name} sequence length mismatch.")

            if X.shape[2] != expected_features:
                raise ValueError(f"{name} feature count mismatch.")

        print("Shape validation passed.")


    def check_soc_bounds(self):

        soc_min = VALIDATION_CONFIG["soc_min"]
        soc_max = VALIDATION_CONFIG["soc_max"]

        for name, y in [
            ("Train", self.y_train),
            ("Validation", self.y_val),
            ("Test", self.y_test)
        ]:

            if np.any(y < soc_min) or np.any(y > soc_max):
                raise ValueError(f"{name} SOC values out of bounds.")

        print("SOC bounds validation passed.")


    def check_nan_values(self):

        datasets = [
            self.X_train,
            self.X_val,
            self.X_test,
            self.y_train,
            self.y_val,
            self.y_test
        ]

        for data in datasets:

            if np.isnan(data).any():
                raise ValueError("Dataset contains NaN values.")

            if np.isinf(data).any():
                raise ValueError("Dataset contains infinite values.")

        print("NaN and Inf validation passed.")


    def plot_distributions(self):

        voltage = self.X_train[:, :, 0].flatten()
        current = self.X_train[:, :, 1].flatten()
        temperature = self.X_train[:, :, 2].flatten()

        soc = self.y_train

        fig, axs = plt.subplots(2, 2, figsize=(10, 8))

        axs[0, 0].hist(voltage, bins=50)
        axs[0, 0].set_title("Voltage Distribution")

        axs[0, 1].hist(current, bins=50)
        axs[0, 1].set_title("Current Distribution")

        axs[1, 0].hist(temperature, bins=50)
        axs[1, 0].set_title("Temperature Distribution")

        axs[1, 1].hist(soc, bins=50)
        axs[1, 1].set_title("SOC Distribution")

        plt.tight_layout()
        plt.show()


    def run_full_validation(self):

        print("Starting dataset validation...")

        self.load_dataset()

        self.check_shapes()
        self.check_soc_bounds()
        self.check_nan_values()

        print("Dataset validation successful.")

        print("Generating distribution plots...")
        self.plot_distributions()