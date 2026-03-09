import os
import json
from datetime import datetime


class ExperimentManager:

    def __init__(self):

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        self.exp_dir = f"experiments/exp_{timestamp}"

        os.makedirs(self.exp_dir, exist_ok=True)


    def save_config(self, config):

        with open(f"{self.exp_dir}/config.json", "w") as f:

            json.dump(config, f, indent=4)


    def log_metrics(self, metrics):

        with open(f"{self.exp_dir}/metrics.json", "w") as f:

            json.dump(metrics, f, indent=4)