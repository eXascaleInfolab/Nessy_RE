"""Tune Hyperparameters for VAD.
Learning rate is fixed – lr = 1e-3
p_theta comes from VAE tuning
q_psi should be fine-tuned
"""

import argparse
import json
import os
import subprocess

from experiments_list import experiments_title, experiments_employee, experiments_top_members


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=str, help="Path to the dataset")
    parser.add_argument("dataset_name", type=str, help="TITLE, TOP_MEMBERS or EMPLOYEE")
    parser.add_argument("vocab_dir", type=str)
    parser.add_argument("--log_dir", type=str, default="logs")
    args = parser.parse_args()
    config = {
              "device": "/device:GPU:0",
              "vad_model": {
                            "model": "MultiTextVADTransfer",
                            "p_theta_dims_1": [32, 64],
                            "lr": 0.001
                           }
              }
    config_filename = "config_vad_hp.json"
    dims_list = [32]
    beta_list = [0.001, 0.01, 0.1, 1, 10]
    beta = 1
    alpha_list = [1, 10]
    seed = 31
    if args.dataset_name == "TITLE":
        experiments = experiments_title
    elif args.dataset_name == "EMPLOYEE":
        experiments = experiments_employee
    elif args.dataset_name == "TOP_MEMBERS":
        experiments = experiments_top_members

    for experiment_name, patterns in experiments.items():
        commandline = ["python",
                   os.path.join(
                       os.path.dirname(os.path.abspath(__file__)),
                       "train_vad.py"
                   ),
                   args.data_dir,
                   args.dataset_name,
                   os.path.join(
                       os.path.dirname(os.path.abspath(__file__)),
                       config_filename
                   ),
                   "--vocab_dir={}".format(args.vocab_dir),
                   "--print_every=25",
                   "--num_epoch=10",
                   "--rules={}".format(patterns),
                   "--seed={}".format(seed),
                   "--dropout=1",
                   "--log_dir={}".format(args.log_dir),
                   "--model_name={} {}".format(experiment_name, seed)
                   ]
        for qpsi_dims in dims_list:
            for alpha in alpha_list:
                config["vad_model"]["q_psi_dims_1"] = [qpsi_dims]
                config["vad_model"]["beta"] = beta
                config["vad_model"]["alpha"] = alpha
                with open(config_filename, "w") as f:
                    f.write(json.dumps(config, indent=2))
                print(f"Q dim = {qpsi_dims}, Beta = {beta}")
                subprocess.run(commandline)


if __name__ == "__main__":
    main()
